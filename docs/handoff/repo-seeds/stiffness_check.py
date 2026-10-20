"""Is RydIQule's Doppler-broadened time solve stiff? (rydiqule 2.1.3, SciPy 1.17)

Problem: the 4-level Rb ladder from rq_setup.py, with the probe (780 nm) and coupling (480 nm)
beams counter-propagating, RydIQule's default 561-class Doppler mesh, the RF field switched on at
t = 0 with 30 % amplitude modulation at 0.5 MHz, 4 us, and RydIQule's defaults (RK45,
rtol = atol = 1e-6).

Part 1 integrates single velocity classes with explicit and implicit methods.
Part 2 runs stock rq.solve_time over the whole stack: the default RK45, then method="Radau" with a
block-diagonal jac_sparsity (velocity classes are uncoupled). solve_time forwards extra keyword
arguments to scipy.integrate.solve_ivp. Pass n_slices=1, or build the pattern per slice, because
the pattern must match the size of each solve.

Expected output on a 2-core cloud VM (timings scale with your CPU; step counts do not):

  class      shift Mrad/s |  RK45 steps   nfev  max err | Radau steps nfev  LU  max err
  resonant              0 |         45     278  4.0e-06 |         50   394  44  8.1e-07
  0.4 vP              780 |       2525   15176  7.7e-05 |          6    44  12  4.7e-08
  1 vP               1949 |       7228   43394  1.6e-04 |          6    44  12  3.0e-09
  2 vP               3899 |      15906   95468  2.6e-04 |          6    44  12  3.7e-10

  RK45 (RydIQule default)       : 45.3 s  nfev  95408  probe-signal error 2.0e-06 of RF swing
  Radau + block jac_sparsity    :  3.0 s  nfev   1471  probe-signal error 8.7e-07 of RF swing
  (rtol=1e-7, atol=1e-9)

"probe-signal error" is the max over time of the error in the Doppler-summed Im(rho_01), divided
by its peak-to-peak change over the run. The reference is Radau at rtol=1e-10, atol=1e-12. A
per-class DOP853 reference (rtol=1e-11, atol=1e-13) over all 561 classes agrees with it to
4e-14 absolute (1.5e-9 of the swing).

Two more results from the full per-class runs (not repeated here, ~6 min):
  - Per-class RK45 needs 4.6x fewer member-RHS evaluations in total than the shared-step solve;
    step count tracks the Doppler shift (r = 0.999).
  - In the shared-step solve, the 2 vP class ends with max error 4.4e-3, 17x its stand-alone
    error, because SciPy's error test is an RMS over all 8,415 unknowns.
"""
import time

import numpy as np
import rydiqule as rq
import rydiqule.stack_solvers.scipy_solver as ss
import scipy.sparse as sp
from scipy.integrate import solve_ivp as _solve_ivp

from rq_setup import (GAMMA_P, GAMMA_RYD, GAMMA_TRANSIT, RABI_COUPLE, RABI_PROBE,
                      obs_index)

TWO_PI = 2 * np.pi
K_PROBE = TWO_PI / 780e-9 * 1e-6    # Mrad/m
K_COUPLE = TWO_PI / 480e-9 * 1e-6   # Mrad/m
VP = 242.0                          # m/s, Rb-85 most probable speed near room temperature
END_TIME, NUM_PTS = 4.0, 401        # us

nfev_log = []


def _counting_solve_ivp(*args, **kwargs):
    sol = _solve_ivp(*args, **kwargs)
    nfev_log.append(sol.nfev)
    return sol


ss.solve_ivp = _counting_solve_ivp   # RydIQule's scipy backend calls this name


def make_sensor():
    s = rq.Sensor(4, vP=VP)
    s.add_coupling((0, 1), detuning=0.0, rabi_frequency=RABI_PROBE,
                   kvec=(K_PROBE, 0, 0), label="probe")
    s.add_coupling((1, 2), detuning=0.0, rabi_frequency=RABI_COUPLE,
                   kvec=(-K_COUPLE, 0, 0), label="couple")
    s.add_coupling((2, 3), detuning=0.0, rabi_frequency=5.0, label="rf",
                   time_dependence=lambda t: 1.0 + 0.3 * np.sin(TWO_PI * 0.5 * t))
    s.add_decoherence((1, 0), GAMMA_P)
    s.add_decoherence((2, 1), GAMMA_RYD)
    s.add_decoherence((3, 2), GAMMA_RYD)
    s.add_transit_broadening(GAMMA_TRANSIT)
    return s


def capture_stack(sensor):
    """Grab the stacked equations through solve_time's public `solver=` hook, without solving."""
    cap = {}

    def capture_only(eoms_base, const, eom_time_r, const_r, eom_time_i, const_i,
                     time_inputs, t_eval, init_cond, **kw):
        cap.update(A=eoms_base, b=const, Br=eom_time_r, br=const_r, Bi=eom_time_i,
                   bi=const_i, f=time_inputs, y0=init_cond)
        return np.zeros(eoms_base.shape[:-1] + (len(t_eval),))

    rq.solve_time(sensor, END_TIME, NUM_PTS, doppler=True, sum_doppler=False, n_slices=1,
                  solver=capture_only)
    return cap


def part1(sensor):
    cap = capture_stack(sensor)
    n = cap["A"].shape[-1]
    A = cap["A"].reshape(-1, n, n)
    b = np.broadcast_to(cap["b"], cap["A"].shape[:-1]).reshape(-1, n)
    y0 = np.broadcast_to(cap["y0"], cap["A"].shape[:-1]).reshape(-1, n)
    vel = rq.doppler_utils.doppler_classes()
    t_eval = np.linspace(0, END_TIME, NUM_PTS)
    print("class      shift Mrad/s |  RK45 steps   nfev  max err | Radau steps nfev  LU  max err")
    for label, v in [("resonant", 0.0), ("0.4 vP", 0.4), ("1 vP", 1.0), ("2 vP", 2.0)]:
        i = int(np.argmin(np.abs(vel - v)))
        # _derEqn_comp is RydIQule's (internal) per-stack right-hand side
        rhs = ss._derEqn_comp(A[i:i + 1], b[i:i + 1], cap["Br"], cap["br"], cap["Bi"],
                              cap["bi"], cap["f"])
        ref = _solve_ivp(rhs, (0, END_TIME), y0[i], method="DOP853", rtol=1e-11, atol=1e-13,
                         t_eval=t_eval).y
        row = f"{label:9s} {abs(vel[i]) * VP * K_PROBE:13.0f} |"
        for m in ("RK45", "Radau"):
            s = _solve_ivp(rhs, (0, END_TIME), y0[i], method=m, rtol=1e-6)
            y = _solve_ivp(rhs, (0, END_TIME), y0[i], method=m, rtol=1e-6, t_eval=t_eval).y
            err = np.abs(y - ref).max()
            if m == "RK45":
                row += f" {len(s.t) - 1:10d} {s.nfev:7d}  {err:.1e} |"
            else:
                row += f" {len(s.t) - 1:10d} {s.nfev:5d} {s.nlu:3d}  {err:.1e}"
        print(row)


def part2(sensor):
    k, _ = obs_index(sensor)
    M, n = len(rq.doppler_utils.doppler_classes()), sensor.basis_size**2 - 1
    pattern = sp.block_diag([np.ones((n, n))] * M, format="csc")

    def run(**kw):
        t0 = time.perf_counter()
        sol = rq.solve_time(sensor, END_TIME, NUM_PTS, doppler=True, n_slices=1, **kw)
        return sol.rho[..., k], time.perf_counter() - t0, nfev_log[-1]

    ref, _, _ = run(method="Radau", jac_sparsity=pattern, rtol=1e-10, atol=1e-12)
    swing = np.ptp(ref)
    print()
    for label, kw in [("RK45 (RydIQule default)", {}),
                      ("Radau + block jac_sparsity",
                       dict(method="Radau", jac_sparsity=pattern, rtol=1e-7, atol=1e-9))]:
        obs, wall, nfev = run(**kw)
        err = np.abs(obs - ref).max() / swing
        print(f"{label:30s}: {wall:4.1f} s  nfev {nfev:6d}  probe-signal error {err:.1e} of RF swing")
    print("(rtol=1e-7, atol=1e-9)")


if __name__ == "__main__":
    s = make_sensor()
    part1(s)
    part2(s)
