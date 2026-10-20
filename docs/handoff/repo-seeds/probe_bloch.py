"""Probe cubie's per-run adaptive stepping on a Doppler-detuned two-level
optical Bloch system (the smallest analogue of the RydIQule stack).

Runs under NUMBA_ENABLE_CUDASIM=1 on CPU.  Each run has a different
detuning Delta; the per-save iteration counters tell us how many steps each
run took, which is the whole point of per-member adaptive stepping.

Usage: NUMBA_ENABLE_CUDASIM=1 python -I probe_bloch.py [method] [precision]
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp

os.environ.setdefault("CUBIE_CACHE_DIR", os.path.join(os.path.dirname(__file__), "generated"))

import cubie as qb  # noqa: E402

METHOD = sys.argv[1] if len(sys.argv) > 1 else "rk45"
PREC = np.float64 if (len(sys.argv) > 2 and sys.argv[2] == "f64") else np.float32
DELTAS = np.array([0.0, 2.0, 20.0, 200.0])   # units of Gamma
OMEGA, GAMMA = 1.0, 1.0
DURATION, SAVE_EVERY = 20.0, 2.0
RTOL, ATOL = 1e-7, 1e-9

BLOCH = qb.create_ODE_system(
    """
    du = -0.5*G*u + D*v
    dv = -D*u - 0.5*G*v - W*w
    dw = W*v - G*(w + 1.0)
    """,
    parameters={"G": GAMMA, "W": OMEGA, "D": 0.0},
    states={"u": 0.0, "v": 0.0, "w": -1.0},
    precision=PREC,
    name="Bloch2L",
)

t_start = time.perf_counter()
res = qb.solve_ivp(
    BLOCH,
    y0={"u": 0.0, "v": 0.0, "w": -1.0},
    parameters={"D": DELTAS},
    method=METHOD,
    duration=DURATION,
    save_every=SAVE_EVERY,
    output_types=["state", "time", "iteration_counters"],
    rtol=RTOL,
    atol=ATOL,
    precision=PREC,
    time_logging_level=None,
)
elapsed = time.perf_counter() - t_start
print(f"method={METHOD} precision={PREC.__name__} wall={elapsed:.1f}s (CUDASIM)")
print("status codes:", res.status_codes, res.status_messages if hasattr(res, "status_messages") else "")

state = np.asarray(res.state)            # (time, var, run)
time_arr = np.asarray(res.time)
print("res.time shape", time_arr.shape, "first rows:", time_arr.reshape(time_arr.shape[0], -1)[:3].tolist())
tgrid = state[:, 3, 0] if state.shape[1] > 3 else np.arange(0.0, DURATION + 1e-9, SAVE_EVERY)
counters = np.asarray(res.iteration_counters)  # (n_saves, 4, n_runs): newton, krylov, steps, rejected
print("state shape", state.shape, "tgrid", tgrid[:3], "...", tgrid[-1])
print("counters shape", counters.shape)

# Reference: scipy DOP853 at tight tolerance, per run.
def rhs(t, y, D):
    u, v, w = y
    return [-0.5 * GAMMA * u + D * v, -D * u - 0.5 * GAMMA * v - OMEGA * w, OMEGA * v - GAMMA * (w + 1.0)]

print(f"\n{'Delta':>7} {'steps':>7} {'rejected':>8} {'newton':>7} {'krylov':>7} {'max|err| (vs DOP853 1e-13)':>28}")
for r, D in enumerate(DELTAS):
    ref = scipy_solve_ivp(rhs, (0.0, DURATION), [0.0, 0.0, -1.0], args=(D,), method="DOP853",
                          rtol=1e-13, atol=1e-15, t_eval=tgrid)
    # cubie saves variables in state order: u, v, w
    err = np.max(np.abs(state[:, :3, r] - ref.y.T))
    steps = counters[:, 2, r].sum()
    rej = counters[:, 3, r].sum()
    newt = counters[:, 0, r].sum()
    kry = counters[:, 1, r].sum()
    print(f"{D:7.1f} {steps:7d} {rej:8d} {newt:7d} {kry:7d} {err:28.3e}")
