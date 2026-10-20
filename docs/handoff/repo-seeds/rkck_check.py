"""Check the Cash-Karp step in CUDA_UCN_SPIN-master against a corrected one (CPU port, no GPU).

CUDA_rkck_SPIN (CUDA_RK45_SPIN.cu, lines 224-235) and CUDA_rkck_XVS (UCN_CUDA_ALL_KERNEL.cu,
lines 576-587) compute each stage input (spinortemp / xvstemp) but then evaluate stages 2, 3, 4
and 6 at the step's starting state, derivs(t, spinor, ...), and stage 5 at time t instead of
t + h. This file ports that step line for line and compares it with the textbook step, using
the same Numerical Recipes controller and tolerances the host sets in UCN_RUN.cpp:
SAFETY 0.9, PGROW -0.2, PSHRNK -0.25, MAXERR1 1e-12, TINY1 1e-13, and a first step of 1e-9 s.

Test problem: the SPIN kernel's spinor equations (CUDA_derivs_SPIN, lines 195-198) in a constant
1 T field, RF off. This is linear, so expm gives the exact answer.

Expected output (about 25 s; most of it is the as-written adaptive run):

  fixed-step global error at T = 1e-7 s
     N    as written      fixed
    100   4.9e-01        8.1e-09
    200   2.2e-01        2.5e-10     slopes 1.14  5.01
    400   1.1e-01        7.8e-12     slopes 1.07  5.00
    800   5.2e-02        2.5e-13     slopes 1.03  4.96
  as written, one step: error estimate / true local error = 0.0386

  kernel controller, 20 ns at 1 T (target: 1e-12 relative per step)
    fixed     :    126 accepted,  2 rejected, max error 2.5e-13
    as written: 331479 accepted,  7 rejected, max error 4.7e-06

  polcalc on the host's classical spin layout (1, sx, sy, sz):
    spin along z -> P_z = 0.0, spin along x -> P_z = 1.0, spin along y -> P_x = 1.0
"""
import time

import numpy as np
from scipy.linalg import expm

MU, HBAR = -9.662364e-27, 1.0545717e-34                 # UCN_RUN.cpp defaults
B = np.array([np.sin(0.3) * np.cos(0.7), np.sin(0.3) * np.sin(0.7), np.cos(0.3)])  # 1 T
b = MU / HBAR                                            # btoomega
M = b * np.array([[0.0,   B[2], -B[1],  B[0]],
                  [-B[2], 0.0,  -B[0], -B[1]],
                  [B[1],  B[0],  0.0,  -B[2]],
                  [-B[0], B[1],  B[2],  0.0]])
y0 = np.array([1.0, 0.0, 0.0, 0.0])


def f(t, y):
    return M @ y


b21 = 0.2; b31, b32 = 3/40, 9/40; b41, b42, b43 = 0.3, -0.9, 1.2
b51, b52, b53, b54 = -11/54, 2.5, -70/27, 35/27
b61, b62, b63, b64, b65 = 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096
c1, c3, c4, c6 = 37/378, 250/621, 125/594, 512/1771
dc5 = -277/14336
dc1, dc3, dc4, dc6 = c1 - 2825/27648, c3 - 18575/48384, c4 - 13525/55296, c6 - 0.25


def rkck_as_written(t, y, h):
    k1 = f(t, y)
    k2 = f(t, y)                                          # derivs(t, spinor, ak2)
    k3 = f(t, y)                                          # derivs(t, spinor, ak3)
    k4 = f(t, y)                                          # derivs(t, spinor, ak4)
    k5 = f(t, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))  # derivs(t, spinortemp, ak5)
    k6 = f(t, y)                                          # derivs(t, spinor, ak6)
    return (y + h*(c1*k1 + c3*k3 + c4*k4 + c6*k6),
            h*(dc1*k1 + dc3*k3 + dc4*k4 + dc5*k5 + dc6*k6))


def rkck_fixed(t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.2*h, y + h*b21*k1)
    k3 = f(t + 0.3*h, y + h*(b31*k1 + b32*k2))
    k4 = f(t + 0.6*h, y + h*(b41*k1 + b42*k2 + b43*k3))
    k5 = f(t + h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
    k6 = f(t + 0.875*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
    return (y + h*(c1*k1 + c3*k3 + c4*k4 + c6*k6),
            h*(dc1*k1 + dc3*k3 + dc4*k4 + dc5*k5 + dc6*k6))


def fixed_step_order():
    T = 1e-7
    print(f"fixed-step global error at T = {T:.0e} s")
    print("   N    as written      fixed")
    prev = None
    for N in (100, 200, 400, 800):
        h, errs = T / N, []
        for step in (rkck_as_written, rkck_fixed):
            y = y0.copy()
            for i in range(N):
                y, _ = step(i * h, y, h)
            errs.append(np.abs(y - expm(M * T) @ y0).max())
        tail = "" if prev is None else (f"     slopes {np.log2(prev[0] / errs[0]):.2f}  "
                                        f"{np.log2(prev[1] / errs[1]):.2f}")
        print(f"  {N:4d}   {errs[0]:.1e}        {errs[1]:.1e}{tail}")
        prev = errs
    h = 1e-10
    y1, est = rkck_as_written(0.0, y0, h)
    print(f"as written, one step: error estimate / true local error = "
          f"{np.abs(est).max() / np.abs(y1 - expm(M * h) @ y0).max():.4f}")


def kernel_controller(step, t_end):
    """CUDA_rkqs_SINGLE_ATTEMPT_SPIN's accept/reject logic, one attempt per loop pass."""
    SAFETY, PGROW, PSHRNK, MAXERR1, TINY1, HMAX = 0.9, -0.2, -0.25, 1e-12, 1e-13, 1e-6
    t, y, h, acc, rej = 0.0, y0.copy(), 1e-9, 0, 0
    while t_end - t > 1e-12 * t_end:
        h = min(h, t_end - t)
        scal = np.abs(y) + np.abs(f(t, y) * h) + TINY1
        ytmp, yerr = step(t, y, h)
        err = np.max(np.abs(yerr / scal)) / MAXERR1
        if err <= 1.0:
            t, y, acc = t + h, ytmp, acc + 1
            h = min(SAFETY * h * err**PGROW if err > 0 else 5 * h, 5 * h, HMAX)
        else:
            rej += 1
            h = max(SAFETY * h * err**PSHRNK, 0.1 * h)
    return acc, rej, np.abs(y - expm(M * t) @ y0).max()


def polcalc(Bv, s):
    """CUDA_polcalc_SPIN / CUDA_polcalc_XVS (spinor mapping)."""
    bx, by, bz = Bv / np.linalg.norm(Bv)
    n = s @ s
    return ((2 * (s[0] * s[2] + s[1] * s[3]) / n) * bx + (2 * (s[0] * s[3] - s[1] * s[2]) / n) * by
            + ((s[0]**2 + s[1]**2 - s[2]**2 - s[3]**2) / n) * bz)


if __name__ == "__main__":
    fixed_step_order()
    print("\nkernel controller, 20 ns at 1 T (target: 1e-12 relative per step)")
    for name, step in (("fixed", rkck_fixed), ("as written", rkck_as_written)):
        t0 = time.perf_counter()
        acc, rej, err = kernel_controller(step, 2e-8)
        print(f"  {name:10s}: {acc:6d} accepted, {rej:2d} rejected, max error {err:.1e}"
              f"   ({time.perf_counter() - t0:.0f} s)")
    print("\npolcalc on the host's classical spin layout (1, sx, sy, sz), UCN_RUN.cpp lines 198-205:")
    z, x = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    print(f"  spin along z -> P_z = {polcalc(z, np.array([1.0, 0, 0, 1])):.1f}, "
          f"spin along x -> P_z = {polcalc(z, np.array([1.0, 1, 0, 0])):.1f}, "
          f"spin along y -> P_x = {polcalc(x, np.array([1.0, 0, 1, 0])):.1f}")
