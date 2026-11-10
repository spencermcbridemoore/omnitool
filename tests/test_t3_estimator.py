"""T3 — error-estimator scaling and non-vanishing, one step on L with Delta = 2 (spec §8).

The estimate must scale as ``h^5`` (``log2`` ratio between 4.5 and 5.5 per
halving), must bound the true one-step error, and must not vanish on a
linear constant-coefficient problem (the ROS3P failure of ADR 0003).
"""

from __future__ import annotations

import numpy as np

from omnitool.reference import dopri5_numpy as ref
from problems import L_BY_DELTA

H_VALUES = (0.5, 0.25, 0.125, 0.0625, 0.03125)
CASE = L_BY_DELTA[2.0]


def one_step(backend: str, h: float) -> tuple[np.ndarray, np.ndarray]:
    """``(y_1, h (b - b̂)·k)`` after one DOPRI5 step from ``y0`` on ``backend``."""
    y0 = CASE.y0.astype(np.float64)
    p = CASE.params.astype(np.float64)
    if backend == "cuda":
        from omnitool.backends.cuda import step_cuda

        y1, err_vec = step_cuda(CASE.problem, y0[None, :], p[None, :], 0.0, h, np.float64)
        return y1[0], err_vec[0]
    assert backend == "cpu", backend
    rhs = CASE.problem.rhs_numpy
    k1 = rhs(np.float64(0.0), y0, p)
    y1, _k, err_vec = ref.step(rhs, np.float64(0.0), y0, np.float64(h), p, k1, np.float64)
    return y1, err_vec


def test_estimator_scaling_and_bound(backend: str) -> None:
    est = []
    true = []
    for h in H_VALUES:
        y1, err_vec = one_step(backend, h)
        y_exact = CASE.reference(np.array([h]))[0]
        est.append(float(np.max(np.abs(err_vec))))
        true.append(float(np.max(np.abs(y1 - y_exact))))
    for i in range(len(H_VALUES) - 1):
        ratio = np.log2(est[i] / est[i + 1])
        assert 4.5 <= ratio <= 5.5, f"h={H_VALUES[i]}: log2 ratio {ratio:.3f}, est={est}"
    for h, e, t in zip(H_VALUES, est, true, strict=True):
        assert e >= t, f"h={h}: estimate {e:.3e} below true error {t:.3e}"
        assert e > 1e-12, f"h={h}: estimate {e:.3e} has vanished"
