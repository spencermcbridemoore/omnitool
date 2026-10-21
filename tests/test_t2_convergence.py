"""T2 — convergence slope with a fixed step on S1 and S2 (spec §8).

FP64 is the order test: the endpoint error must fall as ``h^5``. FP32 is a
precision-consistency test: the FP32 and FP64 endpoints must agree to
``100 * eps32 * max(1, ymax)`` at every step count, and a slope is fitted
only over the points whose FP32 error is still above ``100 * eps32``, when
three or more such points exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from problems import S1, S2, Case, solve

N_STEPS_FP64 = (40, 80, 160, 320, 640)
N_STEPS_FP32 = (20, 40, 80, 160, 320, 640)
EPS32 = float(np.finfo(np.float32).eps)
FLOOR_FP32 = 100 * EPS32
CASES = [pytest.param(c, id=c.name) for c in (S1, S2)]


def _endpoints(
    backend: str,
    case: Case,
    precision: str,
    n_steps: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """``(h, y(T))`` per step count, ``y`` upcast to float64."""
    hs = []
    ys = []
    for n in n_steps:
        h = case.t_end / n
        res = solve(
            backend,
            case,
            precision=precision,
            t_save=np.array([0.0, case.t_end]),
            rtol=1e-4,
            atol=1e-6,
            fixed_step=h,
        )
        assert res.status[0] == 0
        assert res.n_rejected[0] == 0
        hs.append(h)
        ys.append(res.y[0, -1].astype(np.float64))
    return np.array(hs), np.array(ys)


@pytest.mark.parametrize("case", CASES)
def test_convergence_slope_fp64(backend: str, case: Case) -> None:
    y_ref = case.reference(np.array([case.t_end]))[0]
    hs, ys = _endpoints(backend, case, "fp64", N_STEPS_FP64)
    errs = np.max(np.abs(ys - y_ref), axis=1)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 4.5 <= slope <= 5.5, f"slope {slope:.3f}, errors {errs}"
    assert errs[-1] < 5e-9, f"err(N=640) = {errs[-1]:.3e}"


@pytest.mark.parametrize("case", CASES)
def test_precision_consistency_fp32(backend: str, case: Case) -> None:
    y_ref = case.reference(np.array([case.t_end]))[0]
    ymax = float(np.max(np.abs(y_ref)))
    hs, y32 = _endpoints(backend, case, "fp32", N_STEPS_FP32)
    _, y64 = _endpoints(backend, case, "fp64", N_STEPS_FP32)
    gap = np.max(np.abs(y32 - y64), axis=1)
    bound = FLOOR_FP32 * max(1.0, ymax)
    assert np.all(gap <= bound), f"FP32/FP64 gap {gap} exceeds {bound:.3e} at N={N_STEPS_FP32}"
    errs = np.max(np.abs(y32 - y_ref), axis=1)
    keep = errs > FLOOR_FP32
    if keep.sum() >= 3:
        slope = np.polyfit(np.log(hs[keep]), np.log(errs[keep]), 1)[0]
        assert slope >= 4.0, f"slope {slope:.3f} over {keep.sum()} points, errors {errs}"
