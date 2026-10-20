"""T2 — convergence slope with a fixed step on S1 and S2 (spec §8)."""

from __future__ import annotations

import numpy as np
import pytest

from problems import S1, S2, Case, solve

N_STEPS_FP64 = (40, 80, 160, 320, 640)
# Coarser steps leave the asymptotic regime; from N = 160 on, FP32 roundoff is
# of the order of the truncation error (spec T2, FP32 variant).
N_STEPS_FP32 = (20, 40, 80)
CASES = [pytest.param(c, id=c.name) for c in (S1, S2)]


def _endpoint_errors(
    backend: str,
    case: Case,
    precision: str,
    rtol: float,
    atol: float,
    n_steps: tuple[int, ...],
):
    y_ref = case.reference(np.array([case.t_end]))[0]
    hs = []
    errs = []
    for n in n_steps:
        h = case.t_end / n
        res = solve(
            backend,
            case,
            precision=precision,
            t_save=np.array([0.0, case.t_end]),
            rtol=rtol,
            atol=atol,
            fixed_step=h,
        )
        assert res.status[0] == 0
        assert res.n_rejected[0] == 0
        hs.append(h)
        errs.append(float(np.max(np.abs(res.y[0, -1].astype(np.float64) - y_ref))))
    return np.array(hs), np.array(errs)


@pytest.mark.parametrize("case", CASES)
def test_convergence_slope_fp64(backend: str, case: Case) -> None:
    hs, errs = _endpoint_errors(backend, case, "fp64", rtol=1e-7, atol=1e-9, n_steps=N_STEPS_FP64)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 4.5 <= slope <= 5.5, f"slope {slope:.3f}, errors {errs}"
    assert errs[-1] < 5e-9, f"err(N=640) = {errs[-1]:.3e}"


@pytest.mark.parametrize("case", CASES)
def test_convergence_slope_fp32(backend: str, case: Case) -> None:
    hs, errs = _endpoint_errors(backend, case, "fp32", rtol=1e-4, atol=1e-6, n_steps=N_STEPS_FP32)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert slope >= 4.0, f"slope {slope:.3f}, errors {errs}"
    assert errs[-1] < 1e-4, f"err(N=80) = {errs[-1]:.3e}"
