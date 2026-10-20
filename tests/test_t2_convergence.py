"""T2 — convergence slope with a fixed step on S1 and S2 (spec §8)."""

from __future__ import annotations

import numpy as np
import pytest

from problems import S1, S2, Case, solve

N_STEPS = (40, 80, 160, 320, 640)
CASES = [pytest.param(c, id=c.name) for c in (S1, S2)]


def _endpoint_errors(backend: str, case: Case, precision: str, rtol: float, atol: float):
    y_ref = case.reference(np.array([case.t_end]))[0]
    hs = []
    errs = []
    for n in N_STEPS:
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
    hs, errs = _endpoint_errors(backend, case, "fp64", rtol=1e-7, atol=1e-9)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 4.5 <= slope <= 5.5, f"slope {slope:.3f}, errors {errs}"
    assert errs[-1] < 5e-9, f"err(N=640) = {errs[-1]:.3e}"


S2_FP32_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "measured on cpu: FP32 endpoint errors for N = 40..640 are "
        "5.29e-05, 2.84e-06, 5.71e-07, 2.71e-06, 5.71e-07; the FP32 floor is reached by "
        "N = 160 and the N = 320 point above 1e-6 is roundoff, so the fitted slope over the "
        "three kept points is 1.23. The spec filter (err > 1e-6) does not separate truncation "
        "from roundoff on S2; the assertion is left as specified for the spec owner to rule on."
    ),
)
CASES_FP32 = [
    pytest.param(S1, id=S1.name),
    pytest.param(S2, id=S2.name, marks=S2_FP32_XFAIL),
]


@pytest.mark.parametrize("case", CASES_FP32)
def test_convergence_slope_fp32(backend: str, case: Case) -> None:
    hs, errs = _endpoint_errors(backend, case, "fp32", rtol=1e-4, atol=1e-6)
    keep = errs > 1e-6
    assert keep.sum() >= 3, f"only {keep.sum()} points above 1e-6: {errs}"
    slope = np.polyfit(np.log(hs[keep]), np.log(errs[keep]), 1)[0]
    assert slope >= 4.0, f"slope {slope:.3f}, errors {errs}"
