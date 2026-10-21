"""T4 — agreement with the reference under tolerance-scaled bounds (spec §8).

Adaptive runs on every problem, every save time of the 51-point grid, both
precisions, two tolerance pairs each. The save times are almost never step
endpoints, so this is also the T5 clause on dense output at save times.
"""

from __future__ import annotations

import numpy as np
import pytest

from omnitool import Status
from problems import ALL_CASES, Case, reference_on_grid, save_grid, solve

TOLERANCE_PAIRS = {
    "fp64": [(1e-7, 1e-9), (1e-10, 1e-12)],
    "fp32": [(1e-4, 1e-6), (2e-5, 1e-7)],  # rtol floor is 100 * eps32 = 1.19e-5
}
BOUND_FACTOR = 10.0
CASES = [pytest.param(c, id=c.name) for c in ALL_CASES]


@pytest.mark.parametrize("pair_index", [0, 1], ids=["tol0", "tol1"])
@pytest.mark.parametrize("case", CASES)
def test_tolerance_scaled_agreement(
    backend: str, precision: str, case: Case, pair_index: int
) -> None:
    rtol, atol = TOLERANCE_PAIRS[precision][pair_index]
    t_save = save_grid(case.t_end)
    y_ref = reference_on_grid(case, t_save)
    ymax = float(np.max(np.abs(y_ref)))
    bound = BOUND_FACTOR * (rtol * ymax + atol)
    res = solve(backend, case, precision=precision, t_save=t_save, rtol=rtol, atol=atol)
    assert res.status[0] == Status.SUCCESS
    err = np.max(np.abs(res.y[0].astype(np.float64) - y_ref))
    assert err <= bound, f"max error {err:.3e} exceeds {bound:.3e} (rtol={rtol}, atol={atol})"
