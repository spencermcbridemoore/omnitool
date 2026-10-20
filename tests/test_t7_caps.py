"""T7 — step and evaluation caps on the adaptive runs (spec §8).

The lower bounds catch a dead estimator; the upper bounds catch a controller
that thrashes. The counter identity ties ``n_fev`` to the step counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from omnitool import Status
from omnitool.reference import dopri5_numpy as ref
from problems import L_BY_DELTA, S1, S2, Case, solve

TOLERANCES = {"fp64": (1e-7, 1e-9), "fp32": (1e-4, 1e-6)}

# (case, fp64 n_fev range, fp32 n_fev range)
ROWS = [
    pytest.param(L_BY_DELTA[0.0], (150, 500), (50, 200), id="L_delta0"),
    pytest.param(L_BY_DELTA[20.0], (2500, 8500), (800, 2500), id="L_delta20"),
    pytest.param(L_BY_DELTA[200.0], (20000, 60000), (5000, 16000), id="L_delta200"),
    pytest.param(S1, (700, 2200), (200, 700), id="S1_chirp"),
    pytest.param(S2, (400, 1200), (100, 400), id="S2_duffing"),
]


@pytest.mark.parametrize(("case", "fp64_range", "fp32_range"), ROWS)
def test_evaluation_caps(
    backend: str,
    precision: str,
    case: Case,
    fp64_range: tuple[int, int],
    fp32_range: tuple[int, int],
) -> None:
    rtol, atol = TOLERANCES[precision]
    lo, hi = fp64_range if precision == "fp64" else fp32_range
    res = solve(backend, case, precision=precision, rtol=rtol, atol=atol)
    n_steps = int(res.n_accepted[0]) + int(res.n_rejected[0])
    assert res.status[0] == Status.SUCCESS
    assert res.n_fev[0] == 1 + 6 * n_steps + 1, (res.n_fev, res.n_accepted, res.n_rejected)
    assert res.n_rejected[0] <= 0.3 * res.n_accepted[0], (res.n_accepted, res.n_rejected)
    assert lo <= res.n_fev[0] <= hi, f"n_fev = {res.n_fev[0]} outside [{lo}, {hi}]"
    assert np.all(np.isfinite(res.y))
    assert res.h_last[0] > 0


def test_fev_identity_with_explicit_h0(backend: str) -> None:
    """With ``h0`` given, ``hinit`` is skipped and its evaluation is not spent."""
    res = solve(backend, S2, precision="fp64", rtol=1e-7, atol=1e-9, h0=0.05)
    n_steps = int(res.n_accepted[0]) + int(res.n_rejected[0])
    assert res.status[0] == Status.SUCCESS
    assert res.n_fev[0] == 1 + 6 * n_steps


def test_status_enum_matches_reference_constants() -> None:
    assert Status.NOT_RUN == ref.NOT_RUN
    assert Status.SUCCESS == ref.SUCCESS
    assert Status.MAX_STEPS == ref.MAX_STEPS
    assert Status.STEP_TOO_SMALL == ref.STEP_TOO_SMALL
    assert Status.NON_FINITE == ref.NON_FINITE
