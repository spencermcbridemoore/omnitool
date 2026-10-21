"""T8 — status codes (spec §8 and §6)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import omnitool
from omnitool import Problem, Status
from problems import L_BY_DELTA, S2, save_grid, solve

STATUS_H = Path(__file__).resolve().parents[1] / "src" / "omnitool" / "status.h"
TOLERANCES = {"fp64": (1e-7, 1e-9), "fp32": (1e-4, 1e-6)}


def test_max_steps_reports_and_leaves_nan(backend: str, precision: str) -> None:
    case = L_BY_DELTA[200.0]
    rtol, atol = TOLERANCES[precision]
    t_save = save_grid(case.t_end)
    res = solve(backend, case, precision=precision, t_save=t_save, rtol=rtol, atol=atol,
                max_steps=100)  # fmt: skip
    assert res.status[0] == Status.MAX_STEPS
    assert res.n_accepted[0] + res.n_rejected[0] == 100
    finite = np.all(np.isfinite(res.y[0]), axis=1)
    assert finite[0], "t0 is always reached"
    assert not finite[-1], "t_end was not reached"
    # reached save times form a prefix of the grid
    first_nan = int(np.argmin(finite))
    assert np.all(finite[:first_nan])
    assert not np.any(finite[first_nan:])


def _singular_rhs(t, y, p):
    return np.array([1.0 / (t - 5.0)], dtype=y.dtype)


SINGULAR = Problem(name="singular", n_state=1, n_params=0, rhs_numpy=_singular_rhs,
                   rhs_cuda="dydt[0] = 1 / (t - 5);")  # fmt: skip


def test_singularity_never_reports_success(backend: str, precision: str) -> None:
    rtol, atol = TOLERANCES[precision]
    res = omnitool.solve_batch(
        SINGULAR,
        np.zeros((1, 1)),
        np.zeros((1, 0)),
        t_span=(0.0, 10.0),
        t_save=np.array([0.0, 10.0]),
        rtol=rtol,
        atol=atol,
        precision=precision,
        backend=backend,
    )
    assert res.status[0] in (Status.STEP_TOO_SMALL, Status.NON_FINITE), Status(res.status[0])
    assert np.isnan(res.y[0, -1, 0])


def test_nan_initial_state(backend: str, precision: str) -> None:
    rtol, atol = TOLERANCES[precision]
    res = omnitool.solve_batch(
        S2.problem,
        np.array([[np.nan, 0.0]]),
        np.zeros((1, 0)),
        t_span=(0.0, 10.0),
        t_save=np.array([0.0, 10.0]),
        rtol=rtol,
        atol=atol,
        precision=precision,
        backend=backend,
    )
    assert res.status[0] == Status.NON_FINITE
    assert res.n_accepted[0] + res.n_rejected[0] <= 1


def test_enum_matches_header() -> None:
    text = STATUS_H.read_text(encoding="utf-8")
    found = {
        name: int(value)
        for name, value in re.findall(r"#define OMNITOOL_STATUS_(\w+)\s+\(?(-?\d+)\)?", text)
    }
    assert found == {s.name: int(s) for s in Status}


def test_not_run_is_never_returned(backend: str) -> None:
    res = solve(backend, S2, precision="fp64")
    assert res.status[0] != Status.NOT_RUN


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ({"rtol": 1e-9}, "below 100"),
        ({"atol": -1.0}, "non-negative"),
        ({"max_steps": 0}, "max_steps"),
        ({"backend": "numba"}, "backend must be one of"),
    ],
)
def test_validation_errors(bad: dict, match: str) -> None:
    kwargs = {"backend": "cpu", "rtol": 1e-4, "atol": 1e-6, **bad}
    with pytest.raises(ValueError, match=match):
        omnitool.solve_batch(
            S2.problem,
            S2.y0[None, :],
            S2.params[None, :],
            t_span=(0.0, 10.0),
            t_save=np.array([0.0, 10.0]),
            precision="fp32",
            **kwargs,
        )


def test_unbuilt_known_backend_raises_not_implemented() -> None:
    import omnitool.backends

    unbuilt = [b for b in omnitool.backends.KNOWN if b not in omnitool.backends.available()]
    if not unbuilt:
        pytest.skip("every known backend is implemented")
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        omnitool.backends.get(unbuilt[0])
