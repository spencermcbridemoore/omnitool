"""T6 — batch-order invariance (spec §8).

Permuting the members of a batch must permute the outputs and nothing else,
bitwise, on every backend and precision. A batch holds one ``Problem``, so
two batches are used: L with the four detunings cycled through the members
and ``y0`` perturbed per member, and S2 with ``y0`` perturbed per member.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import omnitool
from omnitool import BatchResult, Problem, Status
from problems import L_CASES, S2, save_grid

N_MEMBERS = 64
PERTURBATION = 1e-3
TOLERANCES = {"fp64": (1e-7, 1e-9), "fp32": (1e-4, 1e-6)}
DTYPES = {"fp64": np.float64, "fp32": np.float32}

Batch = tuple[Problem, np.ndarray, np.ndarray]


def _l_batch(dtype: type[np.floating]) -> Batch:
    base = L_CASES[0]
    y0 = np.empty((N_MEMBERS, base.problem.n_state), dtype=dtype)
    params = np.empty((N_MEMBERS, base.problem.n_params), dtype=dtype)
    for m in range(N_MEMBERS):
        case = L_CASES[m % len(L_CASES)]
        y0[m] = case.y0 + PERTURBATION * m
        params[m] = case.params
    return base.problem, y0, params


def _s2_batch(dtype: type[np.floating]) -> Batch:
    y0 = np.stack([S2.y0 + PERTURBATION * m for m in range(N_MEMBERS)]).astype(dtype)
    params = np.zeros((N_MEMBERS, 0), dtype=dtype)
    return S2.problem, y0, params


BATCHES = [pytest.param(_l_batch, id="L_cycled"), pytest.param(_s2_batch, id="S2")]


def _run(
    backend: str, precision: str, problem: Problem, y0: np.ndarray, params: np.ndarray
) -> BatchResult:
    rtol, atol = TOLERANCES[precision]
    return omnitool.solve_batch(
        problem,
        y0,
        params,
        t_span=(0.0, 10.0),
        t_save=save_grid(),
        rtol=rtol,
        atol=atol,
        precision=precision,
        backend=backend,
    )


@pytest.mark.parametrize("make_batch", BATCHES)
def test_permuting_members_only_permutes_outputs(
    backend: str, precision: str, make_batch: Callable[[type[np.floating]], Batch]
) -> None:
    problem, y0, params = make_batch(DTYPES[precision])
    direct = _run(backend, precision, problem, y0, params)
    assert np.all(direct.status == Status.SUCCESS)

    perm = np.random.default_rng(0).permutation(N_MEMBERS)
    inv = np.argsort(perm)
    permuted = _run(backend, precision, problem, y0[perm], params[perm])

    assert np.array_equal(permuted.y[inv], direct.y)
    assert np.array_equal(permuted.status[inv], direct.status)
    assert np.array_equal(permuted.n_accepted[inv], direct.n_accepted)
    assert np.array_equal(permuted.n_rejected[inv], direct.n_rejected)
    assert np.array_equal(permuted.n_fev[inv], direct.n_fev)
    assert np.array_equal(permuted.h_last[inv], direct.h_last)
    if make_batch is _l_batch:
        # Per-member stepping: members with different detunings take different paths.
        assert len(np.unique(direct.n_accepted)) > 1
