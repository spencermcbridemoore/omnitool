"""``cpu`` backend: the NumPy reference, one member at a time (spec §5)."""

from __future__ import annotations

import numpy as np

from omnitool.problem import BatchResult, Problem
from omnitool.reference import dopri5_numpy as ref

__all__ = ["solve_batch_cpu"]


def solve_batch_cpu(
    problem: Problem,
    y0: np.ndarray,
    params: np.ndarray,
    t0: float,
    t_end: float,
    t_save: np.ndarray,
    rtol: float,
    atol: float,
    dtype: type[np.floating],
    max_steps: int,
    h0: float | None,
    fixed_step: float | None,
) -> BatchResult:
    n_members = y0.shape[0]
    n_save = t_save.shape[0]
    y = np.empty((n_members, n_save, problem.n_state), dtype=dtype)
    status = np.empty(n_members, dtype=np.int32)
    n_accepted = np.empty(n_members, dtype=np.int32)
    n_rejected = np.empty(n_members, dtype=np.int32)
    n_fev = np.empty(n_members, dtype=np.int32)
    h_last = np.empty(n_members, dtype=dtype)
    for m in range(n_members):
        r = ref.integrate(
            problem.rhs_numpy,
            y0[m],
            params[m],
            t0,
            t_end,
            t_save,
            rtol,
            atol,
            dtype,
            max_steps,
            h0,
            fixed_step,
        )
        y[m] = r.y
        status[m] = r.status
        n_accepted[m] = r.n_accepted
        n_rejected[m] = r.n_rejected
        n_fev[m] = r.n_fev
        h_last[m] = r.h_last
    return BatchResult(y, status, n_accepted, n_rejected, n_fev, h_last)
