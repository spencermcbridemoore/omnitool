"""The ``Problem`` and ``BatchResult`` types of the public API (spec §2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["BatchResult", "Problem", "RhsNumpy"]

RhsNumpy = Callable[[np.floating, np.ndarray, np.ndarray], np.ndarray]
"""``rhs_numpy(t, y, p) -> dydt`` with ``y`` and ``p`` one-dimensional arrays."""


@dataclass(frozen=True)
class Problem:
    """A right-hand side in two renderings: NumPy for the reference, C for the device.

    ``rhs_cuda`` is the body of
    ``void rhs(real_t t, const real_t* y, real_t* dydt, const real_t* p)``.
    Floating constants in it are written ``REAL(0.5)`` so that they take the
    precision of the build; a bare ``0.5`` would promote an FP32 kernel to
    double arithmetic.
    """

    name: str
    n_state: int
    n_params: int
    rhs_numpy: RhsNumpy
    rhs_cuda: str


@dataclass(frozen=True)
class BatchResult:
    """Host-side results of ``solve_batch``; every field is a NumPy array."""

    y: np.ndarray
    """``(n_members, n_save, n_state)``: solution at the save times."""
    status: np.ndarray
    """``(n_members,)`` int32: :class:`omnitool.status.Status` codes."""
    n_accepted: np.ndarray
    """``(n_members,)`` int32: accepted steps."""
    n_rejected: np.ndarray
    """``(n_members,)`` int32: rejected steps."""
    n_fev: np.ndarray
    """``(n_members,)`` int32: right-hand-side evaluations."""
    h_last: np.ndarray
    """``(n_members,)`` real: last accepted step size."""
