"""The public entry point ``solve_batch`` (spec §2): validation and dispatch."""

from __future__ import annotations

import warnings

import numpy as np

from omnitool.problem import BatchResult, Problem

__all__ = ["BACKENDS", "PRECISIONS", "solve_batch"]

PRECISIONS: dict[str, type[np.floating]] = {"fp32": np.float32, "fp64": np.float64}
BACKENDS = ("cpu", "cudasim", "cuda")


def _cast_checked(name: str, arr: np.ndarray, dtype: type[np.floating]) -> np.ndarray:
    out = np.ascontiguousarray(arr, dtype=dtype)
    if (
        np.result_type(arr).kind == "f"
        and np.finfo(np.result_type(arr)).bits > np.finfo(dtype).bits
    ):
        back = out.astype(np.result_type(arr))
        if not np.array_equal(back, np.asarray(arr), equal_nan=True):
            warnings.warn(
                f"{name} loses information when cast to {dtype.__name__}",
                UserWarning,
                stacklevel=3,
            )
    return out


def solve_batch(
    problem: Problem,
    y0: np.ndarray,
    params: np.ndarray,
    t_span: tuple[float, float],
    t_save: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    precision: str = "fp64",
    backend: str = "cuda",
    max_steps: int = 100_000,
    h0: float | None = None,
    fixed_step: float | None = None,
) -> BatchResult:
    """Integrate every member of a batch with per-member adaptive DOPRI5.

    See ``docs/PHASE0_SPEC.md`` §2 for the contract on shapes, casting and
    error conditions.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {tuple(PRECISIONS)}, got {precision!r}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    dtype = PRECISIONS[precision]
    eps = float(np.finfo(dtype).eps)
    if rtol < 100 * eps:
        raise ValueError(f"rtol={rtol} is below 100 * eps({precision}) = {100 * eps:.3g}")
    if atol < 0:
        raise ValueError(f"atol must be non-negative, got {atol}")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if fixed_step is not None and fixed_step <= 0:
        raise ValueError("fixed_step must be positive")
    if h0 is not None and h0 <= 0:
        raise ValueError("h0 must be positive")

    t0, t_end = (float(t) for t in t_span)
    if not t_end > t0:
        raise ValueError(f"t_end must exceed t0, got t_span={t_span}")
    t_save_arr = np.asarray(t_save, dtype=np.float64)
    if t_save_arr.ndim != 1 or t_save_arr.shape[0] < 2:
        raise ValueError("t_save must be one-dimensional with at least two entries")
    if not np.all(np.diff(t_save_arr) > 0):
        raise ValueError("t_save must be strictly increasing")
    if t_save_arr[0] != t0 or t_save_arr[-1] != t_end:
        raise ValueError("t_save must start at t0 and end at t_end")

    y0_arr = np.asarray(y0)
    params_arr = np.asarray(params)
    if y0_arr.ndim != 2 or y0_arr.shape[1] != problem.n_state:
        raise ValueError(f"y0 must have shape (n_members, {problem.n_state}), got {y0_arr.shape}")
    if params_arr.ndim != 2 or params_arr.shape[1] != problem.n_params:
        raise ValueError(
            f"params must have shape (n_members, {problem.n_params}), got {params_arr.shape}"
        )
    if params_arr.shape[0] != y0_arr.shape[0]:
        raise ValueError("y0 and params must have the same number of members")
    y0_c = _cast_checked("y0", y0_arr, dtype)
    params_c = _cast_checked("params", params_arr, dtype)
    t_save_c = t_save_arr.astype(dtype)

    if backend == "cpu":
        from omnitool.backends.cpu import solve_batch_cpu

        return solve_batch_cpu(
            problem, y0_c, params_c, t0, t_end, t_save_c, rtol, atol, dtype,
            max_steps, h0, fixed_step,
        )  # fmt: skip
    raise NotImplementedError(f"backend {backend!r} is not implemented yet (Phase 0 in progress)")
