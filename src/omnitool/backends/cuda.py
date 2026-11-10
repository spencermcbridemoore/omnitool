"""``cuda`` backend: ``dopri5.cu`` on a CUDA device, one thread per member (spec §4-§5).

The kernel source is rendered by :func:`omnitool.kernels.render_dopri5` and
compiled only through :func:`omnitool.backends.nvrtc.compile_module` (ADR
0004: direct NVRTC with the IEEE option set, never ``RawModule(code=...)``).
CuPy owns the device arrays and the stream. Device arrays are member-fastest
(``[n_state, n_members]``, ``[n_save, n_state, n_members]``); this module
transposes to and from the API shapes.

Besides :func:`solve_batch_cuda` there are two thin wrappers,
:func:`step_cuda` and :func:`dense_cuda`, over the kernel's test entry points
for one ``step()`` and one ``dense()`` evaluation (spec T3 and T5).

CuPy is imported lazily so the package imports without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from omnitool import kernels
from omnitool.backends import nvrtc
from omnitool.problem import BatchResult, Problem
from omnitool.status import Status

__all__ = ["THREADS_PER_BLOCK", "dense_cuda", "solve_batch_cuda", "step_cuda"]

THREADS_PER_BLOCK = 128
"""Launch configuration of spec §4; the grid is rounded up."""

_INT32_MAX = 2**31 - 1
# n_fev = 2 + 6 * (accepted + rejected) is an int32 on the device and in BatchResult.
_MAX_STEPS_LIMIT = (_INT32_MAX - 2) // 6
# The member index is an int; the last block is padded to THREADS_PER_BLOCK threads.
_MAX_MEMBERS = _INT32_MAX - THREADS_PER_BLOCK

# (rhs body, n_state, n_params, dtype name, device id) -> loaded cupy.RawModule
_MODULES: dict[tuple[str, int, int, str, int], Any] = {}


def _cupy() -> Any:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "the cuda backend needs CuPy: install omnitool[cuda12] or omnitool[cuda13], "
            "matching the CUDA major version of the driver"
        ) from exc
    return cp


def _module(problem: Problem, dtype: type[np.floating]) -> Any:
    """The compiled kernel module for ``problem`` at ``dtype`` on the current device."""
    cp = _cupy()
    key = (
        problem.rhs_cuda,
        problem.n_state,
        problem.n_params,
        np.dtype(dtype).name,
        cp.cuda.Device().id,
    )
    mod = _MODULES.get(key)
    if mod is None:
        source = kernels.render_dopri5(problem.rhs_cuda, problem.n_state, problem.n_params, dtype)
        name = f"dopri5_{problem.name}_{kernels.C_TYPES[dtype]}"
        mod = nvrtc.compile_module(source, "".join(c if c.isalnum() else "_" for c in name))
        _MODULES[key] = mod
    return mod


def _grid(n_members: int) -> tuple[int]:
    return ((n_members + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,)


def _to_device(arr: np.ndarray, dtype: type[np.floating]) -> Any:
    """``(n_members, n)`` host array -> ``[max(n, 1), n_members]`` device array.

    A problem without parameters still gets one (unread) row, so the kernel
    is never handed a null pointer.
    """
    cp = _cupy()
    host = np.ascontiguousarray(np.asarray(arr, dtype=dtype).T)
    if host.shape[0] == 0:
        host = np.zeros((1, host.shape[1]), dtype=dtype)
    return cp.asarray(host)


def solve_batch_cuda(
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
    cp = _cupy()
    n_members = y0.shape[0]
    n_save = t_save.shape[0]
    n_state = problem.n_state
    if max_steps > _MAX_STEPS_LIMIT:
        raise ValueError(
            f"max_steps={max_steps} exceeds {_MAX_STEPS_LIMIT} on the cuda backend: "
            "n_fev is an int32 counter"
        )
    if n_members > _MAX_MEMBERS:
        raise ValueError(f"n_members={n_members} exceeds {_MAX_MEMBERS} on the cuda backend")
    kernel = _module(problem, dtype).get_function("dopri5_solve")

    modes = kernels.step_modes()
    if fixed_step is not None:
        step_mode, h_given = modes["FIXED"], fixed_step
    elif h0 is not None:
        step_mode, h_given = modes["H0"], h0
    else:
        step_mode, h_given = modes["AUTO"], 0.0

    y0_d = _to_device(y0, dtype)
    params_d = _to_device(params, dtype)
    t_save_d = cp.asarray(np.ascontiguousarray(t_save, dtype=dtype))
    # NaN everywhere; the kernel writes only the save times a member reaches (spec §2).
    out_d = cp.full((n_save, n_state, n_members), cp.nan, dtype=dtype)
    status_d = cp.full(n_members, int(Status.NOT_RUN), dtype=np.int32)
    n_accepted_d = cp.zeros(n_members, dtype=np.int32)
    n_rejected_d = cp.zeros(n_members, dtype=np.int32)
    n_fev_d = cp.zeros(n_members, dtype=np.int32)
    h_last_d = cp.zeros(n_members, dtype=dtype)

    if n_members > 0:
        kernel(
            _grid(n_members),
            (THREADS_PER_BLOCK,),
            (
                np.int32(n_members),
                np.int32(n_save),
                dtype(t0),
                dtype(t_end),
                t_save_d,
                dtype(rtol),
                dtype(atol),
                np.int32(max_steps),
                np.int32(step_mode),
                dtype(h_given),
                y0_d,
                params_d,
                out_d,
                status_d,
                n_accepted_d,
                n_rejected_d,
                n_fev_d,
                h_last_d,
            ),
        )
        cp.cuda.get_current_stream().synchronize()

    return BatchResult(
        y=np.ascontiguousarray(cp.asnumpy(out_d).transpose(2, 0, 1)),
        status=cp.asnumpy(status_d),
        n_accepted=cp.asnumpy(n_accepted_d),
        n_rejected=cp.asnumpy(n_rejected_d),
        n_fev=cp.asnumpy(n_fev_d),
        h_last=cp.asnumpy(h_last_d),
    )


def step_cuda(
    problem: Problem,
    y: np.ndarray,
    params: np.ndarray,
    t: float,
    h: float,
    dtype: type[np.floating] = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """One device ``step()`` of size ``h`` from ``(t, y)`` for every member.

    ``y`` is ``(n_members, n_state)`` and ``params`` ``(n_members, n_params)``.
    Returns ``(y_new, err_vec)``, both ``(n_members, n_state)``, where
    ``err_vec = h (b - b̂)·k`` is the unscaled error estimate. A test entry
    point: unlike ``solve_batch`` it casts ``y`` and ``params`` to ``dtype``
    without checking for loss.
    """
    cp = _cupy()
    n_members = y.shape[0]
    kernel = _module(problem, dtype).get_function("dopri5_step_once")
    y_d = _to_device(y, dtype)
    params_d = _to_device(params, dtype)
    y_new_d = cp.empty((problem.n_state, n_members), dtype=dtype)
    err_d = cp.empty((problem.n_state, n_members), dtype=dtype)
    kernel(
        _grid(n_members),
        (THREADS_PER_BLOCK,),
        (np.int32(n_members), dtype(t), dtype(h), y_d, params_d, y_new_d, err_d),
    )
    cp.cuda.get_current_stream().synchronize()
    return np.ascontiguousarray(cp.asnumpy(y_new_d).T), np.ascontiguousarray(cp.asnumpy(err_d).T)


def dense_cuda(
    problem: Problem,
    y: np.ndarray,
    params: np.ndarray,
    t: float,
    h: float,
    theta: np.ndarray,
    dtype: type[np.floating] = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """One device step of size ``h`` from ``(t, y)``, then ``dense()`` at each ``theta``.

    Returns ``(y_new, dense)`` with shapes ``(n_members, n_state)`` and
    ``(n_members, n_theta, n_state)``. A test entry point: it casts its
    inputs to ``dtype`` without checking for loss.
    """
    cp = _cupy()
    n_members = y.shape[0]
    theta_h = np.ascontiguousarray(np.atleast_1d(theta), dtype=dtype)
    n_theta = theta_h.shape[0]
    kernel = _module(problem, dtype).get_function("dopri5_dense_once")
    y_d = _to_device(y, dtype)
    params_d = _to_device(params, dtype)
    theta_d = cp.asarray(theta_h)
    y_new_d = cp.empty((problem.n_state, n_members), dtype=dtype)
    dense_d = cp.empty((n_theta, problem.n_state, n_members), dtype=dtype)
    kernel(
        _grid(n_members),
        (THREADS_PER_BLOCK,),
        (
            np.int32(n_members),
            np.int32(n_theta),
            dtype(t),
            dtype(h),
            theta_d,
            y_d,
            params_d,
            y_new_d,
            dense_d,
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    return (
        np.ascontiguousarray(cp.asnumpy(y_new_d).T),
        np.ascontiguousarray(cp.asnumpy(dense_d).transpose(2, 0, 1)),
    )
