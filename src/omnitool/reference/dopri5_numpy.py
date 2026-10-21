"""Plain-NumPy Dormand–Prince 5(4) reference, one member at a time.

This is the ``cpu`` backend and the oracle of the verification layer
(ADR 0003, spec §5). It is written from Hairer, Nørsett & Wanner, *Solving
ODEs I* (HNW I) and Hairer's ``dopri5.f``, not from any kernel, and shares
only the tableau module with the device code. It imports ``numpy`` and
``omnitool.tableaus`` (plus the standard library) and nothing else.

Everything is computed in the requested ``dtype``: the state, the time, the
step size, the tableau and every controller constant, so that an FP32 run
mirrors an FP32 device run rather than an FP64 run rounded at the end.

Numerics (spec §3):

- error norm ``err = sqrt(mean(((h (b - b̂)·k)_i / sc_i)^2))`` with
  ``sc_i = atol + rtol max(|y_n,i|, |y_n+1,i|)``; accept when ``err <= 1``;
- PI controller in Gustafsson form with ``beta = 0.04``,
  ``alpha = 0.2 - 0.75 beta``, ``safe = 0.9``, factor clamped to
  ``[0.1, 5]``; ``err_old = max(err, 1e-4)`` after an accepted step, unchanged
  after a rejection; after a rejection ``h_new = h / min(5, err^alpha / safe)``;
  on the first accepted step after a rejection the new step may not grow
  (``h_new = min(h_new, h)``, the ``REJECT`` rule of ``dopri5.f``);
- initial step by Hairer's ``hinit`` (HNW I, §II.4, eq. 4.14) unless ``h0``;
- the last step lands exactly on ``t_end``; following ``dopri5.f`` the step
  is set to ``t_end - t`` as soon as ``t + 1.01 h`` would pass ``t_end``, so
  a step may be stretched by at most one percent and no spurious tiny final
  step is taken. Steps are never clipped to save times;
- ``STEP_TOO_SMALL`` when ``|h| < 16 eps max(|t|, 1)`` before a step;
- dense output by Hairer's fourth-order continuous extension evaluated at
  every save time in ``(t_n, t_n+1]``; a save time equal to ``t_n+1``
  receives ``y_n+1`` itself (``theta = 1``) so the endpoint values are
  bitwise those of the step.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from omnitool.tableaus import dopri5 as tab

__all__ = [
    "MAX_STEPS",
    "NON_FINITE",
    "NOT_RUN",
    "STEP_TOO_SMALL",
    "SUCCESS",
    "DenseData",
    "MemberResult",
    "dense_coefficients",
    "dense_eval",
    "error_norm",
    "hinit",
    "integrate",
    "step",
]

# Status codes, spec §6 (mirrored by omnitool.status.Status; a test checks).
NOT_RUN = -1
SUCCESS = 0
MAX_STEPS = 1
STEP_TOO_SMALL = 2
NON_FINITE = 3

# Controller constants, spec §3 (Hairer's dopri5.f defaults).
BETA = 0.04
ALPHA = 0.2 - 0.75 * BETA
SAFE = 0.9
FAC_MIN = 0.1
FAC_MAX = 5.0
ERR_OLD_FLOOR = 1e-4
STEP_FLOOR_EPS = 16.0
STRETCH = 1.01


class MemberResult(NamedTuple):
    """Result for one member; the cpu backend stacks these into a BatchResult."""

    y: np.ndarray
    """``(n_save, n_state)``; NaN from the first unreached save time on."""
    status: int
    n_accepted: int
    n_rejected: int
    n_fev: int
    h_last: float


class _Consts(NamedTuple):
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    e: np.ndarray
    d: np.ndarray
    alpha: np.floating
    beta: np.floating
    safe: np.floating
    fac_min: np.floating
    fac_max: np.floating
    err_old_floor: np.floating
    one: np.floating
    zero: np.floating


def _consts(dtype: type[np.floating]) -> _Consts:
    arrs = tab.as_arrays(dtype)
    return _Consts(
        a=arrs["a"],
        b=arrs["b"],
        c=arrs["c"],
        e=arrs["e"],
        d=arrs["d"],
        alpha=dtype(ALPHA),
        beta=dtype(BETA),
        safe=dtype(SAFE),
        fac_min=dtype(FAC_MIN),
        fac_max=dtype(FAC_MAX),
        err_old_floor=dtype(ERR_OLD_FLOOR),
        one=dtype(1),
        zero=dtype(0),
    )


def _call_rhs(rhs, t, y, p, dtype):
    return np.asarray(rhs(t, y, p), dtype=dtype).reshape(y.shape)


def step(
    rhs,
    t: np.floating,
    y: np.ndarray,
    h: np.floating,
    p: np.ndarray,
    k1: np.ndarray,
    dtype: type[np.floating] = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One DOPRI5 step from ``(t, y)`` with stage-one derivative ``k1``.

    Returns ``(y_new, k, err_vec)`` where ``k`` is the ``(7, n_state)`` array of
    stage derivatives (``k[6]`` is ``f(t + h, y_new)``, the FSAL stage) and
    ``err_vec = h (b - b̂)·k`` is the unscaled error estimate. Six RHS
    evaluations.
    """
    cs = _consts(dtype)
    n = y.shape[0]
    k = np.empty((tab.N_STAGES, n), dtype=dtype)
    k[0] = k1
    y_new = y
    for i in range(1, tab.N_STAGES):
        y_i = y + h * (cs.a[i, :i] @ k[:i])
        if i == tab.N_STAGES - 1:
            y_new = y_i  # a[6, :] == b: the stage-7 input is the new solution
        k[i] = _call_rhs(rhs, t + cs.c[i] * h, y_i, p, dtype)
    err_vec = h * (cs.e @ k)
    return y_new, k, err_vec


def error_norm(
    err_vec: np.ndarray,
    y_old: np.ndarray,
    y_new: np.ndarray,
    rtol: np.floating,
    atol: np.floating,
) -> np.floating:
    """Scaled RMS norm of the error estimate (spec §3)."""
    sc = atol + rtol * np.maximum(np.abs(y_old), np.abs(y_new))
    q = err_vec / sc
    return np.sqrt(np.mean(q * q))


class DenseData(NamedTuple):
    """Hairer's ``rcont`` vectors for one accepted step, plus both endpoints."""

    y_old: np.ndarray
    y_new: np.ndarray
    r3: np.ndarray
    r4: np.ndarray
    r5: np.ndarray


def dense_coefficients(
    y_old: np.ndarray,
    y_new: np.ndarray,
    h: np.floating,
    k: np.ndarray,
    dtype: type[np.floating] = np.float64,
) -> DenseData:
    """Coefficients of Hairer's continuous extension for the step ``y_old -> y_new``."""
    cs = _consts(dtype)
    dy = y_new - y_old
    r3 = h * k[0] - dy
    r4 = dy - h * k[6] - r3
    r5 = h * (cs.d @ k)
    return DenseData(y_old, y_new, r3, r4, r5)


def dense_eval(
    r: DenseData, theta: np.floating, dtype: type[np.floating] = np.float64
) -> np.ndarray:
    """Evaluate the continuous extension at ``theta`` in ``[0, 1]``.

    ``theta = 0`` returns ``y_n`` and ``theta = 1`` returns ``y_n+1`` bitwise
    (spec §3); the polynomial form would reproduce them only up to rounding.
    """
    if theta == 0:
        return r.y_old.copy()
    if theta == 1:
        return r.y_new.copy()
    one = dtype(1)
    th1 = one - theta
    dy = r.y_new - r.y_old
    return r.y_old + theta * (dy + th1 * (r.r3 + theta * (r.r4 + th1 * r.r5)))


def hinit(
    rhs,
    t0: np.floating,
    y0: np.ndarray,
    f0: np.ndarray,
    p: np.ndarray,
    t_end: np.floating,
    rtol: np.floating,
    atol: np.floating,
    dtype: type[np.floating] = np.float64,
) -> tuple[np.floating, int]:
    """Hairer's initial step guess (HNW I §II.4, eq. 4.14; ``hinit`` of ``dopri5.f``).

    Returns ``(h, n_fev)``; one extra RHS evaluation is spent.
    """
    iord = tab.ORDER
    hmax = abs(t_end - t0)
    sk = atol + rtol * np.abs(y0)
    dnf = np.sum((f0 / sk) ** 2)
    dny = np.sum((y0 / sk) ** 2)
    if dnf <= dtype(1e-10) or dny <= dtype(1e-10):
        h = dtype(1e-6)
    else:
        h = np.sqrt(dny / dnf) * dtype(0.01)
    h = min(h, hmax)
    y1 = y0 + h * f0
    f1 = _call_rhs(rhs, t0 + h, y1, p, dtype)
    der2 = np.sqrt(np.sum(((f1 - f0) / sk) ** 2)) / h
    der12 = max(abs(der2), np.sqrt(dnf))
    if der12 <= dtype(1e-15):
        h1 = max(dtype(1e-6), abs(h) * dtype(1e-3))
    else:
        h1 = (dtype(0.01) / der12) ** (dtype(1) / dtype(iord))
    h = min(dtype(100) * abs(h), h1, hmax)
    return dtype(h), 1


def integrate(
    rhs,
    y0: np.ndarray,
    p: np.ndarray,
    t0: float,
    t_end: float,
    t_save: np.ndarray,
    rtol: float,
    atol: float,
    dtype: type[np.floating] = np.float64,
    max_steps: int = 100_000,
    h0: float | None = None,
    fixed_step: float | None = None,
) -> MemberResult:
    """Integrate one member from ``t0`` to ``t_end``, saving at ``t_save``.

    ``t_save`` is strictly increasing with ``t_save[0] == t0`` and
    ``t_save[-1] == t_end`` (validated by the public API, not here).
    ``fixed_step`` bypasses the controller: every step has that size (the
    last one clipped), nothing is rejected.
    """
    cs = _consts(dtype)
    eps = dtype(np.finfo(dtype).eps)
    y = np.asarray(y0, dtype=dtype).copy()
    p = np.asarray(p, dtype=dtype)
    t_save = np.asarray(t_save, dtype=dtype)
    n_save = t_save.shape[0]
    out = np.full((n_save, y.shape[0]), np.nan, dtype=dtype)

    t = dtype(t0)
    t_end_d = dtype(t_end)
    rtol_d = dtype(rtol)
    atol_d = dtype(atol)
    n_accepted = 0
    n_rejected = 0
    n_fev = 0
    h_last = cs.zero
    err_old = cs.err_old_floor
    rejected_last = False
    status = NOT_RUN

    save_idx = 0
    while save_idx < n_save and t_save[save_idx] <= t:
        out[save_idx] = y
        save_idx += 1

    if not np.all(np.isfinite(y)):
        return MemberResult(out, NON_FINITE, 0, 0, 0, float(h_last))

    k1 = _call_rhs(rhs, t, y, p, dtype)
    n_fev += 1
    if fixed_step is not None:
        h = dtype(fixed_step)
    elif h0 is not None:
        h = dtype(h0)
    else:
        h, extra = hinit(rhs, t, y, k1, p, t_end_d, rtol_d, atol_d, dtype)
        n_fev += extra

    while True:
        if t >= t_end_d:
            status = SUCCESS
            break
        if n_accepted + n_rejected >= max_steps:
            status = MAX_STEPS
            break
        if abs(h) < dtype(STEP_FLOOR_EPS) * eps * max(abs(t), cs.one):
            status = STEP_TOO_SMALL
            break
        last = False
        if t + dtype(STRETCH) * h >= t_end_d:
            h = t_end_d - t
            last = True

        y_new, k, err_vec = step(rhs, t, y, h, p, k1, dtype)
        n_fev += 6
        err = cs.zero if fixed_step is not None else error_norm(err_vec, y, y_new, rtol_d, atol_d)
        if not np.isfinite(err) or not np.all(np.isfinite(y_new)):
            n_rejected += 1
            status = NON_FINITE
            break

        if fixed_step is not None or err <= cs.one:
            n_accepted += 1
            t_new = t_end_d if last else t + h
            if save_idx < n_save and t_save[save_idx] <= t_new:
                r = dense_coefficients(y, y_new, h, k, dtype)
                while save_idx < n_save and t_save[save_idx] <= t_new:
                    ts = t_save[save_idx]
                    if ts == t_new:
                        out[save_idx] = y_new
                    else:
                        theta = (ts - t) / h
                        out[save_idx] = dense_eval(r, theta, dtype)
                    save_idx += 1
            h_last = h
            if fixed_step is not None:
                h_new = dtype(fixed_step)
            else:
                fac = err**cs.alpha / err_old**cs.beta / cs.safe
                fac = min(max(fac, cs.fac_min), cs.fac_max)
                h_new = h / fac
                if rejected_last:
                    # dopri5.f: no growth on the first accept after a reject.
                    h_new = min(h_new, h)
                    rejected_last = False
                err_old = max(err, cs.err_old_floor)
            t = t_new
            y = y_new
            k1 = k[6]
            h = dtype(h_new)
        else:
            n_rejected += 1
            rejected_last = True
            h = dtype(h / min(cs.fac_max, err**cs.alpha / cs.safe))

    return MemberResult(out, status, n_accepted, n_rejected, n_fev, float(h_last))
