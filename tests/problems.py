"""Test problems L, S1 and S2 of ``docs/PHASE0_SPEC.md`` §7, defined once.

Each :class:`Case` bundles a :class:`omnitool.Problem` with the initial
state, parameters, end time and a high-accuracy reference solution. The
NumPy right-hand sides keep the dtype of ``y`` so an FP32 run stays FP32.
The C bodies are placeholders until the kernel exists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import omnitool
from omnitool import BatchResult, Problem

__all__ = [
    "ALL_CASES",
    "L_CASES",
    "S1",
    "S2",
    "T_END",
    "Case",
    "reference_on_grid",
    "save_grid",
    "solve",
]

T_END = 10.0

RHS_CUDA_PLACEHOLDER = "/* C body supplied with the Phase 0 kernel; not written yet */"


@dataclass(frozen=True)
class Case:
    name: str
    problem: Problem
    y0: np.ndarray
    params: np.ndarray
    t_end: float
    reference: Callable[[np.ndarray], np.ndarray]
    """``reference(t) -> (len(t), n_state)`` float64 solution at times ``t``."""


# --------------------------------------------------------------------------
# L — two-level optical Bloch, constant coefficients (affine linear)
# --------------------------------------------------------------------------


def bloch_rhs(t, y, p):
    gamma, omega, delta = p[0], p[1], p[2]
    u, v, w = y[0], y[1], y[2]
    return np.array(
        [
            -0.5 * gamma * u + delta * v,
            -delta * u - 0.5 * gamma * v - omega * w,
            omega * v - gamma * (w + 1),
        ],
        dtype=y.dtype,
    )


BLOCH = Problem(
    name="bloch",
    n_state=3,
    n_params=3,
    rhs_numpy=bloch_rhs,
    rhs_cuda=RHS_CUDA_PLACEHOLDER,
)


def _bloch_closed_form(gamma: float, omega: float, delta: float, y0: np.ndarray):
    """``y(t) = e^{Jt} y0 + J^{-1}(e^{Jt} - I) g`` evaluated through the augmented exponential.

    ``expm([[J, g], [0, 0]] t)`` applied to ``(y0, 1)`` is that expression
    without forming ``J^{-1}``.
    """
    j = np.array(
        [[-0.5 * gamma, delta, 0.0], [-delta, -0.5 * gamma, -omega], [0.0, omega, -gamma]]
    )
    g = np.array([0.0, 0.0, -gamma])
    m = np.zeros((4, 4))
    m[:3, :3] = j
    m[:3, 3] = g
    z0 = np.append(np.asarray(y0, dtype=np.float64), 1.0)

    def reference(t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        return np.stack([(expm(m * ti) @ z0)[:3] for ti in t])

    return reference


def make_bloch_case(delta: float, gamma: float = 1.0, omega: float = 1.0) -> Case:
    y0 = np.array([0.0, 0.0, -1.0])
    return Case(
        name=f"L_delta{delta:g}",
        problem=BLOCH,
        y0=y0,
        params=np.array([gamma, omega, delta]),
        t_end=T_END,
        reference=_bloch_closed_form(gamma, omega, delta, y0),
    )


L_CASES: tuple[Case, ...] = tuple(make_bloch_case(d) for d in (0.0, 2.0, 20.0, 200.0))
L_BY_DELTA: dict[float, Case] = {float(c.params[2]): c for c in L_CASES}


# --------------------------------------------------------------------------
# S1 — chirped Bloch, Delta(t) = 2 + 3 sin(1.5 t) (smooth, non-autonomous)
# --------------------------------------------------------------------------


def chirp_rhs(t, y, p):
    gamma, omega = p[0], p[1]
    delta = 2.0 + 3.0 * np.sin(1.5 * t)
    u, v, w = y[0], y[1], y[2]
    return np.array(
        [
            -0.5 * gamma * u + delta * v,
            -delta * u - 0.5 * gamma * v - omega * w,
            omega * v - gamma * (w + 1),
        ],
        dtype=y.dtype,
    )


CHIRP = Problem(
    name="chirp",
    n_state=3,
    n_params=2,
    rhs_numpy=chirp_rhs,
    rhs_cuda=RHS_CUDA_PLACEHOLDER,
)


# --------------------------------------------------------------------------
# S2 — Duffing oscillator (smooth, nonlinear, Hamiltonian)
# --------------------------------------------------------------------------


def duffing_rhs(t, y, p):
    return np.array([y[1], -y[0] - y[0] * y[0] * y[0]], dtype=y.dtype)


DUFFING = Problem(
    name="duffing",
    n_state=2,
    n_params=0,
    rhs_numpy=duffing_rhs,
    rhs_cuda=RHS_CUDA_PLACEHOLDER,
)


def duffing_energy(y: np.ndarray) -> np.ndarray:
    """Invariant ``H = y1^2/2 + y0^2/2 + y0^4/4`` (equals 3/4 on the test orbit)."""
    return 0.5 * y[..., 1] ** 2 + 0.5 * y[..., 0] ** 2 + 0.25 * y[..., 0] ** 4


def _dop853_reference(problem: Problem, y0: np.ndarray, params: np.ndarray):
    """SciPy DOP853 at ``rtol = 1e-13, atol = 1e-15`` (spec §7)."""
    p = np.asarray(params, dtype=np.float64)

    def reference(t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.empty((t.shape[0], problem.n_state))
        for i, ti in enumerate(t):
            if ti == 0.0:
                out[i] = y0
                continue
            sol = solve_ivp(
                lambda s, y: problem.rhs_numpy(s, y, p),
                (0.0, float(ti)),
                np.asarray(y0, dtype=np.float64),
                method="DOP853",
                rtol=1e-13,
                atol=1e-15,
            )
            assert sol.success, sol.message
            out[i] = sol.y[:, -1]
        return out

    return reference


_S1_Y0 = np.array([0.0, 0.0, -1.0])
_S1_P = np.array([1.0, 1.0])
S1 = Case(
    name="S1_chirp",
    problem=CHIRP,
    y0=_S1_Y0,
    params=_S1_P,
    t_end=T_END,
    reference=_dop853_reference(CHIRP, _S1_Y0, _S1_P),
)

_S2_Y0 = np.array([1.0, 0.0])
_S2_P = np.zeros(0)
S2 = Case(
    name="S2_duffing",
    problem=DUFFING,
    y0=_S2_Y0,
    params=_S2_P,
    t_end=T_END,
    reference=_dop853_reference(DUFFING, _S2_Y0, _S2_P),
)

ALL_CASES: tuple[Case, ...] = (*L_CASES, S1, S2)


# --------------------------------------------------------------------------
# Helpers shared by the acceptance tests
# --------------------------------------------------------------------------


def save_grid(t_end: float = T_END, n: int = 51) -> np.ndarray:
    """``linspace(0, T, 51)``, the save grid of the adaptive tests (spec §7)."""
    return np.linspace(0.0, t_end, n)


_REFERENCE_CACHE: dict[tuple[str, bytes], np.ndarray] = {}


def reference_on_grid(case: Case, t: np.ndarray) -> np.ndarray:
    """``case.reference(t)``, memoised: the DOP853 references are slow."""
    t = np.asarray(t, dtype=np.float64)
    key = (case.name, t.tobytes())
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = case.reference(t)
    return _REFERENCE_CACHE[key]


def solve(
    backend: str,
    case: Case,
    *,
    precision: str,
    t_save: np.ndarray | None = None,
    t_end: float | None = None,
    **kwargs,
) -> BatchResult:
    """Run one member of ``case`` on ``backend``."""
    t_end = case.t_end if t_end is None else t_end
    if t_save is None:
        t_save = save_grid(t_end)
    return omnitool.solve_batch(
        case.problem,
        case.y0[None, :],
        case.params[None, :],
        t_span=(0.0, t_end),
        t_save=t_save,
        precision=precision,
        backend=backend,
        **kwargs,
    )
