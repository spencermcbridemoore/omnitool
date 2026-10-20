"""Dormand–Prince 5(4) tableau, transcribed once as exact rationals.

Source: J. R. Dormand and P. J. Prince, "A family of embedded Runge–Kutta
formulae", J. Comput. Appl. Math. 6 (1980) 19–26, as tabulated in
E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
Equations I*, 2nd ed., Table 5.2 (p. 178). Seven stages, FSAL: the seventh
stage of step *n* is the first stage of step *n+1*, so ``A[6, :] == B``.

``B`` is the fifth-order propagating weight vector and ``B_HAT`` the embedded
fourth-order one; the error estimate uses ``E = B - B_HAT``.

Dense output: Hairer's fourth-order continuous extension for DOPRI5
(HNW I, §II.6; the ``D1 … D7`` constants of ``dopri5.f``). With
``dy = y1 - y0`` and the stage derivatives ``k_i``,

    r1 = y0
    r2 = dy
    r3 = h k1 - dy
    r4 = dy - h k7 - r3
    r5 = h Σ_i D_i k_i
    y(θ) = r1 + θ (r2 + (1 - θ) (r3 + θ (r4 + (1 - θ) r5)))

Every other file (the NumPy reference, the device header renderer) takes its
coefficients from here; they are never typed a second time
(``docs/PHASE0_SPEC.md`` §3).
"""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np

__all__ = [
    "B_HAT",
    "N_STAGES",
    "ORDER",
    "ORDER_EMBEDDED",
    "A",
    "B",
    "C",
    "D",
    "E",
    "as_arrays",
    "render_c",
]

N_STAGES = 7
ORDER = 5
ORDER_EMBEDDED = 4

# Nodes c_i (column 1 of Table 5.2).
C: tuple[F, ...] = (
    F(0),
    F(1, 5),
    F(3, 10),
    F(4, 5),
    F(8, 9),
    F(1),
    F(1),
)

# Coupling coefficients a_ij, strictly lower triangular (explicit method).
A: tuple[tuple[F, ...], ...] = (
    (F(0), F(0), F(0), F(0), F(0), F(0), F(0)),
    (F(1, 5), F(0), F(0), F(0), F(0), F(0), F(0)),
    (F(3, 40), F(9, 40), F(0), F(0), F(0), F(0), F(0)),
    (F(44, 45), F(-56, 15), F(32, 9), F(0), F(0), F(0), F(0)),
    (F(19372, 6561), F(-25360, 2187), F(64448, 6561), F(-212, 729), F(0), F(0), F(0)),
    (F(9017, 3168), F(-355, 33), F(46732, 5247), F(49, 176), F(-5103, 18656), F(0), F(0)),
    (F(35, 384), F(0), F(500, 1113), F(125, 192), F(-2187, 6784), F(11, 84), F(0)),
)

# Fifth-order weights b_i (the propagating solution).
B: tuple[F, ...] = (
    F(35, 384),
    F(0),
    F(500, 1113),
    F(125, 192),
    F(-2187, 6784),
    F(11, 84),
    F(0),
)

# Fourth-order embedded weights b̂_i.
B_HAT: tuple[F, ...] = (
    F(5179, 57600),
    F(0),
    F(7571, 16695),
    F(393, 640),
    F(-92097, 339200),
    F(187, 2100),
    F(1, 40),
)

# Error weights e_i = b_i - b̂_i, derived here in exact arithmetic.
E: tuple[F, ...] = tuple(b - bh for b, bh in zip(B, B_HAT, strict=True))

# Dense-output constants D1 … D7 of Hairer's dopri5.f (D2 = 0).
D: tuple[F, ...] = (
    F(-12715105075, 11282082432),
    F(0),
    F(87487479700, 32700410799),
    F(-10690763975, 1880347072),
    F(701980252875, 199316789632),
    F(-1453857185, 822651844),
    F(69997945, 29380423),
)


def as_arrays(dtype: type[np.floating] = np.float64) -> dict[str, np.ndarray]:
    """Return the tableau as NumPy arrays of ``dtype``.

    Each rational is converted through a Python ``float`` (correctly rounded
    to binary64) and then to ``dtype``; for ``float32`` that is a second
    rounding, which is also what a C compiler does with a 17-digit literal
    assigned to a ``float``.
    """
    conv = {
        "c": np.array([float(x) for x in C], dtype=dtype),
        "a": np.array([[float(x) for x in row] for row in A], dtype=dtype),
        "b": np.array([float(x) for x in B], dtype=dtype),
        "b_hat": np.array([float(x) for x in B_HAT], dtype=dtype),
        "e": np.array([float(x) for x in E], dtype=dtype),
        "d": np.array([float(x) for x in D], dtype=dtype),
    }
    return conv


def _c_literal(x: F) -> str:
    """Render a rational to 17 significant digits as a ``real_t`` cast."""
    return f"(real_t){float(x):.17g}"


def render_c() -> str:
    """Render the tableau as C initialisers in terms of ``real_t``.

    The device header is generated from this text (spec §3); nothing in the
    kernel sources carries a coefficient of its own.
    """
    lines = [
        "/* Generated from omnitool/tableaus/dopri5.py; do not edit. */",
        f"#define DOPRI5_N_STAGES {N_STAGES}",
        "static const real_t DOPRI5_C[DOPRI5_N_STAGES] = {",
        "    " + ", ".join(_c_literal(x) for x in C) + "};",
        "static const real_t DOPRI5_A[DOPRI5_N_STAGES][DOPRI5_N_STAGES] = {",
    ]
    lines.extend("    {" + ", ".join(_c_literal(x) for x in row) + "}," for row in A)
    lines.append("};")
    for name, vec in (("B", B), ("B_HAT", B_HAT), ("E", E), ("D", D)):
        lines.append(f"static const real_t DOPRI5_{name}[DOPRI5_N_STAGES] = {{")
        lines.append("    " + ", ".join(_c_literal(x) for x in vec) + "};")
    return "\n".join(lines) + "\n"
