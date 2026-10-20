"""T1 — tableau identities for the Dormand–Prince 5(4) transcription.

Everything here runs in exact rational arithmetic on the transcribed
coefficients (spec §8, T1). The order conditions are generated from rooted
trees rather than typed by hand, so the same checker serves later tableaus.
The dense-output coefficients are checked by the continuous order conditions
of the fourth-order extension as polynomial identities in ``theta``.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction as F
from itertools import combinations_with_replacement, product

import numpy as np
import pytest

from omnitool.tableaus import dopri5 as tab

# --------------------------------------------------------------------------
# Rooted trees and elementary weights (Butcher series)
# --------------------------------------------------------------------------

Tree = tuple  # a tree is the sorted tuple of its subtrees; the leaf is ()


def trees_of_order(n: int) -> list[Tree]:
    """All unordered rooted trees with ``n`` vertices."""
    if n == 1:
        return [()]
    out: list[Tree] = []
    seen: set[Tree] = set()
    for forest in _forests(n - 1):
        t = tuple(sorted(forest))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _forests(total: int) -> list[tuple[Tree, ...]]:
    """All multisets of trees whose orders sum to ``total``."""
    if total == 0:
        return [()]
    results: set[tuple[Tree, ...]] = set()
    for part in _partitions(total):
        by_order = Counter(part)
        choices = [
            list(combinations_with_replacement(trees_of_order(k), m)) for k, m in by_order.items()
        ]
        for combo in product(*choices):
            results.add(tuple(sorted(t for group in combo for t in group)))
    return sorted(results)


def _partitions(n: int, largest: int | None = None) -> list[tuple[int, ...]]:
    largest = n if largest is None else largest
    if n == 0:
        return [()]
    parts: list[tuple[int, ...]] = []
    for k in range(min(n, largest), 0, -1):
        for rest in _partitions(n - k, k):
            parts.append((k, *rest))
    return parts


def order(t: Tree) -> int:
    return 1 + sum(order(s) for s in t)


def gamma(t: Tree) -> int:
    """Density γ(t): γ(τ) = 1, γ([t1..tm]) = |t| Π γ(tk)."""
    g = order(t)
    for s in t:
        g *= gamma(s)
    return g


def phi(t: Tree, a: list[list[F]], c: list[F]) -> list[F]:
    """Elementary weight vector Φ(t)_i = Π_k (A Φ(tk))_i, with Φ(τ) = 1."""
    s = len(c)
    vec = [F(1)] * s
    for sub in t:
        inner = phi(sub, a, c)
        a_inner = [sum(a[i][j] * inner[j] for j in range(s)) for i in range(s)]
        vec = [vec[i] * a_inner[i] for i in range(s)]
    return vec


A = [list(row) for row in tab.A]
C = list(tab.C)
B = list(tab.B)
B_HAT = list(tab.B_HAT)
D = list(tab.D)
S = tab.N_STAGES


def test_tree_counts() -> None:
    """Sanity check of the generator: 1, 1, 2, 4, 9 trees of orders 1–5."""
    assert [len(trees_of_order(n)) for n in range(1, 6)] == [1, 1, 2, 4, 9]


# --------------------------------------------------------------------------
# T1: discrete tableau identities
# --------------------------------------------------------------------------


def test_shapes_and_explicitness() -> None:
    assert len(C) == len(B) == len(B_HAT) == len(D) == S == 7
    assert all(len(row) == S for row in A)
    for i in range(S):
        for j in range(i, S):
            assert A[i][j] == 0, f"a[{i}][{j}] must vanish for an explicit method"


def test_weights_sum_to_one() -> None:
    assert sum(B) == 1
    assert sum(B_HAT) == 1


def test_row_sums_equal_nodes() -> None:
    for i in range(S):
        assert sum(A[i]) == C[i], f"row {i}: sum a = {sum(A[i])} != c = {C[i]}"


def test_fsal() -> None:
    assert A[6] == B
    assert C[6] == 1
    assert B[6] == 0


def test_error_weights_are_differences() -> None:
    assert list(tab.E) == [b - bh for b, bh in zip(B, B_HAT, strict=True)]
    assert sum(tab.E) == 0
    assert any(e != 0 for e in tab.E)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_order_conditions_b(p: int) -> None:
    """b satisfies every order condition through order 5, exactly."""
    for t in trees_of_order(p):
        lhs = sum(b * ph for b, ph in zip(B, phi(t, A, C), strict=True))
        assert lhs == F(1, gamma(t)), f"tree {t}: {lhs} != 1/{gamma(t)}"


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_order_conditions_b_hat(p: int) -> None:
    """b̂ satisfies every order condition through order 4, exactly."""
    for t in trees_of_order(p):
        lhs = sum(b * ph for b, ph in zip(B_HAT, phi(t, A, C), strict=True))
        assert lhs == F(1, gamma(t)), f"tree {t}: {lhs} != 1/{gamma(t)}"


def test_b_hat_is_not_order_five() -> None:
    """The embedded formula must be strictly lower order or the estimate is dead."""
    failures = 0
    for t in trees_of_order(5):
        lhs = sum(b * ph for b, ph in zip(B_HAT, phi(t, A, C), strict=True))
        failures += lhs != F(1, gamma(t))
    assert failures > 0


# --------------------------------------------------------------------------
# Dense output: continuous order conditions as polynomial identities
# --------------------------------------------------------------------------

Poly = list  # coefficients in theta, lowest degree first, Fractions


def _pmul(p: Poly, q: Poly) -> Poly:
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i + j] += x * y
    return out


def _padd(p: Poly, q: Poly) -> Poly:
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else F(0)) + (q[i] if i < len(q) else F(0)) for i in range(n)]


def _pscale(p: Poly, s: F) -> Poly:
    return [x * s for x in p]


def _strip(p: Poly) -> Poly:
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def dense_weights() -> list[Poly]:
    """b_i(θ) of Hairer's continuous extension, from the rcont form.

    y(θ) = y0 + θ dy + θ(1-θ) r3 + θ²(1-θ) r4 + θ²(1-θ)² r5 with
    dy = h Σ b_i k_i, r3 = h k1 - dy, r4 = dy - h k7 - r3, r5 = h Σ d_i k_i.
    """
    th = [F(0), F(1)]
    th_1mth = [F(0), F(1), F(-1)]
    th2_1mth = [F(0), F(0), F(1), F(-1)]
    th2_1mth2 = [F(0), F(0), F(1), F(-2), F(1)]
    weights: list[Poly] = []
    for i in range(S):
        d1 = F(1) if i == 0 else F(0)
        d7 = F(1) if i == 6 else F(0)
        w = _pscale(th, B[i])
        w = _padd(w, _pscale(th_1mth, d1 - B[i]))
        w = _padd(w, _pscale(th2_1mth, 2 * B[i] - d1 - d7))
        w = _padd(w, _pscale(th2_1mth2, D[i]))
        weights.append(_strip(w))
    return weights


def _eval(p: Poly, x: F) -> F:
    return sum((coef * x**k for k, coef in enumerate(p)), F(0))


def test_dense_weights_at_endpoints() -> None:
    w = dense_weights()
    for i in range(S):
        assert _eval(w[i], F(0)) == 0
        assert _eval(w[i], F(1)) == B[i]


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_dense_output_order_conditions(p: int) -> None:
    """Σ b_i(θ) Φ_i(t) == θ^|t| / γ(t) for every tree of order ≤ 4."""
    w = dense_weights()
    for t in trees_of_order(p):
        ph = phi(t, A, C)
        lhs: Poly = [F(0)]
        for i in range(S):
            lhs = _padd(lhs, _pscale(w[i], ph[i]))
        rhs: Poly = [F(0)] * p + [F(1, gamma(t))]
        assert _strip(lhs) == _strip(rhs), f"tree {t}: {lhs} != {rhs}"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_LITERAL = re.compile(r"\(real_t\)(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")


def _flat_rationals() -> list[F]:
    out = list(tab.C)
    for row in tab.A:
        out.extend(row)
    out.extend(tab.B)
    out.extend(tab.B_HAT)
    out.extend(tab.E)
    out.extend(tab.D)
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["fp32", "fp64"])
def test_rendered_header_within_one_ulp(dtype: type[np.floating]) -> None:
    text = tab.render_c()
    literals = [dtype(tok) for tok in _LITERAL.findall(text)]
    exact = _flat_rationals()
    assert len(literals) == len(exact) == 7 + 49 + 4 * 7
    for lit, frac in zip(literals, exact, strict=True):
        ref = dtype(float(frac))
        assert abs(float(lit) - float(ref)) <= abs(float(np.spacing(ref))), (lit, frac)
    assert "float" not in text.replace("(real_t)", "")
    assert "double" not in text


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["fp32", "fp64"])
def test_as_arrays_matches_rationals(dtype: type[np.floating]) -> None:
    arrs = tab.as_arrays(dtype)
    assert arrs["a"].shape == (7, 7)
    for name, exact in (
        ("c", tab.C),
        ("b", tab.B),
        ("b_hat", tab.B_HAT),
        ("e", tab.E),
        ("d", tab.D),
    ):
        for x, frac in zip(arrs[name], exact, strict=True):
            assert x == dtype(float(frac)), (name, x, frac)
    for i in range(7):
        for j in range(7):
            assert arrs["a"][i, j] == dtype(float(tab.A[i][j]))
