"""T5 — dense-output accuracy on S2 (spec §8).

One fixed step from ``y0`` of size ``h`` in {0.2, 0.1, 0.05}, interpolated at
``theta = 0.5`` and compared with DOP853 at ``t = h/2``. The T4 clause of T5
(bound at save times that are not step endpoints) runs with T4.
"""

from __future__ import annotations

import numpy as np
import pytest

from omnitool.reference import dopri5_numpy as ref
from problems import S2, solve

H_VALUES = (0.2, 0.1, 0.05)


def one_step_dense(backend: str, h: float):
    """``(y0, y1, dense(0), dense(0.5), dense(1))`` for one step of size ``h`` on S2."""
    if backend != "cpu":
        pytest.xfail(f"device dense() entry point for {backend!r} not implemented yet")
    y0 = S2.y0.astype(np.float64)
    p = S2.params.astype(np.float64)
    rhs = S2.problem.rhs_numpy
    k1 = rhs(np.float64(0.0), y0, p)
    y1, k, _ = ref.step(rhs, np.float64(0.0), y0, np.float64(h), p, k1, np.float64)
    r = ref.dense_coefficients(y0, y1, np.float64(h), k, np.float64)
    return (
        y0,
        y1,
        ref.dense_eval(r, np.float64(0.0), np.float64),
        ref.dense_eval(r, np.float64(0.5), np.float64),
        ref.dense_eval(r, np.float64(1.0), np.float64),
    )


def test_dense_output_order(backend: str) -> None:
    errs = []
    for h in H_VALUES:
        _y0, _y1, _d0, d_half, _d1 = one_step_dense(backend, h)
        y_ref = S2.reference(np.array([h / 2]))[0]
        errs.append(float(np.max(np.abs(d_half - y_ref))))
    for i in range(len(H_VALUES) - 1):
        ratio = errs[i] / errs[i + 1]
        assert ratio >= 2**4.5, f"h={H_VALUES[i]}: error ratio {ratio:.2f}, errors {errs}"


@pytest.mark.parametrize("h", H_VALUES)
def test_dense_endpoints_bitwise(backend: str, h: float) -> None:
    y0, y1, d0, _d_half, d1 = one_step_dense(backend, h)
    assert np.array_equal(d0, y0)
    assert np.array_equal(d1, y1)


@pytest.mark.parametrize("h", H_VALUES)
def test_driver_uses_same_dense_output(backend: str, h: float) -> None:
    """The driver's save at ``h/2`` is the same evaluation as the direct call."""
    res = solve(
        backend,
        S2,
        precision="fp64",
        t_end=h,
        t_save=np.array([0.0, h / 2, h]),
        fixed_step=h,
    )
    _y0, y1, _d0, d_half, _d1 = one_step_dense(backend, h)
    assert res.status[0] == 0
    assert res.n_accepted[0] == 1
    assert np.array_equal(res.y[0, 0], S2.y0)
    assert np.array_equal(res.y[0, 1], d_half)
    assert np.array_equal(res.y[0, 2], y1)
