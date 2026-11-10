"""T9 — GPU-vs-CPU agreement (spec §8).

The reference-independence half of T9 lives in ``test_scaffold.py``
(``test_reference_package_is_independent_of_backends``). This file compares
every implemented device backend with ``cpu`` on the T4 problems. With no
device backend registered, no device present, or the device backend excluded
by ``--backend``, the comparison is skipped, not xfailed.
"""

from __future__ import annotations

import numpy as np
import pytest

import omnitool.backends
from conftest import cuda_available
from omnitool import Status
from problems import ALL_CASES, Case, reference_on_grid, save_grid, solve

TOLERANCE_PAIRS = {
    "fp64": [(1e-7, 1e-9), (1e-10, 1e-12)],
    "fp32": [(1e-4, 1e-6), (2e-5, 1e-7)],
}
DEVICE_BACKENDS = [b for b in omnitool.backends.available() if b != "cpu"]
if not DEVICE_BACKENDS:
    pytest.skip("no device backend implemented yet (T9 cross-backend)", allow_module_level=True)
CASES = [pytest.param(c, id=c.name) for c in ALL_CASES]


@pytest.fixture(params=[pytest.param(b, marks=pytest.mark.gpu) for b in DEVICE_BACKENDS])
def device_backend(request: pytest.FixtureRequest) -> str:
    name = request.param
    wanted = request.config.getoption("--backend")
    if wanted and name not in wanted:
        pytest.skip(f"{name} excluded by --backend")
    if name == "cuda" and not cuda_available():
        pytest.skip("no CUDA device or CuPy not installed")
    return name


@pytest.mark.parametrize("pair_index", [0, 1], ids=["tol0", "tol1"])
@pytest.mark.parametrize("case", CASES)
def test_device_agrees_with_cpu(
    device_backend: str, precision: str, case: Case, pair_index: int
) -> None:
    rtol, atol = TOLERANCE_PAIRS[precision][pair_index]
    t_save = save_grid(case.t_end)
    y_ref = reference_on_grid(case, t_save)
    bound = 10.0 * (rtol * float(np.max(np.abs(y_ref))) + atol)
    cpu = solve("cpu", case, precision=precision, t_save=t_save, rtol=rtol, atol=atol)
    dev = solve(device_backend, case, precision=precision, t_save=t_save, rtol=rtol, atol=atol)
    assert cpu.status[0] == Status.SUCCESS
    assert dev.status[0] == Status.SUCCESS
    gap = np.max(np.abs(dev.y[0].astype(np.float64) - cpu.y[0].astype(np.float64)))
    assert gap <= 2 * bound, f"{device_backend} vs cpu: {gap:.3e} > {2 * bound:.3e}"
    n_cpu = int(cpu.n_accepted[0])
    assert abs(int(dev.n_accepted[0]) - n_cpu) <= max(2, 0.02 * n_cpu)
