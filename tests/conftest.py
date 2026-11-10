"""Shared fixtures.

The ``backend`` fixture parameterises every test that takes it over the
backends that are **implemented in this checkout**
(:func:`omnitool.backends.available`), never over names the spec merely
reserves (ADR 0005), so an unbuilt backend produces no xfails or skips.
``cuda`` is additionally skipped when CuPy or a device is
absent, so the same files run on a laptop, in GitHub Actions and on the
self-hosted 4090 runner.

Restrict with ``pytest --backend cpu``.
"""

from __future__ import annotations

import pytest

import omnitool.backends


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--backend",
        action="append",
        choices=omnitool.backends.KNOWN,
        default=None,
        help="restrict the backend fixture to these backends (repeatable)",
    )


def cuda_available() -> bool:
    try:
        import cupy  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # driver missing, no device, etc.
        return False


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "backend" not in metafunc.fixturenames:
        return
    implemented = omnitool.backends.available()
    wanted = metafunc.config.getoption("--backend") or list(implemented)
    params = []
    for name in implemented:
        if name not in wanted:
            continue
        marks: list[pytest.MarkDecorator] = []
        if name == "cuda":
            marks.append(pytest.mark.gpu)
            if not cuda_available():
                marks.append(pytest.mark.skip(reason="no CUDA device or CuPy not installed"))
        params.append(pytest.param(name, id=name, marks=marks))
    metafunc.parametrize("backend", params)


@pytest.fixture(params=("fp32", "fp64"))
def precision(request: pytest.FixtureRequest) -> str:
    """Both precisions are first-class (ADR 0002)."""
    return request.param
