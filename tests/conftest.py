"""Shared fixtures.

The ``backend`` fixture parameterises every test that takes it over the three
execution backends of ``docs/PHASE0_SPEC.md`` section 5. ``cuda`` skips
cleanly when CuPy or a device is absent, so the same test files run on a
laptop, in GitHub Actions and on the self-hosted 4090 runner.

Restrict with ``pytest --backend cpu --backend cudasim``.
"""

from __future__ import annotations

import pytest

BACKENDS = ("cpu", "cudasim", "cuda")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--backend",
        action="append",
        choices=BACKENDS,
        default=None,
        help="restrict the backend fixture to these backends (repeatable)",
    )


def _cuda_available() -> bool:
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
    wanted = metafunc.config.getoption("--backend") or list(BACKENDS)
    params = []
    for name in BACKENDS:
        if name not in wanted:
            continue
        marks: list[pytest.MarkDecorator] = []
        if name == "cuda":
            marks.append(pytest.mark.gpu)
            if not _cuda_available():
                marks.append(pytest.mark.skip(reason="no CUDA device or CuPy not installed"))
        params.append(pytest.param(name, id=name, marks=marks))
    metafunc.parametrize("backend", params)


@pytest.fixture(params=("fp32", "fp64"))
def precision(request: pytest.FixtureRequest) -> str:
    """Both precisions are first-class (ADR 0002)."""
    return request.param
