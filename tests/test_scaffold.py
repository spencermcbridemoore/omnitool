"""Smoke tests for the scaffold itself. Replaced by real tests in Phase 0."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import omnitool

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_citation() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", omnitool.__version__)
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {omnitool.__version__}" in cff


def test_backend_fixture_parametrises(backend: str) -> None:
    import omnitool.backends

    assert backend in omnitool.backends.available()
    if backend == "cuda":
        import cupy  # type: ignore[import-not-found]

        assert cupy.cuda.runtime.getDeviceCount() > 0


def test_precision_fixture(precision: str) -> None:
    assert precision in {"fp32", "fp64"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_reference_package_is_independent_of_backends() -> None:
    """ADR 0003: the NumPy reference never imports the backends or the kernels."""
    ref_dir = ROOT / "src" / "omnitool" / "reference"
    forbidden = ("omnitool.backends", "omnitool.kernels", "cupy", "scipy")
    for path in ref_dir.rglob("*.py"):
        for name in _imported_modules(path):
            assert not name.startswith(forbidden), f"{path} imports {name}"


def test_subpackages_import() -> None:
    for name in ("backends", "reference", "tableaus", "kernels"):
        importlib.import_module(f"omnitool.{name}")
