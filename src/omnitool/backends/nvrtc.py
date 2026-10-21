"""Compile CUDA C through NVRTC with the project's IEEE option set (ADR 0004).

``cupy.RawModule(code=...)`` appends ``-ftz=true`` to every compile after the
caller's options, which flushes FP32 subnormals and breaks ADR 0002. This
module calls NVRTC directly through CuPy's binding with exactly
:data:`NVRTC_OPTIONS`, writes the cubin to a per-process directory and loads
it with ``cupy.RawModule(path=...)``. CuPy still owns arrays, streams and
memory; this is the only place NVRTC options are defined.

CuPy is imported lazily so the package imports without it.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["FAST_MATH_FLAGS", "NVRTC_OPTIONS", "compile_module", "device_arch"]

NVRTC_OPTIONS: tuple[str, ...] = (
    "-std=c++11",
    "-lineinfo",
    "-ftz=false",
    "-prec-div=true",
    "-prec-sqrt=true",
    "-fmad=true",
)
"""The validated option set; ``-arch=sm_XY`` is appended per device."""

FAST_MATH_FLAGS: frozenset[str] = frozenset(
    {
        "--use_fast_math",
        "-use_fast_math",
        "--ftz=true",
        "-ftz=true",
        "--prec-div=false",
        "-prec-div=false",
        "--prec-sqrt=false",
        "-prec-sqrt=false",
    }
)
"""Flags that break IEEE 754 semantics; never present in a validation build."""

_CUBIN_DIR: tempfile.TemporaryDirectory[str] | None = None


def _cubin_dir() -> Path:
    global _CUBIN_DIR
    if _CUBIN_DIR is None:
        _CUBIN_DIR = tempfile.TemporaryDirectory(prefix="omnitool-nvrtc-")
    return Path(_CUBIN_DIR.name)


def device_arch(device_id: int | None = None) -> str:
    """``"XY"`` for ``sm_XY`` of the given (default: current) device."""
    import cupy as cp  # type: ignore[import-not-found]

    dev = cp.cuda.Device() if device_id is None else cp.cuda.Device(device_id)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    return f"{props['major']}{props['minor']}"


def compile_module(source: str, name: str, arch: str | None = None) -> Any:
    """Compile ``source`` for ``sm_<arch>`` and return a loaded ``cupy.RawModule``.

    Raises ``RuntimeError`` with the NVRTC log on a compile error. The option
    set is :data:`NVRTC_OPTIONS` plus ``-arch``; nothing else is ever added.
    """
    import cupy as cp  # type: ignore[import-not-found]
    from cupy_backends.cuda.libs import nvrtc  # type: ignore[import-not-found]

    arch = device_arch() if arch is None else arch
    options = [*NVRTC_OPTIONS, f"-arch=sm_{arch}"]
    bad = [o for o in options if o in FAST_MATH_FLAGS]
    if bad:
        raise RuntimeError(f"fast-math flags in NVRTC options: {bad}")

    digest = hashlib.sha256(("\0".join(options) + "\0" + source).encode()).hexdigest()[:16]
    path = _cubin_dir() / f"{name}-{digest}.cubin"
    if not path.exists():
        prog = nvrtc.createProgram(source, f"{name}.cu", (), ())
        try:
            nvrtc.compileProgram(prog, options)
        except Exception as exc:
            log = nvrtc.getProgramLog(prog)
            raise RuntimeError(f"NVRTC failed for {name}:\n{log}") from exc
        path.write_bytes(nvrtc.getCUBIN(prog))
    return cp.RawModule(path=str(path))
