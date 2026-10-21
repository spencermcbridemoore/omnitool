"""The NVRTC option set of ADR 0004 and the FP32 subnormal probe (ADR 0002).

The option checks run everywhere. The device probe runs only where CuPy and
a CUDA device exist (the workstation, the self-hosted runner); it is the
permanent form of the spike's finding that ``RawModule(code=...)`` flushes
subnormals.
"""

from __future__ import annotations

import numpy as np
import pytest

from omnitool.backends import nvrtc

SUBNORMAL_SOURCE = r"""
extern "C" __global__
void subnormal(const float* tiny, const float* one, const float* zero, float* out)
{
    /* operands come from memory so nothing can be constant-folded */
    out[0] = tiny[0] * one[0];
    out[1] = tiny[0] + zero[0];
}
"""


def test_option_set_is_ieee() -> None:
    opts = set(nvrtc.NVRTC_OPTIONS)
    assert {"-ftz=false", "-prec-div=true", "-prec-sqrt=true", "-fmad=true"} <= opts
    assert not (opts & nvrtc.FAST_MATH_FLAGS)
    assert not any(o.startswith("-arch") for o in opts), "arch is appended per device"


def _device() -> bool:
    try:
        import cupy  # type: ignore[import-not-found]

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _device(), reason="no CUDA device or CuPy not installed")
def test_fp32_subnormal_survives_on_device() -> None:
    import cupy as cp  # type: ignore[import-not-found]

    mod = nvrtc.compile_module(SUBNORMAL_SOURCE, "subnormal_probe")
    kern = mod.get_function("subnormal")
    tiny_h = np.array([1e-40], dtype=np.float32)
    assert tiny_h[0] != 0
    assert abs(float(tiny_h[0])) < np.finfo(np.float32).tiny  # a subnormal
    tiny = cp.asarray(tiny_h)
    one = cp.asarray(np.array([1.0], dtype=np.float32))
    zero = cp.zeros(1, dtype=np.float32)
    out = cp.zeros(2, dtype=np.float32)
    kern((1,), (1,), (tiny, one, zero, out))
    cp.cuda.Device().synchronize()
    got = cp.asnumpy(out)
    assert got[0] == tiny_h[0], f"1e-40 * 1 flushed to {got[0]!r}"
    assert got[1] == tiny_h[0], f"1e-40 + 0 flushed to {got[1]!r}"
