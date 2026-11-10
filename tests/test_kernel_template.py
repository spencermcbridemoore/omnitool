"""The kernel templater (``omnitool.kernels``), which is pure Python and so runs in CPU CI.

The last test needs NVRTC and is the device-side check of the precision rule
(ADR 0002, spec §3): an FP32 build of the kernel contains no double-precision
instruction, so nothing in ``dopri5.cu`` promotes silently.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import cuda_available
from omnitool import Problem, Status, kernels
from omnitool.backends import nvrtc
from omnitool.tableaus import dopri5 as tab
from problems import BLOCH, DUFFING

BODY = "dydt[0] = -y[0] * p[0];"


def test_step_modes_are_read_from_the_kernel_source() -> None:
    modes = kernels.step_modes()
    assert set(modes) == {"AUTO", "H0", "FIXED"}
    assert len(set(modes.values())) == 3


@pytest.mark.parametrize(("dtype", "ctype"), [(np.float32, "float"), (np.float64, "double")])
def test_render_supplies_the_compile_time_parameters(dtype: type[np.floating], ctype: str) -> None:
    src = kernels.render_dopri5(BODY, 4, 2, dtype)
    assert f"typedef {ctype} real_t;" in src
    assert "#define N_STATE 4\n" in src
    assert "#define N_PARAMS 2\n" in src
    assert BODY in src
    assert tab.render_c() in src  # the tableau is rendered, not retyped
    eps = re.search(r"#define REAL_EPS \(\(real_t\)(\S+)\)", src)
    assert eps is not None
    assert float(eps.group(1)) == float(np.finfo(dtype).eps)
    for status in Status:
        assert re.search(rf"#define OMNITOOL_STATUS_{status.name}\s+\(?{int(status)}\)?\n", src)
    # prelude first, then the kernel file; rhs is defined before it is used
    assert src.index("static __device__ void rhs(") < src.index('#line 1 "dopri5.cu"')
    assert src.count('extern "C" __global__') == 3


def test_kernel_file_has_no_scalar_type_of_its_own() -> None:
    """``real_t`` is the only scalar type in ``dopri5.cu`` (spec §3)."""
    src = kernels.render_dopri5(BODY, 1, 1, np.float64)
    body = src[src.index('#line 1 "dopri5.cu"') :]
    code = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    assert not re.search(r"\b(float|double)\b", code)
    # every floating literal sits inside REAL(...)
    bare = re.sub(r"REAL\([^()]*\)", "", code)
    assert not re.search(r"\d\.\d|\d[eE][-+]?\d", bare)


def test_render_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="dtype"):
        kernels.render_dopri5(BODY, 1, 1, np.float16)
    with pytest.raises(ValueError, match="n_state"):
        kernels.render_dopri5(BODY, 0, 1, np.float64)
    with pytest.raises(ValueError, match="n_params"):
        kernels.render_dopri5(BODY, 1, -1, np.float64)


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="no CUDA device or CuPy not installed")
@pytest.mark.parametrize("problem", [BLOCH, DUFFING], ids=lambda p: p.name)
def test_fp32_build_has_no_double_precision_instruction(problem: Problem) -> None:
    from cupy_backends.cuda.libs import nvrtc as nv  # type: ignore[import-not-found]

    src = kernels.render_dopri5(problem.rhs_cuda, problem.n_state, problem.n_params, np.float32)
    prog = nv.createProgram(src, "dopri5_fp32.cu", (), ())
    nv.compileProgram(prog, [*nvrtc.NVRTC_OPTIONS, f"-arch=sm_{nvrtc.device_arch()}"])
    ptx = nv.getPTX(prog)
    ptx = ptx.decode() if isinstance(ptx, bytes) else ptx
    assert ".f32" in ptx
    doubles = [line.strip() for line in ptx.splitlines() if ".f64" in line]
    assert not doubles, doubles[:5]
