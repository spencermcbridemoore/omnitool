"""ADR 0004 spike: CuPy RawModule + NVRTC on the local device.

Throwaway script. It checks the four points ADR 0004 names before the
backend choice is confirmed:

1. compile and launch of a trivial kernel through ``cupy.RawModule``;
2. FP32 and FP64 instantiation of the *same* source via a ``real_t``
   parameter substituted from Python (ADR 0004: Python-side templating of
   plain C, no C++ templates);
3. a working ``__ballot_sync`` warp vote;
4. timing with explicit stream synchronisation, compile time reported
   separately from launch time.

It also checks the fast-math rule of ADR 0002 on the device itself: an FP32
subnormal loaded from memory must survive a multiply and an add (``ftz`` off).
Two compile paths are exercised, because ``cupy.RawModule(code=...)``
appends ``-ftz=true`` to every NVRTC compile (``cupy/cuda/compiler.py``,
``_compile_with_cache_cuda``) and a caller-supplied ``-ftz=false`` does not
override it:

- path A: ``cupy.RawModule(code=src, options=...)``;
- path B: NVRTC called directly through ``cupy_backends.cuda.libs.nvrtc``
  with exactly our options, cubin written to disk, loaded with
  ``cupy.RawModule(path=cubin)``.

Run:  uv run --no-sync python benchmarks/spike/nvrtc_spike.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import cupy as cp  # type: ignore[import-not-found]
    from cupy_backends.cuda.libs import nvrtc  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - spike only
    sys.exit(f"cupy not importable: {exc}")

FAST_MATH_FLAGS = (
    "--use_fast_math",
    "-use_fast_math",
    "--ftz=true",
    "-ftz=true",
    "--prec-div=false",
    "-prec-div=false",
    "--prec-sqrt=false",
    "-prec-sqrt=false",
)

# Plain C. ``real_t`` is the only placeholder; the Python layer substitutes it.
SOURCE = r"""
typedef @REAL_T@ real_t;

extern "C" __global__
void axpy(const real_t a, const real_t* x, const real_t* y, real_t* out, const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}

extern "C" __global__
void ballot(const int* flags, unsigned int* masks, const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int full = 0xffffffffu;
    /* every lane of a full warp participates; n is a multiple of 32 */
    const unsigned int m = __ballot_sync(full, i < n && flags[i] != 0);
    if (i < n && (i % 32) == 0) {
        masks[i / 32] = m;
    }
}

extern "C" __global__
void subnormal(const real_t* tiny, const real_t* one, const real_t* zero, real_t* out)
{
    /* operands come from memory so nothing can be constant-folded */
    out[0] = tiny[0] * one[0];
    out[1] = tiny[0] + zero[0];
}
"""

OPTIONS: tuple[str, ...] = ("-std=c++11", "-lineinfo")


def check_no_fast_math(options: tuple[str, ...]) -> None:
    bad = [o for o in options if o in FAST_MATH_FLAGS]
    assert not bad, f"fast-math flags present: {bad}"


def build_rawmodule(src: str) -> tuple[cp.RawModule, float]:
    t0 = time.perf_counter()
    mod = cp.RawModule(code=src, options=OPTIONS, backend="nvrtc")
    mod.compile()
    return mod, time.perf_counter() - t0


def build_direct_nvrtc(src: str, arch: str, out_dir: Path, tag: str) -> tuple[cp.RawModule, float]:
    t0 = time.perf_counter()
    prog = nvrtc.createProgram(src, f"{tag}.cu", (), ())
    try:
        nvrtc.compileProgram(prog, [*OPTIONS, f"-arch=sm_{arch}"])
    except Exception:
        print(nvrtc.getProgramLog(prog))
        raise
    cubin = nvrtc.getCUBIN(prog)
    path = out_dir / f"{tag}.cubin"
    path.write_bytes(cubin)
    mod = cp.RawModule(path=str(path))
    mod.get_function("axpy")  # forces the module load
    return mod, time.perf_counter() - t0


def run_axpy(mod: cp.RawModule, dtype: type, n: int, reps: int) -> tuple[float, float]:
    kern = mod.get_function("axpy")
    rng = np.random.default_rng(0)
    x_h = rng.standard_normal(n).astype(dtype)
    y_h = rng.standard_normal(n).astype(dtype)
    a = dtype(1.5)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        x = cp.asarray(x_h)
        y = cp.asarray(y_h)
        out = cp.empty_like(x)
        block = 128
        grid = (n + block - 1) // block
        args = (a, x, y, out, np.int32(n))
        kern((grid,), (block,), args, stream=stream)  # warm-up
        stream.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            kern((grid,), (block,), args, stream=stream)
        stream.synchronize()  # explicit synchronisation before stopping the timer
        t1 = time.perf_counter()
        out_h = cp.asnumpy(out)
    expected = (a * x_h.astype(np.float64) + y_h.astype(np.float64)).astype(dtype)
    scale = float(np.max(np.abs(expected)))
    rel = float(np.max(np.abs(out_h - expected))) / (scale * float(np.finfo(dtype).eps))
    return (t1 - t0) / reps, rel  # rel is in units of eps * max|expected|


def run_ballot(mod: cp.RawModule) -> None:
    kern = mod.get_function("ballot")
    n = 256  # 8 full warps
    flags_h = np.zeros(n, dtype=np.int32)
    for w in range(n // 32):
        for lane in range(32):
            flags_h[w * 32 + lane] = int((lane * (w + 1)) % 3 == 0)
    flags = cp.asarray(flags_h)
    masks = cp.zeros(n // 32, dtype=np.uint32)
    kern((n // 128,), (128,), (flags, masks, np.int32(n)))
    cp.cuda.Device().synchronize()
    got = cp.asnumpy(masks)
    expected = np.zeros(n // 32, dtype=np.uint32)
    for w in range(n // 32):
        bits = 0
        for lane in range(32):
            if flags_h[w * 32 + lane]:
                bits |= 1 << lane
        expected[w] = bits
    assert np.array_equal(got, expected), f"ballot mismatch: {got} vs {expected}"


def run_subnormal(mod: cp.RawModule, dtype: type) -> tuple[float, float]:
    kern = mod.get_function("subnormal")
    tiny = cp.asarray(np.array([1e-40], dtype=dtype))  # subnormal in FP32, normal in FP64
    one = cp.asarray(np.array([1.0], dtype=dtype))
    zero = cp.zeros(1, dtype=dtype)
    out = cp.zeros(2, dtype=dtype)
    kern((1,), (1,), (tiny, one, zero, out))
    cp.cuda.Device().synchronize()
    r = cp.asnumpy(out)
    return float(r[0]), float(r[1])


def main() -> int:
    check_no_fast_math(OPTIONS)
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    arch = f"{props['major']}{props['minor']}"
    print(f"device: {props['name'].decode()}  cc {props['major']}.{props['minor']}")
    print(
        f"cupy {cp.__version__}, CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}, "
        f"driver {cp.cuda.runtime.driverGetVersion()}, nvrtc {nvrtc.getVersion()}"
    )
    print(f"our NVRTC options: {list(OPTIONS)}  (no fast-math flags)")

    n = 1 << 20
    reps = 200
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for path_name in ("A: RawModule(code=...)", "B: direct NVRTC -> RawModule(path=cubin)"):
            print(f"--- compile path {path_name}")
            for real_t, dtype in (("float", np.float32), ("double", np.float64)):
                src = SOURCE.replace("@REAL_T@", real_t)
                if path_name.startswith("A"):
                    mod, t_compile = build_rawmodule(src)
                else:
                    mod, t_compile = build_direct_nvrtc(src, arch, Path(tmp), real_t)
                t_launch, rel = run_axpy(mod, dtype, n, reps)
                mul, add = run_subnormal(mod, dtype)
                print(
                    f"[{real_t:6s}] compile {t_compile * 1e3:7.1f} ms | axpy n={n} "
                    f"{t_launch * 1e6:6.1f} us/launch (mean of {reps}, stream-synced) | "
                    f"max rel err {rel:.2f} eps | 1e-40*1 = {mul:.3e}, 1e-40+0 = {add:.3e}"
                )
                if rel > 2.0:
                    print("  FAIL: axpy result more than 2 eps off")
                    ok = False
                if real_t == "float":
                    run_ballot(mod)
                    print("  __ballot_sync: 8 warps, masks match host expectation")
                    if mul == 0.0 or add == 0.0:
                        flag = "FAIL" if path_name.startswith("B") else "NOTE"
                        print(f"  {flag}: FP32 subnormal flushed to zero on this path (ftz on)")
                        ok = ok and flag == "NOTE"
    print("SPIKE", "PASS (path B validated; path A flushes subnormals)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
