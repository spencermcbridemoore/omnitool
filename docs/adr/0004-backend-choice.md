# ADR 0004: GPU backend choice for Phase 0

- Status: accepted; spike passed 2026-10-20 (see "Spike result" below)
- Date: 2026-10-19

## Context

Three ways to run per-member adaptive kernels on NVIDIA hardware were
surveyed: numba-cuda (maintenance mode through CUDA 13), numba-cuda-mlir
(cubie's author needed a 2,045-line shim and a private wheel), and CUDA C
through CuPy `RawModule` with NVRTC. The trajectory front door needs
`__ballot_sync`, warp-cooperative bisection, per-thread RNG and control of
FP64 semantics and registers. The linear front door needs per-member sparse LU
and a block-per-member mode.

## Decision

Phase 0 kernels are **CUDA C compiled through CuPy `RawModule` + NVRTC**. The
problem-specific right-hand side is inserted by **Python-side templating** of
the kernel source. CuPy owns device arrays, streams and memory pools; the core
never calls the driver API directly.

Kernel sources stay **backend-agnostic**: stepping, controller and
dense-output code are plain C functions parameterised by scalar type and state
size, so a numba-cuda path for Python-authored RHS can be added later.

A **one-day spike** at the start of Phase 0 confirms the choice: compile and
launch through `RawModule`, FP32 and FP64 instantiation of one template, a
working `__ballot_sync` warp vote, and timing with explicit stream
synchronisation. If any point fails, this ADR is revised, not worked around.

## Consequences

- Runtime dependency: `cupy` for the chosen CUDA major version. NVRTC ships in
  the CuPy wheels, so no local toolkit is needed.
- The `cudasim` backend compiles the same `.cu` template as host C++ through
  a shim header (`__device__`, `__global__` and the thread-index macros
  redefined) and runs members in a serial loop, so the real kernel logic is
  tested on every push. Only `cuda` exercises NVRTC and the hardware.
- Kernel text lives in `.cu` files and templating in Python, so kernels stay
  readable, diffable and compilable with `nvcc`.

## Spike result (2026-10-20)

Run on the workstation RTX 4090 (cc 8.9), driver 610.88 (CUDA 13.3), CuPy
14.2.0 (`cupy-cuda13x`), NVRTC 13.1, with `benchmarks/spike/nvrtc_spike.py`.
NVRTC options passed: `-std=c++11 -lineinfo`; no fast-math flags.

| point | result |
|---|---|
| compile and launch through `RawModule` | passes; `axpy` on 2^20 elements exact to ≤ 0.5 ulp |
| FP32 and FP64 from one source via a `real_t` typedef substituted in Python | passes |
| `__ballot_sync` full-warp vote | passes; 8 warp masks equal the host expectation |
| timing with explicit stream synchronisation, compile time separate | passes; compile 26–112 ms, launch 7–9 µs for 2^20 elements |

**Finding.** `cupy.RawModule(code=...)` appends `-ftz=true` to every NVRTC
compile (`cupy/cuda/compiler.py`, `_compile_with_cache_cuda`), after the
caller's options, so a caller's `-ftz=false` is overridden. On the device an
FP32 subnormal loaded from memory came back as zero after `x * 1` and
`x + 0`. That violates the fast-math rule of ADR 0002 ("no `ftz`"). FP64 is
unaffected.

Calling NVRTC directly through `cupy_backends.cuda.libs.nvrtc` with exactly
our options, writing the cubin, and loading it with `RawModule(path=...)`
keeps subnormals and passes every point above at the same launch cost.

**Decision stands, with one amendment to the consequences.** The `cuda`
backend compiles through the NVRTC binding directly and loads the cubin with
`RawModule(path=...)`; it never uses `RawModule(code=...)`. CuPy still owns
arrays, streams and memory. A test in the verification layer asserts on the
device that an FP32 subnormal survives a multiply.
