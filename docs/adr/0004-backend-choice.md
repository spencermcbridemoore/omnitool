# ADR 0004: GPU backend choice for Phase 0

- Status: accepted, pending the Phase 0 spike
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
