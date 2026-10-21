# ADR 0005: No host-compiled `cudasim` backend in Phase 0

- Status: accepted
- Date: 2026-10-21
- Amends: ADR 0004 (consequences), spec §5

## Context

The handoff listed three test backends, `cpu` / `cudasim` / `cuda`, where
`cudasim` meant numba's CPU simulator. ADR 0004 chose CUDA C through NVRTC,
for which no simulator exists, and sketched a replacement: compile the same
`.cu` template as host C++ through a compatibility header that maps
`threadIdx` and the warp votes onto a sequential 32-lane loop. The Phase 0
kernel (DOPRI5, thread-per-member) uses no warp intrinsics and no shared
memory; the trajectory front door of Phase 2 uses both. The spike confirmed
the kernels run on the workstation's RTX 4090, and the gated GPU workflow
exists.

## Decision

`cudasim` is **not built in Phase 0**. The backend registry
(`omnitool.backends.available()`) lists only implemented backends, and the
pytest `backend` fixture parameterises over that registry, so an unbuilt
backend produces no xfails. Phase 0 verifies the kernel through `cuda` on the
4090 (locally before every kernel push, and on the dispatched runner) against
the `cpu` oracle.

**Revisit trigger:** when the first kernel with `__ballot_sync` or shared
memory is specified (Phase 2 events), or earlier if a kernel defect reaches
the 4090 that host execution of the kernel source would have caught in CPU
CI. The revisit must weigh the compatibility header's cost against the
alternative of a longer-lived GPU runner.

## Why not now

Building the header for the Phase 0 kernel is cheap, but it buys little:
that kernel is a direct templating of the algorithm the NumPy oracle already
checks, and the templating and indexing errors it would catch surface on the
first `cuda` run of the same tests. The expensive part of the header, warp
emulation, is exactly what Phase 0 does not need, so building it now means
building it twice. The real cost of deferral is that CPU-only CI cannot see
kernel regressions; the mitigations are the local 4090 rule in `AGENTS.md`
and the dispatched runner.

## Consequences

- Spec §5 has two backends; `cudasim` is removed from CI and the fixture.
- `omnitool.backends.KNOWN` is `("cpu", "cuda")`; a request for `cuda`
  before it is registered raises `NotImplementedError`, an unknown name
  raises `ValueError`.
- Kernel pull requests state in their description that the `cuda` tests
  passed on the 4090, with the driver and CuPy versions.
