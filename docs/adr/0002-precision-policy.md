# ADR 0002: Precision policy

- Status: accepted
- Date: 2026-10-19

## Context

Stiff implicit steps lose 4–6 digits to conditioning in FP32. cubie's own
documentation caps FP32 at rtol ≈ 1e-6, while spectroscopy needs rtol 1e-7 /
atol 1e-9, and weak-RF differential signals (4.5e-8 of the carrier) fail
outright in FP32 without one step of FP64 refinement. Fast-math flags
reorder sums and break IEEE 754. The local RTX 4090 runs FP64 at 1/64 rate;
the H100 on UCF Newton runs it at 1/2 rate.

## Decision

- **FP64 is a first-class, benchmarked configuration.** Every kernel, test and
  benchmark is parameterised over FP32 and FP64.
- **Fast-math is off by default.** Flags that break IEEE 754 (`arcp`, `afn`,
  `ftz`, `reassoc`, NVRTC's `--use_fast_math`) are never on in validation
  builds. FMA contraction is allowed.
- **GPU-vs-CPU tests use tolerances, never bitwise equality**, because FMA
  contraction alone changes the last bits.
- **Every reported result states its precision mode** (FP64, FP32, or mixed
  with FP64 refinement) and its FMA setting. Nothing downcasts silently.
- **Hardware split:** the 4090 is for FP32 development and correctness work;
  the H100 on Newton is for FP64 production and all published numbers.
- Agreement with references on well-conditioned problems: ≤ 1e-10 for FP64
  and mixed, ≤ 1e-6 for FP32. Adaptive-solver comparisons use the
  tolerance-scaled bounds of ADR 0003.

## Consequences

- Kernel templates take the scalar type as a parameter; no literal `float` or
  `double` outside the template header.
- Benchmarks run at both precisions before any speedup is quoted; FP32
  numbers from the 4090 are never presented as production results.
- A fast-math build option may exist for exploration, but tests mark it
  non-validating and CI never runs it.
- Mixed-precision refinement is a later feature; Phase 0 ships pure FP32 and
  pure FP64 only.
