# ADR 0008: Dense LU with partial pivoting per member

- Status: accepted (ruling R3 of 2026-11-19); spike pending
- Date: 2026-11-19

## Context

Every Rodas5P step attempt factorises `M = I/(h·γ) − A(t_n)` and solves
eight right-hand sides. cubie and modax use sparse LU on a static pattern
without pivoting. Measured on the 15-state problem D (spec §7):

- Without pivoting, element growth reaches 512 on the stiffest member at
  `h = 4`. The relative solve error is then 5.2e-5 in FP32 against 1.8e-7
  with partial pivoting (1.2e-13 against 7.0e-15 in FP64).
- The pattern has 65 of 225 entries and L+U has 91–92. Dense costs about
  1125 flops per factorisation plus 1800 for the solves: sparsity is worth a
  factor of a few at this size.

## Decision

The first cut uses **dense LU with partial pivoting, in thread-local
memory, one thread per member**, for `n_state ≤ 35` on the `cuda` backend.
The matrix `A(t)` is still assembled on-chip from shared sparse tables; only
the factorisation is dense. A zero pivot ends the member with status
`SINGULAR_MATRIX`.

Before the kernel is written, a **one-day LU spike** measures thread-local
against shared memory at 15, 35 and 63 states at both precisions and records
the result here, as ADR 0004 did. If thread-local fails at 35 states, this
ADR is revised, not worked around.

## Consequences

- Pivoting costs branches inside a warp; it buys about 2.5 digits in FP32.
- Per-thread memory is about 3 kB at 15 states and 13 kB at 35 in FP64
  (estimates).
- Sparse or structured factorisation and a block-per-member layout for 63+
  states are second-cut decisions, taken from the measured share of kernel
  time spent in LU.
- The `cpu` oracle solves with LAPACK, so the backends agree to tolerance,
  never bitwise.
