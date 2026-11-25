# ADR 0012: On-chip observables and deterministic group sums

- Status: accepted (ruling R8 of 2026-11-19)
- Date: 2026-11-19

## Context

Phase 0 returns the full state of every member at every save time. For the
linear front door that is 262,144 members × 401 save times × 15 states × 8
bytes = 12.6 GB, more than the workstation's card can spare. Callers want a
few linear functionals of the state, and often a weighted sum of those over
groups of members (a Doppler average). `AGENTS.md` requires reduction
on-chip.

## Decision

1. **Observables (first cut).** `solve_batch(..., observables=W)` with `W`
   of shape `(n_obs, n_state)` makes the kernel write `W · y` at each save
   time. `save_state=False` drops the full state from the device output.
   One observable in the case above is 0.84 GB.
2. **Group sums (later in Phase 1, after the kernel).**
   `groups` (one integer per member) and `weights` (one real per member)
   produce `obs_grouped[g] = Σ w_m · obs_m` over the members of group `g`,
   computed on the device by a second kernel that walks each group's members
   sequentially **in the caller's member order**.

## Consequences

- Per-member outputs stay bitwise independent of batch order and of
  `sort_key`. A group sum is bitwise independent of `sort_key`, because the
  summation order is the caller's and not the launch order. Under a
  permutation of the caller's members a group sum agrees only to rounding
  (spec T22 states the bound).
- No atomics are used for reductions: an `atomicAdd` sum would depend on
  thread timing.
- Affine observables (a constant offset) are applied by the caller.
- Nonlinear reductions and reductions over save times are out of scope.
