# ADR 0001: Scope and front doors

- Status: accepted
- Date: 2026-10-19
- Supersedes: the September 2026 "RydIQule GPU backend" brief

## Context

The project began as a GPU backend for RydIQule. The research phase showed
that the same machinery serves RydIQule, pylcp, MoleCool and the UCN spin
tracker. The closest prior art, cubie, has no events, RNG, early termination,
field tables or block-per-member mode, and is pinned to a privately patched
compiler. No existing package is a sound base.

## Decision

`omnitool` is a **common GPU ensemble ODE core** that depends on nothing
physics-specific. Downstream packages are **optional extras**, each with its
own adapter and tests; upstream hooks go in as pull requests from forks. The
project never carries a fork of a downstream package.

Two **front doors** share one engine:

1. **Affine-linear master equations**
   dρ/dt = (A₀ + Σₖ fₖ(t) Bₖ) ρ + c₀ + Σₖ fₖ(t) dₖ, with the exact Jacobian
   known analytically, so Rosenbrock-W and Radau steps are cheap and stiff
   Doppler stacks are handled per member.
2. **Nonlinear trajectories** of 10–15 states with a user device RHS or a
   `TabulatedField`, plus events (collisions, hazard-driven jumps, early
   termination), per-thread RNG, warp-vote event detection and cost-sorted
   scheduling.

Shared: per-member adaptive controller, dense output at shared save times,
reductions, thread-per-member and warp/block-per-member layouts, status
codes, FP32 and FP64 everywhere. Phase 0 builds one vertical slice
(Dormand–Prince 5(4), thread-per-member) for both doors.

## Consequences

- Core code, docs and tests speak of members, RHS and save times, never
  atoms or density matrices.
- Each adapter lives under its own extra and can be removed without
  touching the core.
- Lockstep (vmap-style) stepping is out: the win is per-member adaptivity
  plus cost-ordered scheduling.
- Numerical-Recipes-derived code (the old UCN kernel) and GPL code (PQLS)
  never enter the repo.
