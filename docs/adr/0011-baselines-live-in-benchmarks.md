# ADR 0011: Competitor baselines live under `benchmarks/`, not in the package

- Status: accepted (ruling R6 of 2026-11-19)
- Date: 2026-11-19

## Context

Phase 1 opens with a RydIQule-to-cubie baseline that sets the bar the core
must beat. cubie pins a privately patched compiler wheel. A second design
appeared on 28 September 2026: modax (Berry et al., arXiv:2609.35661; MIT),
one trajectory per CUDA thread with Rodas5P and per-thread LU.

## Decision

Baselines are **benchmarks-only tools** under `benchmarks/baselines/<name>/`,
each with its own pinned environment and lock file. They are never an
`omnitool` extra and never imported by the package or its tests.

The shared piece is the extraction of `A₀, Bₖ, c₀, dₖ` from a RydIQule
sensor by unit perturbations. It belongs to `omnitool.adapters.rydiqule`
(extra `omnitool[rydiqule]`, `rydiqule==2.1.3`), is built first, and the
baseline scripts import it.

**cubie baseline.** The 561-class Doppler run of
`docs/handoff/repo-seeds/stiffness_check.py`, one member per class, float64,
fastmath off, `rodas3p` and `ros34pw2`, members sorted by Doppler shift. It
reports, over a tolerance sweep: achieved error against a Radau reference
(per class, and the Doppler-summed signal as a fraction of its swing);
kernel time with compile time separate; cubie's own step, rejection and
linear-solve counters; the same-node CPU rows (stock RydIQule RK45, and
Radau with block sparsity); and the same run replicated to at least 262,144
members. Every run writes a manifest.

**modax baseline.** A second baseline, time-boxed to half a day. If it does
not run on the workstation in that time, the attempt and reason are
recorded.

## Consequences

- The core's dependency resolution never sees a competitor's pins.
- Speedups are quoted at matched achieved accuracy from the sweeps, never at
  matched requested tolerance.
- `AGENTS.md` lists modax as a competitor.
- A baseline that cannot be rebuilt is a recorded limitation, not a blocker.
