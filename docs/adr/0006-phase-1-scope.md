# ADR 0006: Phase 1 scope: the linear front door, first cut

- Status: accepted (rulings R2, R7, R9 of 2026-11-19)
- Date: 2026-11-19
- Amends: ADR 0001 (wording of front door 1)

## Context

ADR 0001 gives the linear front door "Rosenbrock-W and Radau". Probes made
before the Phase 1 spec showed that one method covers a first cut, and that
several listed pieces are performance work of unmeasured value.

## Decision

The **first cut** (`docs/PHASE1_SPEC.md`) is one implicit method, Rodas5P
(ADR 0007), on a new `AffineProblem` (ADR 0009), with dense pivoted LU per
member (ADR 0008), static cost sorting (ADR 0010), on-chip observables
(ADR 0012), the RydIQule adapter as an optional extra, and the baselines
(ADR 0011).

**Radau IIA(5) is not in the first cut.** It is built only if Rodas5P needs
more than twice the linear-algebra work of SciPy's Radau on the stiff test
set at `rtol = 1e-7`. Measured on problem D in `"dense"` mode: Rodas5P 757
step attempts against SciPy Radau's 1546 steps, about 0.4 of its work. The
gate is not met; it is re-evaluated on the RydIQule 561-class run.

Initial conditions are passed in. Precision is pure FP32 or pure FP64. Done
needs `cpu` in CI and `cuda` on the 4090; Newton H100 manifests are needed
before any number leaves the repo. Upstream issues are drafted as files and
filed by the maintainer.

## Consequences

- ADR 0001's "Rosenbrock-W" is read as "Rosenbrock–Wanner": Rodas5P needs
  the exact Jacobian, which this front door has by construction.
- Deferred to a second cut, each decided from measurements: Radau, sparse
  LU, block-per-member, an on-chip steady state for `y0`, tabulated
  coefficients, suspend/resume, mixed precision. Events, RNG and
  sensitivities stay out of Phase 1.
- ADR 0005's `cudasim` trigger is not hit.
