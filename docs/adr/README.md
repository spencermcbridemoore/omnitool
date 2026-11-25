# Architecture decision records

Short records (context, decision, consequences; under 300 words) of the
decisions that shape `omnitool`. New decisions get the next number; a
superseded record keeps its file and gains a "Superseded by" line.

| ADR | Title |
|-----|-------|
| [0001](0001-scope-and-front-doors.md) | Scope and front doors |
| [0002](0002-precision-policy.md) | Precision policy |
| [0003](0003-verification-layer.md) | Verification layer |
| [0004](0004-backend-choice.md) | GPU backend choice for Phase 0 |
| [0005](0005-no-cudasim-in-phase-0.md) | No host-compiled `cudasim` backend in Phase 0 |
| [0006](0006-phase-1-scope.md) | Phase 1 scope: the linear front door, first cut |
| [0007](0007-rodas5p-and-its-coefficient-source.md) | Rodas5P, and where its coefficients come from |
| [0008](0008-dense-pivoted-lu-per-member.md) | Dense LU with partial pivoting per member |
| [0009](0009-affine-problem-and-coefficient-functions.md) | `AffineProblem` and its coefficient functions |
| [0010](0010-static-cost-sorted-scheduling.md) | Static cost-sorted scheduling |
| [0011](0011-baselines-live-in-benchmarks.md) | Competitor baselines live under `benchmarks/`, not in the package |
| [0012](0012-observables-and-group-sums.md) | On-chip observables and deterministic group sums |
| [0013](0013-save-time-policy-for-rodas5p.md) | Save-time policy for Rodas5P |
