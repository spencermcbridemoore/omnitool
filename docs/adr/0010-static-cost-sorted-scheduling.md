# ADR 0010: Static cost-sorted scheduling

- Status: accepted (ruling R5 of 2026-11-19)
- Date: 2026-11-19

## Context

With one thread per member, a warp runs until its slowest lane finishes. The
Phase 0 benchmark batch cycles four detunings through the members, so every
warp holds a 6712-step member. Probes on the released Phase 0 kernel (RTX
4090, FP64, 262,144 members, three save points, end-to-end wall time, no
manifest):

| member order | 128 threads/block | 32 threads/block |
|---|---|---|
| cycled | 1285 ms | 1286 ms |
| sorted by detuning | 384 ms | 385 ms |

Sorting gains a factor 3.35 of the 3.44 available. Block size changes
nothing. At 16,384 members sorting changes nothing either (100 against
101 ms): that plateau is the serial step count of the slowest member.

## Decision

`solve_batch` takes an optional per-member `sort_key`. The host sorts the
members stably by it, launches, and returns results in the caller's order.
There is no default key. The result carries `lane_utilisation`: useful over
warp-padded step attempts in launch order.

## Consequences

- A member's arithmetic does not depend on its lane, so results are bitwise
  independent of `sort_key` (spec T16).
- The key only has to cluster similar members. Cost is not monotone in the
  natural key for an implicit method: on problem D the step attempts by
  shift are 113, 404, 271, 72, 43.
- Sorting does not shorten a launch set by one member, nor address the
  Windows 2 s display watchdog.
- **Trigger for suspend/resume** (bounded launches with state saved to device
  arrays and members compacted between launches): `lane_utilisation < 0.7`
  after sorting on a production workload, or a launch that cannot be kept
  under the watchdog. Member state is self-contained, so nothing here blocks
  it.
