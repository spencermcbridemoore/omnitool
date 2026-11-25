# ADR 0013: Save-time policy for Rodas5P

- Status: accepted (ruling of 2026-11-24, relayed by the Phase 0 main
  session); proposed by the main session 2026-11-19
- Date: 2026-11-19

## Context

Phase 0 never clips a step to a save time: DOPRI5's dense output is kept
in check by the step controller as a side effect. Calibrating Phase 1
showed this does not carry over to a stiffly accurate method.

On a stiff member on its slow manifold, Rodas5P's step-end error and
estimate become very small, so the controller takes large steps. The
quartic continuous extension still has an interpolation error of about
`2e-4 h^5 |y^(5)|`. Measured on the Prothero–Robinson problem with
`lambda = 1e5` at `(rtol, atol) = (1e-7, 1e-9)`: 9 steps; error 3.2e-9 at
`t_end`; 4.0e-5 at the interpolated save times, 400 times
`rtol * ymax + atol`, with status `SUCCESS`. Clipping steps to the save
times gives 0.01 of that unit in 44 steps.

## Decision

`solve_batch` takes `save_mode`:

- `"clip"`, the default for `rodas5p`: steps land on every save time and
  every saved value is a step endpoint.
- `"dense"`, opt-in: Phase 0's behaviour with Rodas5P's continuous
  extension, documented as not error-controlled on stiff members.

`"clip"` is not offered for DOPRI5 in Phase 1.

## Consequences

- With `"clip"` a member takes at least `n_save - 1` steps: on a 401-point
  grid a member that needs 6 steps takes 400. `"dense"` keeps the stiff
  advantage for callers whose stiff members move little between save times.
- Benchmarks state the save mode and compare at matched achieved accuracy.
- Not taken: dense only (silently inaccurate here); clip only (the fast
  path is lost).
- Open follow-up: an error-controlled dense output, for example from the
  interpolant's defect at each save time. Not calibrated, not in the spec.
