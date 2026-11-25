# ADR 0009: `AffineProblem` and its coefficient functions

- Status: accepted (rulings R4 and R9 of 2026-11-19)
- Date: 2026-11-19

## Context

ADR 0001 writes the linear front door as
`y' = (A₀ + Σₖ fₖ(t) Bₖ) y + c₀ + Σₖ fₖ(t) dₖ`. Scanned parameters, which
differ between members, and time dependence enter through the same K
scalars. Rodas5P (ADR 0007) also needs the time derivative of the right-hand
side exactly; a finite difference in FP32 is good to about 3e-4.

## Decision

A new frozen dataclass `AffineProblem` holds the shared blocks
`a0, c0, b[K], d[K]` and one coefficient function in two renderings, as
`Problem` holds its right-hand side:

```c
void coeff(real_t t, const real_t* p, real_t* g, real_t* dg);
```

It returns the K coefficients `g_k(t, p)` and their time derivatives `dg_k`
for one member, so that `A = A₀ + Σ g_k Bₖ`, `c = c₀ + Σ g_k dₖ` and
`∂f/∂t = Σ dg_k (Bₖ y + dₖ)`. The NumPy twin has the same contract. A flag
`time_dependent=False` declares that `g` does not depend on `t`.

`solve_batch(problem, ..., method=...)` selects the solver: `"dopri5"` for a
`Problem`, `"rodas5p"` for an `AffineProblem`. `omnitool.check_coefficients`
compares `dg` with a central difference of `g` on the host and raises on a
mismatch.

## Consequences

- A wrong `dg` lowers the order from 5 to about 2 (spec T11) with no
  failure status, which is why the check exists.
- The shared blocks are per problem; a member holds only `n_params` scalars.
- An adapter whose upstream package takes arbitrary Python callables for
  time dependence (RydIQule) must ask for a C expression and its derivative
  per time-dependent coupling, and check them against the callable.
- Tabulated coefficient functions with spline interpolation are deferred;
  they would share machinery with Phase 2's `TabulatedField`.
