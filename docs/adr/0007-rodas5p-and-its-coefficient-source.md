# ADR 0007: Rodas5P, and where its coefficients come from

- Status: accepted (ruling R1 of 2026-11-19)
- Date: 2026-11-19
- Amends: the tableau-transcription rule in `AGENTS.md`

## Context

The affine front door needs an L-stable method that holds its order when a
member follows a slowly driven quasi-steady state (the Prothero–Robinson
regime). At `rtol = 1e-7` third order is uncompetitive: cubie's `rodas3p`
took 388 steps on problem L where DOPRI5 took 64. Classic RODAS and Rodas5
fall to about order 1 in the stiff limit; Rodas5P keeps order 3 (Steinebach
2023, Table 6).

The paper does not print the coefficients; it points to OrdinaryDiffEq.jl
(MIT). They come from a numerical optimisation: decimals that cannot be
re-derived.

## Decision

The first cut has **one tableau: Rodas5P** (G. Steinebach, BIT Numer. Math.
63:27, 2023): 8 stages, order 5 with an embedded order 4, stiffly accurate,
A- and L-stable, with a fourth-order continuous extension.

The coefficients are copied **verbatim as decimal strings** from
`lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl` at commit
`24e2fee87faf6580bf813f4ba305c9b4814c3905` (sha256 in the spec, §3.1),
once, into `src/omnitool/tableaus/rodas5p.py`, with the MIT attribution.
The transcription is checked against the paper, not the Julia code: as
exact `Fraction`s it must satisfy the 17 order conditions of Table 2 and
the stiff-accuracy relations (3.1)–(3.3) to a tolerance (spec T10).

## Consequences

- Tableaus printed as rationals keep exact identity tests; decimal-only
  tableaus are copied verbatim from the source the paper designates and
  tested to a stated tolerance.
- Rodas5P is not a W-method: the Jacobian and the time derivative must be
  exact (ADR 0009), and the matrix is refactorised on every step attempt.
- Its estimator does not vanish on linear constant-coefficient problems or
  pure quadrature (spec T12). It under-reads on weakly damped oscillations
  that the method damps numerically (spec §3.7).
- ROS3P remains excluded (ADR 0003).
