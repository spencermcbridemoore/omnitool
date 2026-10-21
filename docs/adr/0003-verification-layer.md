# ADR 0003: Verification layer

- Status: accepted
- Date: 2026-10-19

## Context

cubie's default Rosenbrock tableau (ROS3P) has an error estimate that
vanishes identically on every linear constant-coefficient problem: at
rtol 1e-7 it returned errors of 1e-2 to 1e-1 with status 0. cubie's tests
missed this because its CPU reference uses the same tableau. The old UCN CUDA
kernel evaluated Cash–Karp stages at the wrong states, collapsed to forward
Euler, and passed its own tests. Both were checked only against themselves.

## Decision

Correctness comes from a **verification layer independent of the device
tableaus**; every solver passes it before it is benchmarked ("test before you
benchmark"):

1. **Analytic references**: linear systems with known exponentials, and
   invariants such as trace and norm.
2. **SciPy references**: DOP853 and Radau at 1e-12, within tolerance-scaled
   bounds at both precisions.
3. **Convergence-slope tests**: global error versus fixed step size shows the
   method's order on a smooth problem.
4. **Tableau identities**: row sums and order conditions checked from the
   transcribed coefficients.
5. **Error-estimator tests**: the embedded estimate scales as h^(p+1) and
   does **not vanish** on a linear constant-coefficient problem. This catches
   ROS3P.
6. **Batch-order invariance**: permuting members only permutes outputs.
7. **nfev caps**: step and RHS-evaluation counts stay within budgets.
8. **A plain-NumPy CPU reference** of every tableau, written from the paper,
   not the kernel, so the GPU and CPU paths can disagree.

## Consequences

- Each new tableau arrives with its NumPy reference, identities test and
  estimator test in one pull request.
- Tests run under the implemented backends (`cpu` on every push, `cuda` on
  the 4090; see ADR 0005 on the deferred `cudasim`).
- Adapters are tested against their upstream package, not the core.
- The ROS3P finding is filed upstream to cubie with `ros3p_numpy.py`; ROS3P
  is never a default here.
