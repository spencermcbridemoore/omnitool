# Phase 1 specification: the linear front door, first cut

- Status: accepted 2026-11-24 (rulings of 2026-11-19 and 2026-11-24)
- Governing ADRs: 0001–0003 (scope, precision, verification), 0004–0005
  (backend), 0006 (Phase 1 scope), 0007 (Rodas5P), 0008 (LU), 0009
  (`AffineProblem`), 0010 (scheduling), 0011 (baselines), 0012 (outputs),
  0013 (save-time policy).
- Owner of numerics and tableau transcription: the main session (see
  `AGENTS.md`); the coder agent implements modules from this document and
  never changes it.
- Every numeric threshold below was measured on a throwaway NumPy
  implementation before it was written here, as in Phase 0. "Measured"
  values are from that scratch implementation in IEEE arithmetic on the
  host; they are what a correct implementation produces, give or take the
  last bits. The scratch code is not in the repository and is not a source
  for the implementation.

## 1. Purpose

Phase 1 adds the first implicit path: affine-linear systems
`y' = A(t, p) y + c(t, p)` whose matrix and forcing are sums of shared
blocks weighted by per-member scalar coefficients, integrated with the
L-stable Rosenbrock–Wanner method **Rodas5P** using the exact Jacobian, a
**dense LU with partial pivoting per member**, **static cost-sorted
scheduling**, **on-chip linear observables**, and the Phase 0 machinery
(per-member controller, save times, FP32 and FP64, status codes). It adds
the **RydIQule adapter** as an optional extra and opens with the **cubie and
modax baselines**. Nothing physics-specific appears in the core.

Out of scope for the first cut (ADR 0006): Radau, sparse or structured LU,
block-per-member layout, more than 35 states on `cuda`, tabulated
coefficient functions, an on-chip steady-state solve for `y0`, suspend and
resume, mixed precision, per-member tolerances, events, RNG, sensitivities,
and running DOPRI5 on an `AffineProblem`.

## 2. Public API additions

```python
@dataclass(frozen=True)
class AffineProblem:
    name: str
    n_state: int
    n_params: int
    a0: np.ndarray         # (n_state, n_state) float64
    c0: np.ndarray         # (n_state,) float64
    b: np.ndarray          # (n_coeff, n_state, n_state) float64
    d: np.ndarray          # (n_coeff, n_state) float64
    coeff_numpy: Callable  # coeff_numpy(t, p) -> (g, dg), each (n_coeff,)
    coeff_cuda: str        # C body of  void coeff(real_t t, const real_t* p,
                           #                      real_t* g, real_t* dg)
    time_dependent: bool = True
```

`n_coeff = b.shape[0]` and may be zero, in which case `coeff` is never
called. The member's system at time `t` is

```
A(t, p) = a0 + sum_k g_k(t, p) * b[k]
c(t, p) = c0 + sum_k g_k(t, p) * d[k]
f(t, y) = A y + c,     df/dy = A,     df/dt = sum_k dg_k(t, p) * (b[k] y + d[k])
```

`dg_k` is the exact time derivative of `g_k`. With `time_dependent=False`
the coefficients are evaluated once per member at `t0`, `dg` is ignored and
`df/dt = 0`. Floating constants in `coeff_cuda` are written `REAL(0.5)`, as
in `Problem.rhs_cuda`.

```python
result = omnitool.solve_batch(
    problem, y0, params, t_span, t_save,   # as in Phase 0
    rtol=1e-7, atol=1e-9, precision="fp64", backend="cuda",
    max_steps=100_000, h0=None, fixed_step=None,
    method=None,        # None -> "dopri5" for Problem, "rodas5p" for AffineProblem
    save_mode=None,     # None -> "dense" for dopri5, "clip" for rodas5p (§3.6)
    sort_key=None,      # (n_members,) real, or None (§4.3)
    observables=None,   # (n_obs, n_state) real, or None (§4.4)
    save_state=True,    # False needs observables; then result.y is None
    groups=None,        # (n_members,) int in [0, n_groups), needs observables
    weights=None,       # (n_members,) real, needs groups
)
```

`omnitool.check_coefficients(problem, params, t, rel=1e-6)` evaluates
`coeff_numpy` and compares `dg` with the central difference
`(g(t + s) - g(t - s)) / (2 s)`, `s = 1e-5 * max(1, |t|)`, in float64 at
every given `t` and member; it raises `ValueError` naming the coefficient
`k` when `|dg_k - fd_k| > rel * max(1, |g_k|, |dg_k|)`.

`BatchResult` gains:

| field              | shape                           | meaning                               |
|--------------------|---------------------------------|---------------------------------------|
| `n_lu`             | `(n_members,)` int32            | LU factorisations (0 for DOPRI5)      |
| `n_solve`          | `(n_members,)` int32            | triangular solve pairs (0 for DOPRI5) |
| `obs`              | `(n_members, n_save, n_obs)`    | `observables @ y`, or `None`          |
| `obs_grouped`      | `(n_groups, n_save, n_obs)`     | weighted group sums, or `None`        |
| `lane_utilisation` | float                           | §4.3                                  |

For Rodas5P `n_fev` counts evaluations of `A(t) Y + c(t)`.

Rules, in addition to Phase 0's: `method="rodas5p"` with a `Problem`, or
`"dopri5"` with an `AffineProblem`, raises `ValueError`; `save_mode="clip"`
with `"dopri5"` raises `ValueError`; shapes of `a0, c0, b, d` that disagree
with `n_state` raise; `n_state > 35` with `backend="cuda"` raises
`ValueError` naming ADR 0008; `sort_key` containing NaN raises;
`save_state=False` without `observables` raises; `weights` without
`groups`, or `groups` without `observables`, raises; with
`time_dependent=False`, `coeff_numpy` is evaluated at `t0` and `t_end` for
the first 16 members and a difference in `g` raises. The shared blocks are
cast to the requested precision without a warning.

## 3. Numerics

### 3.1 Tableau

Rodas5P: G. Steinebach, "Construction of Rosenbrock–Wanner method Rodas5P
and numerical benchmarks within the Julia Differential Equations package",
BIT Numer. Math. 63:27 (2023), doi:10.1007/s10543-023-00967-x. Eight stages,
order 5, embedded order 4, stiffly accurate, with a quartic continuous
extension. The paper prints the order conditions but not the coefficients;
per ADR 0007 they are transcribed verbatim, as decimal strings, from the
arrays `RODAS5PA`, `RODAS5PC`, `RODAS5Pc`, `RODAS5Pd`, `RODAS5PH` and the
value `gamma` in `lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl`
of SciML/OrdinaryDiffEq.jl at commit
`24e2fee87faf6580bf813f4ba305c9b4814c3905` (sha256 of the file
`60cd2aa9035717bb94252fea6da05380af3efa8b87fe29f151d66b92c3f5d443`, MIT
licence). The transcription lives once in `src/omnitool/tableaus/rodas5p.py`
with that attribution and is rendered into the device header from that file;
it is never typed twice. Coefficients are stored as `Fraction` built from
the decimal strings and rendered to 17 significant digits.

The arrays are in the transformed form of Hairer & Wanner, *Solving ODEs
II*, eq. (IV.7.25). Below, `a` is the 8 × 8 strictly lower-triangular
`RODAS5PA`, `C` the strictly lower-triangular `RODAS5PC` (its eighth column
is zero), `c` and `d` the vectors, `H` the 3 × 8 array, and `γ = gamma`.

### 3.2 One step

From `(t_n, y_n)` with step `h`:

1. `J = A(t_n, p)` and `f_t = sum_k dg_k(t_n, p) (b[k] y_n + d[k])`.
2. `M = I * (1 / (h * γ)) - J`, with the shift computed as written: one
   multiplication, one division, in `real_t`. Factorise `M` (§3.8).
3. For `i = 1..8`: `Y_i = y_n + sum_{j<i} a[i,j] U_j`;
   `r_i = A(t_n + c_i h, p) Y_i + c(t_n + c_i h, p) + sum_{j<i} (C[i,j] / h) U_j + h d_i f_t`;
   solve `M U_i = r_i`.
4. `y_{n+1} = Y_8 + U_8`. The embedded fourth-order solution is `Y_8`, so
   the error vector is `U_8`.

One step attempt costs one factorisation, eight solves and eight
evaluations of `A Y + c`. Rows 6–8 of `a` share their first five entries
and `c_6 = c_7 = c_8 = 1`, so the coefficient function is needed at six
distinct times per attempt; an implementation may reuse them.

### 3.3 Error norm

As Phase 0: `sc_i = atol + rtol * max(|y_n,i|, |y_n+1,i|)`,
`err = sqrt(mean_i((U_8,i / sc_i)^2))`. Accept when `err <= 1`.

### 3.4 Step-size controller

The predictive controller of Gustafsson (Hairer & Wanner II, §IV.8), with
`q = 5` (embedded order plus one), `safe = 0.9`, and step-ratio limits 1/6
and 5:

- `fac = clamp(err^(1/q) / safe, 1/6, 5)`.
- On an accepted step that has an accepted predecessor:
  `fac_gus = clamp((h_acc / h) * (err^2 / err_acc)^(1/q) / safe, 1/6, 5)`
  and `fac = max(fac, fac_gus)`.
- `h_new = h / fac`. After an accepted step `h_acc = h` and
  `err_acc = max(err, 1e-2)`.
- The first accepted step after a rejection may not grow:
  `h_new = min(h_new, h)`.
- With `fixed_step` the controller is bypassed and every step is accepted;
  the norm is still computed.

`max_steps` counts accepted plus rejected attempts. A member stops with
`STEP_TOO_SMALL` when `|h| < 16 * eps * max(|t|, 1)`.

### 3.5 Initial step

Phase 0's `hinit` (spec §3 there) applied to `f(t, y) = A y + c`, with the
same exponent 1/5; it costs two evaluations, which are counted in `n_fev`.
Then `h = max(h, 100 * 16 * eps * max(|t0|, 1))`. The last rule exists
because a member that starts exactly at rest gets `h = 1e-6` from `hinit`,
which is below the FP32 step floor. (Measured: problem L with `Omega = 0`
finishes `SUCCESS` at both precisions with first step 1e-6 at FP64 and
1.9e-4 at FP32.)

### 3.6 Save times (ADR 0013)

Two modes.

**`"clip"` (default for Rodas5P).** Let `t_s` be the next unreached save
time. If `t + 1.01 h >= t_s`, the step is adjusted to land exactly on
`t_s` and `y_{n+1}` is written there. If an adjusted step is accepted and
did not follow a rejection, the next step is `max(h_new, h_before)` where
`h_before` is the step the controller had proposed. Every saved value is a
step endpoint. A member takes at least `n_save - 1` steps.

**`"dense"`.** Steps are clipped only to `t_end`, by Phase 0's rule
(`t + 1.01 h >= t_end`). After each accepted step, for every save time in
`(t_n, t_n+1]` with `θ = (t_s - t_n) / h`:

```
K_m = sum_i H[m,i] U_i,   m = 1, 2, 3
y(θ) = (1 - θ) y_n + θ (y_{n+1} + (1 - θ) (K_1 + θ (K_2 + θ K_3)))
```

A save time equal to `t_n+1` is written as `y_{n+1}` itself.

In `"dense"` mode the interpolant is not error-controlled. Its error is
about `2e-4 h^5 |y^(5)|` in every regime, while on a stiff member that sits
on its slow manifold the step-end error and the estimate are orders of
magnitude smaller, so the controller chooses steps the interpolant cannot
follow. Measured on problem PR with `lambda = 1e5` at
`(rtol, atol) = (1e-7, 1e-9)`: 9 steps, error 3.2e-9 at `t_end` and 4.0e-5
at the interpolated save times, 400 times the T14 unit `rtol * ymax + atol`.
`"clip"` gives 0.01 units on the same problem in 44 steps. `"dense"` is
therefore opt-in, for callers whose stiff members move little between save
times.

### 3.7 What the error estimate does not see

On the imaginary axis the stability functions `R` (order 5) and `R_hat`
(embedded) satisfy, measured from the transcription:

| `y` in `z = i y` | `abs(R - e^z)` | `abs(R - R_hat)` | `abs(R)` |
|-----:|--------:|--------:|-------:|
| 0.5  | 8.0e-8  | 5.1e-6  | 1.0000 |
| 2    | 1.2e-3  | 8.8e-3  | 0.9989 |
| 4.6  | 1.6e-1  | 2.2e-1  | 0.9596 |
| 10   | 1.8     | 6.1e-1  | 0.8348 |
| 50   | 1.1     | 2.4e-1  | 0.2455 |

For `ωh` above about 5 the estimate is smaller than the true error, and
the method damps an oscillation that the exact solution keeps. A weakly
damped oscillation whose amplitude is only a few orders above `atol` is
therefore damped by the method itself, after which nothing flags it. This
is a property of every L-stable one-step method, and
SciPy's Radau shows it on the same problem (3.9e-8 error on member
`s = 0.4` of problem D at `(1e-7, 1e-9)`, 33 units). It is why problem D
has its own bound constant in T14, and it is stated in the user
documentation of `method="rodas5p"`.

### 3.8 Linear algebra

Dense LU with partial pivoting by rows (ADR 0008): at column `k` the pivot
is the entry of largest magnitude in rows `k..n-1`, the first such row on a
tie. A pivot that is not finite ends the member with `NON_FINITE`; a pivot
equal to zero ends it with `SINGULAR_MATRIX`. No scaling, no iterative
refinement. Each solve is a row permutation, a unit-lower forward
substitution and an upper back substitution. On `cpu`, where LAPACK does
the factorisation, a non-finite entry of `M` ends the member with
`NON_FINITE` before the solve and a `LinAlgError` maps to
`SINGULAR_MATRIX`.

### 3.9 Precision

As Phase 0: `real_t` is a template parameter, all constants are rendered in
`real_t`, no fast-math, FMA contraction allowed (ADR 0002).

## 4. Device design

### 4.1 Layout

- One thread per member, 128 threads per block, as Phase 0. Thread *i* owns
  `y`, `Y`, the eight `U_i`, the right-hand side, `f_t`, the assembled
  matrix values, the dense `M` with its pivot indices, the coefficient
  vectors, the controller state and the counters, all in registers or
  thread-local memory. No shared memory and no warp intrinsics.
- The LU spike of ADR 0008 runs before the kernel is written. Its result is
  recorded in that ADR; this section is amended only if the spike fails.
- Device arrays are member-fastest, as Phase 0: `y0_d[n_state, n_members]`,
  `params_d[n_params, n_members]`, `out_d[n_save, n_state, n_members]`,
  `obs_d[n_save, n_obs, n_members]`.
- The kernel template `rodas5p.cu` is plain C in the Phase 0 sense: a
  `coeff` function supplied by templating, `assemble()`, `lu_factor()`,
  `lu_solve()`, `step()`, `controller()`, `dense()` and the driver loop.
  Compile-time parameters: `real_t`, `N_STATE`, `N_PARAMS`, `N_COEFF`,
  `N_OBS`, `TIME_DEPENDENT`, the tables of §4.2 and the `coeff` body.
- Test entry points, as Phase 0 had for `step()` and `dense()`: one step,
  one dense evaluation, one factor-and-solve of a given matrix and
  right-hand side per member, and one assembly of `A(t, p)` and `c(t, p)`.

### 4.2 Assembly

The union sparsity pattern of `a0` and all `b[k]` is computed on the host,
in row-major order: `nnz` index pairs and a value table `V[(n_coeff + 1),
nnz]` whose row 0 holds `a0` and row `k + 1` holds `b[k]` on the pattern.
The forcing has the dense table `[c0; d[0..]]`. Both are rendered into the
kernel source as `static const` arrays in `real_t`, like the tableau. A
member assembles `val[e] = V[0][e] + sum_k g_k V[k+1][e]` and uses the
values twice: scattered into the dense `M`, and in the matrix-vector
products of the stages. No per-member matrix exists outside the thread.

### 4.3 Scheduling

With `sort_key`, the host computes `order = argsort(sort_key, stable)`,
launches the members in that order and writes every per-member output back
to the caller's index (ADR 0010). Without it the launch order is the
caller's order.

`lane_utilisation` is computed on the host from the launch order and the
per-member attempt counts `s_m = n_accepted + n_rejected`: pad the launch
order to a multiple of 32 with zeros, split it into consecutive groups of
32 (the warps), and return `sum(s) / sum_over_warps(32 * max(s in warp))`.
It is defined on every backend, so it is tested in CPU CI.

### 4.4 Outputs

With `observables = W`, the kernel writes `W y` at each save time into
`obs_d`; `W` is rendered into the kernel source. With `save_state=False`
`out_d` is not allocated. Group sums (ADR 0012) are a second kernel with one
thread per output element `(group, save, obs)`; it walks the members of its
group in increasing caller index and accumulates `weights[m] * obs[m]`
sequentially. A member whose status is not `SUCCESS` contributes its NaN.

## 5. Backends

| backend | what runs |
|---------|-----------|
| `cpu`   | `reference/rodas5p_numpy.py`: plain NumPy, one member at a time, written from the paper and Hairer & Wanner II, sharing only the tableau file; linear solves through `numpy.linalg.solve`; group sums by a Python loop in the same order as the device |
| `cuda`  | `rodas5p.cu` compiled by NVRTC through `omnitool.backends.nvrtc` (ADR 0004 option set) and loaded with `RawModule(path=...)` |

`cpu` is the oracle and runs on every CI push; on `cpu` `n_lu` is the number
of step attempts and `n_solve` eight times that, by definition. `cuda` runs
on the workstation's 4090 before every kernel push.

## 6. Status codes

Phase 0's codes are unchanged. One is added to `status.h` and the `IntEnum`:

| code | name              | meaning                                   |
|-----:|-------------------|-------------------------------------------|
|    4 | `SINGULAR_MATRIX` | a pivot of the step matrix was exactly 0  |

## 7. Test problems

Defined once in `tests/problems_affine.py`, each with a NumPy and a C
coefficient function. Save grids: `linspace(0, T, 51)` for L, S1 and Q,
`linspace(0, T, 41)` for PR and D.

**L — constant-coefficient 3-state.** Phase 0's problem L in affine form:
`a0 = 0`, `c0 = 0`, `g = (Gamma, Omega, Delta)`, `time_dependent=False`,

```
b[Gamma] = diag(-1/2, -1/2, -1)        d[Gamma] = (0, 0, -1)
b[Omega] = [[0,0,0],[0,0,-1],[0,1,0]]  d[Omega] = 0
b[Delta] = [[0,1,0],[-1,0,0],[0,0,0]]  d[Delta] = 0
```

`y0 = (0, 0, -1)`, `Gamma = Omega = 1`, `Delta in {0, 2, 20, 200}`, `T = 10`.
Reference: Phase 0's closed form.

**S1 — chirped 3-state.** As L with `params = (Gamma, Omega)`,
`g = (Gamma, Omega, 2 + 3 sin(1.5 t))`, `dg = (0, 0, 4.5 cos(1.5 t))`.
Reference: SciPy DOP853 at `rtol = 1e-13, atol = 1e-15`.

**Q — pure quadrature.** `n_state = 1`, `a0 = 0`, `b[0] = 0`, `d[0] = 1`,
`g = cos t`, `dg = -sin t`, `y0 = 0`, `T = 10`. Reference: `y = sin t`.

**PR — Prothero–Robinson.** `y' = -lambda (y - sin t) + cos t`,
`params = (lambda,)`, `n_coeff = 3`: `g = (lambda, lambda sin t, cos t)`,
`dg = (0, lambda cos t, -sin t)`, `b[0] = [[-1]]`, `b[1] = b[2] = 0`,
`d[0] = 0`, `d[1] = d[2] = 1`. `T = 2`, `lambda in {1, 1e2, 1e5}`. Reference:
`y(t) = sin t + e^{-lambda t} y0`. "On the manifold" means `y0 = 0`, "off"
means `y0 = 1`.

**D — 15-state stiff ladder.** A four-level ladder master equation in a
real basis, built by a generator in the test module with no physics package.
With `rho` a 4 × 4 Hermitian matrix of unit trace,

```
drho/dt = -i [Hm, rho] + sum_(i->j) G_ij (L rho L^T - (L^T L rho + rho L^T L)/2)
          + G_t (|0><0| tr(rho) - rho),          L = |j><i|
Hm = -diag(0, g0, g0 + g1, g0 + g1 + g2)
     + (g3/2)(|0><1| + h.c.) + (g4/2)(|1><2| + h.c.) + (g5/2)(|2><3| + h.c.)
G_10 = 38, G_21 = 0.3, G_32 = 0.1, G_t = 0.6
```

and state `y = (rho_11, rho_22, rho_33, Re rho_01, Re rho_02, Re rho_03,
Re rho_12, Re rho_13, Re rho_23, Im rho_01, ..., Im rho_23)` with
`rho_00 = 1 - rho_11 - rho_22 - rho_33` eliminated. `params` has seven
entries; `g_k = p_k` for `k = 0..4`, `g_5 = p_5 (1 + p_6 sin(pi t))`,
`dg_5 = p_5 p_6 pi cos(pi t)`. The blocks are obtained by unit
perturbations of the six coefficients. The generator is checked by: `a0`
has 17 nonzeros and `sum|a0| = 163`; the `b[k]` have 6, 8, 6, 12, 12, 12
nonzeros; only `d[3]` is nonzero (one entry, 0.5, at index 9); `c0 = 0`;
the union pattern has 65 entries including the whole diagonal; and
`a0 + sum p_k b[k]` reproduces a direct evaluation at 100 random parameter
points to `<= 1e-14` relative (measured 2.1e-16).

Members: shifts `s in {0, 0.1, 0.4, 1, 2}` with
`params = (-2000 s, 3250 s, 0, 3, 15, 5, 0.3)`, `T = 4`. `y0` is the steady
state with the last coupling off: `y0 = -A^{-1} c` at `p_5 = 0`, solved in
float64 on the host. Reference: SciPy Radau with the analytic Jacobian at
`rtol = 1e-13, atol = 1e-16`; it agrees with DOP853 at
`rtol = 1e-13, atol = 1e-17` to 7.6e-15 on every member (a test asserts
`<= 1e-13`). The members' largest state magnitudes are 1.6e-1, 7.4e-3,
1.9e-3, 7.5e-4, 3.7e-4 and their largest swings 1.6e-1, 1.4e-5, 1.8e-7,
1.4e-8, 2.1e-9; the spectral radius of `A` runs from 30 to 6500. An
explicit method needs 113 to 16,368 steps on them (SciPy RK45 at
`(1e-7, 1e-9)`).

**LU matrices.** `P3 = [[0,1,0],[0,0,1],[1,0,0]]`;
`TP = [[1e-20, 1],[1, 1]]`; `W8`, the 8 × 8 matrix with ones on the
diagonal and in the last column and `-1` below the diagonal; `R15` and
`R35`, standard normal matrices from `default_rng(7)` drawn in that order;
and `M = I / (h γ) - A` of problem D for `s in {0.4, 2}`, `h in {0.05, 4}`.
Right-hand sides are `M x` for a standard normal `x` from `default_rng(3)`.

## 8. Acceptance tests (concrete assertions)

Numbering continues from Phase 0. Each test is parameterised over the
implemented backends and, where stated, over precision. `save_mode` is
`"clip"` unless stated. `unit = rtol * ymax + atol`, with `ymax = max|y_ref|`
over the member's save grid. All tests must pass on `cpu` in CI and on
`cuda` on the 4090 before any Phase 1 benchmark number is recorded. The
thresholds were calibrated on the host; one that fails on `cuda` is reported
to the spec owner with the measured number and is not loosened by the
implementer.

**T10 — Tableau identities** (precision-independent, exact `Fraction`
arithmetic on the transcribed decimals). Convert to the standard form:
`Γ = (I/γ − C)^{-1}`, `α = a Γ`, `β = α + Γ`, `m = (a[8,1..7], 1)`,
`b = m Γ`, `m_hat = (a[7,1..6], 1, 0)`, `b_hat = m_hat Γ`. Assert:
`Γ` is lower triangular with every diagonal entry equal to `γ` exactly;
`α` is strictly lower triangular exactly; `|sum_j α[i,j] − c_i| <= 1e-13`
and `|sum_j Γ[i,j] − d_i| <= 1e-13` for every row (measured 4.0e-15,
2.4e-15); rows 6, 7, 8 of `a` agree in entries 1–5 and
`a[7,6] = a[8,6] = a[8,7] = 1` exactly; `b_i = β[8,i]` and
`b_hat_i = β[7,i]` exactly (stiff accuracy); the 17 order conditions of the
paper's Table 2 hold for `b` with residual `<= 1e-13` (measured 5.5e-15);
conditions 1–8 hold for `b_hat` with residual `<= 1e-13` (measured 4.0e-15)
and condition 9 fails for `b_hat` by `>= 1e-3` (measured 7.3e-3), so the
embedded order is exactly 4. In float64 from the same data:
`|R(−1e8)| <= 1e-6` and `|R_hat(−1e8)| <= 1e-7` (measured 1.25e-7, 2.2e-9);
`max |R(iy)| <= 1 + 1e-12` and the same for `R_hat` over
`y = logspace(−3, 6, 4001)`; `|R(−1) − R_hat(−1)| >= 1e-5` (measured
9.4e-5); and the continuous weights
`w(θ) = θ m + θ (1 − θ)(H_1 + θ H_2 + θ² H_3)`, converted by `Γ`, satisfy
conditions 1–8 with right-hand sides scaled by `θ^n` to `<= 1e-12` at
`θ in {0.25, 0.5, 0.75, 1}` (measured 1.8e-14). The rendered device header
reproduces every coefficient to within 1 ulp of `real_t`.

**T11 — Convergence slope (fixed step, FP64)** on S1 with
`fixed_step = T/N`, `N in {80, 160, 320, 640, 1280}`: least-squares slope of
`log(err)` against `log(h)`, `err = max|y(T) − y_ref(T)|`. Assert
`4.5 <= slope <= 5.5` (measured 4.84) and `err(N = 1280) < 1e-10` (measured
1.4e-11). A step that omits the `h d_i f_t` term measures 2.20 and fails.
On PR, on the manifold, `lambda = 1`, `N in {8, 16, 32, 64, 128}`: assert
`4.5 <= slope <= 5.5` (measured 5.04).
*FP32 variant (precision consistency):* on S1 and on PR with `lambda = 1`,
`N in {40, 80, 160, 320}`: assert
`max|y_fp32(T) − y_fp64(T)| <= 300 * eps32 * max(1, ymax)`, `eps32 = 2^-23`
(measured at most 46 `eps32`). No FP32 slope is asserted.

**T12 — Error-estimator scaling and non-vanishing (one step, FP64)**: for
`h in {0.5, 0.25, 0.125, 0.0625, 0.03125}` take one step from `y0` and
record `est = max|U_8|` and `true = max|y_1 − y_exact(h)|`, on L with
`Delta = 2`, L with `Delta = 0`, and Q. Assert for every consecutive pair
`4.5 <= log2(est(h)/est(h/2)) <= 5.5` (measured 5.02–5.43 on L, 4.97–5.00
on Q); `est(h) >= true(h)` for every `h`; and `est(h) > 1e-12` for every
`h` (smallest measured 6.2e-12). The estimate must not vanish on a linear
constant-coefficient problem or on a pure quadrature. *FP32 variant:* on L
with `Delta = 2`, `h in {1, 0.5, 0.25}`: `est_fp32` is within 10 % of
`est_fp64` (measured 0.1 %, 0.1 %, 2.2 %). The same assertions run on
`cuda` through the one-step entry point.

**T13 — Stiff accuracy and L-stability** on PR.
(a) Stiff order, FP64, on the manifold, `lambda = 1e5`, `fixed_step = 2/N`,
`N in {8, 16, 32, 64, 128}`, `err = |y(2) − sin 2|`: every consecutive
ratio satisfies `log2 >= 2.5` (measured 3.02–3.04), `err(N = 8) < 1e-8` and
`err(N = 128) < 1e-12` (measured 9.8e-10, 2.2e-13).
(b) L-stability, both precisions, one step from `y0 = 1`:
`|y_1 − sin h| <= 1e-4` for `(lambda, h) = (1e6, 1)` and `<= 1e-2` for
`(1e5, 0.1)` (measured 1.25e-5 and 1.25e-3 at both precisions).
(c) On the manifold, both precisions, one step with
`(lambda, h) = (1e6, 1)`: `|y_1 − sin h| <= 1e-6` (measured 2.9e-9 at FP64,
3.2e-8 at FP32).

**T14 — Agreement with reference under tolerance-scaled bounds**
(adaptive, every save time). Assert `status == SUCCESS` and
`max|y − y_ref| <= K * unit` for every member, with

| problems       | precision | `(rtol, atol)` pairs                | `K` | measured, in units |
|----------------|-----------|-------------------------------------|----:|--------------------|
| L, S1, Q, PR   | FP64      | `(1e-7, 1e-9)`, `(1e-10, 1e-12)`    |  10 | at most 0.60       |
| L, S1, Q, PR   | FP32      | `(1e-4, 1e-6)`, `(2e-5, 1e-7)`      |  10 | at most 0.61       |
| D              | FP64      | `(1e-7, 1e-9)`, `(1e-10, 1e-14)`    | 200 | at most 57.6, 61.2 |
| D              | FP32      | `(1e-4, 1e-6)`, `(2e-5, 1e-7)`      | 200 | at most 9.0        |

PR runs on the manifold for all three `lambda` and off it for
`lambda = 1e5`. D's constant is larger for the reason given in §3.7; at
`(1e-10, 1e-14)` the bound on member `s = 2` is 0.5 % of that member's
swing, so the assertion is not vacuous. D at `(1e-10, 1e-12)` is not
asserted (measured 288 units on `s = 2`).
*`"dense"` mode:* the same table holds for L, S1, Q, D and for PR with
`lambda in {1, 1e2}` (measured at most 0.70, and 61.2 on D). For PR with
`lambda = 1e5` only the value at `t_end` is asserted, `<= 10 * unit`
(measured at most 0.08); the interpolated save times are not (measured up
to 417 units, §3.6).

**T15 — Dense-output accuracy** (`"dense"` mode): on L with `Delta = 2` and
on S1, one fixed step from `y0` of size `h in {0.4, 0.2, 0.1, 0.05}`;
`e(h)` is the largest error of the interpolant over
`θ in {0.25, 0.5, 0.75}`. Assert `e(h)/e(h/2) >= 2^4.5` for every
consecutive pair (measured 31–32 on L, 46–78 on S1), and that
`dense(θ = 0) == y_n` and `dense(θ = 1) == y_n+1` bitwise.

**T16 — Batch-order and scheduling invariance.**
(a) Two batches of 64 members: L with the four `Delta` values cycled, and D
with the five shifts cycled; member `i` has its last parameter (`Delta`,
respectively the modulation depth `p_6`) multiplied by `1 + 1e-3 i`. Solve;
solve again with the members permuted by a fixed random permutation (seed
0); un-permute. Assert `y`, `obs`, `status` and all counters are bitwise
equal on every backend and precision, and that `n_accepted` is not constant
across the members of each batch.
(b) The same D batch solved with `sort_key = None`, with
`sort_key = shift`, and with a random key: all per-member outputs bitwise
equal.
(c) `lane_utilisation`, pure function: step counts `[1]*32 + [3]*32` in
that launch order give exactly 1; the same counts interleaved give exactly
`2/3`; 33 members of one step give `33/64`.
(d) FP64 `(1e-7, 1e-9)`, 160 members of D, the five shifts cycled, no
perturbation: `lane_utilisation <= 0.55` without a key and `>= 0.99` with
`sort_key = shift` (expected 0.447 and 1 from the measured attempts 113,
404, 271, 72, 43).

**T17 — Step, evaluation and factorisation counts** (FP64 `(1e-7, 1e-9)`;
FP32 `(1e-4, 1e-6)`). Attempts are `n_accepted + n_rejected`:

| problem              | FP64 attempts in | measured | FP32 attempts in | measured |
|----------------------|------------------|---------:|------------------|---------:|
| L Delta=0            | `[50, 100]`      |       56 | `[50, 100]`      |       53 |
| L Delta=2            | `[60, 180]`      |      100 | `[50, 100]`      |       53 |
| L Delta=20           | `[420, 1260]`    |      701 | `[150, 460]`     |      256 |
| L Delta=200          | `[2800, 8300]`   |     4627 | `[760, 2300]`    |     1273 |
| S1                   | `[190, 560]`     |      310 | `[70, 210]`      |      117 |
| Q                    | `[56, 170]`      |       93 | `[50, 100]`      |       56 |
| PR on, each `lambda` | `[40, 80]`       |       44 | `[40, 80]`       |    44–45 |
| PR 1e5 off           | `[55, 170]`      |       92 | `[40, 110]`      |       60 |
| D s=0                | `[68, 200]`      |      113 | `[40, 100]`      |       41 |
| D s=0.1              | `[240, 730]`     |      404 | `[40, 100]`      |       55 |
| D s=0.4              | `[160, 490]`     |      271 | `[40, 100]`      |       41 |
| D s=1                | `[43, 130]`      |       72 | `[40, 100]`      |       41 |
| D s=2                | `[40, 80]`       |       43 | `[40, 100]`      |       41 |

Assert for every member `n_lu == attempts`, `n_solve == 8 * attempts`,
`n_fev == 8 * attempts` plus 2 if `h0` was automatic,
`n_rejected <= 0.4 * n_accepted` (largest measured 0.24, FP32 L with
`Delta = 20`), and `status == SUCCESS`. In `"dense"` mode at FP64, where
the save grid sets no floor: L `Delta = 0` in `[21, 63]`, S1 in
`[170, 520]`, D `s = 2` in `[4, 15]` (measured 35, 288, 6). The last row is
the point of the phase: SciPy RK45 needs 16,368 steps on that member.

**T18 — Status codes.** L `Delta = 200` with `max_steps = 100` returns
`MAX_STEPS` with `n_accepted + n_rejected == 100` and NaN from the first
unreached save time. A member with `y0 = NaN` returns `NON_FINITE` after at
most one attempt. A member with a NaN parameter returns `NON_FINITE`. The
one-state problem `a0 = [[v]]`, `n_coeff = 0`, with `fixed_step = 0.5` and
`v = 1 / (0.5 * γ)` computed in the run's precision, returns
`SINGULAR_MATRIX` with no accepted step. L with `Omega = 0` (a member
exactly at rest) returns `SUCCESS` at both precisions with `y == y0` at
every save time. The `IntEnum` equals the header.

**T19 — CPU reference independence and GPU-vs-CPU agreement.** The NumPy
reference imports only `numpy` and the tableau module; a test asserts that
`reference/rodas5p_numpy.py` contains no import of `omnitool.backends`,
`scipy` or `omnitool.reference.dopri5_numpy`. For every run of T14, each
device backend agrees with `cpu` to `max|Δy| <= 2 * K * unit` and
`|attempts − attempts_cpu| <= max(3, 0.05 * attempts_cpu)`. Bitwise
equality across backends is not asserted. (These two bounds carry over
Phase 0's form with a wider count allowance for the different LU; they
could not be calibrated without a device and are confirmed or reported on
the first `cuda` run.)

**T20 — Factor-and-solve** (both precisions, through the test entry point;
on `cpu` this exercises `numpy.linalg.solve`). For every matrix `M` and
right-hand side `r` of §7: `max|M x − r| <= 2 n eps (norm_inf(M) max|x| +
max|r|)`, evaluated in float64 (measured at most 0.12 of
`n eps (...)`). `TP` and `P3` in particular return the exact solution to
`4 eps`; an implementation without pivoting returns `x_1 = 0` on `TP`. The
matrices `[[1, 2], [2, 4]]` and the 3 × 3 zero matrix report a zero pivot.

**T21 — Coefficient check.** `check_coefficients` passes on S1, Q, PR and D
at 64 times drawn uniformly on `[0, T]` (seed 0; the largest measured
difference is 0.3 % of the threshold). With S1's `dg` replaced by
`(0, 0, 4.5 sin(1.5 t))` it raises `ValueError` naming coefficient 2.
With `time_dependent=False` set on S1, `solve_batch` raises `ValueError`
on every backend, because `g` differs between `t0` and `t_end`.

**T22 — Observables and group sums.** With a random `W` of shape
`(2, n_state)` (seed 0) on the D batch of T16(a):
`|obs − y W^T| <= n_state * eps * (|y| |W|^T)` elementwise; `obs` with
`save_state=False` is bitwise equal to `obs` with `save_state=True`, and
`result.y is None`. With `groups = i mod 4` and `weights` uniform on
`[0, 1)` (seed 1): `obs_grouped` is bitwise independent of `sort_key`; it
equals a sequential host sum in caller order to
`<= n_g * eps * sum|w obs|` per element, `n_g` the group size; and after a
permutation of the caller's members (groups and weights permuted with them)
it changes by at most the same bound (measured 0.006 of it for 561-member
sums).

**T23 — Assembly** (both precisions, through the test entry point). For
problem D at the five shifts and `t in {0, 1.3, 4}`: the assembled matrix
equals the float64 `a0 + sum g_k b[k]` elementwise to
`(n_coeff + 2) * eps * (|a0| + sum |g_k| |b[k]|)`, and likewise the
forcing (measured: exact at FP64, 0.33 `eps32` of the largest entry at
FP32). The rendered pattern has 65 entries.

## 9. RydIQule adapter (optional extra)

`omnitool.adapters.rydiqule`, installed by `omnitool[rydiqule]` with
`rydiqule==2.1.3` pinned. It is the only module that imports `rydiqule`,
and nothing in the core imports it.

Contract:

1. **Extraction.** From a RydIQule `Sensor`, build the shared blocks by
   unit perturbations of every scanned parameter, the Doppler shift per
   class, and every time-dependent coupling. The physics comes from
   RydIQule; none is reimplemented. Which RydIQule entry point is used
   (`solve_time`'s `solver=` hook, as in `stiffness_check.py`, or
   `generate_eom`) is decided in the adapter session from the pinned
   source and recorded in an ADR.
2. **Members.** One member per scan point and Doppler class. The default
   `sort_key` is the magnitude of the member's Doppler shift.
3. **Time dependence.** For each coupling with a `time_dependence`
   callable, the caller supplies a C expression and its derivative; the
   adapter checks the expression's NumPy twin against the callable at 64
   random times and runs `check_coefficients` (ADR 0009).
4. **Outputs.** Observables are selected by `dm_basis` label, never by
   position; axes are mapped by `axis_labels()`; the Doppler average is a
   group sum with RydIQule's own weights.

Adapter tests run in their own CI job with the extra installed:

- **AD1.** On the four-level ladder of `stiffness_check.py`, the blocks
  reproduce RydIQule's equations of motion at 50 random parameter points to
  `<= 1e-13` relative (the research phase measured about 1e-16).
- **AD2.** `−A^{-1} c` from the blocks equals RydIQule's steady state to
  `<= 1e-10` at FP64 (ADR 0002's rule; the research phase measured 1e-15).
- **AD3.** Time domain without Doppler against RydIQule's own solve at its
  tightest usable tolerance, bounded as T14.
- **AD4.** The 561-class Doppler-summed signal against the Radau reference
  of `stiffness_check.py`, as a fraction of the signal's swing.

The constants of AD3 and AD4 cannot be calibrated without the pinned
package, which is not installed in the core's environment. They are
measured in the adapter session and added here by amendment, ruled by the
spec owner, before the adapter is merged.

## 10. Baselines and benchmarks

**Baselines** (ADR 0011) come first: `benchmarks/baselines/cubie/` and
`benchmarks/baselines/modax/`, each with its own lock file, a script that
runs the 561-class problem through the adapter's extraction, and a
manifest. modax is time-boxed to half a day.

**`benchmarks/bench_rodas5p.py`** solves problem D with the five shifts
cycled over the members for `n_members in {1024, 16384, 262144}`, at both
precisions, in both save modes, with and without `sort_key = shift`, with
one observable and `save_state=False`. It follows Phase 0 §9: the T14 gate
on a seeded sample before any timer starts, the stream synchronised before
every timer stops, compile time separate, and a manifest that also records
the method, save mode, whether a sort key was used, `lane_utilisation` and
the totals of `n_lu` and `n_solve`. The CPU baselines are the `cpu` backend
(capped, as in Phase 0) and SciPy Radau with the analytic Jacobian over the
same members at matched achieved accuracy.

**`benchmarks/bench_rydiqule_doppler.py`** runs the 561-class problem and
its replication over a scan axis through the adapter, and reports omnitool
against the baselines at matched achieved accuracy from the tolerance
sweeps. The Radau gate of ADR 0006 is re-evaluated on this run.

No number from either script is quoted until T10–T23 pass on `cuda`, and
none leaves the repository before the Newton H100 runs exist.

## 11. Order of work

1. RydIQule extraction with AD1 and AD2.
2. cubie baseline, then the modax time-box; manifests on the 4090.
3. Tableau transcription and T10 (main session).
4. `AffineProblem`, the API, `check_coefficients`, the NumPy reference and
   the test problems; T11–T18 and T21–T23 on `cpu`.
5. The LU spike of ADR 0008 on the 4090.
6. `rodas5p.cu` and the `cuda` path; T10–T23 on `cuda`.
7. Group sums on the device.
8. `bench_rodas5p.py` and its manifests.
9. The rest of the adapter (AD3, AD4), `bench_rydiqule_doppler.py`, the
   Radau gate, and the two upstream issue drafts under `docs/upstream/`.

Steps 1–2 and 3–5 are independent of each other.

## 12. Definition of done

The first cut of Phase 1 is complete when: T10–T23 pass on `cpu` in GitHub
Actions
and on `cuda` on the 4090, with the Phase 0 tests T1–T9 still passing; the
LU spike result is recorded in ADR 0008; AD1–AD4 pass in the adapter job
with their constants ruled; the cubie baseline has manifests on the 4090
and the modax attempt is recorded either way; `bench_rodas5p.py` and
`bench_rydiqule_doppler.py` have each produced manifests at both precisions
on the 4090; the Radau gate of ADR 0006 has been evaluated on the RydIQule
run and its outcome recorded; the two upstream issues exist as drafts; and
`CHANGELOG.md` lists the phase under `Unreleased`. Publishing any number
additionally needs the Newton H100 manifests.
