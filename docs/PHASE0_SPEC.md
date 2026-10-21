# Phase 0 specification: one vertical slice

- Status: accepted 2026-10-19
- Governing ADRs: 0001 (scope), 0002 (precision), 0003 (verification),
  0004 (backend)
- Owner of numerics and tableau transcription: the main session (see
  `AGENTS.md`); the coder agent implements modules from this document and
  never changes it.

## 1. Purpose

Phase 0 proves the full path from a Python call to a per-member adaptive GPU
integration and back, with the verification layer of ADR 0003 in place
before any benchmark is run. It ships exactly one method, explicit
**Dormand–Prince 5(4)** ("DOPRI5"), in a **thread-per-member** layout, with a
**per-member PI step controller**, **dense output at shared save times**,
**FP32 and FP64** instantiations of the same template, and **status codes**.
Nothing physics-specific appears anywhere in the core.

Out of scope for Phase 0 (deferred to Phases 1–2): implicit methods, events,
RNG, early termination, per-member outputs other than the solution, field
tables, block-per-member layout, mixed precision, per-member tolerances.

## 2. Public API

```python
result = omnitool.solve_batch(
    problem,            # Problem (see below)
    y0,                 # (n_members, n_state) array
    params,             # (n_members, n_params) array
    t_span=(t0, t_end), # floats, t_end > t0
    t_save,             # (n_save,) strictly increasing, t_save[0] == t0,
                        #   t_save[-1] == t_end
    rtol=1e-7, atol=1e-9,
    precision="fp64",   # "fp32" | "fp64"
    backend="cuda",     # "cpu" | "cudasim" | "cuda"
    max_steps=100_000,  # accepted + rejected steps per member
    h0=None,            # None -> automatic initial step
    fixed_step=None,    # float -> controller bypassed (test use only)
)
```

`Problem` is a frozen dataclass:

```python
@dataclass(frozen=True)
class Problem:
    name: str
    n_state: int
    n_params: int
    rhs_numpy: Callable   # rhs_numpy(t, y, p) -> dydt, y and p 1-D arrays
    rhs_cuda: str         # C body of  void rhs(real_t t, const real_t* y,
                          #                    real_t* dydt, const real_t* p)
```

`BatchResult` carries, all on the host as NumPy arrays:

| field        | shape                           | meaning                          |
|--------------|---------------------------------|----------------------------------|
| `y`          | `(n_members, n_save, n_state)`  | solution at the save times       |
| `status`     | `(n_members,)` int32            | status code, §6                  |
| `n_accepted` | `(n_members,)` int32            | accepted steps                   |
| `n_rejected` | `(n_members,)` int32            | rejected steps                   |
| `n_fev`      | `(n_members,)` int32            | RHS evaluations                  |
| `h_last`     | `(n_members,)` real             | last accepted step size          |

Rules: `rtol < 100 * eps(precision)` raises `ValueError`; `atol < 0` raises;
`y0` and `params` are cast to the requested precision with a warning if the
cast loses information; a member whose integration fails has `status != 0`
and `y` is filled with NaN from the first unreached save time onward.

## 3. Numerics

**Tableau.** Dormand & Prince (1980), as tabulated in Hairer, Nørsett &
Wanner, *Solving ODEs I*, Table 5.2 (7 stages, FSAL: stage 7 of step *n* is
stage 1 of step *n+1*). The transcription lives once in
`src/omnitool/tableaus/dopri5.py` and is rendered into the device header from
that file; it is never typed twice. Coefficients are stored as exact
rationals and rendered to 17 significant digits.

**Error norm.** With `sc_i = atol + rtol * max(|y_n,i|, |y_n+1,i|)`,
`err = sqrt(mean_i((h * (b - b̂) · k)_i / sc_i)^2)`. Accept when `err <= 1`.

**PI controller** (Hairer's `dopri5.f`, Gustafsson form): with
`alpha = 0.2 - 0.75 * beta`, `beta = 0.04`, `safe = 0.9`,
`fac = clamp(err^alpha / err_old^beta / safe, 0.1, 5.0)`,
`h_new = h / fac`. After an accepted step `err_old = max(err, 1e-4)`. After a
rejection `h_new = h / min(5.0, err^alpha / safe)` (no growth) and
`err_old` is unchanged.

**Initial step.** Hairer's `hinit` (HNW I, §II.4, eq. 4.14) with the member's
own `y0` and `f(t0, y0)`, unless `h0` is given.

**Step clipping.** The last step is clipped to land exactly on `t_end`.
Steps are never clipped to save times. A member stops with
`STEP_TOO_SMALL` when `|h| < 16 * eps * max(|t|, 1)`.

**Dense output.** Hairer's fourth-order continuous extension for DOPRI5
(HNW I, §II.6; the `rcont` coefficients of `dopri5.f`), evaluated at every
save time in `(t_n, t_n+1]` after each accepted step. At `theta = 0` it
returns `y_n` and at `theta = 1` it returns `y_n+1` exactly.

**Precision.** `real_t` is a template parameter; all constants are rendered
in `real_t`. No `float`/`double` literals outside the template header. FMA
contraction is allowed; no fast-math flags (ADR 0002).

## 4. Device design

- One thread per member. Thread *i* owns `y[i]`, `k1..k7`, `t`, `h`,
  `err_old`, counters and the next save index, all in registers or local
  memory. No shared memory and no warp intrinsics in Phase 0.
- Device arrays are member-fastest: `y0_d[n_state, n_members]`,
  `params_d[n_params, n_members]`, `out_d[n_save, n_state, n_members]`,
  so adjacent threads touch adjacent addresses. The Python layer transposes
  to the API shapes.
- The kernel template `dopri5.cu` is plain C: a `rhs` function supplied by
  templating, `step()`, `controller()`, `dense()` and the driver loop. The
  only compile-time parameters are `real_t`, `N_STATE`, `N_PARAMS` and the
  RHS body. Launch configuration: 128 threads per block, grid rounded up.
- Status codes are defined once in `src/omnitool/status.h` and mirrored by
  an `IntEnum` that a test asserts equal to the header.

## 5. Backends

| backend   | what runs                                                     |
|-----------|---------------------------------------------------------------|
| `cpu`     | `reference/dopri5_numpy.py`: plain NumPy, one member at a time, written from the paper, shares only the tableau file |
| `cudasim` | `dopri5.cu` compiled as host C++ through `shim.h` (redefines `__device__`, `__global__`, thread index), loaded with `ctypes`, members in a serial loop |
| `cuda`    | `dopri5.cu` through CuPy `RawModule`/NVRTC on the device      |

`cpu` is the oracle. `cudasim` runs the real kernel logic on every CI push.
`cuda` runs on the self-hosted 4090 runner only.

## 6. Status codes

| code | name             | meaning                                         |
|-----:|------------------|-------------------------------------------------|
|   -1 | `NOT_RUN`        | initial value; never returned by a finished run |
|    0 | `SUCCESS`        | reached `t_end`                                 |
|    1 | `MAX_STEPS`      | `n_accepted + n_rejected == max_steps`          |
|    2 | `STEP_TOO_SMALL` | `h` fell below the floor in §3                  |
|    3 | `NON_FINITE`     | NaN or Inf in `y` or in the error norm          |

## 7. Test problems

All problems have analytic or SciPy references and are defined once in
`tests/problems.py` with both a NumPy RHS and a C RHS body.

**L — two-level optical Bloch (affine linear, constant coefficients).**
`y = (u, v, w)`, params `(Gamma, Omega, Delta)`:
`u' = -Gamma/2 u + Delta v`, `v' = -Delta u - Gamma/2 v - Omega w`,
`w' = Omega v - Gamma (w + 1)`. `y0 = (0, 0, -1)`, `Gamma = Omega = 1`,
`Delta in {0, 2, 20, 200}`, `T = 10`. Reference: closed form
`y(t) = e^{Jt} y0 + J^{-1}(e^{Jt} - I) g` via `scipy.linalg.expm`.

**S1 — chirped Bloch (smooth, non-autonomous).** As L with
`Delta(t) = 2 + 3 sin(1.5 t)`. Reference: SciPy DOP853 at
`rtol = 1e-13, atol = 1e-15`.

**S2 — Duffing oscillator (smooth, nonlinear, Hamiltonian).**
`y0' = y1`, `y1' = -y0 - y0^3`, `y0 = (1, 0)`, `T = 10`. Reference: DOP853 as
above. Invariant `H = y1^2/2 + y0^2/2 + y0^4/4 = 3/4`.

Save grid for adaptive tests: `t_save = linspace(0, T, 51)`.

## 8. Acceptance tests (concrete assertions)

Each test is parameterised over `backend in {cpu, cudasim, cuda}` (the
`cuda` case skips without a device) and, where stated, over precision. All
tests must pass on `cpu` and `cudasim` in CI and on `cuda` on the 4090
before any benchmark number is recorded.

**T1 — Tableau identities** (precision-independent, on the rational
transcription):
`sum(b) == 1`, `sum(b_hat) == 1`, `sum_j a[i, j] == c[i]` for every row,
`a[6, :] == b` (FSAL), order conditions through order 5 for `b` and through
order 4 for `b_hat` satisfied exactly in rational arithmetic, and the
rendered device header reproduces every coefficient to within 1 ulp of
`real_t`.

**T2 — Convergence slope (fixed step, FP64)** on S1 and S2:
run with `fixed_step = T/N` for `N in {40, 80, 160, 320, 640}`, measure
`max|y(T) - y_ref(T)|`, fit `log(err)` against `log(h)` by least squares.
Assert `4.5 <= slope <= 5.5` and `err(N=640) < 5e-9`.
*FP32 variant (precision consistency):* the FP32 roundoff floor is reached
by `N = 160` on S2, so FP32 is not an order test. Run
`N in {20, 40, 80, 160, 320, 640}` at FP32 and at FP64 and assert at every
`N` that `max|y_fp32(T) - y_fp64(T)| <= 100 * eps32 * max(1, ymax)` with
`ymax = max|y_ref(T)|` and `eps32 = 2^-23`. Keep the points whose FP32
error exceeds `100 * eps32` (about 1.2e-5); when three or more survive,
also assert `slope >= 4.0` over them; otherwise no slope is asserted.
(Measured on the reference: S1 keeps `N = 20, 40, 80`; S2 keeps
`N = 20, 40`; the FP32/FP64 gap is at most 2.7e-6.)

**T3 — Error-estimator scaling and non-vanishing (one step on L, Delta = 2,
FP64)**: for `h in {0.5, 0.25, 0.125, 0.0625, 0.03125}` take one DOPRI5 step
from `y0` and record `est = max|h (b - b̂)·k|` and
`true = max|y_1 - y_exact(h)|`. Assert, for every consecutive pair,
`4.5 <= log2(est(h)/est(h/2)) <= 5.5`; assert `est(h) >= true(h)` for every
`h` and `est(h) > 1e-12` for every `h` (the estimate must not vanish on a
linear problem). The same assertions run on `cudasim` and `cuda` with the
device `step()` exposed through a test entry point.

**T4 — Agreement with reference under tolerance-scaled bounds**
(adaptive, all problems, all save times):
let `ymax = max|y_ref|` over the save grid and
`bound = 10 * (rtol * ymax + atol)`. Assert
`max|y - y_ref| <= bound` and `status == SUCCESS` for every member, for

| precision | `(rtol, atol)` pairs           |
|-----------|--------------------------------|
| FP64      | `(1e-7, 1e-9)`, `(1e-10, 1e-12)` |
| FP32      | `(1e-4, 1e-6)`, `(1e-5, 1e-7)`   |

**T5 — Dense-output accuracy**: on S2, one fixed step from `y0` of size
`h in {0.2, 0.1, 0.05}`, interpolate at `theta = 0.5`, compare with DOP853 at
`t = h/2`. Assert the interpolation error halves at least `2^4.5` per step
halving, that `dense(theta = 0) == y_n` and `dense(theta = 1) == y_n+1`
bitwise, and that in T4 the error at save times that are not step endpoints
satisfies the same `bound`.

**T6 — Batch-order invariance**: build a batch of 64 members by cycling the
four L problems and S1, S2 with per-member perturbations of `y0` by
`1e-3 * member_index`; solve; solve again with the members permuted by a
fixed random permutation (seed 0); un-permute. Assert `y`, `status` and all
counters are **bitwise equal** (`np.array_equal`) on every backend and
precision.

**T7 — Step and evaluation caps** (FP64 `(1e-7, 1e-9)`; FP32 `(1e-4, 1e-6)`):

| problem      | FP64 `n_fev` in | FP32 `n_fev` in |
|--------------|-----------------|-----------------|
| L Delta=0    | `[150, 500]`    | `[50, 200]`     |
| L Delta=20   | `[2500, 8500]`  | `[800, 2500]`   |
| L Delta=200  | `[20000, 60000]`| `[5000, 16000]` |
| S1 chirp     | `[700, 2200]`   | `[200, 700]`    |
| S2 Duffing   | `[400, 1200]`   | `[100, 400]`    |

Assert `n_fev == 1 + 6 * (n_accepted + n_rejected)` plus 1 if `h0` was
automatic, `n_rejected <= 0.3 * n_accepted` for every member, and
`status == SUCCESS`. The lower bounds catch a dead estimator.

**T8 — Status codes**: L Delta = 200 with `max_steps = 100` returns
`MAX_STEPS` with `n_accepted + n_rejected == 100` and NaN at unreached save
times; a problem whose RHS returns `1/(t - 5)` with `t_end = 10` returns
`STEP_TOO_SMALL` or `NON_FINITE`, never `SUCCESS`; a member with `y0 = NaN`
returns `NON_FINITE` after at most one step; the `IntEnum` equals the header.

**T9 — CPU reference independence and GPU-vs-CPU agreement**: the NumPy
reference imports only `numpy` and the tableau module; a test asserts that
`reference/dopri5_numpy.py` contains no `import` of `omnitool.backends`.
For every problem in T4, `cuda` and `cudasim` results agree with `cpu` to
`max|Δy| <= 2 * bound` (the T4 bound) and
`|n_accepted - n_accepted_cpu| <= max(2, 0.02 * n_accepted_cpu)`. FMA and
summation order differ, and one differently-judged accept/reject decision
changes the whole step sequence, so bitwise equality is **not** asserted
across backends.

## 9. First benchmark script

`benchmarks/bench_dopri5.py` solves L (Delta cycled over the four values) for
`n_members in {1024, 16384, 262144}` at both precisions, synchronises the
stream before stopping timers, excludes compile time and reports it
separately, and writes `manifest.json` with: git commit and dirty flag, lock
file sha256, `nvidia-smi --query-gpu=name,driver_version,memory.total`,
CUDA runtime version, precision, FMA flag, seed, sha256 of inputs and
outputs, and the compile / H2D / kernel / D2H split. A same-machine CPU
baseline is the `cpu` backend over the same batch. No number from this script
is quoted until T1–T9 pass on `cuda`.

## 10. Definition of done

Phase 0 is complete when, on the first tagged commit after the scaffold:
T1–T9 pass on `cpu` and `cudasim` in GitHub Actions and on `cuda` on the
4090 runner; the ADR 0004 spike checklist is recorded as passed in ADR 0004;
`bench_dopri5.py` has produced one manifest at each precision; and
`CHANGELOG.md` lists the slice under `Unreleased`.
