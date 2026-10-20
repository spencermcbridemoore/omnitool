# GPU ensemble ODE core — project handoff

*Written 2026-10-18 at the end of a long Cowork research thread (Opus, then Fable 5.1). This is the single document a new session should read first. Evidence files are listed in §7; measured numbers below are from scripts delivered in that thread, not from memory.*

## 1. Status in one paragraph

The project started as "a GPU backend for RydIQule" and is now a **common GPU ensemble ODE core** that serves RydIQule, pylcp, MoleCool, the UCN spin tracker and similar codes through thin adapters, with demonstration calculations delivered as notebooks for Newton (H100), Google Colab (T4) and Jetstream2. The research phase is complete: the architecture is settled (two front doors over shared machinery, §2), the prior art has been surveyed and the closest competitor (cubie) has been audited at source level and probed numerically (§3), reproducibility and repo practices are chosen (§4–5), and the remaining decisions are the ones only Max can make (§6). The next step is to create the repository and build Phase 0 (§5) in Claude Code.

## 2. Architecture decisions (settled)

**D1 — Common core, adapters as optional extras, never a fork.** The core depends on nothing physics-specific. Each downstream package (RydIQule pinned from PyPI, pylcp, MoleCool, …) is an optional extra with its own adapter and tests; upstream hooks go in as pull requests from forks. Diagram: *Solver Sharing Map* (https://claude.ai/artifact/DwUupi5LtENGq4x6V9jV27); earlier comparison page *Bloch Solver Stacks* (https://claude.ai/artifact/6nmmgi4Lvxgj5mYjgQ9L3a).

**D2 — Two front doors over shared machinery.**
(a) *Affine-linear master equations* dρ/dt = (A₀ + Σₖ fₖ(t) Bₖ) ρ + c₀ + Σₖ fₖ(t) dₖ, with the exact Jacobian known analytically, so Rosenbrock-W / Radau stepping is cheap and stiff Doppler stacks are handled per member. RydIQule's EOM were measured to be exactly affine in every scanned parameter (reconstruction error ~1e-16; each Bₖ has 6–12 nonzeros of 225 for the 4-level ladder).
(b) *Nonlinear trajectories* (UCN, pylcp/MoleCool-type), 10–15 states, user device RHS or a `TabulatedField` (cubic B-spline + signed-distance field for walls), with **events**: wall collisions, quantum jumps / spin flips via cumulative hazard, early termination; per-thread RNG; warp-vote event detection (`__ballot_sync`), warp-cooperative bisection, cost-sorted member scheduling.
Shared: per-member adaptive controller, dense output at shared save times, common end time, reductions, thread-per-member and warp/block-per-member layouts, status codes, FP32/FP64 everywhere.

**D3 — FP64 is a first-class, benchmarked configuration; fastmath is off by default.** Stiff implicit steps lose 4–6 digits to conditioning in FP32 and cubie's own docs cap FP32 at rtol ≈ 1e-6; spectroscopy needs rtol 1e-7 / atol 1e-9. The 4090 (FP64 at 1/64 rate) is for FP32 development and correctness; the H100 on UCF Newton (FP64 at 1/2 rate) is for FP64 production and all published numbers. Compile flags that break IEEE 754 (`arcp`, `afn`, `ftz`, `reassoc`) are never on in validation builds; FMA contraction is allowed and is why GPU-vs-CPU tests use tolerances, not bitwise equality.

**D4 — Verification layer independent of the device tableaus.** Reference solutions come from analytic cases, SciPy DOP853/Radau at 1e-12, convergence-slope tests, tableau identities, invariants (trace, norm), nfev caps and batch-order invariance — and the *error estimator itself* is tested (its estimate must scale as h^(p+1) and must not vanish on a linear test problem). This is the test that cubie structurally cannot run and that would have caught its ROS3P default (§3).

**D5 — Key numerical facts to carry forward.**
- RydIQule 2.1.3 time domain: the stacked EOM is integrated as one flattened system with one shared step; per-class RK45 nfev ranges 278–95,468 across 561 Doppler classes (corr 0.999 with |k·v|); Radau needs 6 steps for 361/561 classes. Stock RydIQule with `method="Radau"` + block-diagonal `jac_sparsity` (rtol 1e-7, atol 1e-9): 3.0 s vs 45 s, better observable accuracy. → file upstream as an issue with `stiffness_check.py`.
- Lockstep (vmap-style) GPU work ≈ shared-step work; the win comes from *per-member* adaptive stepping with implicit steps for stiff members, plus ordering members by expected cost so warps stay coherent.
- Steady state (earlier phase): fused Numba kernel 26× one core / 53× two cores vs RydIQule; FP32 fails on weak-RF differential signals (4.5e-8 relative signal) unless one step of FP64 iterative refinement is added.
- The old UCN CUDA kernel is an architectural template only: its Cash–Karp stages evaluate the wrong states (collapses to forward Euler; verified analytically and numerically with `rkck_check.py`), plus eight other defects; Numerical-Recipes-derived code must not enter the new repo.

## 3. cubie: why it is not the starting point, and what it is instead

cubie (cubiepy/cubie, v0.16.0, MIT, single author, Numba-CUDA-MLIR) is the closest prior art: one thread per run, per-run adaptive controllers, ERK/DIRK/FIRK/Rosenbrock-W, symbolic Jacobian with per-thread sparse LU, driver splines, kernel cache, auto-tuner, paired ABBA performance gate, real-GPU CI on Windows and Linux. Probed under its CPU simulator at float64: per-run stepping works (64 vs 8588 steps in one batch at consistent 1e-8 accuracy); 15-state systems codegen in ~10 s, 63-state in ~34 s with 4.4 MB of generated source.

Not the starting point because: no events, RNG, root finding, early termination, mid-run state changes or per-run outputs (its docs: "`dense_output` and event functions have no cubie equivalent"); user RHS is AST-parsed into sympy, no path for field tables; LU is fully unrolled straight-line code per thread, unworkable at 255 states (16-level manifold) and no block-per-member mode; FP64 declared "not a realistic test case, ever" and fastmath on by default; pinned to a privately patched compiler fork (`cubie-numba-cuda-mlir`, 2,045-line shim) — bus factor 1 for both solver and compiler; pre-1.0 breaking changes. Its **default Rosenbrock tableau (ROS3P) has an error estimate that vanishes identically on every linear constant-coefficient problem** (embedded method shares the stability function; verified algebraically and with an independent numpy implementation): at rtol 1e-7 it returned 1e-2–1e-1 errors with status 0. RODAS3P and ROS34PW2 are fine.

What cubie *is*: the **benchmark baseline** and the **first adapter target** for the linear front door (RydIQule → cubie at float64, fastmath off, `rodas3p`/`ros34pw2`, grid sorted by Doppler shift) — a few days of work that yields the 561-class per-member result immediately and sets the bar the core must beat; the **engineering bar** to match (CPU reference layer, ABBA performance gate, gated real-GPU CI); and a **contribution target** (file the ROS3P issue with `ros3p_numpy.py`). Fair comparison rule: same problem, both at float64, matched *achieved* accuracy, compile time separate.

Also a backend warning: an expert who went all-in on numba-cuda-mlir needed a 2k-line shim and a private wheel. Stock `numba-cuda` (maintenance mode through CUDA 13) or CuPy `RawModule`/NVRTC are the safer near-term backends; keep the kernel source backend-agnostic enough to switch.

## 4. Reproducibility and benchmark policy (settled)

Run manifests for every published number: git commit + dirty flag, lockfile hash, container digest, `nvidia-smi --query`, Slurm job ID/`sacct`, precision and FMA flags, seeds via `SeedSequence`, sha256 of inputs/outputs, timing split (compile / H2D / kernel / D2H). Large-run artifacts to Zenodo (50 GB/record) fetched with Pooch `retrieve("doi:…", known_hash=…)`. Small CPU-verifiable checks reproduce everywhere; large runs are documented by hardware/version/manifest (SC Artifact Description/Evaluation style; ACM badge vocabulary). Honest speedup reporting: same-node CPU baseline at matched precision and accuracy, synchronize before stopping timers, compile time separate, include competitors (PQLS, cubie, DiffEqGPU.jl) where they apply. Target venue for the core: JOSS (needs 6 months public history and an AI-usage disclosure).

## 5. Repo plan

Practices (from the survey of similar numerical repos): `src/` layout, `pyproject.toml` with `uv` lock, Python ≥ 3.11, pre-commit (ruff + pyright), backend-parameterized tests (`cpu` / `cudasim` / `cuda`), notebooks under `nbmake` by glob, two-tier benchmarks (quick CPU-verifiable; full GPU with manifests), tag → wheels → PyPI trusted publishing, Keep-a-Changelog, `CITATION.cff` + Zenodo DOI, weekly CI. No free GPU CI exists: a self-hosted 4090 runner gated by `workflow_dispatch`/label/nightly, never for fork PRs. Record design decisions as short ADRs (`docs/adr/`), and keep `AGENTS.md`/`CLAUDE.md` as the operating rules (seed in the project; revise for the common-core framing).

Phases:
0. **Skeleton + one vertical slice** — package, CI, reference layer, Dormand–Prince 5(4) thread-per-member kernel with per-member controller, dense output at save times, FP32/FP64, status codes; tests against SciPy; a first benchmark script with a manifest. Wipe-and-restart is acceptable here (branches/tags, not deleted repos); after Phase 0 the public history matters for JOSS.
1. **Linear front door** — Rosenbrock-W (RODAS-type, L-stable) and Radau with the analytic affine Jacobian, block-diagonal Doppler stacks, cost-sorted members; **RydIQule adapter**; cubie baseline comparison on the 4090 (FP32) and Newton (FP64); upstream RydIQule issue.
2. **Trajectory front door** — `TabulatedField`, events with warp votes, RNG, early termination, per-member outputs; **UCN adapter** (clean rewrite); comparison with CATAPULT/firm3d.
3. **Demonstrations as notebooks** — targets ranked in `gpu-ensemble-targets.xlsx` (Bragg interferometry headline, RydIQule motion run, CRPropa, Sorcha, proton EDM, MoleCool, non-physics candidates), each with a small CPU-verifiable twin.

## 6. Decisions taken 2026-10-19 (details in `first-claude-code-session.md`)

Max chose the name **`omnitool`** and the location **`github.com/spencermcbridemoore/omnitool`**. Claude's recommended defaults, to be overridden in the first Claude Code message if Max disagrees: public from the first tagged commit; Apache-2.0; Python ≥ 3.11; CUDA C kernels through CuPy `RawModule` + NVRTC with Python-side templating of the RHS (numba-cuda kept as a possible later path for Python-authored RHS; confirmed by a one-day spike in Phase 0 and recorded in ADR 0004); WSL2 (Ubuntu 24.04) as the dev environment for parity with Newton; Phase 1 opens with the RydIQule → cubie baseline adapter. Still open: whether to connect a folder on the workstation so Cowork sessions can run the 4090 (≤ 50 % of GPU/CPU/memory, display must stay alive) — moot for Claude Code, which runs on the workstation directly.

## 7. Where things live

- Project docs (this folder): this handoff; `project-brief-rydiqule-backend-2026-09.md` (earlier framing, still-valid measured facts, RydIQule team/PQLS context); `agents-md-seed.md` (repo operating rules, to be revised); `cubie-deep-dive-results.md`; `gpu-ensemble-targets.md` (values-only export of the targets spreadsheet); repo seeds `stiffness_check.py`, `rkck_check.py`, `probe_bloch.py`, `ros3p_numpy.py`.
- Artifacts (account-level, persist): Solver Sharing Map, Bloch Solver Stacks, Rydiqule API Map (https://claude.ai/artifact/Qz47SqhEz33RgbozTGcTUQ).
- Session-only (lost when the Cowork session ends unless downloaded from the chat): `gpu-ensemble-targets.xlsx` with its live formulas, the steady-state prototype scripts (`fused.py`, `bench_*.py`, `precision.py`, `sparsity.py`, `blockcheck.py`), `velocity_class_check.py`, cubie probe logs.
- Memory (account-level): `/areas/rydiqule-gpu.md`, `/areas/ucn-cuda-solver.md`, `/topics/hardware.md` (workstation: Windows 11 Pro, i9-13900K, 128 GB, RTX 4090 24 GB).
