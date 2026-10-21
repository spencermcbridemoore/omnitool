# AGENTS.md — omnitool

`omnitool` is a common GPU ensemble ODE core: one solver engine that
integrates many independent members with per-member adaptive stepping, and
serves RydIQule, pylcp, MoleCool, the UCN spin tracker and similar codes
through thin adapters. `CLAUDE.md` is a symlink to this file.

Read before non-trivial work: `docs/adr/` (the decisions and why),
`docs/PHASE0_SPEC.md` (what is being built now), and
`docs/handoff/00-gpu-ensemble-core-handoff.md` (the research record).
Anything you decide goes into a new ADR under `docs/adr/`, not into chat.

## Architecture

- **A common core, not a backend for one package and never a fork.** The
  core depends on nothing physics-specific: it speaks of members, right-hand
  sides, parameters and save times.
- **Adapters are optional extras.** Each downstream package gets its own
  extra (`omnitool[rydiqule]`, …), its own adapter module and its own tests
  against a **pinned** upstream release. Upstream hooks are submitted as pull
  requests from forks.
- **Two front doors over shared machinery** (ADR 0001): affine-linear master
  equations with an analytic Jacobian, and nonlinear trajectories with
  events. Phase 0 builds the shared explicit slice; see the spec.
- **Never materialise a full parameter stack.** Assemble each member's
  system on-chip, integrate it, reduce on-chip.
- **Kernel sources stay backend-agnostic** (ADR 0004): plain C in `.cu`
  files, templated from Python, compiled by NVRTC through
  `omnitool.backends.nvrtc` with the IEEE option set and never through
  `RawModule(code=...)`, which forces `ftz`.

## Correctness rules

- **Test before you benchmark.** No speedup is measured, recorded or quoted
  until the verification layer (ADR 0003, spec §8) passes on the backend and
  precision being benchmarked.
- **The verification layer is independent of the device code.** The NumPy
  reference for every tableau is written from the paper, not from the
  kernel. The error estimator itself is tested: it must scale as `h^(p+1)`
  and must not vanish on a linear constant-coefficient problem.
- **Transcribe tableaus from the primary source** (Dormand & Prince 1980;
  Hairer, Nørsett & Wanner) as exact rationals, once, with identity tests.
- **Never copy Numerical Recipes or GPL code.** The old UCN CUDA kernel
  (Numerical-Recipes-derived, and wrong) is an architectural template only;
  PQLS is GPL-3.0; cubie is MIT but is a benchmark baseline, not a source.
- **Tolerances by precision mode** for agreement with references on
  well-conditioned problems: ≤ 1e-10 for FP64 and mixed, ≤ 1e-6 for FP32.
  Adaptive runs are judged by tolerance-scaled bounds (spec T4).
- **GPU-vs-CPU comparisons use tolerances, never bitwise equality**; FMA
  contraction alone changes the last bits. Batch-order invariance within one
  backend *is* bitwise.

## Precision rules (ADR 0002)

- FP64 is a first-class, benchmarked configuration. Every kernel, test and
  benchmark is parameterised over FP32 and FP64.
- Fast-math is off in validation builds: no `arcp`, `afn`, `ftz`, `reassoc`,
  no `--use_fast_math`. FMA contraction is allowed.
- Every result states its precision mode (FP64, FP32, or mixed with FP64
  refinement) and its FMA setting. Never downcast silently.
- Scalar type is a template parameter; no `float`/`double` literals outside
  the template header.

## Benchmark rules

- Synchronise the stream before stopping any timer.
- Exclude warm-up and compile time; report compile time separately.
- Report speedup against a same-node CPU baseline at **matched precision and
  matched achieved accuracy**, and include competitors where they apply
  (cubie for per-member adaptive stepping, PQLS for steady state,
  DiffEqGPU.jl). Fair-comparison rule with cubie: same problem, both at
  float64, fastmath off, matched achieved accuracy, compile time separate.
- Every published number has a run manifest: git commit and dirty flag,
  lock-file hash, container digest if any, `nvidia-smi` query, Slurm job ID
  where applicable, precision and FMA flags, seeds via `SeedSequence`,
  sha256 of inputs and outputs, and the compile / H2D / kernel / D2H split.
- Record the hardware: GPU model and variant (SXM or PCIe for an H100),
  driver and CUDA versions, CPU model and thread count.
- Treat speedup claims skeptically, including your own. State assumptions
  and uncertainty.

## Execution environments

- **Workstation (Windows 11, i9-13900K, 128 GB, RTX 4090 24 GB).** The
  primary dev environment is Windows-native: this clone, Git Bash, `uv`.
  The same repo also runs under WSL2 and on Linux nodes, so everything is
  LF (`.gitattributes`), paths go through `pathlib`, and shell scripts are
  POSIX. The 4090 is FP64-weak (1/64 rate): use it for FP32 development,
  correctness runs at both precisions, and the gated GPU CI job.
- **UCF ARCC Newton (H100 80 GB).** Submit Slurm batch jobs with
  `--gres=gpu:h100:1`; record SXM vs PCIe from `nvidia-smi -q`. Newton is
  for FP64 production runs and all published benchmarks. Do not run IDE
  remote servers or heavy processes on login nodes; compute nodes may lack
  internet, so prefetch dependencies.
- **CI.** The `cpu` backend runs on every push and pull request. The `cuda`
  backend runs only on the self-hosted 4090 runner through a
  `workflow_dispatch` job, never for fork pull requests. There is no CPU
  simulator of the kernels in Phase 0 (ADR 0005), so **run the `cuda` tests
  on the workstation before pushing any kernel change** and say so, with
  driver and CuPy versions, in the pull request.

## Roles in Claude Code sessions

- **Main session (Fable)** owns ADRs, kernel numerics, tableau transcription,
  the spec, and the reading of benchmark results.
- **`coder` agent** (`.claude/agents/coder.md`, Opus) implements a module
  from a spec together with its tests. It never changes a spec; if the spec
  is wrong or incomplete it stops and reports.
- **`reviewer` agent** (`.claude/agents/reviewer.md`) receives the diff and
  the spec only, never the coder's summary, and reports findings with
  `file:line`.

## Working conventions

- `src/` layout, `pyproject.toml`, `uv` lock; Python ≥ 3.11.
- `pre-commit` with ruff and pyright must pass before a commit.
- Tests are parameterised by the `backend` fixture over the implemented
  backends (`omnitool.backends.available()`); the `cuda` case skips cleanly
  without a device, and an unbuilt backend produces no xfails.
- `CHANGELOG.md` follows Keep a Changelog; every user-visible change gets a
  line under `Unreleased` in the same commit.
- Work only on the current phase. Flag scope creep instead of acting on it.
- Restarts during Phase 0 are done with branches and tags, never by deleting
  the repository: the public history matters for JOSS.

## Licensing

- `omnitool` is Apache-2.0. RydIQule is Apache-2.0; keep its attribution in
  the adapter. cubie is MIT. PQLS is GPL-3.0: do not copy its code.

## Adapter notes (for Phase 1, kept from the research phase)

- RydIQule's real, ground-removed equations of motion are exactly affine in
  every scanned parameter: `A(p) = A₀ + Σₖ pₖBₖ`, `c(p) = c₀ + Σₖ pₖdₖ`.
  Build the blocks from unit perturbations; get the physics from RydIQule,
  do not reimplement it.
- `generate_eom` is internal API (`rydiqule.sensor_utils`): pin the version
  and let the adapter tests catch drift.
- Map output axes by `axis_labels()`; stack axes follow coupling-label
  order, not insertion order. `dm_basis` labels changed in rydiqule 2.1.3:
  select basis indices by label.
- Stock RydIQule with `method="Radau"` and a block-diagonal `jac_sparsity`
  was 15× faster at better accuracy on the 561-class Doppler run; file this
  upstream with `docs/handoff/repo-seeds/stiffness_check.py`.
- Sensitivities: solve `A x′ = −(Bₖ x + dₖ)` with the existing factorisation;
  never difference two nearly equal solutions.
