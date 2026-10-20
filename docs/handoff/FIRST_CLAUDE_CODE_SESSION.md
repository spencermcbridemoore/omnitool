# First Claude Code session — brief to paste

Before starting: create an empty repo folder on the workstation, unzip this
handoff bundle into `docs/handoff/` inside it, and fill in the decisions block
below. Then open Claude Code in that folder (Fable as the main model) and paste
everything from "Context" down.

## Decisions (filled 2026-10-19; items marked "recommended" are Claude's
## defaults — override any of them in the first message if you disagree)

- Package / repo name: `omnitool` (Max's choice). No released distribution
  under that name on PyPI as of 2026-10-19 (`pip index versions omnitool`
  finds nothing); a reserved-but-empty PyPI project could not be ruled out
  from the research session, so confirm at https://pypi.org/project/omnitool/
  before the first release.
- GitHub: `github.com/spencermcbridemoore/omnitool` (Max's choice).
- Visibility: public from the first tagged commit (recommended — JOSS wants
  six months of public history, and restarts are done with branches and tags,
  never by deleting the repo).
- License: Apache-2.0 (recommended — matches RydIQule's licence, carries a
  patent grant; MIT is the acceptable alternative).
- Python floor: >= 3.11.
- GPU backend for Phase 0: CUDA C kernels compiled through CuPy
  `RawModule` + NVRTC, with the problem-specific RHS inserted by Python-side
  templating; CuPy owns device arrays and streams (recommended — full control
  of registers, warp intrinsics and FP64 semantics, a stable toolchain, and no
  dependence on numba-cuda's maintenance mode or numba-cuda-mlir's maturity).
  Keep the kernel sources backend-agnostic enough that a numba-cuda path for
  Python-authored RHS can be added later. Record the choice in ADR 0004 and
  confirm it with a one-day spike in Phase 0 (compile, launch, FP32/FP64,
  `__ballot_sync`, timing) before writing the real kernel.
- Dev environment: WSL2 (Ubuntu 24.04) on the workstation, repo on the WSL
  filesystem, CUDA via the WSL driver (recommended — same wheels, paths and
  shell scripts as Newton's Linux nodes; Cursor and Claude Code both attach
  to WSL). Windows-native is the fallback if WSL2 GPU access misbehaves.
- Phase 1 opens with the RydIQule -> cubie baseline adapter: yes
  (recommended — it yields the 561-class per-member result in days and sets
  the bar the core must beat).

## Context

Read `docs/handoff/00-gpu-ensemble-core-handoff.md` first, then
`docs/handoff/agents-md-seed.md` and `docs/handoff/cubie-deep-dive-results.md`.
The handoff is the design record of a finished research phase; the repo is now
the source of truth, so anything you decide goes into `docs/adr/` as a short
architecture decision record (ADR), not into chat.

## Session goals (in order; stop after each and summarise)

1. Write `docs/adr/0001-scope-and-front-doors.md`, `0002-precision-policy.md`,
   `0003-verification-layer.md`, `0004-backend-choice.md` from the handoff and
   the decisions above. Each ADR: context, decision, consequences, under 300 words.
2. Write `docs/PHASE0_SPEC.md`: the vertical slice (Dormand-Prince 5(4),
   thread-per-member, per-member adaptive controller with PI step control,
   dense output at shared save times, FP32 and FP64, status codes) with its
   acceptance tests spelled out as concrete assertions: convergence slope 5
   on a smooth test problem, agreement with SciPy DOP853 at 1e-12 to within
   tolerance-scaled bounds at both precisions, error-estimator scaling h^5 and
   non-vanishing on a linear test problem, batch-order invariance, nfev caps,
   and a CPU reference implementation of the same tableau in plain NumPy.
3. Rewrite `docs/handoff/agents-md-seed.md` as the repo's `AGENTS.md`
   (symlink `CLAUDE.md` to it) for the common-core framing: adapters as
   optional extras, precision rules, benchmark rules, the Newton/4090 split,
   "never copy Numerical Recipes or GPL code", "test before you benchmark".
4. Define `.claude/agents/coder.md` (model: opus; implements a module from a
   spec and its tests, never changes a spec) and `.claude/agents/reviewer.md`
   (model: opus or fable; receives the diff and the spec only, never the
   coder's summary; reports findings with file:line). The main session (Fable)
   owns ADRs, kernel numerics, tableau transcription and benchmark reading.
5. Only then scaffold: `src/` layout, `pyproject.toml` with uv lock,
   pre-commit (ruff + pyright), pytest with a `backend` fixture parameterised
   over `cpu` / `cudasim` / `cuda`, GitHub Actions for CPU tests on push and a
   `workflow_dispatch`-gated job for the self-hosted 4090 runner (never fork
   PRs), Keep-a-Changelog, CITATION.cff. Commit as the first tagged state.

Do not write kernel code in this session. The first kernel is the next
session, after the spec and the CPU reference exist.
