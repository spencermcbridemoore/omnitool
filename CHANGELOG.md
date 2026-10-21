# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-21

The Phase 0 design record, the DOPRI5 NumPy reference and the full CPU
side of the verification layer; no device kernel yet.

### Added

- `omnitool.backends.nvrtc`: the single definition of the NVRTC option set
  (`-ftz=false -prec-div=true -prec-sqrt=true -fmad=true`, no fast-math),
  compiling through the NVRTC binding and loading cubins with
  `RawModule(path=...)`. `tests/test_nvrtc_flags.py` makes the spike's FP32
  subnormal probe a permanent device test; ADR 0004 records the option set.
- `src/omnitool/status.h`, the device-side status codes, with a test that it
  matches the `Status` enum.
- Tests T4 (tolerance-scaled agreement), T6 (batch-order invariance, now two
  single-`Problem` batches), T8 (status codes and validation) and T9
  (cross-backend agreement; skipped until a device backend is registered).
- ADR 0004 spike (`benchmarks/spike/nvrtc_spike.py`) passed on the RTX 4090;
  result recorded in the ADR. Finding: `cupy.RawModule(code=...)` forces
  `-ftz=true`, so the `cuda` backend will compile through NVRTC directly and
  load the cubin with `RawModule(path=...)`.
- Dormand–Prince 5(4) tableau and Hairer's dense-output coefficients as exact
  rationals (`omnitool.tableaus.dopri5`) with the T1 identity tests, including
  rooted-tree order conditions through order 5 and the continuous order
  conditions of the dense output.
- Plain-NumPy DOPRI5 reference (`omnitool.reference.dopri5_numpy`): PI
  controller, `hinit`, step clipping, dense output and status codes; the
  `cpu` backend of `omnitool.solve_batch`, with `Problem`, `BatchResult` and
  `Status`.
- Test problems L, S1, S2 (`tests/problems.py`) and the `cpu` rows of
  T2, T3, T5 and T7.
- ADR 0005: no host-compiled `cudasim` backend in Phase 0. The backend
  registry `omnitool.backends.available()` lists implemented backends only,
  and the pytest `backend` fixture parameterises over it, so an unbuilt
  backend produces no xfails. `omnitool.backends.KNOWN` is `("cpu", "cuda")`.

### Changed

- Controller: on the first accepted step after a rejection the step may not
  grow (`h_new = min(h_new, h)`, the `REJECT` rule of `dopri5.f`); spec §3
  and the reference follow it, and every T7 cap still holds at both
  precisions.
- Spec T4: the tighter FP32 tolerance pair is `(2e-5, 1e-7)`; the previous
  `rtol = 1e-5` sat below the spec's own `100 * eps32` floor.
- Spec T2, FP32 variant, is now a precision-consistency test: FP32 and FP64
  endpoints agree to `100 * eps32 * max(1, ymax)` at every `N`, and a slope
  is fitted only when three or more points sit above the `100 * eps32`
  floor. FP64 keeps the order-5 assertion. This supersedes the interim
  `N in {20, 40, 80}` form.
- CI runs the `cpu` backend only; `cudasim` is gone from the workflow, the
  spec's backend table, `AGENTS.md` and the README (ADR 0005).

## [0.0.0] - 2026-10-19

The first tagged state: design record and scaffold, no solver yet.

### Added

- Research-phase handoff bundle under `docs/handoff/`.
- Architecture decision records 0001–0004 (scope and front doors, precision
  policy, verification layer, Phase 0 backend choice).
- `docs/PHASE0_SPEC.md`: the Dormand–Prince 5(4) vertical slice with its
  acceptance tests as concrete assertions.
- `AGENTS.md` (with `CLAUDE.md` symlinked to it) and the `coder` and
  `reviewer` agent definitions.
- Repository scaffold: `src/` layout, `pyproject.toml` with `uv` lock,
  pre-commit (ruff, pyright), pytest `backend` fixture over `cpu` /
  `cudasim` / `cuda`, CPU CI on push, `workflow_dispatch`-gated GPU job for
  the self-hosted RTX 4090 runner, `CITATION.cff`.

[Unreleased]: https://github.com/spencermcbridemoore/omnitool/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/spencermcbridemoore/omnitool/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/spencermcbridemoore/omnitool/releases/tag/v0.0.0
