# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/spencermcbridemoore/omnitool/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/spencermcbridemoore/omnitool/releases/tag/v0.0.0
