# omnitool

A common GPU ensemble ODE core: one solver engine that integrates many
independent members with **per-member adaptive stepping**, dense output at
shared save times, **FP32 and FP64 as first-class configurations**, and a
verification layer that is independent of the device code. Physics packages
(RydIQule, pylcp, MoleCool, a UCN spin tracker, …) attach through thin,
optional adapters.

**Status: pre-alpha, Phase 0.** The repository holds the design record and
the scaffold; the first kernel (Dormand–Prince 5(4), thread-per-member) is
specified in [`docs/PHASE0_SPEC.md`](docs/PHASE0_SPEC.md) and is the next
piece of work. Nothing here solves an ODE yet.

## Where to read

- [`docs/adr/`](docs/adr/) — architecture decision records: scope and front
  doors, precision policy, verification layer, backend choice.
- [`docs/PHASE0_SPEC.md`](docs/PHASE0_SPEC.md) — the vertical slice and its
  acceptance tests as concrete assertions.
- [`AGENTS.md`](AGENTS.md) — operating rules for humans and agents working
  in this repository.
- [`docs/handoff/`](docs/handoff/) — the research-phase handoff, including
  the audit of cubie and the measured RydIQule facts.

## Development setup

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/). Works
Windows-native (Git Bash), under WSL2 and on Linux.

```bash
uv sync --group dev
uv run pre-commit install
uv run pytest
```

Tests are parameterised over three backends: `cpu` (plain-NumPy reference),
`cudasim` (the kernel template compiled for the host) and `cuda` (CuPy +
NVRTC on a device). The `cuda` case skips without a GPU; restrict with
`--backend cpu --backend cudasim`. To run on a GPU:

```bash
uv sync --group dev --extra cuda12   # or cuda13, matching your driver
uv run pytest --backend cuda
```

GPU CI runs only on a self-hosted RTX 4090 runner through a manually
dispatched workflow, never for fork pull requests.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff).
