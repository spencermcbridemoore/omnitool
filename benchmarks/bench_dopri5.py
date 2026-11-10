"""First Phase 0 benchmark: DOPRI5 on problem L at one precision (spec §9).

Solves the two-level Bloch problem L of ``tests/problems.py`` with ``Delta``
cycled over ``{0, 2, 20, 200}`` across members, for ``n_members`` in
``{1024, 16384, 262144}``, on the ``cuda`` backend and on the same machine's
``cpu`` backend, and writes ``manifest.json`` under
``benchmarks/results/<date>-<precision>/``.

Rules followed here (``AGENTS.md``, ADR 0002, spec §9):

- **Gate before timing.** For every batch that is timed, every member must
  finish with ``Status.SUCCESS`` and a seeded sample of members must meet
  the T4 bound ``10 * (rtol * ymax + atol)`` against the closed-form
  reference. If the gate fails the script exits before writing a manifest.
- **Compile time is separate.** The first NVRTC compile of the process is
  timed on its own and reported as ``compile_s``; the warm-up launch is not
  timed.
- **The stream is synchronised before every timer stops.** The split
  recorded per launch is H2D (device allocation, fills and uploads), kernel,
  D2H (device-to-host copies); the host transpose to the API layout is
  recorded separately so the kernel number is not diluted.
- **Same launch path as the backend.** The timed launch uses the backend's
  own module cache, upload helper and launch configuration; on the smallest
  batch its output is checked bitwise against ``solve_batch(backend="cuda")``
  and the result is recorded in the manifest.
- Raw outputs go to ``raw/`` next to the manifest and are ignored by git;
  the manifest carries their sha256.

Run (one precision per invocation)::

    uv run --no-sync python benchmarks/bench_dopri5.py --precision fp64
    uv run --no-sync python benchmarks/bench_dopri5.py --precision fp32
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import omnitool
from omnitool import BatchResult, Status, kernels
from omnitool.api import PRECISIONS
from omnitool.backends import cuda as cuda_backend
from omnitool.backends import nvrtc

REPO = Path(__file__).resolve().parents[1]
# Spec §7 defines the test problems once, in tests/problems.py; the benchmark
# reuses them from there rather than duplicating the right-hand sides.
sys.path.insert(0, str(REPO / "tests"))
from problems import BLOCH, L_BY_DELTA, T_END, reference_on_grid, save_grid  # noqa: E402

DELTA_CYCLE: tuple[float, ...] = (0.0, 2.0, 20.0, 200.0)
SIZES_DEFAULT: tuple[int, ...] = (1024, 16384, 262144)
TOLERANCES: dict[str, tuple[float, float]] = {"fp64": (1e-7, 1e-9), "fp32": (1e-4, 1e-6)}
"""The first T4 pair of each precision."""
BOUND_FACTOR = 10.0
"""T4: ``err <= BOUND_FACTOR * (rtol * ymax + atol)``."""
MAX_STEPS = 100_000
"""``solve_batch``'s default; passed explicitly so the manifest can state it."""
GATE_SAMPLE_MAX = 4096
"""Members per batch whose trajectories are checked against the reference."""


# --------------------------------------------------------------------------
# Batch construction and the correctness gate
# --------------------------------------------------------------------------


def build_batch(n_members: int, dtype: type[np.floating]) -> tuple[np.ndarray, np.ndarray]:
    """``(y0, params)`` for ``n_members`` copies of L with ``Delta`` cycled."""
    case0 = L_BY_DELTA[DELTA_CYCLE[0]]
    y0 = np.tile(case0.y0, (n_members, 1)).astype(dtype)
    params = np.tile(case0.params, (n_members, 1))
    params[:, 2] = np.asarray(DELTA_CYCLE)[np.arange(n_members) % len(DELTA_CYCLE)]
    return y0, params.astype(dtype)


def reference_grid(t_save: np.ndarray) -> np.ndarray:
    """``(len(DELTA_CYCLE), n_save, n_state)`` closed-form solution per ``Delta``."""
    return np.stack([reference_on_grid(L_BY_DELTA[d], t_save) for d in DELTA_CYCLE])


def gate(
    res: BatchResult,
    y_ref: np.ndarray,
    rtol: float,
    atol: float,
    rng: np.random.Generator,
    label: str,
) -> dict[str, Any]:
    """Assert T4 on a seeded sample of members; return what the manifest records."""
    n_members = res.y.shape[0]
    bad = np.flatnonzero(res.status != int(Status.SUCCESS))
    if bad.size:
        codes = sorted({int(Status(int(c))) for c in res.status[bad]})
        names = [Status(c).name for c in codes]
        sys.exit(f"GATE FAILED ({label}): {bad.size} of {n_members} members not SUCCESS: {names}")
    n_sample = min(n_members, GATE_SAMPLE_MAX)
    sample = np.sort(rng.choice(n_members, size=n_sample, replace=False))
    delta_index = sample % len(DELTA_CYCLE)
    ymax = np.max(np.abs(y_ref), axis=(1, 2))  # per Delta, as T4 takes it per case
    bound = BOUND_FACTOR * (rtol * ymax + atol)  # (len(DELTA_CYCLE),)
    err = np.max(np.abs(res.y[sample].astype(np.float64) - y_ref[delta_index]), axis=(1, 2))
    ratio = err / bound[delta_index]
    worst = int(np.argmax(ratio))
    if ratio[worst] > 1.0:
        delta = DELTA_CYCLE[delta_index[worst]]
        sys.exit(
            f"GATE FAILED ({label}): member {sample[worst]} (Delta={delta:g})"
            f" error {err[worst]:.3e} exceeds bound {bound[delta_index[worst]]:.3e}"
        )
    return {
        "rule": "max |y - y_ref| <= 10 * (rtol * ymax + atol), ymax per Delta over the save grid",
        "n_sampled": int(n_sample),
        "n_members": int(n_members),
        "max_error": float(err[worst]),
        "bound": float(bound[delta_index[worst]]),
        "worst_fraction_of_bound": float(ratio[worst]),
        "worst_member": int(sample[worst]),
        "worst_delta": float(DELTA_CYCLE[delta_index[worst]]),
        "max_error_by_delta": {
            f"{d:g}": float(np.max(err[delta_index == i], initial=0.0))
            for i, d in enumerate(DELTA_CYCLE)
        },
    }


def step_stats(res: BatchResult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("n_accepted", "n_rejected", "n_fev"):
        a = getattr(res, name)
        out[name] = {"min": int(a.min()), "max": int(a.max()), "mean": float(a.mean())}
    delta_index = np.arange(res.y.shape[0]) % len(DELTA_CYCLE)
    out["n_accepted_by_delta"] = {
        f"{d:g}": float(res.n_accepted[delta_index == i].mean()) for i, d in enumerate(DELTA_CYCLE)
    }
    return out


# --------------------------------------------------------------------------
# Hashing and provenance
# --------------------------------------------------------------------------


def sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def inputs_sha256(
    y0: np.ndarray, params: np.ndarray, t_save: np.ndarray, rtol: float, atol: float
) -> str:
    h = hashlib.sha256()
    for a in (y0, params, t_save):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((float(rtol), float(atol), MAX_STEPS)).encode())
    return h.hexdigest()


def outputs_sha256(res: BatchResult) -> str:
    return sha256_arrays(res.y, res.status, res.n_accepted, res.n_rejected, res.n_fev, res.h_last)


def _run(cmd: list[str], **kw: Any) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw).stdout.strip()


def git_state(exclude: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=REPO)
    tracked_dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO).returncode != 0
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO).splitlines()
    if exclude.is_relative_to(REPO):
        rel = exclude.relative_to(REPO).as_posix()
        untracked = [u for u in untracked if not u.startswith(rel)]
    return {
        "commit": commit,
        "dirty": tracked_dirty,
        "dirty_rule": "tracked files differ from HEAD; untracked files are listed instead",
        "untracked_files": untracked,
    }


def cpu_model() -> str:
    try:
        if sys.platform == "win32":
            return _run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
            )
        if sys.platform == "darwin":
            return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # provenance only; fall through to platform
        pass
    return platform.processor() or "unknown"


def nvidia_smi() -> dict[str, str]:
    fields = ("name", "driver_version", "memory.total")
    out = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"])
    values = [v.strip() for v in out.split(",")]
    return dict(zip(fields, values, strict=True))


def environment() -> dict[str, Any]:
    import cupy as cp  # type: ignore[import-not-found]
    import scipy
    from cupy_backends.cuda.libs import nvrtc as nvrtc_lib  # type: ignore[import-not-found]

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "nvrtc": ".".join(str(v) for v in nvrtc_lib.getVersion()),
        "nvidia_smi": nvidia_smi(),
        "gpu_arch": f"sm_{nvrtc.device_arch()}",
        "gpu_multiprocessors": int(props["multiProcessorCount"]),
        "cpu_model": cpu_model(),
        "cpu_threads": os.cpu_count(),
    }


# --------------------------------------------------------------------------
# Timed GPU launch (the backend's launch path, with timers between phases)
# --------------------------------------------------------------------------


class GpuRunner:
    """Launch ``dopri5_solve`` the way ``solve_batch_cuda`` does, timing each phase."""

    def __init__(self, dtype: type[np.floating]) -> None:
        import cupy as cp  # type: ignore[import-not-found]

        self.cp = cp
        self.dtype = dtype
        self.stream = cp.cuda.get_current_stream()
        t0 = time.perf_counter()
        module = cuda_backend._module(BLOCH, dtype)  # first call of the process: compiles
        self.stream.synchronize()
        self.compile_s = time.perf_counter() - t0
        self.kernel = module.get_function("dopri5_solve")
        self.step_mode = kernels.step_modes()["AUTO"]

    def run(
        self,
        y0: np.ndarray,
        params: np.ndarray,
        t_save: np.ndarray,
        rtol: float,
        atol: float,
    ) -> tuple[BatchResult, dict[str, float]]:
        cp, dtype, stream = self.cp, self.dtype, self.stream
        n_members, n_save, n_state = y0.shape[0], t_save.shape[0], BLOCH.n_state

        t0 = time.perf_counter()
        y0_d = cuda_backend._to_device(y0, dtype)
        params_d = cuda_backend._to_device(params, dtype)
        t_save_d = cp.asarray(np.ascontiguousarray(t_save, dtype=dtype))
        out_d = cp.full((n_save, n_state, n_members), cp.nan, dtype=dtype)
        status_d = cp.full(n_members, int(Status.NOT_RUN), dtype=np.int32)
        n_accepted_d = cp.zeros(n_members, dtype=np.int32)
        n_rejected_d = cp.zeros(n_members, dtype=np.int32)
        n_fev_d = cp.zeros(n_members, dtype=np.int32)
        h_last_d = cp.zeros(n_members, dtype=dtype)
        stream.synchronize()
        t1 = time.perf_counter()

        self.kernel(
            cuda_backend._grid(n_members),
            (cuda_backend.THREADS_PER_BLOCK,),
            (
                np.int32(n_members),
                np.int32(n_save),
                dtype(0.0),
                dtype(T_END),
                t_save_d,
                dtype(rtol),
                dtype(atol),
                np.int32(MAX_STEPS),
                np.int32(self.step_mode),
                dtype(0.0),
                y0_d,
                params_d,
                out_d,
                status_d,
                n_accepted_d,
                n_rejected_d,
                n_fev_d,
                h_last_d,
            ),
        )
        stream.synchronize()
        t2 = time.perf_counter()

        out_h = cp.asnumpy(out_d)
        status = cp.asnumpy(status_d)
        n_accepted = cp.asnumpy(n_accepted_d)
        n_rejected = cp.asnumpy(n_rejected_d)
        n_fev = cp.asnumpy(n_fev_d)
        h_last = cp.asnumpy(h_last_d)
        stream.synchronize()
        t3 = time.perf_counter()

        y = np.ascontiguousarray(out_h.transpose(2, 0, 1))
        t4 = time.perf_counter()

        res = BatchResult(y, status, n_accepted, n_rejected, n_fev, h_last)
        split = {
            "h2d_s": t1 - t0,
            "kernel_s": t2 - t1,
            "d2h_s": t3 - t2,
            "host_transpose_s": t4 - t3,
            "total_s": t4 - t0,
        }
        return res, split


def results_equal(a: BatchResult, b: BatchResult) -> bool:
    return all(
        np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
        for f in ("y", "status", "n_accepted", "n_rejected", "n_fev", "h_last")
    )


def solve_cpu(y0: np.ndarray, params: np.ndarray, t_save: np.ndarray, rtol: float, atol: float,
              precision: str) -> BatchResult:  # fmt: skip
    return omnitool.solve_batch(
        BLOCH,
        y0,
        params,
        t_span=(0.0, T_END),
        t_save=t_save,
        rtol=rtol,
        atol=atol,
        precision=precision,
        backend="cpu",
        max_steps=MAX_STEPS,
    )


def summarise(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def save_raw(path: Path, res: BatchResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        y=res.y,
        status=res.status,
        n_accepted=res.n_accepted,
        n_rejected=res.n_rejected,
        n_fev=res.n_fev,
        h_last=res.h_last,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--precision", choices=tuple(PRECISIONS), required=True)
    p.add_argument(
        "--sizes",
        type=lambda s: tuple(int(v) for v in s.split(",")),
        default=SIZES_DEFAULT,
        help="comma-separated n_members (default: %(default)s)",
    )
    p.add_argument("--repeats", type=int, default=5, help="timed GPU launches per size")
    p.add_argument(
        "--cpu-max-members",
        type=int,
        default=1024,
        help="run the cpu baseline only for sizes up to this (the cpu backend is a serial loop)",
    )
    p.add_argument("--cpu-repeats", type=int, default=1, help="timed cpu runs per size")
    p.add_argument("--seed", type=int, default=None, help="SeedSequence entropy (default: fresh)")
    p.add_argument("--out", type=Path, default=None, help="results directory (default: by date)")
    p.add_argument("--force", action="store_true", help="overwrite an existing results directory")
    p.add_argument("--no-raw", action="store_true", help="do not save raw outputs under raw/")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.repeats < 1 or args.cpu_repeats < 1:
        sys.exit("--repeats and --cpu-repeats must be at least 1")
    precision: str = args.precision
    dtype = PRECISIONS[precision]
    rtol, atol = TOLERANCES[precision]
    seed_seq = np.random.SeedSequence(args.seed)
    rng = np.random.default_rng(seed_seq)

    today = dt.datetime.now(dt.UTC).date().isoformat()
    out_dir: Path = args.out or REPO / "benchmarks" / "results" / f"{today}-{precision}"
    if out_dir.exists() and not args.force:
        sys.exit(f"{out_dir} exists; pass --force to overwrite or --out for another directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"omnitool {omnitool.__version__}  precision={precision}  rtol={rtol}  atol={atol}")
    env = environment()
    print(f"GPU: {env['nvidia_smi']}  CuPy {env['cupy']}  NVRTC {env['nvrtc']}")
    print(f"CPU: {env['cpu_model']} ({env['cpu_threads']} threads)")

    t_save = save_grid(T_END)
    y_ref = reference_grid(t_save)

    gpu = GpuRunner(dtype)
    print(f"compile: {gpu.compile_s:.3f} s (NVRTC {env['nvrtc']}, {env['gpu_arch']})")

    manifest: dict[str, Any] = {
        "schema": 1,
        "script": "benchmarks/bench_dopri5.py",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "omnitool_version": omnitool.__version__,
        "git": git_state(out_dir),
        "uv_lock_sha256": hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest(),
        "environment": env,
        "problem": {
            "name": "L: two-level optical Bloch (tests/problems.py, spec section 7)",
            "n_state": BLOCH.n_state,
            "n_params": BLOCH.n_params,
            "y0": [0.0, 0.0, -1.0],
            "gamma": 1.0,
            "omega": 1.0,
            "delta_cycle": list(DELTA_CYCLE),
            "delta_rule": "member m gets delta_cycle[m % 4]",
            "t_span": [0.0, T_END],
            "t_save": f"linspace(0, {T_END:g}, {t_save.shape[0]})",
            "reference": "closed form via scipy.linalg.expm (augmented exponential)",
        },
        "precision": precision,
        "dtype": np.dtype(dtype).name,
        "rtol": rtol,
        "atol": atol,
        "max_steps": MAX_STEPS,
        "step_mode": "AUTO (hinit)",
        "nvrtc_options": [*nvrtc.NVRTC_OPTIONS, f"-arch={env['gpu_arch']}"],
        "fma": "contraction allowed (-fmad=true)",
        "fast_math": False,
        "seed": {
            "entropy": int(seed_seq.entropy),  # type: ignore[arg-type]
            "use": "selects the members checked by the correctness gate; inputs are deterministic",
        },
        "timing": {
            "clock": "time.perf_counter; the CUDA stream is synchronised before every stop",
            "warmup": "one untimed launch per size before the timed repeats",
            "compile": "excluded from every launch; the first compile of the process is compile_s",
            "h2d_s": "device allocation, NaN/zero fills and host-to-device uploads",
            "kernel_s": "dopri5_solve launch to stream synchronisation",
            "d2h_s": "device-to-host copies",
            "host_transpose_s": "host transpose to the (n_members, n_save, n_state) API layout",
        },
        "compile_s": gpu.compile_s,
        "launch_path_check": None,
        "cpu_baseline": {
            "backend": "cpu (omnitool.backends.cpu: the NumPy reference, one member at a time)",
            "max_members": args.cpu_max_members,
            "note": "sizes above max_members are not run on the cpu backend; see README",
        },
        "batches": [],
    }

    for n_members in args.sizes:
        print(f"\n== n_members = {n_members}")
        y0, params = build_batch(n_members, dtype)
        in_hash = inputs_sha256(y0, params, t_save.astype(dtype), rtol, atol)
        batch: dict[str, Any] = {"n_members": n_members, "inputs_sha256": in_hash}

        # Warm-up (untimed): first launch at this size, memory-pool growth included.
        res, _ = gpu.run(y0, params, t_save, rtol, atol)
        batch["gpu_gate"] = gate(res, y_ref, rtol, atol, rng, f"cuda n={n_members}")
        print(
            f"  gate cuda: max err {batch['gpu_gate']['max_error']:.3e}"
            f" = {100 * batch['gpu_gate']['worst_fraction_of_bound']:.1f}% of bound"
        )
        if manifest["launch_path_check"] is None:
            via_api = omnitool.solve_batch(
                BLOCH, y0, params, t_span=(0.0, T_END), t_save=t_save, rtol=rtol, atol=atol,
                precision=precision, backend="cuda", max_steps=MAX_STEPS,
            )  # fmt: skip
            same = results_equal(res, via_api)
            manifest["launch_path_check"] = {
                "n_members": n_members,
                "bitwise_equal_to_solve_batch": same,
            }
            if not same:
                sys.exit("benchmark launch path differs from solve_batch(backend='cuda')")
        warm_hash = outputs_sha256(res)
        del res

        splits: list[dict[str, float]] = []
        res = None
        for _ in range(args.repeats):
            res, split = gpu.run(y0, params, t_save, rtol, atol)
            if outputs_sha256(res) != warm_hash:
                sys.exit(f"cuda outputs changed between launches at n={n_members}")
            splits.append(split)
        assert res is not None
        batch["gpu"] = {
            "repeats": args.repeats,
            "runs": splits,
            "summary": {k: summarise([s[k] for s in splits]) for k in splits[0]},
            "outputs_sha256": warm_hash,
            "steps": step_stats(res),
        }
        k = batch["gpu"]["summary"]["kernel_s"]
        print(f"  cuda kernel: median {k['median'] * 1e3:.2f} ms  (min {k['min'] * 1e3:.2f} ms)")
        if not args.no_raw:
            save_raw(out_dir / "raw" / f"cuda-n{n_members}.npz", res)
        del res

        if n_members <= args.cpu_max_members:
            walls: list[float] = []
            res_cpu = None
            for _ in range(args.cpu_repeats):
                t0 = time.perf_counter()
                res_cpu = solve_cpu(y0, params, t_save, rtol, atol, precision)
                walls.append(time.perf_counter() - t0)
            assert res_cpu is not None
            batch["cpu_gate"] = gate(res_cpu, y_ref, rtol, atol, rng, f"cpu n={n_members}")
            batch["cpu"] = {
                "repeats": args.cpu_repeats,
                "wall_s": walls,
                "summary": summarise(walls),
                "outputs_sha256": outputs_sha256(res_cpu),
                "steps": step_stats(res_cpu),
            }
            batch["ratio_cpu_wall_over_gpu_kernel"] = (
                batch["cpu"]["summary"]["median"] / k["median"]
            )
            batch["ratio_cpu_wall_over_gpu_total"] = (
                batch["cpu"]["summary"]["median"] / batch["gpu"]["summary"]["total_s"]["median"]
            )
            print(
                f"  cpu wall: {batch['cpu']['summary']['median']:.2f} s"
                f"  ratio (cpu / cuda kernel): {batch['ratio_cpu_wall_over_gpu_kernel']:.0f}"
            )
            if not args.no_raw:
                save_raw(out_dir / "raw" / f"cpu-n{n_members}.npz", res_cpu)
        else:
            batch["cpu"] = None
            print(f"  cpu baseline skipped (n > {args.cpu_max_members})")
        manifest["batches"].append(batch)

    manifest_path = out_dir / "manifest.json"
    text = json.dumps(manifest, indent=2) + "\n"
    manifest_path.write_text(text, encoding="utf-8", newline="\n")  # the repository is LF
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
