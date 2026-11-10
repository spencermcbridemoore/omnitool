# Benchmarks

`bench_dopri5.py` is the first Phase 0 benchmark (spec §9). Each run solves
problem L (two-level Bloch, `Delta` cycled over `{0, 2, 20, 200}` across
members, save grid `linspace(0, 10, 51)`) at one precision for
`n_members in {1024, 16384, 262144}` on the `cuda` backend and, up to a cap,
on the same machine's `cpu` backend, and writes a run manifest.

```bash
uv run --no-sync python benchmarks/bench_dopri5.py --precision fp64
```

```bash
uv run --no-sync python benchmarks/bench_dopri5.py --precision fp32
```

Every run passes the T4 correctness gate before any timer starts: every
member must finish `SUCCESS`, and a seeded sample of members (all of them
up to 4096) must satisfy `max |y - y_ref| <= 10 * (rtol * ymax + atol)`
against the closed-form reference. No manifest is written if the gate fails.
Manifests live under `results/<date>-<precision>/manifest.json`; raw
outputs under `raw/` next to them are git-ignored and are identified by the
sha256 in the manifest.

## Results: RTX 4090, 2026-11-10

Workstation: Windows 11, i9-13900K (32 threads), RTX 4090 24 GB, driver
610.88 (CUDA 13.3), CuPy 14.2.0, NVRTC 13.1, `sm_89`. NVRTC options
`-ftz=false -prec-div=true -prec-sqrt=true -fmad=true` (no fast-math, FMA
contraction allowed). Commit `aebaea3`, clean tree. Manifests:
[`results/2026-11-10-fp64/manifest.json`](results/2026-11-10-fp64/manifest.json)
and
[`results/2026-11-10-fp32/manifest.json`](results/2026-11-10-fp32/manifest.json).

Tolerances are the first T4 pair of each precision: `(rtol, atol) =
(1e-7, 1e-9)` at FP64 and `(1e-4, 1e-6)` at FP32. Times are medians of five
timed launches after one untimed warm-up; the stream is synchronised before
every timer stops. Compile time is excluded from every launch and reported
separately. The CPU baseline is the `cpu` backend (the NumPy reference, one
member at a time) over the same batch at the same precision and tolerances,
a single timed run.

| members | precision | kernel (ms) | CPU baseline (s) | CPU / kernel | achieved accuracy (max err, bound) |
|--------:|:---------:|------------:|-----------------:|-------------:|:-----------------------------------|
|    1024 | FP64      |       78.3  |           108.3  |     1380     | 1.33e-7 (13 % of 1.01e-6)          |
|   16384 | FP64      |       79.6  |      not run     |      n/a     | 1.33e-7 (13 % of 1.01e-6)          |
|  262144 | FP64      |     1249    |      not run     |      n/a     | 1.33e-7 (13 % of 1.01e-6)          |
|    1024 | FP32      |        1.06 |            31.5  |    29600     | 1.73e-4 (17 % of 1.01e-3)          |
|   16384 | FP32      |        3.92 |      not run     |      n/a     | 1.73e-4 (17 % of 1.01e-3)          |
|  262144 | FP32      |       27.0  |      not run     |      n/a     | 1.73e-4 (17 % of 1.01e-3)          |

Compile time (first NVRTC compile of the process, one kernel per precision):
0.115 s at FP64, 0.097 s at FP32.

The full split per launch (medians, ms):

| members | precision | H2D | kernel | D2H | host transpose | total |
|--------:|:---------:|----:|-------:|----:|---------------:|------:|
|    1024 | FP64      | 0.37 |   78.3 |  0.55 |   0.42 |   79.7 |
|   16384 | FP64      | 0.68 |   79.6 |  3.45 |  14.6  |   98.1 |
|  262144 | FP64      | 5.19 | 1249   | 52.0  | 249    | 1560   |
|    1024 | FP32      | 0.19 |   1.06 |  0.16 |   0.09 |    1.50 |
|   16384 | FP32      | 0.74 |   3.92 |  3.45 |  10.9  |   19.0 |
|  262144 | FP32      | 6.08 |  27.0  | 30.8  | 169    |  230   |

H2D is device allocation, fills and uploads; D2H is the device-to-host copies;
"host transpose" is the NumPy transpose from the member-fastest device layout
to the `(n_members, n_save, n_state)` API layout that `solve_batch_cuda`
performs on the host. The CPU / total ratios (CPU wall over the whole launch
including H2D, D2H and the transpose) are 1360 at FP64 and 21000 at FP32 for
1024 members.

## Caveats

- **These are not production numbers.** The 4090 is the development GPU; its
  FP64 rate is 1/64 of FP32, and FP32 results from it are development
  checks, not results to publish. Published benchmarks come from the H100
  on Newton (AGENTS.md).
- **The CPU baseline is a serial Python loop, not a compiled solver.** The
  `cpu` backend calls the NumPy reference once per member, so its time is
  dominated by Python overhead per step. The CPU / kernel ratio is therefore
  an upper bound on the speedup against real CPU code; a comparison against
  a compiled DOPRI5 (SciPy's Fortran `dopri5`, or cubie) at matched accuracy
  is still to be done and will be far smaller. The baseline was capped at
  1024 members (about 108 s at FP64) because 16384 would take about half an
  hour and 262144 about eight hours.
- **FP32 and FP64 rows are not comparable to each other.** The FP32
  tolerances are looser, so the `Delta = 200` member takes 1754 accepted
  steps at FP32 against 6712 at FP64; the FP32 kernel does roughly 3.8 times
  less work per member before any rate difference.
- **Kernel time is set by the longest member below about 16384 members.**
  At FP64 the kernel takes 78 ms for 1024 members and 80 ms for 16384: one
  thread per member, and every warp waits for its `Delta = 200` members'
  6712 serial steps (0.75 rejections per member on average). Above that the
  device is throughput-bound and time grows linearly (1249 ms at 262144).
- **Host work dominates the end-to-end FP32 time at the largest size.** At
  262144 members the host transpose (169 ms) and D2H (31 ms) outweigh the
  FP32 kernel (27 ms). The kernel column is the right number for the
  device; the total column is what a caller of `solve_batch` sees today.
- **Watchdog.** The 1.25 s FP64 launch at 262144 members ran under the
  Windows display-driver timeout (2 s) on this display-attached GPU. A
  larger FP64 batch, or a tighter tolerance, would exceed it here; that is a
  property of this workstation, not of the kernel.
- **Accuracy.** The max error column is over the sampled members, all
  states and all 51 save times, against the closed-form reference. The
  `cpu` baseline achieves the same error to five digits (its own gate is in
  the manifest). GPU and CPU outputs differ in the last bits (FMA), never
  bitwise.
- **Variance.** The FP32 kernel at 262144 members ranged from 22.9 ms to
  36.6 ms over the five launches (median 27.0 ms); all other kernel medians
  sit within a few percent of their minimum. Single-run CPU times carry the
  usual desktop noise.

The `spike/` directory holds the ADR 0004 throwaway spike and is not a
benchmark.
