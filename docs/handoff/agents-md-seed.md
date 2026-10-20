# AGENTS.md — rydiqule-gpu

This repository is a GPU-accelerated solver backend for RydIQule ([QTC-UMD/rydiqule](https://github.com/QTC-UMD/rydiqule)). Read `PROJECT_BRIEF.md` before non-trivial work; it holds the background, measured baselines, architecture, and milestones.

## Architecture

- **This is a backend, not a fork.** Depend on a pinned `rydiqule` release from PyPI.
- **Get the physics from RydIQule; don't reimplement it.** RydIQule's real, ground-removed equations of motion are exactly affine in every scanned parameter: A(p) = A₀ + Σₖ pₖBₖ and c(p) = c₀ + Σₖ pₖdₖ. Build A₀, Bₖ, c₀ and dₖ from unit perturbations.
- **`generate_eom` is internal API** (`rydiqule.sensor_utils`). Pin the version and let the test harness catch drift. A public hook belongs upstream, submitted as a pull request from the fork.
- **Never materialize the full parameter stack.** Assemble each system on-chip, solve it, and reduce on-chip (observables, Doppler sums).

## Correctness rules

- **Test before you benchmark.** Every kernel change must pass the randomized equivalence harness against RydIQule before anything is benchmarked or reported. Tolerances by precision mode:
  - ≤1e-10 for FP64 and for mixed precision
  - ≤1e-6 for FP32
- **The harness must cover:**
  - random level counts, couplings, decays and parameters
  - far-detuned Doppler velocity classes
  - weak-RF differential signals
- **Map RydIQule output axes by `axis_labels()`; never reshape blindly.** Stack axes follow coupling-label order, not insertion order.
- **`dm_basis` labels changed in rydiqule 2.1.3.** The fix corrected a sign on the imaginary terms, so select basis indices by label and pin the version.

## Precision rules

- **Every result and benchmark states its precision mode:** FP64, FP32, or mixed (FP32 solve with FP64 or float-float refinement).
- **Never downcast silently.**
  - Set `jax_enable_x64` if you use JAX.
  - Don't enable fast-math in validation builds.
- **Sensitivities:** compute derivatives with the existing factorization, solving A x′ = −(Bₖ x + dₖ). Don't difference two nearly equal solutions.

## Benchmark rules

- Synchronize the GPU before stopping any timer.
- Exclude warm-up and JIT or compile time, and report compile time separately.
- Report speedup against **both** RydIQule and the optimized CPU path on the same machine, and include PQLS where it applies.
- Record the hardware for every number:
  - GPU model and variant (for an H100, whether it is SXM or PCIe)
  - driver and CUDA versions
  - CPU model and thread count
  - rydiqule version

## Execution environments

- **Local:** RTX 4090. It is strong in FP32 and weak in FP64 (1:64), so use it for FP32 development and iteration.
- **UCF ARCC Newton (H100-80GB):**
  - Submit SLURM batch jobs with `--gres=gpu:h100:1`. Use Newton for FP64 production runs and published benchmarks.
  - Don't run IDE remote servers or heavy processes on login nodes; per-user thread caps apply.
  - Compute nodes may lack internet, so prefetch dependencies first.

## Licensing and scope

- RydIQule is Apache-2.0: keep its attribution.
- Don't copy code from GPL-licensed projects such as PQLS.
- Work only on the current milestone. Flag scope creep instead of acting on it.
- Treat speedup claims skeptically, including your own. State assumptions and uncertainty.
