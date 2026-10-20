# RydIQule GPU Backend — Project Brief

*Seeded 2026-09-26 from a Cowork research session. Measured numbers come from the prototype scripts (`fused.py`, `bench_*.py`, `precision.py`, `sparsity.py`, `blockcheck.py`) delivered in that session.*

## Goal

A GPU-accelerated solver backend for RydIQule ([QTC-UMD/rydiqule](https://github.com/QTC-UMD/rydiqule), Apache-2.0). The target is ≥10³× over RydIQule on large parameter sweeps, backed by honest benchmarks.

## Background (September 2026)

- **Who maintains RydIQule.** The code comes from the DEVCOM Army Research Laboratory (ARL) and the Naval Air Warfare Center (NAWC), China Lake.
  - Maintainer: David Meyer (@dihm).
  - Lead developer: Ben Miller, who moved from NAWC to ARL.
  - Principal investigator: Kevin Cox.
  - Development happens privately and is mirrored to GitHub. The README invites collaborators to email Meyer or Cox.
- **Release status.** The latest release is v2.1.3 (2026-04-16). The v2 paper appeared in the Journal of Open Source Software (JOSS) on 2026-02-11.
- **What v2 added.**
  - Fine/hyperfine sublevel support through the Alkali Rydberg Calculator (ARC).
  - Coupling groups.
  - Exact analytic Doppler averaging, after Nagib & Walker, *Physical Review Research* 7, 033076 (2025).
  - RydIQule runs on CPU only (NumPy/SciPy/CyRK). It has no GPU backend.
- **Direct competitor: Parallel Quantum Liouvillian Solver (PQLS).** From the University of Toronto, released 2026-09 ([arXiv:2609.12309](https://arxiv.org/abs/2609.12309)). It uses JAX, is GPL-3.0, and solves steady states only. Its benchmark on a 4-level Rb system at 10⁶ points:
  - RydIQule on a Ryzen 9 7950X: 10.7 s.
  - PQLS on the same CPU: 3.3 s.
  - PQLS on an RTX 4090: 0.18 s (about 58×).
- **General GPU Lindblad solvers.** dynamiqs, QuantumToolbox.jl, and QuTiP 5 with qutip-jax. None of these are Rydberg-specific.

## Measured facts (RydIQule 2.1.3, 4-level Rb probe/coupling/RF ladder)

- **Baseline speed.** At 250k points, RydIQule manages about 35k steady-state solves/s on one 2.8 GHz Xeon core.
  - 80–90% of that time goes to building the equations of motion (EOM).
  - Only about 9% goes to `np.linalg.solve`.
- **The EOM is exactly affine in every scanned parameter.**
  - A(p) = A₀ + Σₖ pₖBₖ and c(p) = c₀ + Σₖ pₖdₖ, reconstructed to within about 1e-16.
  - Each Bₖ has only 6–12 nonzeros out of 225 entries.
- **Fused prototype.** A Numba kernel that assembles each system from A₀ and Bₖ in-loop, then solves it with pivoted Gaussian elimination (scalar FP64):
  - 26× faster than RydIQule on one core, 53× on two cores.
  - Results match RydIQule to 1e-15.
- **FP32 accuracy.**
  - Absolute observables are accurate to about 1e-7, including far-detuned Doppler classes (condition numbers up to about 5e4).
  - It fails on weak-RF differential signals. At Ω_rf = 1e-4 Mrad/s the signal is 4.5e-8 of Im ρ₀₁, so FP32 gives 100% error.
  - One step of iterative refinement (FP32 solve, FP64 residual) brings that error down to about 1e-8.
- **Sublevel toy model (16 levels, π couplings).**
  - The EOM is 255×255, 1.8% dense, and splits into 7 independent blocks.
  - In steady state only the 63-row block containing the source term is nonzero. Solving that block alone matches the full solve to 1e-16 and needs about 66× fewer LU flops.

## Hardware

| | FP64 | FP64 tensor core | FP32 | Memory |
|---|---|---|---|---|
| RTX 4090 (local) | 1.3 TFLOPS | — | 82.6 TFLOPS | 24 GB, 1.0 TB/s |
| H100 SXM (UCF ARCC Newton) | 34 TFLOPS | 67 TFLOPS | 67 TFLOPS | 80 GB, 3.35 TB/s |
| i9-13900K (estimate) | ~1.1 TFLOPS | — | ~2.3 TFLOPS | 128 GB |

- **Newton access.** Request an H100 with `--gres=gpu:h100:1`. Record whether the node has the SXM or PCIe variant (`nvidia-smi -q`); PCIe runs about 10–25% lower.
- **Policy.** Use the 4090 for FP32 development and iteration. Use the H100 for FP64 production runs and all published benchmarks.

## Architecture

- **A backend, not a hard fork.**
  - Extract A₀, Bₖ, c₀ and dₖ from RydIQule through unit perturbations. The GPU then sees only parameters.
  - Keep RydIQule as the physics front end (Cell/ARC, coupling groups, hyperfine structure).
  - Map nonlinear inputs such as beam power (Ω ∝ √P) to their linear coefficients first.
- **Fused kernel.**
  - Assemble each system on-chip from the sparse Bₖ.
  - Solve in registers or shared memory.
  - Reduce on-chip (observables, Doppler sums).
  - Never materialize the full parameter stack.
- **Precision modes.**
  - FP64: the default on the H100.
  - FP32.
  - FP32 plus iterative refinement, with a float-float residual on the 4090.
  - For sensitivities, solve A x′ = −(Bₖ x + dₖ) with the same LU factors instead of differencing two solutions.

## Milestones and definition of done

1. **Steady state.** Systems up to about 10 levels. Passes the harness (≤1e-10 in FP64 or mixed precision, ≤1e-6 in FP32). Produces a benchmark table.
2. **Time domain.** Batched Runge–Kutta integration, since A(t) = A₀ + Σₖ fₖ(t)Bₖ. PQLS does not cover this.
3. **Doppler and sublevels.** Treat velocity as another affine term, summed on-chip. Add block decomposition for sublevel systems.

**Stop point:** a JOSS paper after milestone 2.

## Verification and benchmark rules

- **Build the harness first.** Run randomized equivalence tests against RydIQule:
  - random level counts, couplings, decays and parameters;
  - far-detuned Doppler classes;
  - weak-RF differential signals;
  - tolerances set per precision mode.
- **Benchmark rules.**
  - Synchronize CUDA before stopping any timer.
  - Exclude warm-up and compilation time.
  - Report speedup against RydIQule **and** against an optimized CPU path on the same machine.
  - Label the precision and hardware for every number, and include PQLS.
- **Known pitfalls.**
  - RydIQule orders its stack axes by coupling label, so never reshape without mapping axes by label.
  - `dm_basis` labels were only corrected in v2.1.3.
  - JAX defaults to FP32.
  - Fast-math compiler flags reorder sums.

## Licensing and engagement

- RydIQule is Apache-2.0, so keep its attribution.
- PQLS is GPL-3.0, so do not copy its code.
- Approach the ARL team with a working backend and honest benchmarks, not a proposal.
