# cubie v0.16.0 deep dive — probe results

Clone: `github.com/cubiepy/cubie` at `6f1dcf4` (tag v0.16.0, 2026-10-07).
Environment: Python 3.11, `pip install -e ".[test]"` plus
`cubie-numba-cuda-mlir[cu13]==0.5.3.1`, run under `NUMBA_ENABLE_CUDASIM=1`
(cubie's vendored CPU simulator) in the cloud container. No GPU was used, so
no timing below is a GPU timing; step counts, status codes and accuracy are
what the device code computes (the simulator executes the same Python
device functions one thread at a time).

Note for a real-GPU rerun on the 4090: drop `NUMBA_ENABLE_CUDASIM=1`,
install `cubie[cuda13]` in a fresh venv, and keep the batch small
(the scripts use 2–4 runs). `cubie` writes generated kernels under
`CUBIE_CACHE_DIR` (the scripts set `./generated`).

## 1. Per-run adaptive stepping, float64 — `probe_bloch.py`

Two-level optical Bloch equations (u, v, w), Gamma = Omega = 1, swept
detuning Delta = 0, 2, 20, 200; duration 20, `save_every` 2;
`rtol 1e-7, atol 1e-9`, `precision=float64`. Error is the max absolute
difference against SciPy DOP853 at rtol 1e-13 on the save grid.
Counters are cubie's own per-save `iteration_counters`
(newton, krylov/linear solves, steps, rejected steps).

| method  | Delta | steps  | rejected | newton | linear | max abs err |
|---------|------:|-------:|---------:|-------:|-------:|------------:|
| rk45    |   0   |     64 |    0     |   0    |   0    | 2.7e-08 |
| rk45    |   2   |    158 |   12     |   0    |   0    | 3.0e-08 |
| rk45    |  20   |   1361 |  134     |   0    |   0    | 4.2e-08 |
| rk45    | 200   |   8588 |   34     |   0    |   0    | 2.4e-07 |
| radau   |   0   |    116 |    0     |  125   |  241   | 8.8e-10 |
| radau   |   2   |    302 |    4     |  314   |  616   | 1.1e-09 |
| radau   |  20   |   2621 |   62     | 2635   | 5256   | 1.1e-09 |
| radau   | 200   |  16628 |    1     | 16638  | 33266  | 1.1e-08 |
| rodas3p |   0   |    388 |    0     |   0    | 1940   | 4.5e-08 |
| rodas3p |   2   |   1182 |    2     |   0    | 5910   | 4.4e-08 |
| rodas3p |  20   |  10095 |   39     |   0    | 50475  | 6.1e-08 |
| rodas3p | 200   |  56447 |    2     |   0    | 282235 | 2.8e-07 |
| ros3p   |  any  |     24 |    0     |   0    |   72   | **1.6e-02 … 8.4e-02** |

Each run takes its own number of steps (64 vs 8588 in one batch), status
codes are all 0, and float64 works through the whole stack (symbolic
Jacobian, sparse LU, Radau Newton iteration). Radau IIA5 does not help on
this problem because the coherences oscillate at amplitude Omega/Delta
above tolerance for the whole run (unlike the far-detuned RydIQule
classes), so it has to resolve the oscillation like RK45 does.

## 2. ROS3P (cubie's default Rosenbrock tableau) — `probe_ros3p_variants.py`, `ros3p_numpy.py`

`method="ros3p"` (which is what `method="rosenbrock"` resolves to) ran every
run at `dt_max` with zero rejections and returned errors of 1e-2 … 1e-1
against a requested rtol of 1e-7, with status 0. Changing the controller
(gustafsson → pi) or turning the smoothed error off changes nothing; a
fixed step of 0.01 gives 5e-8, so the stepping itself is right and only
the embedded error estimate is dead.

`ros3p_numpy.py` is an independent 30-line implementation of ROS3P in the
Lang & Verwer (2001) transformed form with the paper's Table 5.1
coefficients (identical to cubie's). Its estimate `(m − m̂)·u` is 1e-16
on a linear constant-coefficient system for every step size while the
true local error is 1e-3 … 1e-6. The algebra: for `f(y) = Jy + g`,
`u2 = (1 − 1/γ) u1` exactly, and
`(m1 − m̂1) + (m2 − m̂2)(1 − 1/γ) = −0.11325 + 0.11325 = 0`, while
`m3 = m̂3`. So the embedded method has the same stability function as the
main method (the paper says "a second-order embedding with the same
stability properties") and its error estimate vanishes identically on
every linear problem; on nonlinear problems it only sees the nonlinear
part (Lotka–Volterra: ROS3P adapts but lands at 6.8e-4 vs 1.0e-5 for
RODAS3P at the same tolerance).

This is a property of the published pair, faithfully implemented, but it
makes ROS3P a trap as the *default* on exactly the problem class you care
about (affine-linear master equations), and cubie's test suite cannot see
it because its CPU reference uses the same tableau. RODAS3P, ROS34PW2 and
Rosenbrock23 estimators behave normally.

| variant (Delta 0 / 5, duration 10)       | steps      | max abs err      |
|------------------------------------------|-----------:|-----------------:|
| ros3p, gustafsson (default)              | 14 / 14    | 1.6e-2 / 1.3e-1  |
| ros3p, pi controller                     | 14 / 14    | 1.6e-2 / 1.3e-1  |
| ros3p, use_smoothed_error=False          | 14 / 14    | 1.6e-2 / 1.3e-1  |
| ros3p, fixed dt=0.01                     | 1004       | 5.4e-8 / 7.8e-6  |
| rosenbrock23                             | 368 / 2552 | 6.5e-6 / 5.1e-6  |
| ros2                                     | 13029 / 95760 | 1.2e-7 / 1.1e-7 |
| ros34pw2                                 | 364 / 2532 | 6.4e-8 / 5.9e-8  |
| rodas3p, use_smoothed_error=False        | 355 / 2476 | 4.5e-8 / 4.1e-8  |

## 3. Symbolic codegen scaling on affine-linear systems — `probe_scaling.py`

`dx/dt = A x + c k` with Liouvillian-like sparsity (about 4·N^1.5
nonzeros), one stiff direction, `rodas3p`, float64, 2 runs.

| N states | nnz(J) | generated source | first solve, cold | first solve, warm cache | max abs err |
|---------:|-------:|-----------------:|------------------:|------------------------:|------------:|
| 15       | 225    | 0.25 MB          | 10 s (203 steps simulated) | –              | 2.9e-8 |
| 63       | 1981   | 4.4 MB           | 68 s (42 steps simulated)  | 34 s           | 7.7e-9 |

Codegen alone at N = 63 is roughly 34 s (cold minus warm); the rest is
the simulator stepping. The sparse LU is emitted as fully unrolled
straight-line code per thread (`codegen/lu_solver.py`), so the source
grows about as N^2.1 here; extrapolated to a 16-level system (255 real
states) that is of order 100 MB of generated device source before
fill-in, with `lu_factor` living in per-thread local memory. Not a path
for large Rydberg manifolds; fine for 3–8 levels.

Gotcha found on the way: cubie orders states lexicographically
(`x0, x1, x10, x11, …`) even when you pass them as an ordered list; the
scripts map outputs back by name.

## 4. Test suite under the simulator

`pytest tests/integrators/step_control tests/integrators/algorithms/test_rosenbrock_tableaus.py tests/integrators/algorithms/test_tableau_properties.py -m "not nocudasim and not specific_algos"`
→ 189 passed in 22 s (4 xdist workers).
