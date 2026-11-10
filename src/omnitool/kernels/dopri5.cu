/* Dormand-Prince 5(4), one thread per member (docs/PHASE0_SPEC.md sections 3-4).
 *
 * Plain C apart from the CUDA qualifiers. This file is a template body: the
 * Python templater (omnitool.kernels.render_dopri5) puts a generated prelude
 * in front of it that supplies
 *
 *   typedef <float|double> real_t;
 *   #define N_STATE   <number of state components>
 *   #define N_PARAMS  <number of parameters per member, may be 0>
 *   #define REAL_EPS  <machine epsilon of real_t>
 *   #define REAL(x)   ((real_t)(x))   a floating constant in real_t
 *   #define DOPRI5_ORDER 5
 *   OMNITOOL_STATUS_*            (src/omnitool/status.h, verbatim)
 *   DOPRI5_N_STAGES, DOPRI5_C, DOPRI5_A, DOPRI5_B, DOPRI5_B_HAT, DOPRI5_E,
 *   DOPRI5_D                     (rendered from omnitool/tableaus/dopri5.py)
 *   static __device__ void rhs(real_t t, const real_t* y, real_t* dydt,
 *                              const real_t* p) { <problem body> }
 *
 * To compile it by hand, write that prelude to a file and pass it to nvcc
 * with -include. It is built as C++ (NVRTC has no C mode), so sqrt, pow and
 * fabs resolve to the real_t overloads; nothing here promotes to double in
 * an FP32 build.
 *
 * Written from Hairer, Norsett & Wanner, "Solving ODEs I", and Hairer's
 * dopri5.f. The control flow mirrors omnitool/reference/dopri5_numpy.py
 * decision for decision so that the two can be compared step for step, but
 * shares no code with it: only the tableau is common.
 *
 * Device arrays are member-fastest: element (i, m) of an [n_i, n_members]
 * array is at i * n_members + m, so neighbouring threads touch neighbouring
 * addresses.
 */

/* ---- Template header: the only floating literals in this file -----------
 * Every one is written REAL(...), as in right-hand-side bodies. */

#define R_ZERO REAL(0)
#define R_ONE  REAL(1)

/* PI controller, Gustafsson form (dopri5.f defaults; spec section 3). */
#define CTRL_BETA          REAL(0.04)
#define CTRL_ALPHA         REAL(0.2 - 0.75 * 0.04)
#define CTRL_SAFE          REAL(0.9)
#define CTRL_FAC_MIN       REAL(0.1)
#define CTRL_FAC_MAX       REAL(5.0)
#define CTRL_ERR_OLD_FLOOR REAL(1e-4)

/* |h| < STEP_FLOOR_EPS * eps * max(|t|, 1) stops with STEP_TOO_SMALL. */
#define STEP_FLOOR_EPS REAL(16.0)
/* dopri5.f end rule: the step becomes t_end - t once t + 1.01 h passes t_end. */
#define STEP_STRETCH   REAL(1.01)

/* hinit (HNW I, section II.4, eq. 4.14). */
#define HINIT_TINY_NORM  REAL(1e-10)
#define HINIT_TINY_STEP  REAL(1e-6)
#define HINIT_FRACTION   REAL(0.01)
#define HINIT_TINY_DER   REAL(1e-15)
#define HINIT_SHRINK     REAL(1e-3)
#define HINIT_GROWTH     REAL(100.0)

/* ---- End of template header --------------------------------------------- */

/* How the driver chooses its step sizes. Defined once, here; the Python
 * layer reads these lines (omnitool.kernels.step_modes). */
#define STEP_MODE_AUTO  0 /* hinit, then the controller */
#define STEP_MODE_H0    1 /* given first step, then the controller */
#define STEP_MODE_FIXED 2 /* every step has the given size; nothing is rejected */

/* A zero-length array is not C; a problem without parameters gets one slot. */
#define P_STORAGE (N_PARAMS > 0 ? N_PARAMS : 1)

#define IDX2(i, m, n_members) ((long long)(i) * (n_members) + (m))

/* min and max that return the first argument unless the second is strictly
 * beyond it, so a NaN in the first argument survives (fmin/fmax drop it). */
static __device__ real_t r_min(real_t a, real_t b) { return (b < a) ? b : a; }
static __device__ real_t r_max(real_t a, real_t b) { return (b > a) ? b : a; }

static __device__ int all_finite(const real_t* y)
{
    int ok = 1;
    for (int s = 0; s < N_STATE; ++s) {
        ok = ok && isfinite(y[s]);
    }
    return ok;
}

/* One step from (t, y) of size h. k[0] holds f(t, y) on entry; stages 2..7
 * are written to k[1..6], with k[6] = f(t + h, y_new) (FSAL). err_vec is the
 * unscaled estimate h (b - b_hat).k. Six right-hand-side evaluations. */
static __device__ void step(real_t t, const real_t* y, real_t h, const real_t* p,
                            real_t k[DOPRI5_N_STAGES][N_STATE], real_t* y_new, real_t* err_vec)
{
    real_t y_i[N_STATE];
    for (int i = 1; i < DOPRI5_N_STAGES; ++i) {
        for (int s = 0; s < N_STATE; ++s) {
            real_t acc = R_ZERO;
            for (int j = 0; j < i; ++j) {
                acc += DOPRI5_A[i][j] * k[j][s];
            }
            y_i[s] = y[s] + h * acc;
        }
        if (i == DOPRI5_N_STAGES - 1) {
            /* a[6][:] == b: the stage-7 argument is the new solution */
            for (int s = 0; s < N_STATE; ++s) {
                y_new[s] = y_i[s];
            }
        }
        rhs(t + DOPRI5_C[i] * h, y_i, k[i], p);
    }
    for (int s = 0; s < N_STATE; ++s) {
        real_t acc = R_ZERO;
        for (int i = 0; i < DOPRI5_N_STAGES; ++i) {
            acc += DOPRI5_E[i] * k[i][s];
        }
        err_vec[s] = h * acc;
    }
}

/* err = sqrt(mean_i((err_vec_i / sc_i)^2)), sc_i = atol + rtol max(|y_i|, |y_new_i|). */
static __device__ real_t error_norm(const real_t* err_vec, const real_t* y, const real_t* y_new,
                                    real_t rtol, real_t atol)
{
    real_t sum = R_ZERO;
    for (int s = 0; s < N_STATE; ++s) {
        const real_t sc = atol + rtol * r_max(fabs(y[s]), fabs(y_new[s]));
        const real_t q = err_vec[s] / sc;
        sum += q * q;
    }
    return sqrt(sum / (real_t)N_STATE);
}

/* Step-size controller. Returns the next step size and updates the
 * controller state (*err_old, *rejected_last).
 *
 * accepted: fac = clamp(err^alpha / err_old^beta / safe, FAC_MIN, FAC_MAX),
 *   h_new = h / fac; no growth on the first accept after a reject (the REJECT
 *   rule of dopri5.f); err_old = max(err, floor).
 * rejected: h_new = h / min(FAC_MAX, err^alpha / safe); err_old unchanged. */
static __device__ real_t controller(real_t h, real_t err, int accepted, real_t* err_old,
                                    int* rejected_last)
{
    real_t h_new;
    if (accepted) {
        real_t fac = pow(err, CTRL_ALPHA) / pow(*err_old, CTRL_BETA) / CTRL_SAFE;
        fac = r_min(r_max(fac, CTRL_FAC_MIN), CTRL_FAC_MAX);
        h_new = h / fac;
        if (*rejected_last) {
            h_new = r_min(h_new, h);
            *rejected_last = 0;
        }
        *err_old = r_max(err, CTRL_ERR_OLD_FLOOR);
    } else {
        *rejected_last = 1;
        h_new = h / r_min(CTRL_FAC_MAX, pow(err, CTRL_ALPHA) / CTRL_SAFE);
    }
    return h_new;
}

/* Hairer's rcont vectors r3, r4, r5 for the accepted step y -> y_new. */
static __device__ void dense_coefficients(const real_t* y, const real_t* y_new, real_t h,
                                          real_t k[DOPRI5_N_STAGES][N_STATE], real_t* r3,
                                          real_t* r4, real_t* r5)
{
    for (int s = 0; s < N_STATE; ++s) {
        const real_t dy = y_new[s] - y[s];
        real_t acc = R_ZERO;
        for (int i = 0; i < DOPRI5_N_STAGES; ++i) {
            acc += DOPRI5_D[i] * k[i][s];
        }
        r3[s] = h * k[0][s] - dy;
        r4[s] = dy - h * k[DOPRI5_N_STAGES - 1][s] - r3[s];
        r5[s] = h * acc;
    }
}

/* Hairer's fourth-order continuous extension at theta in [0, 1], written to
 * out[s * stride] for s < N_STATE. theta == 0 and theta == 1 return y and
 * y_new themselves, so the endpoints are bitwise those of the step. */
static __device__ void dense(real_t theta, const real_t* y, const real_t* y_new,
                             const real_t* r3, const real_t* r4, const real_t* r5, real_t* out,
                             long long stride)
{
    if (theta == R_ZERO) {
        for (int s = 0; s < N_STATE; ++s) {
            out[s * stride] = y[s];
        }
    } else if (theta == R_ONE) {
        for (int s = 0; s < N_STATE; ++s) {
            out[s * stride] = y_new[s];
        }
    } else {
        const real_t th1 = R_ONE - theta;
        for (int s = 0; s < N_STATE; ++s) {
            const real_t dy = y_new[s] - y[s];
            out[s * stride] =
                y[s] + theta * (dy + th1 * (r3[s] + theta * (r4[s] + th1 * r5[s])));
        }
    }
}

/* Hairer's initial step guess from y0 and f0 = f(t0, y0); one more
 * right-hand-side evaluation. */
static __device__ real_t hinit(real_t t0, const real_t* y0, const real_t* f0, const real_t* p,
                               real_t t_end, real_t rtol, real_t atol)
{
    const real_t hmax = fabs(t_end - t0);
    real_t sk[N_STATE];
    real_t y1[N_STATE];
    real_t f1[N_STATE];
    real_t dnf = R_ZERO;
    real_t dny = R_ZERO;
    for (int s = 0; s < N_STATE; ++s) {
        sk[s] = atol + rtol * fabs(y0[s]);
        const real_t qf = f0[s] / sk[s];
        const real_t qy = y0[s] / sk[s];
        dnf += qf * qf;
        dny += qy * qy;
    }
    real_t h;
    if (dnf <= HINIT_TINY_NORM || dny <= HINIT_TINY_NORM) {
        h = HINIT_TINY_STEP;
    } else {
        h = sqrt(dny / dnf) * HINIT_FRACTION;
    }
    h = r_min(h, hmax);
    for (int s = 0; s < N_STATE; ++s) {
        y1[s] = y0[s] + h * f0[s];
    }
    rhs(t0 + h, y1, f1, p);
    real_t d2 = R_ZERO;
    for (int s = 0; s < N_STATE; ++s) {
        const real_t q = (f1[s] - f0[s]) / sk[s];
        d2 += q * q;
    }
    const real_t der2 = sqrt(d2) / h;
    const real_t der12 = r_max(fabs(der2), sqrt(dnf));
    real_t h1;
    if (der12 <= HINIT_TINY_DER) {
        h1 = r_max(HINIT_TINY_STEP, fabs(h) * HINIT_SHRINK);
    } else {
        h1 = pow(HINIT_FRACTION / der12, R_ONE / (real_t)DOPRI5_ORDER);
    }
    return r_min(r_min(HINIT_GROWTH * fabs(h), h1), hmax);
}

static __device__ void load_member(int m, int n_members, const real_t* y_d, const real_t* params_d,
                                   real_t* y, real_t* p)
{
    for (int s = 0; s < N_STATE; ++s) {
        y[s] = y_d[IDX2(s, m, n_members)];
    }
    for (int j = 0; j < N_PARAMS; ++j) {
        p[j] = params_d[IDX2(j, m, n_members)];
    }
}

/* The driver: integrates member m from t0 to t_end and writes the solution
 * at the shared save times.
 *
 *   y0_d     [N_STATE, n_members]           initial states
 *   params_d [N_PARAMS, n_members]          parameters
 *   t_save   [n_save]                       strictly increasing, t_save[0] == t0,
 *                                           t_save[n_save - 1] == t_end
 *   out_d    [n_save, N_STATE, n_members]   solution; the host fills it with NaN
 *                                           and the kernel writes only the save
 *                                           times it reaches
 *   status_d, n_accepted_d, n_rejected_d, n_fev_d, h_last_d   [n_members]
 */
extern "C" __global__ void dopri5_solve(int n_members, int n_save, real_t t0, real_t t_end,
                                        const real_t* t_save, real_t rtol, real_t atol,
                                        int max_steps, int step_mode, real_t h_given,
                                        const real_t* y0_d, const real_t* params_d, real_t* out_d,
                                        int* status_d, int* n_accepted_d, int* n_rejected_d,
                                        int* n_fev_d, real_t* h_last_d)
{
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_members) {
        return;
    }
    const long long stride = (long long)n_members;
    const int fixed = (step_mode == STEP_MODE_FIXED);

    real_t y[N_STATE];
    real_t y_new[N_STATE];
    real_t err_vec[N_STATE];
    real_t r3[N_STATE];
    real_t r4[N_STATE];
    real_t r5[N_STATE];
    real_t k[DOPRI5_N_STAGES][N_STATE];
    real_t p[P_STORAGE];
    load_member(m, n_members, y0_d, params_d, y, p);

    real_t t = t0;
    real_t h = R_ZERO;
    real_t h_last = R_ZERO;
    real_t err_old = CTRL_ERR_OLD_FLOOR;
    int rejected_last = 0;
    int n_accepted = 0;
    int n_rejected = 0;
    int n_fev = 0;
    int status = OMNITOOL_STATUS_NOT_RUN;
    int save_idx = 0;

    while (save_idx < n_save && t_save[save_idx] <= t) {
        for (int s = 0; s < N_STATE; ++s) {
            out_d[IDX2((long long)save_idx * N_STATE + s, m, n_members)] = y[s];
        }
        ++save_idx;
    }

    if (!all_finite(y)) {
        status = OMNITOOL_STATUS_NON_FINITE;
    } else {
        rhs(t, y, k[0], p);
        n_fev += 1;
        if (step_mode == STEP_MODE_AUTO) {
            h = hinit(t, y, k[0], p, t_end, rtol, atol);
            n_fev += 1;
        } else {
            h = h_given;
        }
    }

    while (status == OMNITOOL_STATUS_NOT_RUN) {
        if (t >= t_end) {
            status = OMNITOOL_STATUS_SUCCESS;
            break;
        }
        if (n_accepted + n_rejected >= max_steps) {
            status = OMNITOOL_STATUS_MAX_STEPS;
            break;
        }
        if (fabs(h) < STEP_FLOOR_EPS * REAL_EPS * r_max(fabs(t), R_ONE)) {
            status = OMNITOOL_STATUS_STEP_TOO_SMALL;
            break;
        }
        int last = 0;
        if (t + STEP_STRETCH * h >= t_end) {
            h = t_end - t;
            last = 1;
        }

        step(t, y, h, p, k, y_new, err_vec);
        n_fev += 6;
        const real_t err = fixed ? R_ZERO : error_norm(err_vec, y, y_new, rtol, atol);
        if (!isfinite(err) || !all_finite(y_new)) {
            n_rejected += 1;
            status = OMNITOOL_STATUS_NON_FINITE;
            break;
        }

        if (fixed || err <= R_ONE) {
            n_accepted += 1;
            const real_t t_new = last ? t_end : t + h;
            if (save_idx < n_save && t_save[save_idx] <= t_new) {
                dense_coefficients(y, y_new, h, k, r3, r4, r5);
                while (save_idx < n_save && t_save[save_idx] <= t_new) {
                    const real_t ts = t_save[save_idx];
                    /* a save time on the step end gets y_new itself */
                    const real_t theta = (ts == t_new) ? R_ONE : (ts - t) / h;
                    dense(theta, y, y_new, r3, r4, r5,
                          out_d + IDX2((long long)save_idx * N_STATE, m, n_members), stride);
                    ++save_idx;
                }
            }
            h_last = h;
            const real_t h_new = fixed ? h_given : controller(h, err, 1, &err_old, &rejected_last);
            t = t_new;
            for (int s = 0; s < N_STATE; ++s) {
                y[s] = y_new[s];
                k[0][s] = k[DOPRI5_N_STAGES - 1][s];
            }
            h = h_new;
        } else {
            n_rejected += 1;
            h = controller(h, err, 0, &err_old, &rejected_last);
        }
    }

    status_d[m] = status;
    n_accepted_d[m] = n_accepted;
    n_rejected_d[m] = n_rejected;
    n_fev_d[m] = n_fev;
    h_last_d[m] = h_last;
}

/* Test entry point for step() (spec T3): one step of size h from (t, y) for
 * every member, with k1 = f(t, y) evaluated here.
 *
 *   y_d, y_new_d, err_d   [N_STATE, n_members]
 *   params_d              [N_PARAMS, n_members]
 */
extern "C" __global__ void dopri5_step_once(int n_members, real_t t, real_t h, const real_t* y_d,
                                            const real_t* params_d, real_t* y_new_d,
                                            real_t* err_d)
{
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_members) {
        return;
    }
    real_t y[N_STATE];
    real_t y_new[N_STATE];
    real_t err_vec[N_STATE];
    real_t k[DOPRI5_N_STAGES][N_STATE];
    real_t p[P_STORAGE];
    load_member(m, n_members, y_d, params_d, y, p);

    rhs(t, y, k[0], p);
    step(t, y, h, p, k, y_new, err_vec);
    for (int s = 0; s < N_STATE; ++s) {
        y_new_d[IDX2(s, m, n_members)] = y_new[s];
        err_d[IDX2(s, m, n_members)] = err_vec[s];
    }
}

/* Test entry point for dense() (spec T5): one step of size h from (t, y),
 * then the continuous extension at each of the n_theta values.
 *
 *   theta_d    [n_theta]
 *   y_new_d    [N_STATE, n_members]
 *   dense_d    [n_theta, N_STATE, n_members]
 */
extern "C" __global__ void dopri5_dense_once(int n_members, int n_theta, real_t t, real_t h,
                                             const real_t* theta_d, const real_t* y_d,
                                             const real_t* params_d, real_t* y_new_d,
                                             real_t* dense_d)
{
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_members) {
        return;
    }
    real_t y[N_STATE];
    real_t y_new[N_STATE];
    real_t err_vec[N_STATE];
    real_t r3[N_STATE];
    real_t r4[N_STATE];
    real_t r5[N_STATE];
    real_t k[DOPRI5_N_STAGES][N_STATE];
    real_t p[P_STORAGE];
    load_member(m, n_members, y_d, params_d, y, p);

    rhs(t, y, k[0], p);
    step(t, y, h, p, k, y_new, err_vec);
    dense_coefficients(y, y_new, h, k, r3, r4, r5);
    for (int s = 0; s < N_STATE; ++s) {
        y_new_d[IDX2(s, m, n_members)] = y_new[s];
    }
    for (int i = 0; i < n_theta; ++i) {
        dense(theta_d[i], y, y_new, r3, r4, r5,
              dense_d + IDX2((long long)i * N_STATE, m, n_members), (long long)n_members);
    }
}
