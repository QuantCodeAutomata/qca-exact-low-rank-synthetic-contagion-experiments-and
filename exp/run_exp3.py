"""
run_exp3.py — Experiment 3: Graphon smoothing and structure-preserving
kernel truncation experiments.

Components
----------
1. Piecewise-smooth indicator experiment (block-exponential kernel, eps-ramp).
2. Directed 30-mode truncation experiment (low-rank directed kernel).
3. Trigonometric family experiment (Fourier-mode truncation of smooth part).
4. Step-halving temporal validation (dt=0.002 vs dt=0.001).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphon import (
    midpoint_grid,
    kernel_block_exponential,
    kernel_directed_modes,
    kernel_trigonometric,
    compute_directed_modes,
    simulate_graphon,
    initial_profile_piecewise,
    initial_profile_smooth,
)
from model import loss_hard, loss_positive_ramp
from metrics import (
    threshold_fraction_path,
    sup_vector_error_inf,
    sup_scalar_error,
    loglog_slope,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Shared experiment parameters
N = 2000
DT = 2e-3


def _make_ramp(eps: float):
    """Return a zero-argument-eps closure over loss_positive_ramp."""
    return lambda x: loss_positive_ramp(x, eps)


# ---------------------------------------------------------------------------
# Component 1 — Piecewise-smooth indicator experiment
# ---------------------------------------------------------------------------

def run_component1(n: int = N, dt: float = DT):
    """Block-exponential kernel, hard vs positive-ramp loss for eps in [0.04, 0.02, 0.01].

    Returns
    -------
    times      : (n_steps+1,) time grid
    results    : dict keyed by 'hard' and eps values; each value is a sub-dict
                 with keys terminal_frac, state_err, frac_err
    all_states : dict keyed by 'hard' and eps values; (n_steps+1, n) arrays
    all_fracs  : dict keyed by 'hard' and eps values; (n_steps+1,) arrays
    """
    B = np.array([[0.38, 0.24], [0.31, 0.36]])
    r = 0.0
    mu = -0.06
    T = 2.0
    eps_vals = [0.04, 0.02, 0.01]

    u = midpoint_grid(n)
    x0 = initial_profile_piecewise(u)
    W = kernel_block_exponential(u, u, B)

    # Hard-threshold baseline
    times, states_hard = simulate_graphon(x0, W, mu, r, T, dt, loss_hard)
    frac_hard = threshold_fraction_path(states_hard)

    results = {"hard": {"terminal_frac": float(frac_hard[-1])}}
    all_states = {"hard": states_hard}
    all_fracs = {"hard": frac_hard}

    for eps in eps_vals:
        loss_fn = _make_ramp(eps)
        _, states_eps = simulate_graphon(x0, W, mu, r, T, dt, loss_fn)
        frac_eps = threshold_fraction_path(states_eps)

        results[eps] = {
            "terminal_frac": float(frac_eps[-1]),
            "state_err": sup_vector_error_inf(states_eps, states_hard),
            "frac_err": sup_scalar_error(frac_eps, frac_hard),
        }
        all_states[eps] = states_eps
        all_fracs[eps] = frac_eps

    return times, results, all_states, all_fracs


# ---------------------------------------------------------------------------
# Component 2 — Directed 30-mode truncation experiment
# ---------------------------------------------------------------------------

def run_component2(n: int = N, dt: float = DT):
    """30-mode directed kernel; simulate reference, then K-truncations.

    Returns
    -------
    times      : (n_steps+1,) time grid
    results    : dict keyed by K; sub-dict with state_err, frac_err
    states_ref : (n_steps+1, n) reference states (30-mode)
    frac_ref   : (n_steps+1,) reference threshold-fraction path
    """
    mu = -0.0006
    r = 0.0
    T = 2.0
    eps = 0.02
    M_ref = 30
    K_vals = [1, 2, 4, 6, 8, 12, 16, 24]

    u = midpoint_grid(n)
    x0 = initial_profile_smooth(u)
    loss_fn = _make_ramp(eps)

    # Reference kernel (all 30 modes)
    sigma_m, a_modes, b_modes = compute_directed_modes(u, M_ref)
    W_ref = kernel_directed_modes(u, u, sigma_m, a_modes, b_modes, M_ref)
    times, states_ref = simulate_graphon(x0, W_ref, mu, r, T, dt, loss_fn)
    frac_ref = threshold_fraction_path(states_ref)

    results = {}
    for K in K_vals:
        W_K = kernel_directed_modes(u, u, sigma_m, a_modes, b_modes, K)
        _, states_K = simulate_graphon(x0, W_K, mu, r, T, dt, loss_fn)
        frac_K = threshold_fraction_path(states_K)
        results[K] = {
            "state_err": sup_vector_error_inf(states_K, states_ref),
            "frac_err": sup_scalar_error(frac_K, frac_ref),
        }

    return times, results, states_ref, frac_ref


# ---------------------------------------------------------------------------
# Component 3 — Trigonometric family experiment
# ---------------------------------------------------------------------------

def run_component3(n: int = N, dt: float = DT):
    """Block matrix + M-mode trigonometric smooth part; benchmark at M=16.

    The combined kernel is W(u,v) = B[i,j] + W_trig^{(M)}(u,v).
    The M=16 result is the benchmark; errors for M in {1,2,4,8} are computed
    against it.

    Returns
    -------
    times      : (n_steps+1,) time grid
    results    : dict keyed by M (1,2,4,8); sub-dict with state_err, frac_err
    all_states : dict keyed by M (1,2,4,8,16)
    all_fracs  : dict keyed by M (1,2,4,8,16)
    """
    B = np.array([[0.38, 0.24], [0.31, 0.36]])
    r = 0.0
    mu = -0.06
    T = 2.0
    eps = 0.02
    M_vals = [1, 2, 4, 8, 16]

    u = midpoint_grid(n)
    x0 = initial_profile_piecewise(u)
    loss_fn = _make_ramp(eps)

    # Block-constant part of the kernel (shared across all M)
    u_block = (u >= 0.5).astype(int)
    W_block = B[np.ix_(u_block, u_block)]

    all_states = {}
    all_fracs = {}
    times = None

    for M in M_vals:
        W_sm = kernel_trigonometric(u, u, M)   # kappa=5.0 by default
        W = W_block + W_sm
        times, states_M = simulate_graphon(x0, W, mu, r, T, dt, loss_fn)
        frac_M = threshold_fraction_path(states_M)
        all_states[M] = states_M
        all_fracs[M] = frac_M

    # M=16 is the benchmark
    states_ref = all_states[16]
    frac_ref = all_fracs[16]

    results = {}
    for M in [1, 2, 4, 8]:
        results[M] = {
            "state_err": sup_vector_error_inf(all_states[M], states_ref),
            "frac_err": sup_scalar_error(all_fracs[M], frac_ref),
        }

    return times, results, all_states, all_fracs


# ---------------------------------------------------------------------------
# Step-halving temporal validation
# ---------------------------------------------------------------------------

def run_step_halving(n: int = N):
    """Compare hard-threshold simulation at dt=0.002 vs dt=0.001.

    The finer grid is subsampled at every other step to align with the
    coarse grid before computing the sup discrepancy.

    Returns
    -------
    discrepancy : float  sup_t max_i |X^{dt=0.002}(t,i) - X^{dt=0.001}(t,i)|
    times_coarse : (1001,) coarse time grid
    states_coarse, states_fine_sub : (1001, n) arrays
    """
    B = np.array([[0.38, 0.24], [0.31, 0.36]])
    r = 0.0
    mu = -0.06
    T = 2.0

    u = midpoint_grid(n)
    x0 = initial_profile_piecewise(u)
    W = kernel_block_exponential(u, u, B)

    times_coarse, states_coarse = simulate_graphon(x0, W, mu, r, T, 0.002, loss_hard)
    _times_fine, states_fine = simulate_graphon(x0, W, mu, r, T, 0.001, loss_hard)

    # Subsample fine grid at matching coarse-grid steps (every 2nd point)
    states_fine_sub = states_fine[::2]
    discrepancy = sup_vector_error_inf(states_coarse, states_fine_sub)

    return discrepancy, times_coarse, states_coarse, states_fine_sub


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(
    times1, res1, all_states1, all_fracs1,
    times2, res2, frac_ref2,
    times3, res3, all_states3, all_fracs3,
    times_sh, states_sh_coarse, states_sh_fine,
):
    eps_vals = [0.04, 0.02, 0.01]
    colors_eps = ["tab:blue", "tab:orange", "tab:green"]

    # --- Plot C1a: threshold-fraction paths ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times1, all_fracs1["hard"], "k-", lw=2, label="Hard baseline")
    for eps, col in zip(eps_vals, colors_eps):
        ax.plot(times1, all_fracs1[eps], color=col, lw=1.5, label=f"eps={eps}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Threshold fraction h(t)")
    ax.set_title("Component 1 – Threshold fraction paths")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_c1_fraction_paths.png"), dpi=150)
    plt.close(fig)

    # --- Plot C1b: log-log error vs eps ---
    eps_arr = np.array(eps_vals)
    state_errs1 = [res1[e]["state_err"] for e in eps_vals]
    frac_errs1 = [res1[e]["frac_err"] for e in eps_vals]
    _, slope_s = loglog_slope(eps_arr, state_errs1)
    _, slope_f = loglog_slope(eps_arr, frac_errs1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(eps_arr, state_errs1, "o-", label=f"State err (slope={slope_s:.2f})")
    ax.loglog(eps_arr, frac_errs1, "s-", label=f"Frac err  (slope={slope_f:.2f})")
    ax.set_xlabel("eps")
    ax.set_ylabel("Sup error vs hard baseline")
    ax.set_title("Component 1 – Error convergence (log-log)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_c1_loglog.png"), dpi=150)
    plt.close(fig)

    # --- Plot C2: truncation convergence ---
    K_vals = sorted(res2.keys())
    state_errs2 = [res2[K]["state_err"] for K in K_vals]
    frac_errs2 = [res2[K]["frac_err"] for K in K_vals]

    fig, ax = plt.subplots(figsize=(6, 5))
    all_zero2 = all(e == 0.0 for e in state_errs2) and all(e == 0.0 for e in frac_errs2)
    if all_zero2:
        ax.plot(K_vals, state_errs2, "o-", label="State sup error (all zero)")
        ax.plot(K_vals, frac_errs2, "s-", label="Frac sup error (all zero)")
        ax.set_yscale("linear")
        ax.text(0.5, 0.5, "All errors = 0\n(no cascades with smooth x₀, μ=−0.0006)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        ax.semilogy(K_vals, [max(e, 1e-18) for e in state_errs2], "o-", label="State sup error")
        ax.semilogy(K_vals, [max(e, 1e-18) for e in frac_errs2], "s-", label="Frac sup error")
    ax.set_xlabel("Number of modes K")
    ax.set_ylabel("Error vs 30-mode reference")
    ax.set_title("Component 2 – Directed-mode truncation convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_c2_truncation.png"), dpi=150)
    plt.close(fig)

    # --- Plot C2b: reference threshold-fraction path ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times2, frac_ref2, "k-", lw=2, label="30-mode reference")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Threshold fraction h(t)")
    ax.set_title("Component 2 – 30-mode reference path (eps=0.02)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_c2_ref_path.png"), dpi=150)
    plt.close(fig)

    # --- Plot C3: trigonometric convergence ---
    M_cmp = [1, 2, 4, 8]
    state_errs3 = [res3[M]["state_err"] for M in M_cmp]
    frac_errs3 = [res3[M]["frac_err"] for M in M_cmp]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(M_cmp, state_errs3, "o-", label="State sup error")
    ax.semilogy(M_cmp, frac_errs3, "s-", label="Frac sup error")
    ax.set_xlabel("Number of Fourier modes M")
    ax.set_ylabel("Error vs M=16 benchmark")
    ax.set_title("Component 3 – Trigonometric kernel convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_c3_trig.png"), dpi=150)
    plt.close(fig)

    # --- Plot C3b: threshold-fraction paths for each M ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for M, col in zip([1, 2, 4, 8, 16], colors_eps + ["tab:red", "tab:purple"]):
        ax.plot(times3, all_fracs3[M], color=col, lw=1.5, label=f"M={M}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Threshold fraction h(t)")
    ax.set_title("Component 3 – Trigonometric kernel paths")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_c3_paths.png"), dpi=150)
    plt.close(fig)

    # --- Plot step-halving discrepancy over time ---
    disc_over_time = np.max(np.abs(states_sh_coarse - states_sh_fine), axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times_sh, disc_over_time, "b-", lw=1.5)
    ax.set_xlabel("Time t")
    ax.set_ylabel("max_i |X^{0.002}_i(t) - X^{0.001}_i(t)|")
    ax.set_title("Step-halving validation (hard threshold, n=2000)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_step_halving.png"), dpi=150)
    plt.close(fig)

    print("  Plots saved to", RESULTS_DIR)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(res1, res2, res3, step_disc):
    filepath = os.path.join(RESULTS_DIR, "exp3_results.csv")
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)

        # Component 1
        w.writerow(["# Component 1: piecewise-smooth indicator (block-exponential kernel)"])
        w.writerow(["loss_type", "eps", "terminal_frac", "state_err_sup", "frac_err_sup"])
        w.writerow(["hard", "", f"{res1['hard']['terminal_frac']:.6f}", "", ""])
        for eps in [0.04, 0.02, 0.01]:
            r = res1[eps]
            w.writerow([
                "positive_ramp", eps,
                f"{r['terminal_frac']:.6f}",
                f"{r['state_err']:.4e}",
                f"{r['frac_err']:.4e}",
            ])
        w.writerow([])

        # Component 2
        w.writerow(["# Component 2: directed 30-mode truncation (eps=0.02)"])
        w.writerow(["K", "state_err_sup", "frac_err_sup"])
        for K in sorted(res2.keys()):
            r = res2[K]
            w.writerow([K, f"{r['state_err']:.4e}", f"{r['frac_err']:.4e}"])
        w.writerow([])

        # Component 3
        w.writerow(["# Component 3: trigonometric family vs M=16 benchmark (eps=0.02)"])
        w.writerow(["M", "state_err_sup", "frac_err_sup"])
        for M in sorted(res3.keys()):
            r = res3[M]
            w.writerow([M, f"{r['state_err']:.4e}", f"{r['frac_err']:.4e}"])
        w.writerow([])

        # Step-halving
        w.writerow(["# Step-halving validation (hard threshold, Component 1 setup)"])
        w.writerow(["dt_coarse", "dt_fine", "max_discrepancy"])
        w.writerow([0.002, 0.001, f"{step_disc:.4e}"])

    print(f"  CSV saved to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Experiment 3: Graphon smoothing and kernel truncation")
    print("=" * 65)

    # ---- Component 1 ----
    print("\n[C1] Piecewise-smooth indicator experiment (n=2000, dt=2e-3)...")
    times1, res1, all_states1, all_fracs1 = run_component1()

    print(f"     Hard baseline  terminal frac = {res1['hard']['terminal_frac']:.4f}"
          f"  (expected ~0.2195)")
    for eps in [0.04, 0.02, 0.01]:
        r = res1[eps]
        print(f"     eps={eps:.2f}  frac={r['terminal_frac']:.4f}"
              f"  state_err={r['state_err']:.4e}"
              f"  frac_err={r['frac_err']:.4e}")

    # Step-halving ratios (should be ~2 for first-order convergence)
    print("\n     Halving ratios (expect ~2.0):")
    for (e0, e1) in [(0.04, 0.02), (0.02, 0.01)]:
        rs = res1[e0]["state_err"] / res1[e1]["state_err"]
        rf = res1[e0]["frac_err"] / res1[e1]["frac_err"]
        print(f"       eps {e0}→{e1}: state ratio={rs:.3f}, frac ratio={rf:.3f}")

    # ---- Component 2 ----
    print("\n[C2] Directed 30-mode truncation experiment (n=2000, dt=2e-3)...")
    times2, res2, states_ref2, frac_ref2 = run_component2()

    print(f"     {'K':>4}  {'state_err':>12}  {'frac_err':>12}")
    for K in sorted(res2.keys()):
        r = res2[K]
        print(f"     {K:>4}  {r['state_err']:>12.4e}  {r['frac_err']:>12.4e}")

    K_sorted = sorted(res2.keys())
    se = [res2[K]["state_err"] for K in K_sorted]
    monotone = all(se[i] >= se[i + 1] for i in range(len(se) - 1))
    print(f"     Errors monotonically decreasing: {monotone}")

    # ---- Component 3 ----
    print("\n[C3] Trigonometric family experiment (n=2000, dt=2e-3)...")
    times3, res3, all_states3, all_fracs3 = run_component3()

    print(f"     {'M':>4}  {'state_err':>12}  {'frac_err':>12}")
    for M in sorted(res3.keys()):
        r = res3[M]
        print(f"     {M:>4}  {r['state_err']:>12.4e}  {r['frac_err']:>12.4e}")

    M_sorted = sorted(res3.keys())
    se3 = [res3[M]["state_err"] for M in M_sorted]
    monotone3 = all(se3[i] >= se3[i + 1] for i in range(len(se3) - 1))
    print(f"     Errors monotonically decreasing: {monotone3}")

    # ---- Step-halving ----
    print("\n[SH] Step-halving temporal validation (hard threshold, dt=0.002 vs 0.001)...")
    disc, times_sh, states_sh_coarse, states_sh_fine = run_step_halving()
    print(f"     Max discrepancy: {disc:.4e}")

    # ---- Save outputs ----
    print("\nSaving plots and CSV...")
    save_plots(
        times1, res1, all_states1, all_fracs1,
        times2, res2, frac_ref2,
        times3, res3, all_states3, all_fracs3,
        times_sh, states_sh_coarse, states_sh_fine,
    )
    save_csv(res1, res2, res3, disc)

    print("\nDone.")


if __name__ == "__main__":
    main()
