"""
run_exp6.py — Experiment 6: EBA full-sample 116-institution contagion.

116 institutions (N=116) × 38 sovereign buckets (K=38).
Demonstrates:
  - Baseline contagion dynamics with empirical loadings
  - Direct / reduced system exact equivalence
  - Robustness to 'Other sovereigns' factor and alternative normalization
  - Scenario-aligned rank-2 surrogate vs generic SVD truncations
  - Italian-book perturbation sensitivity
  - Resampling error decay and log-log convergence rates
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eba_loader import generate_synthetic_116
from src.empirical_kernel import (
    build_empirical_loadings,
    build_empirical_exposure,
    build_initial_buffers,
    add_other_factor,
    perturb_italian_holdings,
)
from src.model import simulate_direct, loss_negative_ramp, mean_smoothed_distress
from src.low_rank import simulate_reduced
from src.metrics import (
    sup_vector_error_inf,
    frobenius_relative_error,
    svd_truncate,
    sup_feedback_error,
    wasserstein1_empirical,
    loglog_slope,
)
from src.sampling import resample_from_empirical

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
MU = 0.0
R = 0.0
T = 3.0
DT = 1e-3
EPS = 0.05
LAMBDA_E = 0.075
X_STAR = 0.40


def loss_fn(x: np.ndarray) -> np.ndarray:
    return loss_negative_ramp(x, EPS)


def reduced_factors(A: np.ndarray, B: np.ndarray, lambda_E: float) -> tuple:
    """Return scaled (A_r, B_r) so simulate_reduced matches simulate_direct with E=lambda_E*A@B.T."""
    K = A.shape[1]
    scale = np.sqrt(K * lambda_E)
    return scale * A, scale * B


def run_direct_reduced(x0, A, B, E, lambda_E, label=""):
    """Run direct + reduced, assert equivalence, return (times, states, distress, betas)."""
    times, states = simulate_direct(x0, E, MU, R, T, DT, loss_fn)
    A_r, B_r = reduced_factors(A, B, lambda_E)
    _, states_r, betas = simulate_reduced(x0, A_r, B_r, MU, R, T, DT, loss_fn)
    err = sup_vector_error_inf(states, states_r)
    assert err < 1e-10, f"[{label}] direct/reduced gap = {err:.3e}"
    distress = mean_smoothed_distress(states, loss_fn)
    return times, states, distress, betas


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate synthetic 116-institution data
    # ------------------------------------------------------------------
    S, banks, factors = generate_synthetic_116(seed=42)
    N, K = S.shape  # 116, 38
    print(f"Generated synthetic data: N={N}, K={K}")
    print(f"  Total holdings: {S.sum():.1f}")

    # ------------------------------------------------------------------
    # 2. Data cleaning
    # ------------------------------------------------------------------
    # a. No aggregate placeholder rows in synthetic data
    # b. Set values in [-0.001, 0) to zero
    S[(-0.001 <= S) & (S < 0.0)] = 0.0
    # c. Retain all K=38 buckets
    assert S.shape == (116, 38)

    # ------------------------------------------------------------------
    # 3. Build empirical loadings
    # ------------------------------------------------------------------
    A, B = build_empirical_loadings(S, LAMBDA_E)

    # ------------------------------------------------------------------
    # 4. Validate normalization identities
    # ------------------------------------------------------------------
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-10), "A row sums != 1"
    assert np.allclose(B.mean(axis=0), 1.0, atol=1e-10), "B col means != 1"
    print(f"  A row-sum range:   [{A.sum(axis=1).min():.10f}, {A.sum(axis=1).max():.10f}]")
    print(f"  B col-mean range:  [{B.mean(axis=0).min():.10f}, {B.mean(axis=0).max():.10f}]")

    # ------------------------------------------------------------------
    # 5. Build exposure matrix E (rank-38)
    # ------------------------------------------------------------------
    E = build_empirical_exposure(A, B, LAMBDA_E)
    E_rank = np.linalg.matrix_rank(E)
    print(f"  E shape: {E.shape}, rank: {E_rank}")

    # ------------------------------------------------------------------
    # 6. Stress scenario: FR=0.35, DE=0.15, IT=0.45, NL=0.05
    # ------------------------------------------------------------------
    # Factor indices in the 38-factor list:
    # France=0, Germany=1, Italy=2, Spain=3, Netherlands=4
    eta = np.zeros(K)
    eta[0] = 0.35   # France
    eta[1] = 0.15   # Germany
    eta[2] = 0.45   # Italy
    eta[4] = 0.05   # Netherlands

    # ------------------------------------------------------------------
    # 7. Initial buffers
    # ------------------------------------------------------------------
    x0 = build_initial_buffers(A, eta, X_STAR)
    print(f"\n  x0 range: [{x0.min():.4f}, {x0.max():.4f}]")

    # ------------------------------------------------------------------
    # 8–9. Baseline simulation (direct + reduced equivalence)
    # ------------------------------------------------------------------
    print("\n--- Baseline simulation ---")
    times, states, distress, betas_pop = run_direct_reduced(
        x0, A, B, E, LAMBDA_E, label="baseline"
    )

    # ------------------------------------------------------------------
    # 10. Baseline metrics
    # ------------------------------------------------------------------
    td_base = float(distress[-1])
    tf_base = float((states[-1] <= 0.0).mean())
    n_distressed = int(round(tf_base * N))
    print(f"  Terminal mean smoothed distress:  {td_base:.4f}  (expected ≈ 0.0246)")
    print(f"  Terminal threshold fraction:      {tf_base:.4f}  ≈ {n_distressed}/{N}  (expected ≈ 4/116)")

    # ------------------------------------------------------------------
    # 11. Robustness: 'Other sovereigns' factor (K=39)
    # ------------------------------------------------------------------
    print("\n--- Other sovereigns (K=39) ---")
    # Assume 38 factors cover 99.704% of total sovereign book
    coverage = 0.99704
    full_row_totals = S.sum(axis=1) / coverage
    S_ext = add_other_factor(S, full_row_totals)  # (116, 39)
    A_ext, B_ext = build_empirical_loadings(S_ext, LAMBDA_E)
    E_ext = build_empirical_exposure(A_ext, B_ext, LAMBDA_E)
    eta_ext = np.append(eta, 0.0)  # 39-dim, 'Other' unstressed
    x0_ext = build_initial_buffers(A_ext, eta_ext, X_STAR)
    _, states_ext, distress_ext, _ = run_direct_reduced(
        x0_ext, A_ext, B_ext, E_ext, LAMBDA_E, label="other_factor"
    )
    td_ext = float(distress_ext[-1])
    tf_ext = float((states_ext[-1] <= 0.0).mean())
    print(f"  Terminal distress (K=39): {td_ext:.5f}  vs baseline: {td_base:.5f}")
    print(f"  Terminal thresh  (K=39): {int(round(tf_ext*N))}/{N}  vs baseline: {n_distressed}/{N}")
    print(f"  (expected ≈ 0.02459 vs 0.02462, same threshold fraction)")

    # ------------------------------------------------------------------
    # 12. Alternative normalization: c_E=0.30 fixed → lambda_E_alt = 0.30/K
    # ------------------------------------------------------------------
    print("\n--- Alternative normalization: c_E=0.30 ---")
    lambda_E_alt = 0.30 / K   # ≈ 0.00789
    E_alt = lambda_E_alt * (A @ B.T)   # A, B unchanged; only intensity changes
    _, states_alt = simulate_direct(x0, E_alt, MU, R, T, DT, loss_fn)
    distress_alt = mean_smoothed_distress(states_alt, loss_fn)
    td_alt = float(distress_alt[-1])
    tf_alt = float((states_alt[-1] <= 0.0).mean())
    print(f"  lambda_E_alt = 0.30/{K} ≈ {lambda_E_alt:.5f}")
    print(f"  Terminal distress: {td_alt:.4f}  (expected ≈ 0.0138)")
    print(f"  Terminal thresh:   {int(round(tf_alt*N))}/{N}  (expected same frac)")

    # ------------------------------------------------------------------
    # 13. Reduction comparison: scenario-aligned rank-2 surrogate
    # ------------------------------------------------------------------
    print("\n--- Scenario-aligned rank-2 surrogate vs generic SVD ---")

    # Stressed sovereign column indices: FR=0, DE=1, IT=2, NL=4
    stress_cols = [0, 1, 2, 4]
    S_4 = S[:, stress_cols]   # (116, 4)
    # Build A_4, B_4 restricted to the 4 stressed factors
    row_sums_4 = S_4.sum(axis=1, keepdims=True)
    col_means_4 = S_4.mean(axis=0, keepdims=True)
    # Some banks may have zero holdings in all 4 stressed countries; guard
    row_sums_4 = np.where(row_sums_4 == 0, 1.0, row_sums_4)
    col_means_4 = np.where(col_means_4 == 0, 1.0, col_means_4)
    A_4 = S_4 / row_sums_4    # (116, 4) receiver loadings for stressed factors
    B_4 = S_4 / col_means_4   # (116, 4) sender loadings for stressed factors
    E_4 = LAMBDA_E * (A_4 @ B_4.T)   # rank≤4
    E_scen_r2 = svd_truncate(E_4, 2)  # scenario-aligned rank-2

    _, states_scen_r2 = simulate_direct(x0, E_scen_r2, MU, R, T, DT, loss_fn)
    err_scen_r2 = sup_vector_error_inf(states, states_scen_r2)
    print(f"  Scenario-aligned rank-2 path error: {err_scen_r2:.4e}  (expected ≈ 7.20e-4)")

    # Generic SVD truncations of full E
    generic_ranks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    generic_errors = []
    for rk in generic_ranks:
        E_gen = svd_truncate(E, rk)
        _, states_gen = simulate_direct(x0, E_gen, MU, R, T, DT, loss_fn)
        err_gen = sup_vector_error_inf(states, states_gen)
        generic_errors.append((rk, err_gen))
        if rk == 12:
            print(f"  Generic rank-12 SVD path error:    {err_gen:.4e}  (expected ≈ 1.12e-2)")

    print("  Generic rank → path error:")
    for rk, err in generic_errors:
        print(f"    rank={rk:2d}: {err:.4e}")

    # ------------------------------------------------------------------
    # 14. Perturbation: Italian-book delta sweep
    # ------------------------------------------------------------------
    print("\n--- Italian-book perturbation ---")
    italy_col = 2   # Italy is factor index 2
    deltas = np.linspace(-0.20, 0.20, 9)
    pert_results = []
    for delta in deltas:
        S_pert = perturb_italian_holdings(S, italy_col, delta)
        A_pert, B_pert = build_empirical_loadings(S_pert, LAMBDA_E)
        E_pert = build_empirical_exposure(A_pert, B_pert, LAMBDA_E)
        x0_pert = build_initial_buffers(A_pert, eta, X_STAR)
        _, states_pert = simulate_direct(x0_pert, E_pert, MU, R, T, DT, loss_fn)
        max_gap = float(np.max(np.abs(states_pert[-1] - states[-1])))
        pert_results.append((delta, max_gap))
        print(f"  delta={delta:+.2f}: max |ΔX_T^i| = {max_gap:.4e}")

    # ------------------------------------------------------------------
    # 15–19. Resampling convergence
    # ------------------------------------------------------------------
    print("\n--- Resampling convergence (n_replications=50) ---")
    N_resample_vals = [50, 100, 200, 500, 1000]
    n_replications = 50
    rng = np.random.default_rng(seed=0)

    # Population: run reduced on full 116-institution system
    scale_K = np.sqrt(K * LAMBDA_E)
    A_r_pop, B_r_pop = scale_K * A, scale_K * B
    _, states_pop, betas_pop_path = simulate_reduced(
        x0, A_r_pop, B_r_pop, MU, R, T, DT, loss_fn
    )

    resample_rows = []
    mean_feed_errs = []
    mean_w1_errs = []

    for N_s in N_resample_vals:
        feed_errs = []
        w1_errs = []
        for _ in range(n_replications):
            x0_s, A_s, B_s = resample_from_empirical(N_s, x0, A, B, rng)
            A_s_r = scale_K * A_s
            B_s_r = scale_K * B_s
            _, states_s, betas_s = simulate_reduced(
                x0_s, A_s_r, B_s_r, MU, R, T, DT, loss_fn
            )
            feed_errs.append(sup_feedback_error(betas_s, betas_pop_path))
            w1_errs.append(wasserstein1_empirical(states_s[-1], states_pop[-1]))

        mfe = float(np.mean(feed_errs))
        mw1 = float(np.mean(w1_errs))
        mean_feed_errs.append(mfe)
        mean_w1_errs.append(mw1)
        resample_rows.append({"N_resample": N_s, "mean_feed_error": mfe, "mean_w1": mw1})
        print(f"  N={N_s:5d}:  feed_err={mfe:.4e},  W1={mw1:.4e}")

    # Log-log slopes
    N_arr = np.array(N_resample_vals, dtype=float)
    _, slope_feed = loglog_slope(N_arr, np.array(mean_feed_errs))
    _, slope_w1   = loglog_slope(N_arr, np.array(mean_w1_errs))
    print(f"  Log-log slope (feed error): {slope_feed:.3f}  (expected ≈ -0.43 to -0.49)")
    print(f"  Log-log slope (W1):         {slope_w1:.3f}  (expected ≈ -0.43 to -0.49)")

    # ------------------------------------------------------------------
    # Save results CSV
    # ------------------------------------------------------------------
    rows = []

    rows.append({
        "experiment": "baseline",
        "label": "K=38",
        "lambda_E": LAMBDA_E,
        "K": K,
        "terminal_distress": td_base,
        "terminal_frac": tf_base,
        "path_error": float("nan"),
        "extra": float("nan"),
    })
    rows.append({
        "experiment": "other_factor",
        "label": "K=39",
        "lambda_E": LAMBDA_E,
        "K": K + 1,
        "terminal_distress": td_ext,
        "terminal_frac": tf_ext,
        "path_error": float("nan"),
        "extra": float("nan"),
    })
    rows.append({
        "experiment": "alt_norm",
        "label": f"c_E=0.30",
        "lambda_E": lambda_E_alt,
        "K": K,
        "terminal_distress": td_alt,
        "terminal_frac": tf_alt,
        "path_error": float("nan"),
        "extra": float("nan"),
    })
    rows.append({
        "experiment": "surrogate",
        "label": "scenario_r2",
        "lambda_E": LAMBDA_E,
        "K": 2,
        "terminal_distress": float(mean_smoothed_distress(states_scen_r2, loss_fn)[-1]),
        "terminal_frac": float((states_scen_r2[-1] <= 0.0).mean()),
        "path_error": err_scen_r2,
        "extra": float("nan"),
    })
    for rk, err in generic_errors:
        rows.append({
            "experiment": "surrogate",
            "label": f"generic_r{rk}",
            "lambda_E": LAMBDA_E,
            "K": rk,
            "terminal_distress": float("nan"),
            "terminal_frac": float("nan"),
            "path_error": err,
            "extra": float("nan"),
        })
    for delta, gap in pert_results:
        rows.append({
            "experiment": "perturbation",
            "label": f"delta={delta:+.2f}",
            "lambda_E": LAMBDA_E,
            "K": K,
            "terminal_distress": float("nan"),
            "terminal_frac": float("nan"),
            "path_error": float("nan"),
            "extra": gap,
        })
    for rec in resample_rows:
        rows.append({
            "experiment": "resampling",
            "label": f"N={rec['N_resample']}",
            "lambda_E": LAMBDA_E,
            "K": K,
            "terminal_distress": float("nan"),
            "terminal_frac": float("nan"),
            "path_error": rec["mean_feed_error"],
            "extra": rec["mean_w1"],
        })
    rows.append({
        "experiment": "resampling_slope",
        "label": "slope_feed",
        "lambda_E": LAMBDA_E,
        "K": K,
        "terminal_distress": float("nan"),
        "terminal_frac": float("nan"),
        "path_error": slope_feed,
        "extra": slope_w1,
    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp6_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ------------------------------------------------------------------
    # Plot 1: singular values of E
    # ------------------------------------------------------------------
    svs = np.linalg.svd(E, compute_uv=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(np.arange(1, min(21, K + 1)), svs[:20], color="steelblue", alpha=0.8)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular values of $E$ (top-20 of 38)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    # Mean smoothed distress over time for baseline
    ax.plot(times, distress, "k-", linewidth=2, label="baseline")
    ax.plot(times, distress_alt, "--", color="steelblue", label=f"c_E=0.30 (λ={lambda_E_alt:.4f})")
    ax.plot(times, distress_ext, ":", color="firebrick", label="K=39 (Other factor)")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Mean smoothed distress")
    ax.set_title("Distress paths: baseline vs robustness checks")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(RESULTS_DIR, "exp6_singular_values.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Plot saved: {path1}")

    # ------------------------------------------------------------------
    # Plot 2: resampling error decay (log-log)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(N_resample_vals, mean_feed_errs, "o-", color="navy", label="Feedback error")
    # Reference line: slope = -0.5
    ref_x = np.array([50, 1000], dtype=float)
    ref_y = mean_feed_errs[0] * (ref_x / N_resample_vals[0]) ** (-0.5)
    ax.loglog(ref_x, ref_y, "k--", linewidth=0.8, label="slope $-0.5$")
    ax.set_xlabel("$N$ (resampled institutions)")
    ax.set_ylabel("Mean sup feedback error")
    ax.set_title(f"Feedback path error (slope≈{slope_feed:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(N_resample_vals, mean_w1_errs, "s-", color="firebrick", label="Wasserstein-1")
    ref_y2 = mean_w1_errs[0] * (ref_x / N_resample_vals[0]) ** (-0.5)
    ax.loglog(ref_x, ref_y2, "k--", linewidth=0.8, label="slope $-0.5$")
    ax.set_xlabel("$N$ (resampled institutions)")
    ax.set_ylabel("Mean terminal Wasserstein-1")
    ax.set_title(f"Terminal $W_1$ error (slope≈{slope_w1:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, "exp6_resampling_error.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Plot saved: {path2}")

    # ------------------------------------------------------------------
    # Plot 3: perturbation sensitivity curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    delta_vals = [d for d, _ in pert_results]
    gap_vals   = [g for _, g in pert_results]
    ax.plot(delta_vals, gap_vals, "o-", color="darkgreen", linewidth=2)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Italian holdings perturbation $\\delta$")
    ax.set_ylabel("$\\max_i |\\Delta X_T^i|$")
    ax.set_title("Terminal state sensitivity to Italian book perturbation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(RESULTS_DIR, "exp6_perturbation.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"Plot saved: {path3}")

    print("\nExperiment 6 complete.")


if __name__ == "__main__":
    main()
