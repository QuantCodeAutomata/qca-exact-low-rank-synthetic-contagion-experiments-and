"""
run_exp5.py — Experiment 5: EBA six-bank sovereign-holdings contagion application.

Six banks (N=6) × four sovereign factors (K=4: France, Germany, Italy, Netherlands).
Demonstrates:
  - Empirical loading construction and normalization validation
  - Direct / reduced system exact equivalence
  - Sensitivity to full-book denominator, exposure intensity, and SVD truncation rank
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

from src.eba_loader import load_six_bank_data, load_six_bank_full_book
from src.empirical_kernel import (
    build_empirical_loadings,
    build_empirical_exposure,
    build_initial_buffers,
    build_full_book_loadings,
    perturb_italian_holdings,
)
from src.model import simulate_direct, loss_negative_ramp, mean_smoothed_distress
from src.low_rank import simulate_reduced, build_factorised_exposure
from src.metrics import sup_vector_error_inf, frobenius_relative_error, svd_truncate

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
ETA = np.array([0.35, 0.15, 0.45, 0.05])


def loss_fn(x: np.ndarray) -> np.ndarray:
    return loss_negative_ramp(x, EPS)


def reduced_factors(A: np.ndarray, B: np.ndarray, lambda_E: float) -> tuple:
    """Scale A, B so simulate_reduced matches simulate_direct with E=lambda_E*A@B.T.

    The factorised exposure used by simulate_reduced is (A_r @ B_r.T) / K.
    Setting A_r = sqrt(K*lambda_E)*A and B_r = sqrt(K*lambda_E)*B gives
    (A_r @ B_r.T) / K = lambda_E * (A @ B.T) = E.
    """
    K = A.shape[1]
    scale = np.sqrt(K * lambda_E)
    return scale * A, scale * B


def run_baseline(A, B, E, x0, N, K, label="baseline"):
    """Run direct + reduced simulation and assert equivalence."""
    times, states_d = simulate_direct(x0, E, MU, R, T, DT, loss_fn)

    A_r, B_r = reduced_factors(A, B, LAMBDA_E)
    _, states_r, betas = simulate_reduced(x0, A_r, B_r, MU, R, T, DT, loss_fn)

    err = sup_vector_error_inf(states_d, states_r)
    assert err < 1e-10, f"[{label}] direct/reduced gap = {err:.3e}"

    distress = mean_smoothed_distress(states_d, loss_fn)
    term_distress = float(distress[-1])
    term_frac = float((states_d[-1] <= 0.0).mean())
    return times, states_d, distress, term_distress, term_frac


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load six-bank data
    # ------------------------------------------------------------------
    S, banks, factors = load_six_bank_data()
    N, K = S.shape   # 6, 4
    print(f"Loaded six-bank data: N={N}, K={K}")
    print(f"  Banks:   {banks}")
    print(f"  Factors: {factors}")

    # ------------------------------------------------------------------
    # 2. Build empirical loadings
    # ------------------------------------------------------------------
    A, B = build_empirical_loadings(S, LAMBDA_E)

    # ------------------------------------------------------------------
    # 3. Validate normalization identities
    # ------------------------------------------------------------------
    row_sum_A = A.sum(axis=1)
    col_mean_B = B.mean(axis=0)
    assert np.allclose(row_sum_A, 1.0, atol=1e-10), "A row sums != 1"
    assert np.allclose(col_mean_B, 1.0, atol=1e-10), "B col means != 1"
    print("\nNormalization check:")
    print(f"  sum_k a_{{i,k}} for each i: {row_sum_A}")
    print(f"  (1/N) sum_i b_{{i,k}} for each k: {col_mean_B}")

    # ------------------------------------------------------------------
    # 4. Build exposure matrix E (exact rank-4)
    # ------------------------------------------------------------------
    E = build_empirical_exposure(A, B, LAMBDA_E)
    print(f"\nE shape: {E.shape}, rank: {np.linalg.matrix_rank(E)}")

    # ------------------------------------------------------------------
    # 5–7. Stress scenario and initial buffers
    # ------------------------------------------------------------------
    x0 = build_initial_buffers(A, ETA, X_STAR)
    print(f"\nInitial buffers x0:\n  {dict(zip(banks, x0.round(4)))}")

    # ------------------------------------------------------------------
    # 8–10. Baseline direct + reduced simulation
    # ------------------------------------------------------------------
    print("\n--- Baseline simulation ---")
    times, states_base, distress_base, td_base, tf_base = run_baseline(
        A, B, E, x0, N, K, label="baseline"
    )
    print(f"  Terminal mean smoothed distress:  {td_base:.4f}  (expected ≈ 0.1694)")
    print(f"  Terminal threshold fraction:      {tf_base:.4f}  = {int(round(tf_base*N))}/{N}  (expected 2/6)")

    # ------------------------------------------------------------------
    # 11. Full-book denominator sensitivity
    # ------------------------------------------------------------------
    print("\n--- Full-book denominator sensitivity ---")
    S_full, full_row_totals = load_six_bank_full_book()
    A_fb = build_full_book_loadings(S_full, full_row_totals)
    # B sender loadings: standard col-mean normalization from the same S
    col_means = S.mean(axis=0, keepdims=True)
    B_fb = S / col_means
    E_fb = LAMBDA_E * (A_fb @ B_fb.T)
    x0_fb = X_STAR - A_fb @ ETA
    times_fb, states_fb = simulate_direct(x0_fb, E_fb, MU, R, T, DT, loss_fn)
    distress_fb = mean_smoothed_distress(states_fb, loss_fn)
    td_fb = float(distress_fb[-1])
    tf_fb = float((states_fb[-1] <= 0.0).mean())
    print(f"  Terminal mean smoothed distress:  {td_fb:.4f}  (expected → ~0)")
    print(f"  Terminal threshold fraction:      {tf_fb:.4f}  = {int(round(tf_fb*N))}/{N}")

    # ------------------------------------------------------------------
    # 12. Intensity sweep: lambda_E in [0.5x, 1x, 2x]
    # ------------------------------------------------------------------
    print("\n--- Intensity sweep ---")
    sweep_lambdas = [0.5 * LAMBDA_E, LAMBDA_E, 2.0 * LAMBDA_E]
    sweep_labels  = ["0.5×", "1×", "2×"]
    sweep_results = []
    for lam, lbl in zip(sweep_lambdas, sweep_labels):
        E_sw = lam * (A @ B.T)   # scale E; A, B, x0 fixed
        _, states_sw = simulate_direct(x0, E_sw, MU, R, T, DT, loss_fn)
        d_sw = mean_smoothed_distress(states_sw, loss_fn)
        td_sw = float(d_sw[-1])
        tf_sw = float((states_sw[-1] <= 0.0).mean())
        sweep_results.append((lbl, lam, td_sw, tf_sw))
        print(f"  λ_E={lam:.5f} ({lbl}):  distress={td_sw:.4f},  thresh={int(round(tf_sw*N))}/{N}")
    print("  Expected: 0.5×→0.0750/2/6, 1×→0.1694/2/6, 2×→0.5682/5/6")

    # ------------------------------------------------------------------
    # 13. SVD truncation analysis
    # ------------------------------------------------------------------
    print("\n--- SVD truncation analysis ---")
    svd_results = []
    for rank in [1, 2, 3, 4]:
        E_trunc = svd_truncate(E, rank)
        frob_err = frobenius_relative_error(E, E_trunc)
        _, states_trunc = simulate_direct(x0, E_trunc, MU, R, T, DT, loss_fn)
        d_trunc = mean_smoothed_distress(states_trunc, loss_fn)
        td_trunc = float(d_trunc[-1])
        tf_trunc = float((states_trunc[-1] <= 0.0).mean())
        path_gap = sup_vector_error_inf(states_base, states_trunc)
        svd_results.append((rank, frob_err, td_trunc, tf_trunc, path_gap))
        print(
            f"  rank={rank}: Frobenius err={frob_err:.4f}, "
            f"distress={td_trunc:.4f}, thresh={int(round(tf_trunc*N))}/{N}, "
            f"path gap={path_gap:.4e}"
        )
    print("  Expected: rank-1(0.2155,0.1203), rank-2(0.1027,0.1565), "
          "rank-3(0.0334,0.1565), rank-4(0,0.1694)")

    # ------------------------------------------------------------------
    # Save results CSV
    # ------------------------------------------------------------------
    rows = []

    # Baseline
    rows.append({
        "experiment": "baseline",
        "label": "1×",
        "lambda_E": LAMBDA_E,
        "rank": K,
        "frobenius_err": 0.0,
        "terminal_distress": td_base,
        "terminal_frac": tf_base,
        "path_gap": 0.0,
    })

    # Full-book
    rows.append({
        "experiment": "full_book",
        "label": "full_book",
        "lambda_E": LAMBDA_E,
        "rank": K,
        "frobenius_err": float("nan"),
        "terminal_distress": td_fb,
        "terminal_frac": tf_fb,
        "path_gap": float("nan"),
    })

    # Intensity sweep
    for lbl, lam, td, tf in sweep_results:
        rows.append({
            "experiment": "intensity_sweep",
            "label": lbl,
            "lambda_E": lam,
            "rank": K,
            "frobenius_err": float("nan"),
            "terminal_distress": td,
            "terminal_frac": tf,
            "path_gap": float("nan"),
        })

    # SVD truncation
    for rank, ferr, td, tf, pg in svd_results:
        rows.append({
            "experiment": "svd_truncation",
            "label": f"rank-{rank}",
            "lambda_E": LAMBDA_E,
            "rank": rank,
            "frobenius_err": ferr,
            "terminal_distress": td,
            "terminal_frac": tf,
            "path_gap": pg,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp5_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ------------------------------------------------------------------
    # Plot 1: distress paths per bank + mean smoothed distress
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for i, name in enumerate(banks):
        ax.plot(times, states_base[:, i], label=name, linewidth=1.5)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Buffer $X_t^i$")
    ax.set_title("Bank state paths (baseline)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, distress_base, "k-", linewidth=2, label="baseline λ=0.075")
    for lbl, lam, _, _ in sweep_results:
        E_sw = lam * (A @ B.T)
        _, states_sw = simulate_direct(x0, E_sw, MU, R, T, DT, loss_fn)
        d_sw = mean_smoothed_distress(states_sw, loss_fn)
        ax.plot(times, d_sw, linestyle="--", label=f"λ={lam:.4f} ({lbl})")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Mean smoothed distress")
    ax.set_title("Mean smoothed distress — intensity sweep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(RESULTS_DIR, "exp5_distress_paths.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Plot saved: {path1}")

    # ------------------------------------------------------------------
    # Plot 2: SVD truncation sensitivity
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ranks = [r[0] for r in svd_results]
    ferrs = [r[1] for r in svd_results]
    distresses = [r[2] for r in svd_results]
    path_gaps = [r[4] for r in svd_results]

    ax = axes[0]
    ax.bar(ranks, ferrs, color="steelblue", alpha=0.8)
    ax.set_xlabel("SVD truncation rank")
    ax.set_ylabel("Frobenius relative error")
    ax.set_title("Frobenius error of rank-$r$ approximation to $E$")
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(ranks, distresses, "o-", color="navy", label="Terminal distress")
    ax2.plot(ranks, path_gaps, "s--", color="firebrick", label="Max path gap")
    ax.set_xlabel("SVD truncation rank")
    ax.set_ylabel("Terminal mean smoothed distress", color="navy")
    ax2.set_ylabel("Sup-path gap vs rank-4", color="firebrick")
    ax.set_title("SVD truncation: distress and path gap")
    ax.set_xticks(ranks)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, "exp5_sensitivity.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Plot saved: {path2}")

    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
