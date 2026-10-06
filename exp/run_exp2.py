"""
run_exp2.py — Directedness and multilayer amplification experiment.

Compares r=0.0 (symmetric exposures) vs r=0.05 (directed exposures):
  - Group 0: net lender  (high a=2.0, low b=0.6) — outbound > inbound
  - Group 1: net borrower (low a=0.6, high b=2.0) — inbound > outbound

The directedness imbalance Λ_i shifts lenders upward and borrowers downward,
amplifying the separation in group distress levels.

Population limit (eq. 74):
  - r=0: exact ODE (solve_population_ode)
  - r>0: ODE not valid; N=1600 finite-N simulation used as large-N proxy
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import loss_hard, simulate_direct
from src.low_rank import (
    build_factorised_exposure,
    simulate_reduced,
    solve_population_ode,
    population_threshold_fraction,
)
from src.sampling import build_quantile_matched_network
from src.metrics import threshold_fraction_path, group_threshold_fractions

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

G = 2
K = 1
p = np.array([0.5, 0.5])
m_g = np.array([0.50, 0.45])
sigma_g = np.array([0.18, 0.18])
A_g = np.array([[2.0], [0.6]])   # Group 0: net lender (high a)
B_g = np.array([[0.6], [2.0]])   # Group 1: net borrower (high b)
T = 2.0
dt = 5e-4
mu = 0.0

R_VALUES = [0.0, 0.05]
N_MAIN = 400      # primary experimental network size
N_PROXY = 1600    # large-N proxy for r>0 population limit

GROUP_LABELS = ["Group 0 (net lender, high $a$, low $b$)",
                "Group 1 (net borrower, low $a$, high $b$)"]

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_direct_case(r: float, N: int) -> dict:
    """Build a quantile-matched network, run the direct solver, return metrics."""
    x0, A, B, group_ids = build_quantile_matched_network(
        N, p, m_g, sigma_g, A_g, B_g
    )
    E = build_factorised_exposure(A, B)
    times, states = simulate_direct(x0, E, mu, r, T, dt, loss_hard)
    h_N = threshold_fraction_path(states)
    h_g = group_threshold_fractions(states, group_ids, G)
    return {
        "N": N,
        "r": r,
        "times": times,
        "h_N": h_N,
        "h_g": h_g,
        "h_terminal": float(h_N[-1]),
        "h_g_terminal": h_g[-1].copy(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 2: Directedness and multilayer amplification")
    print(f"  Config: G={G}, K={K}, T={T}, dt={dt}, mu={mu}")
    print(f"  r values: {R_VALUES},  N_main={N_MAIN},  N_proxy={N_PROXY}")
    print("=" * 70)

    results = {}
    limit_h = {}

    for r in R_VALUES:
        print(f"\n--- r = {r:.3f} ---")

        # Step 3a-3d: N_MAIN simulation
        res = run_direct_case(r, N_MAIN)
        results[r] = res
        h_g_T = res["h_g_terminal"]
        print(f"  N={N_MAIN}: Group 0 lender  terminal h = {h_g_T[0]:.4f}")
        print(f"  N={N_MAIN}: Group 1 borrower terminal h = {h_g_T[1]:.4f}")
        print(f"  N={N_MAIN}: Aggregate        terminal h = {res['h_terminal']:.4f}")

        # Step 4: Population limit
        if r == 0.0:
            # Exact ODE limit (eq. 74), valid for r=0
            print("  Solving population limit ODE ...")
            times_ode, betas_ode = solve_population_ode(
                p, m_g, sigma_g, A_g, B_g, mu, T, dt
            )
            h_lim = population_threshold_fraction(
                betas_ode, times_ode, p, m_g, sigma_g, A_g, mu
            )
            limit_h[r] = {
                "times": times_ode,
                "h": h_lim,
                "source": "ODE (eq. 74)",
            }
            print(f"  ODE limit terminal h = {h_lim[-1]:.4f}")
        else:
            # ODE not valid for r>0; use N_PROXY as large-N proxy
            print(f"  Running N={N_PROXY} large-N proxy for r={r} ...")
            proxy = run_direct_case(r, N_PROXY)
            limit_h[r] = {
                "times": proxy["times"],
                "h": proxy["h_N"],
                "h_g": proxy["h_g"],
                "source": f"N={N_PROXY} proxy",
            }
            print(
                f"  N={N_PROXY} proxy terminal h = {proxy['h_terminal']:.4f}  "
                f"(G0={proxy['h_g_terminal'][0]:.4f}, "
                f"G1={proxy['h_g_terminal'][1]:.4f})"
            )

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(
        f"{'r':>6}  {'G0 lender':>12}  {'G1 borrower':>12}  "
        f"{'G1-G0':>10}  {'agg h_T':>10}  {'limit h_T':>10}"
    )
    print("-" * 70)
    for r in R_VALUES:
        h_g_T = results[r]["h_g_terminal"]
        h_T = results[r]["h_terminal"]
        lim_T = float(limit_h[r]["h"][-1])
        print(
            f"{r:>6.3f}  {h_g_T[0]:>12.4f}  {h_g_T[1]:>12.4f}  "
            f"{h_g_T[1] - h_g_T[0]:>10.4f}  {h_T:>10.4f}  {lim_T:>10.4f}"
        )
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_rows = []
    for r in R_VALUES:
        h_g_T = results[r]["h_g_terminal"]
        csv_rows.append({
            "r": r,
            "N": N_MAIN,
            "h_terminal_group0_lender": h_g_T[0],
            "h_terminal_group1_borrower": h_g_T[1],
            "h_terminal_group1_minus_group0": h_g_T[1] - h_g_T[0],
            "h_terminal_aggregate": results[r]["h_terminal"],
            "h_terminal_limit_or_proxy": float(limit_h[r]["h"][-1]),
            "limit_source": limit_h[r]["source"],
        })
    csv_path = RESULTS_DIR / "exp2_results.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nResults saved → {csv_path}")

    # ------------------------------------------------------------------
    # Plot 1: Per-group distress trajectories (one panel per group)
    # ------------------------------------------------------------------
    r_colors = {0.0: "steelblue", 0.05: "firebrick"}
    r_labels = {0.0: "$r=0$  (no directedness)", 0.05: "$r=0.05$  (directed)"}

    fig, axes = plt.subplots(1, G, figsize=(11, 4), sharey=False)
    for g in range(G):
        ax = axes[g]
        for r in R_VALUES:
            ax.plot(
                results[r]["times"], results[r]["h_g"][:, g],
                color=r_colors[r], lw=1.8, label=r_labels[r],
            )
        ax.set_xlabel("time $t$")
        ax.set_ylabel(f"$h_t^{{(g={g})}}$")
        ax.set_title(GROUP_LABELS[g])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Group distress trajectories: directedness effect ($N=" + str(N_MAIN) + "$)",
        y=1.02,
    )
    fig.tight_layout()
    png1 = RESULTS_DIR / "exp2_group_distress.png"
    fig.savefig(png1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {png1}")

    # ------------------------------------------------------------------
    # Plot 2: Lender vs borrower breakdown for each r (one panel per r)
    # ------------------------------------------------------------------
    group_colors = ["steelblue", "darkorange"]
    group_line_styles = ["-", "--"]

    fig, axes = plt.subplots(1, len(R_VALUES), figsize=(11, 4), sharey=False)
    for ax, r in zip(axes, R_VALUES):
        for g in range(G):
            ax.plot(
                results[r]["times"], results[r]["h_g"][:, g],
                color=group_colors[g], ls=group_line_styles[g],
                lw=1.5, label=GROUP_LABELS[g],
            )
        ax.plot(
            results[r]["times"], results[r]["h_N"],
            "k:", lw=1.5, label="Aggregate",
        )
        ax.set_xlabel("time $t$")
        ax.set_ylabel("threshold fraction")
        ax.set_title(r_labels[r])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Lender vs borrower separation under directedness")
    fig.tight_layout()
    png2 = RESULTS_DIR / "exp2_lender_borrower.png"
    fig.savefig(png2, dpi=120)
    plt.close(fig)
    print(f"Saved → {png2}")

    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    main()
