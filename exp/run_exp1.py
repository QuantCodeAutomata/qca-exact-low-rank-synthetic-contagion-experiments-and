"""
run_exp1.py — Exact low-rank synthetic contagion experiments and
direct-vs-reduced closure validation.

Rank-1 two-group Gaussian mixture surrogate (paper parameters unarchived):
  G=2, K=1, p=[0.5, 0.5], m_g=[0.50, 0.45], sigma_g=[0.18, 0.18]
  A_g=[[2.0], [0.6]], B_g=[[0.6], [2.0]], T=2.0, dt=5e-4, mu=0.0, r=0.0

Validates:
  1. Machine-precision direct/reduced solver equivalence
  2. Finite-N threshold fraction convergence to population limit (ODE eq. 74)
  3. Finite-N feedback path convergence
  4. Step-halving Euler consistency
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
from src.metrics import (
    sup_scalar_error,
    sup_vector_error_inf,
    sup_feedback_error,
    threshold_fraction_path,
    group_threshold_fractions,
)
from src import contagion_solver

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

G = 2
K = 1
p = np.array([0.5, 0.5])
m_g = np.array([0.50, 0.45])
sigma_g = np.array([0.18, 0.18])
A_g = np.array([[2.0], [0.6]])
B_g = np.array([[0.6], [2.0]])
T = 2.0
dt = 5e-4
mu = 0.0
r = 0.0
N_VALUES = [400, 1600]

RESULTS_DIR = Path(__file__).parent.parent / "results"
EQUIV_TOL = 1e-10


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 1: Exact low-rank closure validation")
    print(f"  Config: G={G}, K={K}, T={T}, dt={dt}, mu={mu}, r={r}")
    print(f"  N values: {N_VALUES}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Population limit ODE — solved once, shared across all N
    # ------------------------------------------------------------------
    print("\nSolving population limit ODE (eq. 74, r=0) ...")
    times_ode, betas_ode = solve_population_ode(
        p, m_g, sigma_g, A_g, B_g, mu, T, dt
    )
    h_limit = population_threshold_fraction(
        betas_ode, times_ode, p, m_g, sigma_g, A_g, mu
    )
    print(f"  Terminal limit fraction h(T) = {h_limit[-1]:.6f}")

    csv_rows = []
    plot_data = {}

    for N in N_VALUES:
        print(f"\n--- N = {N} ---")

        # Step 3a: Build quantile-matched network
        x0, A, B, group_ids = build_quantile_matched_network(
            N, p, m_g, sigma_g, A_g, B_g
        )
        E = build_factorised_exposure(A, B)

        # Step 3b: Direct solver
        times, states_direct = simulate_direct(x0, E, mu, r, T, dt, loss_hard)

        # Step 3c: Reduced solver
        _, states_reduced, betas_reduced = simulate_reduced(
            x0, A, B, mu, r, T, dt, loss_hard
        )

        # Step 3d: Machine-precision equivalence assertion
        err_dr = sup_vector_error_inf(states_direct, states_reduced)
        assert err_dr < EQUIV_TOL, (
            f"Direct/reduced equivalence FAILED for N={N}: "
            f"max_err = {err_dr:.3e} >= {EQUIV_TOL:.3e}"
        )
        print(f"  Direct/reduced equivalence:  max_err = {err_dr:.3e}  [PASSED]")

        # Step 3f: sup_t |h_N(t) - h_limit(t)|
        h_N = threshold_fraction_path(states_direct)
        err_h = sup_scalar_error(h_N, h_limit)
        print(f"  sup_t |h_N - h_limit|        = {err_h:.6f}")

        # Step 3g: sup_t ||beta_N(t) - beta_limit(t)||_inf
        err_beta = sup_feedback_error(betas_reduced, betas_ode)
        print(f"  sup_t ||beta_N - beta_ode||  = {err_beta:.6f}")

        # Group-resolved threshold fractions
        h_g = group_threshold_fractions(states_direct, group_ids, G)

        # Step 4: Step-halving validation (dt → dt/2)
        disc = contagion_solver.step_halving_check(x0, E, mu, r, T, dt, loss_hard)
        print(f"  Step-halving discrepancy:    {disc:.3e}")

        csv_rows.append({
            "N": N,
            "err_direct_reduced": err_dr,
            "err_h_N_limit": err_h,
            "err_beta_N_limit": err_beta,
            "step_halving_disc": disc,
            "h_terminal_N": float(h_N[-1]),
            "h_terminal_limit": float(h_limit[-1]),
        })

        plot_data[N] = {
            "times": times,
            "h_N": h_N,
            "h_g": h_g,
        }

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = RESULTS_DIR / "exp1_results.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nResults saved → {csv_path}")

    # ------------------------------------------------------------------
    # Formatted error table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(
        f"{'N':>6}  {'err_direct_reduced':>20}  {'err_h':>10}  "
        f"{'err_beta':>10}  {'step_halv':>10}"
    )
    print("-" * 70)
    for row in csv_rows:
        print(
            f"{row['N']:>6}  {row['err_direct_reduced']:>20.3e}  "
            f"{row['err_h_N_limit']:>10.6f}  {row['err_beta_N_limit']:>10.6f}  "
            f"{row['step_halving_disc']:>10.3e}"
        )
    print("=" * 70)

    # ------------------------------------------------------------------
    # Plot 1: Aggregate threshold fraction paths
    # ------------------------------------------------------------------
    colors = {400: "steelblue", 1600: "darkorange"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in N_VALUES:
        ax.plot(
            plot_data[N]["times"], plot_data[N]["h_N"],
            color=colors[N], lw=1.5, label=f"$h_t^N$, $N={N}$",
        )
    ax.plot(times_ode, h_limit, "k--", lw=2.0, label="$h_t$ (ODE limit)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("threshold fraction $h_t$")
    ax.set_title("Threshold fraction paths: finite-$N$ vs population limit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png1 = RESULTS_DIR / "exp1_threshold_fractions.png"
    fig.savefig(png1, dpi=120)
    plt.close(fig)
    print(f"Saved → {png1}")

    # ------------------------------------------------------------------
    # Plot 2: Group-resolved threshold fractions
    # ------------------------------------------------------------------
    group_labels = [
        "Group 0  (high $a$, low $b$)",
        "Group 1  (low $a$, high $b$)",
    ]
    fig, axes = plt.subplots(1, G, figsize=(11, 4), sharey=False)
    for g in range(G):
        ax = axes[g]
        for N in N_VALUES:
            ax.plot(
                plot_data[N]["times"], plot_data[N]["h_g"][:, g],
                color=colors[N], lw=1.5, label=f"$N={N}$",
            )
        ax.set_xlabel("time $t$")
        ax.set_ylabel(f"$h_t^{{(g={g})}}$")
        ax.set_title(group_labels[g])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Group-resolved threshold fractions")
    fig.tight_layout()
    png2 = RESULTS_DIR / "exp1_group_fractions.png"
    fig.savefig(png2, dpi=120)
    plt.close(fig)
    print(f"Saved → {png2}")

    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
