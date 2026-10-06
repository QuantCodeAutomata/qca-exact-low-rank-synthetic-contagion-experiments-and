"""
run_exp4.py — Experiment 4: Osgood boundary and hard-solution selection.

For each F in {F_linear, F_log_osgood, F_sqrt}:
  • Solve the positive-ramp regularised ODE β'_ε = averaged_drift(β, ε)
    for eps in {2^{-4}, 2^{-5}, …, 2^{-12}}.
  • Compare against the greatest hard solution β^+(t) and fit a log-log slope.
  • Expected slopes: linear ≈ 1.0000, log-Osgood ≈ 0.3650, sqrt ≈ 0.4997.

For F_sqrt also solve the negative-ramp ODE and verify convergence to β^- = 0.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from osgood import (
    F_linear,
    F_log_osgood,
    F_sqrt,
    averaged_drift_linear,
    averaged_drift_log,
    averaged_drift_sqrt,
    averaged_drift_sqrt_negative,
    solve_regularised_ode,
    beta_hard_linear,
    beta_hard_log,
    beta_hard_sqrt_greatest,
    beta_hard_sqrt_minimal,
)
from metrics import loglog_slope

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Solver settings (shared across all runs)
T = 1.0
N_OUTPUT = 2001
SOLVER_KWARGS = dict(rtol=2e-12, atol=2e-14, max_step=0.002)

# eps grid: 2^{-4}, 2^{-5}, …, 2^{-12}
EPS_VALS = 2.0 ** np.arange(-4, -13, -1)


# ---------------------------------------------------------------------------
# Core convergence routine
# ---------------------------------------------------------------------------

def run_convergence_study(averaged_drift_fn, beta_hard_fn, eps_vals):
    """Solve the regularised ODE for each eps and compute max-time error.

    Parameters
    ----------
    averaged_drift_fn : callable(beta_array, eps) → array
    beta_hard_fn      : callable(t_array) → array  (hard benchmark)
    eps_vals          : sequence of eps values

    Returns
    -------
    errors : (len(eps_vals),) array of max_t |beta_eps(t) - beta^+(t)|
    paths  : dict  eps → (t_array, beta_array)
    """
    errors = []
    paths = {}
    for eps in eps_vals:
        t, beta = solve_regularised_ode(
            averaged_drift_fn,
            eps,
            T=T,
            n_output=N_OUTPUT,
            **SOLVER_KWARGS,
        )
        beta_ref = beta_hard_fn(t)
        err = float(np.max(np.abs(beta - beta_ref)))
        errors.append(err)
        paths[eps] = (t, beta)
    return np.array(errors), paths


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _safe_loglog_slope(eps_vals, errors):
    """Wrapper that returns nan for series with all-zero (or near-zero) errors."""
    if np.all(errors == 0.0) or np.any(~np.isfinite(errors)):
        return float("nan"), float("nan")
    return loglog_slope(eps_vals, errors)


def _plot_loglog(ax, eps_vals, errors, gamma, label, marker):
    """Draw a log-log error line on ax, skipping series with all-zero errors."""
    if np.all(errors == 0.0):
        ax.annotate(
            f"{label}: errors exactly 0 (exact minimal solution)",
            xy=(0.05, 0.05), xycoords="axes fraction", fontsize=8,
        )
        return
    gamma_str = f"{gamma:.4f}" if np.isfinite(gamma) else "exact"
    ax.loglog(eps_vals, errors, marker + "-", label=f"{label}  (slope={gamma_str})")


def save_loglog_plot(eps_vals, results_dict):
    """Log-log convergence plot for all F functions.

    Parameters
    ----------
    results_dict : dict  name → (errors, gamma)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"F_linear": "o", "F_log_osgood": "s", "F_sqrt_pos": "^", "F_sqrt_neg": "v"}
    for name, (errors, gamma) in results_dict.items():
        marker = markers.get(name, "D")
        _plot_loglog(ax, eps_vals, errors, gamma, name, marker)

    # Reference lines
    eps_plot = np.array([eps_vals[0], eps_vals[-1]])
    ax.loglog(eps_plot, 0.5 * eps_plot ** 1.0, "k--", lw=0.8, label="slope 1")
    ax.loglog(eps_plot, 0.5 * eps_plot ** 0.5, "k:",  lw=0.8, label="slope 0.5")

    ax.set_xlabel("eps")
    ax.set_ylabel(r"$\max_t\,|\beta_\varepsilon(t) - \beta^+(t)|$")
    ax.set_title("Experiment 4 – Convergence of regularised ODE to hard solution")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_loglog_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_path_plots(eps_vals, paths_dict, hard_fns_dict):
    """Time-path plots for selected eps values.

    Parameters
    ----------
    paths_dict    : dict  name → paths  (paths = dict eps → (t, beta))
    hard_fns_dict : dict  name → hard_fn callable
    """
    # Select three representative eps values: smallest, middle, largest
    sel_eps = [eps_vals[0], eps_vals[len(eps_vals) // 2], eps_vals[-1]]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    t_ref = np.linspace(0, T, N_OUTPUT)
    names = list(paths_dict.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), sharey=False)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        hard_fn = hard_fns_dict[name]
        paths = paths_dict[name]
        ax.plot(t_ref, hard_fn(t_ref), "k--", lw=2, label=r"$\beta^+$")
        for eps, col in zip(sel_eps, colors):
            t, beta = paths[eps]
            ax.plot(t, beta, color=col, lw=1.5, label=f"eps={eps:.4f}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\beta(t)$")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Experiment 4 – Regularised ODE paths vs hard solution")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_time_paths.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_sqrt_negative_plot(eps_vals, paths_sqrt_neg):
    """Time-path plot for the F_sqrt negative-ramp ODE."""
    sel_eps = [eps_vals[0], eps_vals[len(eps_vals) // 2], eps_vals[-1]]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    t_ref = np.linspace(0, T, N_OUTPUT)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="k", linestyle="--", lw=2, label=r"$\beta^- = 0$")
    for eps, col in zip(sel_eps, colors):
        t, beta = paths_sqrt_neg[eps]
        ax.plot(t, beta, color=col, lw=1.5, label=f"eps={eps:.4f}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\beta_\varepsilon^-(t)$")
    ax.set_title(r"$F_{\sqrt{\cdot}}$ negative ramp: convergence to $\beta^- = 0$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_sqrt_negative.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(eps_vals, all_results):
    """Save error table to CSV.

    Parameters
    ----------
    all_results : dict  name → (errors, gamma)
    """
    filepath = os.path.join(RESULTS_DIR, "exp4_results.csv")
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["F_name", "eps", "max_error", "fitted_slope"])
        for name, (errors, gamma) in all_results.items():
            slope_str = (f"{gamma:.4f}" if np.isfinite(gamma) else "exact")
            for i, eps in enumerate(eps_vals):
                w.writerow([
                    name,
                    f"{eps:.8e}",
                    f"{errors[i]:.6e}",
                    slope_str if i == 0 else "",
                ])
    print(f"  CSV saved to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Experiment 4: Osgood boundary and hard-solution selection")
    print("=" * 65)
    print(f"\neps grid: {EPS_VALS}")
    print(f"Solver:   DOP853, rtol={SOLVER_KWARGS['rtol']}, atol={SOLVER_KWARGS['atol']}, "
          f"max_step={SOLVER_KWARGS['max_step']}, n_output={N_OUTPUT}, T={T}")

    eps_vals = EPS_VALS

    # ---- F_linear ----
    print("\n[1] F_linear — positive ramp → beta^+(t) = 0")
    errors_linear, paths_linear = run_convergence_study(
        averaged_drift_linear, beta_hard_linear, eps_vals
    )
    _, gamma_lin = _safe_loglog_slope(eps_vals, errors_linear)
    print(f"    Fitted log-log slope: {gamma_lin:.4f}  (expected ~1.0000)")

    # ---- F_log_osgood ----
    print("\n[2] F_log_osgood — positive ramp → beta^+(t) = 0")
    errors_log, paths_log = run_convergence_study(
        averaged_drift_log, beta_hard_log, eps_vals
    )
    _, gamma_log = _safe_loglog_slope(eps_vals, errors_log)
    print(f"    Fitted log-log slope: {gamma_log:.4f}  (expected ~0.3650)")

    # ---- F_sqrt (positive ramp) ----
    print("\n[3] F_sqrt (positive ramp) — beta^+(t) = t²/4")
    errors_sqrt, paths_sqrt = run_convergence_study(
        averaged_drift_sqrt, beta_hard_sqrt_greatest, eps_vals
    )
    _, gamma_sqrt = _safe_loglog_slope(eps_vals, errors_sqrt)
    print(f"    Fitted log-log slope: {gamma_sqrt:.4f}  (expected ~0.4997)")

    # ---- F_sqrt (negative ramp) ----
    print("\n[4] F_sqrt (negative ramp) — beta^-(t) = 0")
    errors_sqrt_neg, paths_sqrt_neg = run_convergence_study(
        averaged_drift_sqrt_negative, beta_hard_sqrt_minimal, eps_vals
    )
    _, gamma_sqrt_neg = _safe_loglog_slope(eps_vals, errors_sqrt_neg)
    slope_str = f"{gamma_sqrt_neg:.4f}" if np.isfinite(gamma_sqrt_neg) else "exact (all errors = 0)"
    print(f"    Fitted log-log slope: {slope_str}")

    # ---- Summary table ----
    print("\n  Summary of fitted slopes:")
    print(f"  {'F':20s}  {'slope':>8}  {'expected':>10}")
    print(f"  {'-'*44}")
    expected = {
        "F_linear":      "~1.0000",
        "F_log_osgood":  "~0.3650",
        "F_sqrt_pos":    "~0.4997",
        "F_sqrt_neg":    "(select β⁻)",
    }
    def _fmt_gamma(g):
        return f"{g:8.4f}" if np.isfinite(g) else "    exact"

    for name, gamma in [
        ("F_linear",     gamma_lin),
        ("F_log_osgood", gamma_log),
        ("F_sqrt_pos",   gamma_sqrt),
        ("F_sqrt_neg",   gamma_sqrt_neg),
    ]:
        print(f"  {name:20s}  {_fmt_gamma(gamma)}  {expected[name]:>10}")

    # ---- Convergence detail printout ----
    print("\n  Detailed error tables:")
    for name, errors, gamma in [
        ("F_linear",     errors_linear,   gamma_lin),
        ("F_log_osgood", errors_log,      gamma_log),
        ("F_sqrt_pos",   errors_sqrt,     gamma_sqrt),
        ("F_sqrt_neg",   errors_sqrt_neg, gamma_sqrt_neg),
    ]:
        slope_label = f"{gamma:.4f}" if np.isfinite(gamma) else "exact"
        print(f"\n  {name}  (slope={slope_label})")
        print(f"    {'eps':>12}  {'max_error':>14}")
        for eps, err in zip(eps_vals, errors):
            print(f"    {eps:12.6e}  {err:14.6e}")

    # ---- Save outputs ----
    print("\nSaving plots and CSV...")

    all_results = {
        "F_linear":     (errors_linear,   gamma_lin),
        "F_log_osgood": (errors_log,      gamma_log),
        "F_sqrt_pos":   (errors_sqrt,     gamma_sqrt),
        "F_sqrt_neg":   (errors_sqrt_neg, gamma_sqrt_neg),
    }

    save_loglog_plot(eps_vals, all_results)

    save_path_plots(
        eps_vals,
        paths_dict={
            "F_linear":     paths_linear,
            "F_log_osgood": paths_log,
            "F_sqrt_pos":   paths_sqrt,
        },
        hard_fns_dict={
            "F_linear":     beta_hard_linear,
            "F_log_osgood": beta_hard_log,
            "F_sqrt_pos":   beta_hard_sqrt_greatest,
        },
    )

    save_sqrt_negative_plot(eps_vals, paths_sqrt_neg)

    save_csv(eps_vals, all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
