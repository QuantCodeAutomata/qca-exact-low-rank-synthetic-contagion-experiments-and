"""
contagion_solver.py — High-level solver interface for finite-N contagion dynamics.

Provides direct and reduced solvers with exposure validation,
direct-vs-reduced equivalence checks, and step-halving validation.
"""

from __future__ import annotations

import numpy as np

from src.model import simulate_direct, threshold_fraction, mean_smoothed_distress
from src.low_rank import (
    build_factorised_exposure,
    simulate_reduced,
    imbalance_coefficient,
)
from src.metrics import (
    sup_scalar_error,
    sup_vector_error_inf,
    sup_feedback_error,
    threshold_fraction_path,
)


# ---------------------------------------------------------------------------
# Main solver entry point
# ---------------------------------------------------------------------------

def run_direct_and_reduced(
    x0: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    mu: float,
    r: float,
    T: float,
    dt: float,
    loss_fn,
) -> dict:
    """Solve both the direct and reduced systems and return paths + errors.

    Parameters
    ----------
    x0      : (N,) initial states
    A       : (N, K) receiver loadings
    B       : (N, K) sender loadings
    mu      : drift μ
    r       : imbalance parameter r
    T       : horizon
    dt      : Euler step Δt
    loss_fn : callable(x) → array

    Returns
    -------
    dict with keys:
      times, states_direct, states_reduced, betas_reduced,
      h_direct, h_reduced,
      err_states_inf, err_states_mean, err_h, err_beta
    """
    E = build_factorised_exposure(A, B)

    # Direct solver
    times, states_direct = simulate_direct(x0, E, mu, r, T, dt, loss_fn)

    # Reduced solver
    _, states_reduced, betas_reduced = simulate_reduced(x0, A, B, mu, r, T, dt, loss_fn)

    # Threshold fractions
    h_direct = threshold_fraction_path(states_direct)
    h_reduced = threshold_fraction_path(states_reduced)

    # Errors
    err_states_inf = sup_vector_error_inf(states_direct, states_reduced)
    err_states_mean = float(np.max(np.abs(states_direct - states_reduced).mean(axis=1)))
    err_h = sup_scalar_error(h_direct, h_reduced)

    return {
        "times": times,
        "states_direct": states_direct,
        "states_reduced": states_reduced,
        "betas_reduced": betas_reduced,
        "h_direct": h_direct,
        "h_reduced": h_reduced,
        "err_states_inf": err_states_inf,
        "err_states_mean": err_states_mean,
        "err_h": err_h,
    }


def validate_direct_reduced_equivalence(
    x0: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    mu: float,
    r: float,
    T: float,
    dt: float,
    loss_fn,
    tol: float = 1e-10,
) -> tuple[bool, float]:
    """Assert machine-precision equivalence between direct and reduced solvers.

    Parameters
    ----------
    tol : tolerance for floating-point equivalence check

    Returns
    -------
    (passed, max_error) : bool and maximum state error
    """
    result = run_direct_and_reduced(x0, A, B, mu, r, T, dt, loss_fn)
    max_err = result["err_states_inf"]
    passed = max_err < tol
    return passed, max_err


# ---------------------------------------------------------------------------
# Step-halving validation
# ---------------------------------------------------------------------------

def step_halving_check(
    x0: np.ndarray,
    E: np.ndarray,
    mu: float,
    r: float,
    T: float,
    dt: float,
    loss_fn,
) -> float:
    """Run at dt and dt/2, return max sup-norm difference.

    Parameters
    ----------
    x0, E, mu, r, T : model parameters
    dt              : reference Euler step
    loss_fn         : callable

    Returns
    -------
    max_discrepancy : float
    """
    _, states_dt = simulate_direct(x0, E, mu, r, T, dt, loss_fn)
    _, states_half = simulate_direct(x0, E, mu, r, T, dt / 2, loss_fn)

    # Align times: states_dt has n+1 points, states_half has 2n+1 points
    # Compare at every other step of the finer grid
    n = states_dt.shape[0]
    states_half_aligned = states_half[::2]   # pick every other row
    # Make sure lengths match
    min_len = min(n, states_half_aligned.shape[0])
    return sup_vector_error_inf(
        states_dt[:min_len],
        states_half_aligned[:min_len],
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_sweep(
    x0_fn,
    A: np.ndarray,
    B: np.ndarray,
    mu: float,
    r: float,
    T: float,
    dt: float,
    loss_fn,
    lambda_E_values: list[float],
    eta: np.ndarray,
    x_star: float,
) -> list[dict]:
    """Run dynamics for multiple λ_E values.

    Parameters
    ----------
    x0_fn        : callable(A, eta, x_star, lambda_E) → (N,) initial states
                   (lambda_E scales the stress scenario)
    lambda_E_values : list of multiplier values

    Returns
    -------
    results : list of dicts with terminal metrics
    """
    results = []
    for lam in lambda_E_values:
        # Rebuild exposure and initial conditions at this intensity
        E = lam * (A @ B.T)
        x0 = x0_fn(A, eta, x_star, lam)
        _, states = simulate_direct(x0, E, mu, r, T, dt, loss_fn)
        h_terminal = float((states[-1] <= 0).mean())
        mean_distress_terminal = float(loss_fn(states[-1]).mean())
        results.append({
            "lambda_E": lam,
            "h_terminal": h_terminal,
            "mean_distress_terminal": mean_distress_terminal,
        })
    return results
