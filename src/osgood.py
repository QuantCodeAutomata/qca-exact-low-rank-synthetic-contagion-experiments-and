"""
osgood.py — Scalar reduced contagion ODE and Osgood-condition experiments (exp_4).

Scalar reduced equation (W≡1, μ=r=0, eq. 75):
    β'(t) = F(β(t)),  β(0) = 0

Positive-ramp regularised equation:
    β'_ε(t) = (1/ε) ∫_0^ε F(β_ε(t) + s) ds,  β_ε(0) = 0

Negative-ramp regularised equation (for minimal-solution selection):
    β'_ε(t) = (1/ε) ∫_{-ε}^0 F(β_ε(t) + s) ds  [minimal selector]
    or equivalently from ℓ_ε^-(x): averaged loss from the left

Test functions:
    F(a) = a                  (linear, Osgood unique)
    F(a) = a log(e/a)         (log-Osgood, unique)
    F(a) = sqrt(a)            (non-Osgood, nonunique)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# F functions and their averaged drifts (closed-form positive ramp)
# ---------------------------------------------------------------------------

def F_linear(a: float | np.ndarray) -> float | np.ndarray:
    """F(a) = a."""
    return a


def F_log_osgood(a: float | np.ndarray) -> float | np.ndarray:
    """F(a) = a log(e/a), with F(0) = 0 by continuity."""
    a = np.asarray(a, dtype=float)
    return np.where(a <= 0.0, 0.0, a * np.log(np.e / a))


def F_sqrt(a: float | np.ndarray) -> float | np.ndarray:
    """F(a) = sqrt(a) for a ≥ 0."""
    a = np.asarray(a, dtype=float)
    return np.sqrt(np.maximum(a, 0.0))


# ---------------------------------------------------------------------------
# Averaged positive-ramp drifts: (1/ε) ∫_0^ε F(β + s) ds
# ---------------------------------------------------------------------------

def averaged_drift_linear(beta: np.ndarray, eps: float) -> np.ndarray:
    """(1/ε) ∫_0^ε (β+s) ds = β + ε/2."""
    return beta + eps / 2.0


def averaged_drift_log(beta: np.ndarray, eps: float) -> np.ndarray:
    """(1/ε) ∫_0^ε (β+s) log(e/(β+s)) ds (closed form).

    ∫_0^ε (β+s) log(e/(β+s)) ds = [-(β+s)^2/2 log((β+s)/e) + (β+s)^2/4]_0^ε
    = -(β+ε)^2/2 log((β+ε)/e) + (β+ε)^2/4 + β^2/2 log(β/e) - β^2/4
    Division by ε gives the average.
    Guard against β+s = 0 by using F(0)=0.
    """
    beta = np.asarray(beta, dtype=float)
    b0 = np.maximum(beta, 0.0)
    b1 = b0 + eps

    def antideriv(x: np.ndarray) -> np.ndarray:
        # Antiderivative of x log(e/x): -(x^2/2) log(x/e) + x^2/4
        safe_x = np.where(x > 0, x, 1.0)
        val = -(safe_x ** 2 / 2.0) * np.log(safe_x / np.e) + safe_x ** 2 / 4.0
        return np.where(x <= 0, 0.0, val)

    return (antideriv(b1) - antideriv(b0)) / eps


def averaged_drift_sqrt(beta: np.ndarray, eps: float) -> np.ndarray:
    """(1/ε) ∫_0^ε sqrt(β+s) ds = (2/(3ε)) [(β+ε)^{3/2} - β^{3/2}]."""
    beta = np.asarray(beta, dtype=float)
    b0 = np.maximum(beta, 0.0)
    b1 = b0 + eps
    return (2.0 / (3.0 * eps)) * (b1 ** 1.5 - b0 ** 1.5)


# ---------------------------------------------------------------------------
# Negative-ramp averaged drift (minimal solution selector)
# ---------------------------------------------------------------------------

def averaged_drift_sqrt_negative(beta: np.ndarray, eps: float) -> np.ndarray:
    """Negative-ramp averaged drift for F_sqrt: (1/ε) ∫_{-ε}^0 F(β+s) ds.

    = (1/ε) ∫_{-ε}^0 sqrt(β+s) ds, where β+s may dip below 0.
    Guard: sqrt(max(0, β+s)).
    = (2/(3ε)) [max(0,β)^{3/2} - max(0, β-ε)^{3/2}]
    """
    beta = np.asarray(beta, dtype=float)
    upper = np.maximum(beta, 0.0)
    lower = np.maximum(beta - eps, 0.0)
    return (2.0 / (3.0 * eps)) * (upper ** 1.5 - lower ** 1.5)


# ---------------------------------------------------------------------------
# ODE solvers using DOP853
# ---------------------------------------------------------------------------

def _make_rhs(averaged_drift_fn, eps: float):
    """Return a scipy-compatible RHS from an averaged drift function."""
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return averaged_drift_fn(y, eps)
    return rhs


def solve_regularised_ode(
    averaged_drift_fn,
    eps: float,
    T: float = 1.0,
    n_output: int = 2001,
    rtol: float = 2e-12,
    atol: float = 2e-14,
    max_step: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve β'_ε = averaged_drift_fn(β, ε) with DOP853 on [0, T].

    Parameters
    ----------
    averaged_drift_fn : callable(beta_array, eps) → array
    eps               : ramp width
    T                 : horizon
    n_output          : number of output times
    rtol, atol        : tolerances
    max_step          : maximum step size

    Returns
    -------
    t_eval : (n_output,) evaluation times
    beta   : (n_output,) regularised path β_ε(t)
    """
    t_eval = np.linspace(0.0, T, n_output)
    rhs = _make_rhs(averaged_drift_fn, eps)

    sol = solve_ivp(
        rhs,
        (0.0, T),
        [0.0],
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"DOP853 solve failed: {sol.message}")
    return sol.t, sol.y[0]


# ---------------------------------------------------------------------------
# Exact hard solutions (theoretical benchmarks)
# ---------------------------------------------------------------------------

def beta_hard_linear(t: np.ndarray) -> np.ndarray:
    """Greatest hard solution for F(a)=a: β^+(t) = exp(t) - 1 ... 
    
    Actually β' = β, β(0)=0 → β(t) = 0 (trivially), but the averaged drift
    pushes to the nontrivial solution.
    
    NOTE: For β'=β with β(0)=0 the unique solution is β(t)=0.
    The positive-ramp adds a constant offset ε/2, yielding a positive trajectory.
    The paper's benchmark β^+ is obtained as the limit of the ε-regularised path.
    For the linear case, β'_ε = β + ε/2 → β^+(t) = (ε/2)(e^t - 1)/... 
    but actually the hard equation β'=β,β(0)=0 has only the trivial solution.
    
    Instead, interpret: x_0 = F^{-1} mapping → x_0 ~ Uniform[0,1] so β^+(t)
    is the non-trivial solution. For the linear F(a)=a the hard ODE  
    with the initial mass has β^+(t) = ?  
    
    From the paper: the scalar reduced ODE is β' = F(β) from eq (75).
    The hard equation: β'=β, β(0)=0 → β=0 is the unique (trivial) solution.
    But the regularised drift β'_ε = β+ε/2 → β_ε(t) = (ε/2)(e^t - 1).
    As ε→0, β_ε → 0. So in the linear case β^+ = 0.
    The fitted slope of max_t|β_ε - β^+| = max_t (ε/2)(e^t-1) ≈ (ε/2)(e^1-1) ~ ε × const,
    slope ~1.0.
    """
    return np.zeros_like(t)


def beta_hard_log(t: np.ndarray) -> np.ndarray:
    """Greatest hard solution for F(a)=a log(e/a): β^+(t) = 0 (trivial).

    The positive-ramp solution will approximate this from above.
    Convergence slope ~0.365.
    """
    return np.zeros_like(t)


def beta_hard_sqrt_greatest(t: np.ndarray) -> np.ndarray:
    """Greatest hard solution for F(a)=sqrt(a): β^+(t) = t^2/4."""
    return t ** 2 / 4.0


def beta_hard_sqrt_minimal(t: np.ndarray) -> np.ndarray:
    """Minimal hard solution for F(a)=sqrt(a): β^-(t) = 0."""
    return np.zeros_like(t)
