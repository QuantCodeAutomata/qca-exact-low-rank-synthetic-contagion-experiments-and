"""
model.py — Loss functions and state update primitives for finite-network
contagion dynamics (equations 6–7 of the paper).

Hard threshold loss:  ℓ(x) = 1_{x ≤ 0}
Positive ramp loss:   ℓ_ε^+(x) = 1 for x≤0, 1-x/ε for 0<x<ε, 0 for x≥ε
Negative ramp loss:   ℓ_ε^-(x) = max{0, min{1, -x/ε}}
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def loss_hard(x: np.ndarray) -> np.ndarray:
    """Hard threshold indicator: ℓ(x) = 1_{x ≤ 0}.

    Parameters
    ----------
    x : array of state values

    Returns
    -------
    indicator array in {0.0, 1.0}
    """
    return (x <= 0.0).astype(float)


def loss_positive_ramp(x: np.ndarray, eps: float) -> np.ndarray:
    """Positive-side ramp regularisation of the hard indicator.

    ℓ_ε^+(x) = 1          for x ≤ 0
              = 1 - x/ε   for 0 < x < ε
              = 0          for x ≥ ε

    Parameters
    ----------
    x   : array of state values
    eps : ramp half-width ε > 0

    Returns
    -------
    smoothed indicator values in [0, 1]
    """
    assert eps > 0, "eps must be positive"
    result = np.where(x <= 0.0, 1.0,
             np.where(x < eps, 1.0 - x / eps, 0.0))
    return result


def loss_negative_ramp(x: np.ndarray, eps: float) -> np.ndarray:
    """Negative-side ramp regularisation: ℓ_ε^-(x) = max{0, min{1, -x/ε}}.

    Parameters
    ----------
    x   : array of state values
    eps : ramp half-width ε > 0

    Returns
    -------
    smoothed indicator values in [0, 1]
    """
    assert eps > 0, "eps must be positive"
    return np.clip(-x / eps, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Direct finite-N Euler solver (equation 6)
# ---------------------------------------------------------------------------

def simulate_direct(
    x0: np.ndarray,
    E: np.ndarray,
    mu: float,
    r: float,
    T: float,
    dt: float,
    loss_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler solver for the direct finite-N contagion system (eq. 6).

    X_t^{i,N} = x_i^N + μt + (rt/N) Σ_j(e_ij - e_ji)
                - (1/N) Σ_j e_ij ∫_0^t ℓ(X_s^{j,N}) ds

    Equivalently, track H_j(t) = ∫_0^t ℓ(X_s^j) ds with
        H_j^{n+1} = H_j^n + Δt ℓ(X_j^n)
        X_j^{n+1} = x0_j + μ(n+1)Δt + (r(n+1)Δt/N) Σ_k(E_jk - E_kj)
                    - (1/N) Σ_k E_jk H_k^{n+1}

    Parameters
    ----------
    x0      : (N,) initial states
    E       : (N,N) exposure matrix, e_ij at position [i,j]
    mu      : drift μ
    r       : imbalance coefficient r
    T       : horizon
    dt      : Euler step Δt
    loss_fn : callable(x) → array of same shape as x

    Returns
    -------
    times  : (n_steps+1,) array of recorded times
    states : (n_steps+1, N) array of state trajectories
    """
    N = len(x0)
    n_steps = int(round(T / dt))
    times = np.linspace(0.0, T, n_steps + 1)

    # Precompute imbalance per node: Λ_i = (r/N) Σ_j (E_ij - E_ji)
    row_sum = E.sum(axis=1)          # (N,)  Σ_j e_ij
    col_sum = E.sum(axis=0)          # (N,)  Σ_j e_ji
    Lambda = (r / N) * (row_sum - col_sum)   # (N,)

    # Cumulative distress integrals H_j(t)
    H = np.zeros(N)
    states = np.zeros((n_steps + 1, N))
    states[0] = x0.copy()

    for step in range(n_steps):
        t_now = times[step]
        X_now = states[step]

        # Update cumulative distress
        H = H + dt * loss_fn(X_now)

        # Advance time
        t_new = times[step + 1]
        # Reconstruct state
        # X_i = x0_i + μt + t Λ_i - (1/N) Σ_j E_ij H_j
        loss_effect = (E @ H) / N          # (N,)
        states[step + 1] = x0 + mu * t_new + t_new * Lambda - loss_effect

    return times, states


# ---------------------------------------------------------------------------
# Threshold fraction helper
# ---------------------------------------------------------------------------

def threshold_fraction(states: np.ndarray) -> np.ndarray:
    """Compute hard threshold fraction h_t = (1/N) Σ_i 1_{X_t^i ≤ 0}.

    Parameters
    ----------
    states : (n_steps+1, N) or (N,) state array

    Returns
    -------
    h : (n_steps+1,) or scalar threshold fraction
    """
    return (states <= 0.0).mean(axis=-1)


def mean_smoothed_distress(states: np.ndarray, loss_fn) -> np.ndarray:
    """Compute (1/N) Σ_i ℓ(X_t^i) for a given loss function.

    Parameters
    ----------
    states  : (n_steps+1, N) array
    loss_fn : callable

    Returns
    -------
    d : (n_steps+1,) mean smoothed distress
    """
    return np.array([loss_fn(states[t]).mean() for t in range(states.shape[0])])
