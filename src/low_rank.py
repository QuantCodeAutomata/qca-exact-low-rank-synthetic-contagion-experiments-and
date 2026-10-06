"""
low_rank.py — Factorised exposure matrix construction and reduced K-dimensional
feedback system for exact low-rank contagion dynamics (equations 7–12, 74).

Exposure factorisation:  e_ij^N = (1/K) Σ_k a_{i,k}^N b_{j,k}^N  (eq. 7)
Reduced feedback:        β_k^N(t) = ∫_0^t m_k^N(s) ds,
                         m_k^N(t) = (1/N) Σ_j b_{j,k}^N ℓ(X_t^{j,N})

Imbalance coefficient:   Λ_i^N = (r/K) Σ_k (a_{i,k}^N b̄_k^N - b_{i,k}^N ā_k^N)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as scipy_norm

from src.model import loss_hard, threshold_fraction


# ---------------------------------------------------------------------------
# Exposure matrix construction
# ---------------------------------------------------------------------------

def build_factorised_exposure(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Build the rank-K factorised exposure matrix E with 1/N scaling applied later.

    e_ij = (1/K) Σ_k a_{i,k} b_{j,k}  — equation (7)

    E[i,j] = (1/K) A[i,:] @ B[j,:]

    Parameters
    ----------
    A : (N, K) receiver loadings
    B : (N, K) sender loadings

    Returns
    -------
    E : (N, N) exposure matrix (dense)
    """
    K = A.shape[1]
    return (A @ B.T) / K


def imbalance_coefficient(
    A: np.ndarray,
    B: np.ndarray,
    r: float,
) -> np.ndarray:
    """Per-institution imbalance Λ_i^N (equation in methodology section).

    Λ_i = (r/K) Σ_k (a_{i,k} b̄_k - b_{i,k} ā_k)
    where ā_k = (1/N) Σ_i a_{i,k}, b̄_k = (1/N) Σ_i b_{i,k}.

    Parameters
    ----------
    A : (N, K) receiver loadings
    B : (N, K) sender loadings
    r : imbalance parameter

    Returns
    -------
    Lambda : (N,) imbalance vector
    """
    K = A.shape[1]
    a_bar = A.mean(axis=0)    # (K,)
    b_bar = B.mean(axis=0)    # (K,)
    # Λ_i = (r/K) Σ_k (a_{i,k} b̄_k - b_{i,k} ā_k)
    Lambda = (r / K) * ((A * b_bar).sum(axis=1) - (B * a_bar).sum(axis=1))
    return Lambda


# ---------------------------------------------------------------------------
# Reduced K-dimensional feedback solver
# ---------------------------------------------------------------------------

def simulate_reduced(
    x0: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    mu: float,
    r: float,
    T: float,
    dt: float,
    loss_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduced finite-N solver using K feedback coordinates β_k^N(t).

    State reconstruction:
        X_t^i = x0_i + μt + t Λ_i - (1/K) Σ_k a_{i,k} β_k^N(t)

    Feedback update:
        β_k^{n+1} = β_k^n + Δt m_k^n,   m_k^n = (1/N) Σ_j b_{j,k} ℓ(X_j^n)

    Parameters
    ----------
    x0      : (N,) initial states
    A       : (N, K) receiver loadings
    B       : (N, K) sender loadings
    mu      : drift μ
    r       : imbalance parameter
    T       : horizon
    dt      : Euler step Δt
    loss_fn : callable(x) → array

    Returns
    -------
    times  : (n_steps+1,) times
    states : (n_steps+1, N) reconstructed states
    betas  : (n_steps+1, K) feedback coordinates β_k^N(t)
    """
    N, K = A.shape
    n_steps = int(round(T / dt))
    times = np.linspace(0.0, T, n_steps + 1)

    Lambda = imbalance_coefficient(A, B, r)   # (N,)

    beta = np.zeros(K)          # β_k^N(0) = 0
    betas = np.zeros((n_steps + 1, K))
    states = np.zeros((n_steps + 1, N))

    # Initial state
    states[0] = x0.copy()

    for step in range(n_steps):
        X_now = states[step]
        t_new = times[step + 1]

        # Compute m_k^n = (1/N) Σ_j b_{j,k} ℓ(X_j^n)
        lv = loss_fn(X_now)         # (N,)
        m_k = (B * lv[:, None]).mean(axis=0)   # (K,)

        # Update β
        beta = beta + dt * m_k
        betas[step + 1] = beta

        # Reconstruct state
        # X_i = x0_i + μt + t Λ_i - (1/K) Σ_k a_{i,k} β_k
        contagion_effect = (A @ beta) / K      # (N,)
        states[step + 1] = x0 + mu * t_new + t_new * Lambda - contagion_effect

    return times, states, betas


# ---------------------------------------------------------------------------
# Population limit ODE (equation 74) — Gaussian mixture, r=0
# ---------------------------------------------------------------------------

def gaussian_mixture_ode_rhs(
    beta: np.ndarray,
    t: float,
    p: np.ndarray,
    m_g: np.ndarray,
    sigma_g: np.ndarray,
    A_g: np.ndarray,
    B_g: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Right-hand side of the rank-K population ODE (eq. 74), r=0.

    dβ_k/dt = Σ_g p_g b_{g,k} Φ((-m_g - μt + (1/K) Σ_ℓ a_{g,ℓ} β_ℓ(t)) / σ_g)

    Parameters
    ----------
    beta    : (K,) current feedback coordinates
    t       : current time
    p       : (G,) group probabilities
    m_g     : (G,) group means
    sigma_g : (G,) group standard deviations
    A_g     : (G, K) group receiver loadings
    B_g     : (G, K) group sender loadings
    mu      : drift μ

    Returns
    -------
    dbeta : (K,) time derivative
    """
    K = len(beta)
    G = len(p)
    # Argument for CDF: (-m_g - μt + (1/K) Σ_ℓ a_{g,ℓ} β_ℓ) / σ_g
    contagion_shift = (A_g @ beta) / K    # (G,)
    arg = (-m_g - mu * t + contagion_shift) / sigma_g   # (G,)
    Phi = scipy_norm.cdf(arg)              # (G,)

    # dβ_k/dt = Σ_g p_g b_{g,k} Φ_g
    # dbeta[k] = Σ_g p_g B_g[g,k] Phi[g]
    dbeta = (B_g * (p * Phi)[:, None]).sum(axis=0)   # (K,)
    return dbeta


def solve_population_ode(
    p: np.ndarray,
    m_g: np.ndarray,
    sigma_g: np.ndarray,
    A_g: np.ndarray,
    B_g: np.ndarray,
    mu: float,
    T: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the rank-K population ODE (eq. 74) with explicit Euler.

    Parameters
    ----------
    p       : (G,) group probabilities
    m_g     : (G,) group initial means
    sigma_g : (G,) group standard deviations
    A_g     : (G, K) group receiver loadings
    B_g     : (G, K) group sender loadings
    mu      : drift μ
    T       : horizon
    dt      : Euler step

    Returns
    -------
    times : (n_steps+1,) times
    betas : (n_steps+1, K) feedback path β_k(t)
    """
    K = B_g.shape[1]
    n_steps = int(round(T / dt))
    times = np.linspace(0.0, T, n_steps + 1)

    beta = np.zeros(K)
    betas = np.zeros((n_steps + 1, K))

    for step in range(n_steps):
        t = times[step]
        rhs = gaussian_mixture_ode_rhs(beta, t, p, m_g, sigma_g, A_g, B_g, mu)
        beta = beta + dt * rhs
        betas[step + 1] = beta

    return times, betas


def population_threshold_fraction(
    betas: np.ndarray,
    times: np.ndarray,
    p: np.ndarray,
    m_g: np.ndarray,
    sigma_g: np.ndarray,
    A_g: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Compute limiting threshold fraction h_t = Σ_g p_g Φ((-m_g - μt + (1/K) Σ_k a_{g,k} β_k(t))/σ_g).

    Parameters
    ----------
    betas  : (n_steps+1, K) feedback path
    times  : (n_steps+1,) times
    p      : (G,) group probabilities
    m_g    : (G,) group means
    sigma_g: (G,) group standard deviations
    A_g    : (G, K) group receiver loadings
    mu     : drift

    Returns
    -------
    h : (n_steps+1,) threshold fraction path
    """
    K = betas.shape[1]
    n = len(times)
    h = np.zeros(n)
    for step in range(n):
        beta = betas[step]
        t = times[step]
        contagion_shift = (A_g @ beta) / K
        arg = (-m_g - mu * t + contagion_shift) / sigma_g
        Phi = scipy_norm.cdf(arg)
        h[step] = (p * Phi).sum()
    return h
