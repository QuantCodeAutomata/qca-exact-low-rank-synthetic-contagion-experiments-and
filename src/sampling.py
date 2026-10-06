"""
sampling.py — Deterministic quantile-matched and iid Monte Carlo constructions
for finite-N synthetic low-rank networks.

Quantile matching:
  - Allocate group counts n_g by rounding N * p_g to integers summing to N.
  - Within group g, assign initial conditions via Gaussian quantiles:
      x_{g,m} = m_g + σ_g Φ^{-1}((m - 0.5) / n_g), m = 1, ..., n_g.
  - Assign deterministic factor loadings a_{g,k}, b_{g,k} to each institution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as scipy_norm


# ---------------------------------------------------------------------------
# Group count allocation
# ---------------------------------------------------------------------------

def allocate_group_counts(N: int, p: np.ndarray) -> np.ndarray:
    """Allocate group counts n_g ≈ N p_g with integers summing exactly to N.

    Strategy: round N*p_g and then adjust the largest-remainder group to
    ensure the total equals N.

    Parameters
    ----------
    N : total sample size
    p : (G,) group probabilities, must sum to 1

    Returns
    -------
    n_g : (G,) integer group counts summing to N
    """
    G = len(p)
    assert abs(p.sum() - 1.0) < 1e-10, "Group probabilities must sum to 1"
    assert all(p > 0), "All group probabilities must be positive"

    raw = N * p
    floored = np.floor(raw).astype(int)
    remainder = N - floored.sum()

    # Largest-remainder rule: add 1 to groups with largest fractional parts
    fracs = raw - floored
    indices = np.argsort(-fracs)
    for i in range(remainder):
        floored[indices[i]] += 1

    assert floored.sum() == N, "Group counts do not sum to N"
    return floored


# ---------------------------------------------------------------------------
# Quantile-matched initial conditions
# ---------------------------------------------------------------------------

def quantile_initial_conditions(
    n_g: np.ndarray,
    m_g: np.ndarray,
    sigma_g: np.ndarray,
) -> np.ndarray:
    """Generate Gaussian-quantile initial conditions within each group.

    x_{g,m} = m_g + σ_g Φ^{-1}((m - 0.5) / n_g), m = 1, ..., n_g

    Parameters
    ----------
    n_g     : (G,) group counts
    m_g     : (G,) group means
    sigma_g : (G,) group standard deviations

    Returns
    -------
    x0 : (N,) initial conditions (concatenated across groups)
    """
    x_groups = []
    for g, ng in enumerate(n_g):
        quantile_positions = (np.arange(1, ng + 1) - 0.5) / ng
        x_g = m_g[g] + sigma_g[g] * scipy_norm.ppf(quantile_positions)
        x_groups.append(x_g)
    return np.concatenate(x_groups)


def build_quantile_matched_network(
    N: int,
    p: np.ndarray,
    m_g: np.ndarray,
    sigma_g: np.ndarray,
    A_g: np.ndarray,
    B_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct full quantile-matched finite-N network.

    Parameters
    ----------
    N       : total sample size
    p       : (G,) group probabilities
    m_g     : (G,) group initial means
    sigma_g : (G,) group standard deviations
    A_g     : (G, K) group receiver loadings
    B_g     : (G, K) group sender loadings

    Returns
    -------
    x0      : (N,) initial states
    A       : (N, K) receiver loadings per institution
    B       : (N, K) sender loadings per institution
    group_ids : (N,) integer group membership array
    """
    n_g = allocate_group_counts(N, p)
    x0 = quantile_initial_conditions(n_g, m_g, sigma_g)

    G, K = A_g.shape
    A = np.zeros((N, K))
    B = np.zeros((N, K))
    group_ids = np.zeros(N, dtype=int)

    idx = 0
    for g, ng in enumerate(n_g):
        A[idx: idx + ng] = A_g[g]
        B[idx: idx + ng] = B_g[g]
        group_ids[idx: idx + ng] = g
        idx += ng

    return x0, A, B, group_ids


# ---------------------------------------------------------------------------
# IID Monte Carlo sampling
# ---------------------------------------------------------------------------

def iid_type_sample(
    N: int,
    p: np.ndarray,
    m_g: np.ndarray,
    sigma_g: np.ndarray,
    A_g: np.ndarray,
    B_g: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw iid types from the group mixture distribution.

    Parameters
    ----------
    N       : sample size
    p       : (G,) group probabilities
    m_g     : (G,) group means
    sigma_g : (G,) group standard deviations
    A_g     : (G, K) group receiver loadings
    B_g     : (G, K) group sender loadings
    rng     : numpy random generator

    Returns
    -------
    x0        : (N,) initial states
    A         : (N, K) receiver loadings
    B         : (N, K) sender loadings
    group_ids : (N,) group membership
    """
    G, K = A_g.shape
    group_ids = rng.choice(G, size=N, p=p)

    x0 = np.array([
        rng.normal(m_g[g], sigma_g[g]) for g in group_ids
    ])
    A = A_g[group_ids]
    B = B_g[group_ids]

    return x0, A, B, group_ids


# ---------------------------------------------------------------------------
# Empirical resampling from a type law
# ---------------------------------------------------------------------------

def iid_sample_network(
    N: int,
    x0_pop: np.ndarray,
    A_pop: np.ndarray,
    B_pop: np.ndarray,
    ids_pop: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample N institutions iid from an empirical finite population with group IDs.

    Parameters
    ----------
    N       : sample size
    x0_pop  : (M,) population initial states
    A_pop   : (M, K) population receiver loadings
    B_pop   : (M, K) population sender loadings
    ids_pop : (M,) population group membership IDs
    rng     : numpy random generator

    Returns
    -------
    x0_s  : (N,) resampled initial states
    A_s   : (N, K) resampled receiver loadings
    B_s   : (N, K) resampled sender loadings
    ids_s : (N,) resampled group IDs
    """
    M = len(x0_pop)
    indices = rng.choice(M, size=N, replace=True)
    return x0_pop[indices], A_pop[indices], B_pop[indices], ids_pop[indices]


def resample_from_empirical(
    N: int,
    x0_pop: np.ndarray,
    A_pop: np.ndarray,
    B_pop: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample N institutions iid from the empirical type distribution.

    Parameters
    ----------
    N       : resample size
    x0_pop  : (M,) population initial states
    A_pop   : (M, K) population receiver loadings
    B_pop   : (M, K) population sender loadings
    rng     : numpy random generator

    Returns
    -------
    x0_s : (N,) resampled initial states
    A_s  : (N, K) resampled receiver loadings
    B_s  : (N, K) resampled sender loadings
    """
    M = len(x0_pop)
    indices = rng.choice(M, size=N, replace=True)
    return x0_pop[indices], A_pop[indices], B_pop[indices]
