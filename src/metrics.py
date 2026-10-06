"""
metrics.py — Sup-norm errors, threshold-fraction paths, Wasserstein-1, and
related diagnostic statistics for finite-N contagion experiments.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Path error metrics
# ---------------------------------------------------------------------------

def sup_scalar_error(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """sup_t |path_a(t) - path_b(t)|.

    Parameters
    ----------
    path_a, path_b : (n_steps+1,) arrays

    Returns
    -------
    supremum absolute error (float)
    """
    return float(np.max(np.abs(path_a - path_b)))


def sup_vector_error_inf(
    states_a: np.ndarray,
    states_b: np.ndarray,
) -> float:
    """sup_t max_i |X_a(t)_i - X_b(t)_i| (uniform over time and institutions).

    Parameters
    ----------
    states_a, states_b : (n_steps+1, N) state arrays

    Returns
    -------
    sup norm error (float)
    """
    return float(np.max(np.abs(states_a - states_b)))


def sup_vector_error_mean(
    states_a: np.ndarray,
    states_b: np.ndarray,
) -> float:
    """sup_t (1/N) Σ_i |X_a(t)_i - X_b(t)_i| (mean norm, sup over time).

    Parameters
    ----------
    states_a, states_b : (n_steps+1, N) state arrays

    Returns
    -------
    sup-mean error (float)
    """
    return float(np.max(np.abs(states_a - states_b).mean(axis=1)))


def sup_feedback_error(betas_a: np.ndarray, betas_b: np.ndarray) -> float:
    """sup_t ||β_a(t) - β_b(t)||_∞ (max over K coordinates and time).

    Parameters
    ----------
    betas_a, betas_b : (n_steps+1, K) feedback arrays

    Returns
    -------
    supremum infinity-norm error (float)
    """
    return float(np.max(np.abs(betas_a - betas_b)))


# ---------------------------------------------------------------------------
# Threshold fraction path
# ---------------------------------------------------------------------------

def threshold_fraction_path(states: np.ndarray) -> np.ndarray:
    """h_t^N = (1/N) Σ_i 1_{X_t^i ≤ 0}.

    Parameters
    ----------
    states : (n_steps+1, N) or (N,) array

    Returns
    -------
    h : (n_steps+1,) or scalar
    """
    return (states <= 0.0).mean(axis=-1)


def group_threshold_fractions(
    states: np.ndarray,
    group_ids: np.ndarray,
    G: int,
) -> np.ndarray:
    """Group-resolved threshold fractions h_t^{(g)}.

    Parameters
    ----------
    states    : (n_steps+1, N) state array
    group_ids : (N,) integer group membership
    G         : number of groups

    Returns
    -------
    h_g : (n_steps+1, G) group threshold fractions
    """
    n_t = states.shape[0]
    h_g = np.zeros((n_t, G))
    for g in range(G):
        mask = group_ids == g
        if mask.sum() > 0:
            h_g[:, g] = (states[:, mask] <= 0.0).mean(axis=1)
    return h_g


# ---------------------------------------------------------------------------
# Wasserstein-1 distance (1D, CDF-integral method)
# ---------------------------------------------------------------------------

def wasserstein1_empirical(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """One-dimensional Wasserstein-1 distance between two empirical distributions.

    W_1(μ, ν) = ∫ |F_μ(x) - F_ν(x)| dx

    Computed via the sorted-quantile formula:
        W_1 = (1/n) Σ_i |sort(sample_a)[i] - sort(sample_b)[i]|
    when both samples have size n, otherwise via CDF-integral on a common grid.

    Parameters
    ----------
    sample_a, sample_b : 1D arrays of observations

    Returns
    -------
    W1 : float Wasserstein-1 distance
    """
    na, nb = len(sample_a), len(sample_b)
    # Use quantile interpolation for unequal sizes
    all_vals = np.sort(np.concatenate([sample_a, sample_b]))
    F_a = np.searchsorted(np.sort(sample_a), all_vals, side='right') / na
    F_b = np.searchsorted(np.sort(sample_b), all_vals, side='right') / nb
    # Integrate |F_a - F_b| over all_vals using trapezoidal rule
    diff = np.abs(F_a - F_b)
    # dx between consecutive values
    dx = np.diff(all_vals)
    # Midpoint rule on intervals
    integrand = 0.5 * (diff[:-1] + diff[1:])
    return float(np.dot(integrand, dx))


# ---------------------------------------------------------------------------
# Log-log slope fitting
# ---------------------------------------------------------------------------

def loglog_slope(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> tuple[float, float]:
    """Fit log(y) = α + γ log(x) by OLS, return (α, γ).

    Parameters
    ----------
    x_vals : x values (e.g., epsilon or N values), must be > 0
    y_vals : y values (errors), must be > 0

    Returns
    -------
    alpha : intercept
    gamma : slope (convergence rate)
    """
    log_x = np.log(x_vals)
    log_y = np.log(y_vals)
    gamma, alpha = np.polyfit(log_x, log_y, 1)
    return float(alpha), float(gamma)


# ---------------------------------------------------------------------------
# SVD truncation utilities
# ---------------------------------------------------------------------------

def svd_truncate(E: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-`rank` approximation of matrix E via truncated SVD.

    Parameters
    ----------
    E    : (N, M) exposure matrix
    rank : number of singular values to retain

    Returns
    -------
    E_approx : (N, M) rank-`rank` approximation
    """
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    s_trunc = np.zeros_like(s)
    s_trunc[:rank] = s[:rank]
    return (U * s_trunc) @ Vt


def frobenius_relative_error(E: np.ndarray, E_approx: np.ndarray) -> float:
    """||E_approx - E||_F / ||E||_F.

    Parameters
    ----------
    E       : exact matrix
    E_approx: approximated matrix

    Returns
    -------
    relative Frobenius error (float)
    """
    norm_E = np.linalg.norm(E, 'fro')
    if norm_E == 0.0:
        return 0.0
    return float(np.linalg.norm(E_approx - E, 'fro') / norm_E)
