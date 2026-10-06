"""
empirical_kernel.py — Empirical receiver/sender factor loadings and
common-exposure kernel construction from sovereign holdings data (eq. 76–77).

Receiver loadings:  a_{i,k}^{emp} = s_{i,k} / Σ_ℓ s_{i,ℓ}
Sender loadings:    b_{i,k}^{emp} = s_{i,k} / [(1/N) Σ_m s_{m,k}]
Exposure kernel:    e_{ij}^{emp}  = λ_E Σ_k a_{i,k}^{emp} b_{j,k}^{emp}

Validation identities (eq. 76–77):
  Σ_k a_{i,k}^{emp} = 1   ∀ i
  (1/N) Σ_i b_{i,k}^{emp} = 1   ∀ k
  (1/N) Σ_j e_{ij}^{emp} = λ_E  ∀ i
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Factor loading construction (eq. 76)
# ---------------------------------------------------------------------------

def build_empirical_loadings(
    S: np.ndarray,
    lambda_E: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct empirical receiver and sender loadings from holdings matrix S.

    a_{i,k} = s_{i,k} / Σ_ℓ s_{i,ℓ}   (row-normalised receiver loadings)
    b_{i,k} = s_{i,k} / [(1/N) Σ_m s_{m,k}]  (column-mean-normalised senders)

    Parameters
    ----------
    S        : (N, K) holdings matrix (non-negative)
    lambda_E : intensity scaling λ_E

    Returns
    -------
    A : (N, K) receiver loadings
    B : (N, K) sender loadings
    """
    N, K = S.shape
    # Receiver loadings: row-normalised
    row_sums = S.sum(axis=1, keepdims=True)          # (N, 1)
    assert np.all(row_sums > 0), "All institutions must have positive total holdings"
    A = S / row_sums

    # Sender loadings: column-mean-normalised
    col_means = S.mean(axis=0, keepdims=True)         # (1, K)
    assert np.all(col_means > 0), "All sovereign factors must have positive mean holdings"
    B = S / col_means

    # Validate identities
    row_sum_A = A.sum(axis=1)
    assert np.allclose(row_sum_A, 1.0, atol=1e-10), \
        f"Row sums of A not all 1; max dev = {np.max(np.abs(row_sum_A - 1))}"

    col_mean_B = B.mean(axis=0)
    assert np.allclose(col_mean_B, 1.0, atol=1e-10), \
        f"Col means of B not all 1; max dev = {np.max(np.abs(col_mean_B - 1))}"

    return A, B


def build_empirical_exposure(
    A: np.ndarray,
    B: np.ndarray,
    lambda_E: float,
) -> np.ndarray:
    """Build the exact rank-K common-exposure matrix E (eq. 77).

    e_{ij}^{emp} = λ_E Σ_k a_{i,k} b_{j,k}

    Parameters
    ----------
    A        : (N, K) receiver loadings
    B        : (N, K) sender loadings
    lambda_E : intensity λ_E

    Returns
    -------
    E : (N, N) exposure matrix
    """
    E = lambda_E * (A @ B.T)
    # Validate row-mean identity: (1/N) Σ_j e_{ij} = λ_E
    row_means = E.mean(axis=1)
    assert np.allclose(row_means, lambda_E, atol=1e-10), \
        f"Row means of E not equal to λ_E={lambda_E}; max dev = {np.max(np.abs(row_means - lambda_E))}"
    return E


# ---------------------------------------------------------------------------
# Initial buffer construction (eq. 79)
# ---------------------------------------------------------------------------

def build_initial_buffers(
    A: np.ndarray,
    eta: np.ndarray,
    x_star: float,
) -> np.ndarray:
    """Compute initial capital buffers x_i = x* - Σ_k η_k a_{i,k} (eq. 79).

    Parameters
    ----------
    A      : (N, K) receiver loadings
    eta    : (K,) stress vector η_k
    x_star : baseline buffer level x*

    Returns
    -------
    x0 : (N,) initial buffers
    """
    return x_star - A @ eta


# ---------------------------------------------------------------------------
# Full-book alternative denominator (sensitivity analysis, exp_5)
# ---------------------------------------------------------------------------

def build_full_book_loadings(
    S: np.ndarray,
    full_row_totals: np.ndarray,
) -> np.ndarray:
    """Receiver loadings using full sovereign book denominator.

    a_{i,k}^{full} = s_{i,k} / (full sovereign book of bank i)

    Parameters
    ----------
    S               : (N, K) selected-factor holdings
    full_row_totals : (N,) or (N,1) full sovereign book totals

    Returns
    -------
    A_full : (N, K) receiver loadings with full-book denominator
    """
    full_totals = np.asarray(full_row_totals).reshape(-1, 1)
    assert np.all(full_totals > 0), "Full-book totals must be positive"
    return S / full_totals


# ---------------------------------------------------------------------------
# Robustness: adding 'Other sovereigns' factor
# ---------------------------------------------------------------------------

def add_other_factor(
    S: np.ndarray,
    full_row_totals: np.ndarray,
) -> np.ndarray:
    """Append an 'Other sovereigns' column to the holdings matrix.

    The new column captures the residual between the full book and
    the selected K factors.

    Parameters
    ----------
    S               : (N, K) selected holdings
    full_row_totals : (N,) full sovereign book totals

    Returns
    -------
    S_ext : (N, K+1) extended holdings matrix
    """
    full = np.asarray(full_row_totals).reshape(-1)
    residual = np.maximum(0.0, full - S.sum(axis=1))
    return np.column_stack([S, residual])


# ---------------------------------------------------------------------------
# Perturbation: Italian book scaling (eq. 80)
# ---------------------------------------------------------------------------

def perturb_italian_holdings(
    S: np.ndarray,
    italy_col: int,
    delta: float,
) -> np.ndarray:
    """Apply uniform Italian-book perturbation s_{i,IT} → (1+δ) s_{i,IT}.

    Parameters
    ----------
    S          : (N, K) holdings matrix
    italy_col  : column index for Italy
    delta      : perturbation scalar δ ∈ [-0.20, 0.20]

    Returns
    -------
    S_perturbed : (N, K) perturbed holdings matrix
    """
    S_pert = S.copy()
    S_pert[:, italy_col] = (1.0 + delta) * S[:, italy_col]
    return S_pert
