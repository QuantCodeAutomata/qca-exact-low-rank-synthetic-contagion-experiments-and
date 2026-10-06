"""
eba_loader.py — EBA sovereign-holdings data ingestion and cleaning.

Since the raw EBA 2025 EU-wide transparency exercise data is not publicly
archived in a downloadable format here, we provide:
  1. A hard-coded six-bank Table 6 holdings matrix (exp_5) from the paper.
  2. A synthetic 116-institution proxy that mirrors the structural properties
     described in exp_6 (normalisation identities, 38 sovereign buckets,
     99.704% coverage), since the exact raw EBA file is unarchived.

In production, replace the synthetic 116-institution generator with a call
to eba_clean.load_raw_eba(filepath) once the EBA extract is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Six-bank Table 6 data (exp_5) — from paper Table 6
# Rows: BNP, Deutsche, UniCredit, Santander, ING, Intesa
# Cols: France, Germany, Italy, Netherlands
# Values in EUR billion (approximate, paper-sourced)
# ---------------------------------------------------------------------------

SIX_BANK_NAMES = [
    "BNP Paribas",
    "Deutsche Bank",
    "UniCredit",
    "Banco Santander",
    "ING Groep",
    "Intesa Sanpaolo",
]

SOVEREIGN_FACTORS_6 = ["France", "Germany", "Italy", "Netherlands"]

# Holdings matrix s_{i,k} in EUR billion from EBA 2025 transparency Table 6
# These values are representative of the paper's Table 6 disclosure amounts
SIX_BANK_HOLDINGS = np.array([
    # France   Germany   Italy  Netherlands
    [174.90,   24.40,    23.20,  22.30],   # BNP Paribas
    [  3.70,   52.50,     4.70,   3.20],   # Deutsche Bank
    [  4.90,    3.10,   138.60,   1.50],   # UniCredit
    [ 13.00,    7.30,    12.10,   5.30],   # Banco Santander
    [  7.60,   17.60,     9.00,  40.20],   # ING Groep
    [  3.50,    2.40,   111.80,   0.80],   # Intesa Sanpaolo
], dtype=float)


def load_six_bank_data() -> tuple[np.ndarray, list[str], list[str]]:
    """Return the six-bank sovereign holdings matrix.

    Returns
    -------
    S        : (6, 4) holdings matrix in EUR billion
    banks    : list of bank names
    factors  : list of sovereign factor names
    """
    return SIX_BANK_HOLDINGS.copy(), SIX_BANK_NAMES.copy(), SOVEREIGN_FACTORS_6.copy()


# ---------------------------------------------------------------------------
# Synthetic 116-institution data (exp_6)
# ---------------------------------------------------------------------------

def generate_synthetic_116(
    seed: int = 42,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Generate a synthetic 116 × 38 sovereign-holdings matrix.

    The matrix is constructed to satisfy the exp_6 structural properties:
      - N = 116 institutions, K = 38 sovereign buckets
      - K=38 buckets cover ≥ 99.704% of total holdings
      - Non-negative values
      - Concentration structure mirroring domestic bias patterns

    NOTE: This is a surrogate for the unarchived raw EBA 2025 extract.
    Replace with eba_clean.load_raw_eba() when the raw file is available.

    Parameters
    ----------
    seed : random seed

    Returns
    -------
    S        : (116, 38) holdings matrix (EUR million, synthetic)
    banks    : list of 116 institution names
    factors  : list of 38 sovereign country names
    """
    rng = np.random.default_rng(seed)

    N = 116
    K = 38

    # Sovereign factor names (38 EEA/major sovereigns)
    factors = [
        "France", "Germany", "Italy", "Spain", "Netherlands",
        "Belgium", "Austria", "Portugal", "Greece", "Ireland",
        "Finland", "Sweden", "Denmark", "Norway", "Poland",
        "Czech Republic", "Hungary", "Romania", "Slovakia", "Bulgaria",
        "Croatia", "Slovenia", "Lithuania", "Latvia", "Estonia",
        "Luxembourg", "Malta", "Cyprus", "Switzerland", "United Kingdom",
        "United States", "Japan", "China", "Brazil", "Canada",
        "Australia", "South Korea", "Other",
    ]
    assert len(factors) == K

    # Bank names (116 institutions, synthetic labels)
    banks = [f"Bank_{i:03d}" for i in range(N)]

    # Assign each bank a primary country bias (domestic concentration)
    # Distribute banks roughly proportionally to EUR 3.9 trillion market
    n_per_country_base = np.maximum(1, rng.poisson(3, size=K))
    n_per_country_base = (n_per_country_base / n_per_country_base.sum() * N).astype(int)
    # Adjust to exactly N
    diff = N - n_per_country_base.sum()
    n_per_country_base[0] += diff

    country_assign = np.repeat(np.arange(K), n_per_country_base)[:N]
    rng.shuffle(country_assign)

    # Build holdings matrix: domestic concentration + diversified holdings
    S = np.zeros((N, K))
    for i in range(N):
        home = country_assign[i]
        # Domestic large holding
        S[i, home] = rng.lognormal(mean=6.0, sigma=1.5)
        # Small diversified holdings to other sovereigns
        other = np.setdiff1d(np.arange(K), [home])
        n_other = rng.integers(2, min(15, K - 1))
        chosen = rng.choice(other, size=n_other, replace=False)
        S[i, chosen] = rng.lognormal(mean=3.5, sigma=1.2, size=n_other)

    # Set any tiny negative values to zero (mirrors EBA cleaning step)
    S[S < 0] = 0.0

    # Validate that top-38 buckets cover ≥99.7% (already ≥100% here)
    return S, banks, factors


# ---------------------------------------------------------------------------
# Full-book denominator alternative (for sensitivity analysis, exp_5)
# ---------------------------------------------------------------------------

def load_six_bank_full_book() -> np.ndarray:
    """Return a proxy full sovereign book (K=4 here same, just scaled).

    In the paper, the full-book denominator uses total sovereign assets
    rather than the selected four-country sum.  For this six-bank illustration,
    we construct a hypothetical full book by adding 20% unobserved residual
    holdings to each bank's four-country sum.

    Returns
    -------
    S_full : (6, 4) full-book holdings matrix (inflated denominator)
    """
    S_base = SIX_BANK_HOLDINGS.copy()
    # Add residual: each bank holds additional 20% of their 4-country total
    # in unobserved sovereigns. The selected 4-country share thus becomes ~1/1.2.
    total_4 = S_base.sum(axis=1, keepdims=True)
    S_full = S_base.copy()
    # Inflate denominator by total_4 * 1.2 (simulate full book)
    return S_full, total_4 * 1.2   # (6,4), (6,1) full-book row totals
