"""
tests/test_empirical_kernel.py — Unit tests for src/empirical_kernel.py

Tests for empirical loading normalization identities (eq. 76–77),
exposure matrix row-mean identity, and sensitivity constructions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.empirical_kernel import (
    build_empirical_loadings,
    build_empirical_exposure,
    build_initial_buffers,
    build_full_book_loadings,
    add_other_factor,
    perturb_italian_holdings,
)
from src.eba_loader import load_six_bank_data


# ---------------------------------------------------------------------------
# Normalization identity tests (eq. 76–77)
# ---------------------------------------------------------------------------

class TestEmpiricalNormalization:
    def _get_six_bank(self):
        S, _, _ = load_six_bank_data()
        return S

    def test_row_sum_A_equals_one(self):
        """Σ_k a_{i,k} = 1 for all institutions."""
        S = self._get_six_bank()
        A, B = build_empirical_loadings(S, lambda_E=0.075)
        row_sums = A.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(S)), atol=1e-10)

    def test_col_mean_B_equals_one(self):
        """(1/N) Σ_i b_{i,k} = 1 for all sovereign factors."""
        S = self._get_six_bank()
        A, B = build_empirical_loadings(S, lambda_E=0.075)
        col_means = B.mean(axis=0)
        np.testing.assert_allclose(col_means, np.ones(S.shape[1]), atol=1e-10)

    def test_row_mean_E_equals_lambda(self):
        """(1/N) Σ_j e_{ij} = λ_E for all institutions."""
        S = self._get_six_bank()
        lambda_E = 0.075
        A, B = build_empirical_loadings(S, lambda_E=lambda_E)
        E = build_empirical_exposure(A, B, lambda_E)
        row_means = E.mean(axis=1)
        np.testing.assert_allclose(row_means, lambda_E * np.ones(len(S)), atol=1e-10)

    def test_rank_equals_K(self):
        """Exposure matrix should be exactly rank K (number of factors)."""
        S = self._get_six_bank()
        K = S.shape[1]
        A, B = build_empirical_loadings(S, lambda_E=0.075)
        E = build_empirical_exposure(A, B, 0.075)
        rank = np.linalg.matrix_rank(E, tol=1e-10)
        assert rank <= K, f"Rank {rank} > K={K}"

    def test_nonneg_exposure(self):
        """Exposure matrix should be non-negative for non-negative holdings."""
        S = self._get_six_bank()
        A, B = build_empirical_loadings(S, lambda_E=0.075)
        E = build_empirical_exposure(A, B, 0.075)
        assert np.all(E >= -1e-14)

    def test_normalization_with_synthetic_data(self):
        """Test with a random non-negative holdings matrix."""
        rng = np.random.default_rng(0)
        N, K = 20, 5
        S = np.abs(rng.standard_normal((N, K))) + 0.1
        lambda_E = 0.1
        A, B = build_empirical_loadings(S, lambda_E=lambda_E)
        np.testing.assert_allclose(A.sum(axis=1), np.ones(N), atol=1e-10)
        np.testing.assert_allclose(B.mean(axis=0), np.ones(K), atol=1e-10)
        E = build_empirical_exposure(A, B, lambda_E)
        np.testing.assert_allclose(E.mean(axis=1), lambda_E * np.ones(N), atol=1e-10)


# ---------------------------------------------------------------------------
# Initial buffer tests
# ---------------------------------------------------------------------------

class TestInitialBuffers:
    def test_formula(self):
        """x_i = x* - Σ_k η_k a_{i,k}."""
        A = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        eta = np.array([0.35, 0.15, 0.05])
        x_star = 0.4
        x0 = build_initial_buffers(A, eta, x_star)
        expected = x_star - A @ eta
        np.testing.assert_allclose(x0, expected, atol=1e-14)

    def test_positive_star_large_stress(self):
        """With large stress vector, some buffers may go negative."""
        A = np.eye(3)  # each bank exposed to exactly one factor
        eta = np.array([0.5, 0.5, 0.5])
        x_star = 0.3
        x0 = build_initial_buffers(A, eta, x_star)
        # Each x_i = 0.3 - 0.5 = -0.2
        np.testing.assert_allclose(x0, -0.2 * np.ones(3), atol=1e-14)


# ---------------------------------------------------------------------------
# Perturbation tests
# ---------------------------------------------------------------------------

class TestPerturbItalianHoldings:
    def test_zero_delta(self):
        S = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        S_pert = perturb_italian_holdings(S, italy_col=2, delta=0.0)
        np.testing.assert_array_equal(S_pert, S)

    def test_double_italy(self):
        S = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        S_pert = perturb_italian_holdings(S, italy_col=2, delta=1.0)
        expected = np.array([[1.0, 2.0, 6.0], [4.0, 5.0, 12.0]])
        np.testing.assert_allclose(S_pert, expected, atol=1e-14)

    def test_non_italy_unchanged(self):
        S = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        S_pert = perturb_italian_holdings(S, italy_col=2, delta=0.15)
        # Columns 0 and 1 unchanged
        np.testing.assert_array_equal(S_pert[:, :2], S[:, :2])

    def test_original_unchanged(self):
        """Perturbation should not modify original matrix in-place."""
        S = np.array([[1.0, 2.0, 3.0]])
        S_orig = S.copy()
        perturb_italian_holdings(S, italy_col=2, delta=0.5)
        np.testing.assert_array_equal(S, S_orig)


# ---------------------------------------------------------------------------
# Full-book denominator tests
# ---------------------------------------------------------------------------

class TestFullBookLoadings:
    def test_four_factor_vs_full_denominator(self):
        """With full book = 4-factor sum, should match A exactly."""
        S = np.abs(np.random.randn(5, 3)) + 0.1
        A_selected, _ = build_empirical_loadings(S, lambda_E=0.1)
        full_totals = S.sum(axis=1)  # same as 4-factor sum
        A_full = build_full_book_loadings(S, full_totals)
        np.testing.assert_allclose(A_full, A_selected, atol=1e-14)

    def test_larger_full_book_smaller_loadings(self):
        """With larger full denominator, A_full values should be smaller."""
        S = np.abs(np.random.randn(4, 2)) + 0.1
        A_selected, _ = build_empirical_loadings(S, lambda_E=0.1)
        full_totals = S.sum(axis=1) * 1.5  # 50% larger book
        A_full = build_full_book_loadings(S, full_totals)
        assert np.all(A_full <= A_selected + 1e-14)


# ---------------------------------------------------------------------------
# Add other factor tests
# ---------------------------------------------------------------------------

class TestAddOtherFactor:
    def test_shape(self):
        S = np.ones((5, 3))
        full = 4.0 * np.ones(5)
        S_ext = add_other_factor(S, full)
        assert S_ext.shape == (5, 4)

    def test_residual_correct(self):
        S = np.array([[1.0, 2.0], [3.0, 1.0]])
        full = np.array([5.0, 6.0])
        S_ext = add_other_factor(S, full)
        # Residual: [5-3, 6-4] = [2, 2]
        np.testing.assert_allclose(S_ext[:, 2], [2.0, 2.0], atol=1e-14)

    def test_no_negative_residuals(self):
        """Residuals should be clamped to zero if full < sum of selected."""
        S = np.array([[3.0, 3.0]])
        full = np.array([4.0])  # less than selected sum 6
        S_ext = add_other_factor(S, full)
        assert S_ext[0, 2] == 0.0  # max(0, 4-6) = 0
