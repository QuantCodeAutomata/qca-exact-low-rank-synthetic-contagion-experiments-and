"""
tests/test_low_rank.py — Unit tests for src/low_rank.py

Tests cover:
  - Factorised exposure construction
  - Direct/reduced equivalence (core closure claim)
  - Imbalance coefficient
  - Population ODE consistency
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import loss_hard, simulate_direct
from src.low_rank import (
    build_factorised_exposure,
    imbalance_coefficient,
    simulate_reduced,
    gaussian_mixture_ode_rhs,
    solve_population_ode,
    population_threshold_fraction,
)
from src.sampling import build_quantile_matched_network
from src.metrics import sup_vector_error_inf


# ---------------------------------------------------------------------------
# Factorised exposure tests
# ---------------------------------------------------------------------------

class TestBuildFactorisedExposure:
    def test_rank_one(self):
        N = 5
        A = np.random.rand(N, 1)
        B = np.random.rand(N, 1)
        E = build_factorised_exposure(A, B)
        expected = A @ B.T  # K=1 → divide by 1
        np.testing.assert_allclose(E, expected, atol=1e-14)

    def test_rank_two(self):
        N = 4
        K = 2
        A = np.random.rand(N, K)
        B = np.random.rand(N, K)
        E = build_factorised_exposure(A, B)
        expected = A @ B.T / K
        np.testing.assert_allclose(E, expected, atol=1e-14)

    def test_shape(self):
        N = 6
        K = 3
        A = np.random.rand(N, K)
        B = np.random.rand(N, K)
        E = build_factorised_exposure(A, B)
        assert E.shape == (N, N)

    def test_nonneg_when_inputs_nonneg(self):
        A = np.abs(np.random.randn(8, 2))
        B = np.abs(np.random.randn(8, 2))
        E = build_factorised_exposure(A, B)
        assert np.all(E >= 0)


# ---------------------------------------------------------------------------
# Imbalance coefficient tests
# ---------------------------------------------------------------------------

class TestImbalanceCoefficient:
    def test_zero_when_r_zero(self):
        A = np.random.rand(6, 2)
        B = np.random.rand(6, 2)
        Lambda = imbalance_coefficient(A, B, r=0.0)
        np.testing.assert_array_equal(Lambda, np.zeros(6))

    def test_antisymmetric_structure(self):
        """If a_g = b_g for all institutions, imbalance = 0 (symmetric case)."""
        A = np.random.rand(8, 2)
        B = A.copy()
        Lambda = imbalance_coefficient(A, B, r=0.5)
        np.testing.assert_allclose(Lambda, np.zeros(8), atol=1e-14)

    def test_shape(self):
        N = 10
        A = np.random.rand(N, 3)
        B = np.random.rand(N, 3)
        Lambda = imbalance_coefficient(A, B, r=1.0)
        assert Lambda.shape == (N,)


# ---------------------------------------------------------------------------
# Direct/reduced equivalence (core closure claim)
# ---------------------------------------------------------------------------

class TestDirectReducedEquivalence:
    """Machine-precision equivalence between direct and reduced solvers."""

    def _run_equivalence(
        self, N, K, seed=0, mu=0.0, r=0.0, T=0.5, dt=1e-2
    ):
        rng = np.random.default_rng(seed)
        x0 = rng.standard_normal(N) * 0.2 + 0.2
        A = np.abs(rng.standard_normal((N, K))) + 0.1
        B = np.abs(rng.standard_normal((N, K))) + 0.1

        E = build_factorised_exposure(A, B)
        _, states_direct = simulate_direct(x0, E, mu=mu, r=r, T=T, dt=dt,
                                           loss_fn=loss_hard)
        _, states_reduced, _ = simulate_reduced(x0, A, B, mu=mu, r=r, T=T, dt=dt,
                                                loss_fn=loss_hard)
        err = sup_vector_error_inf(states_direct, states_reduced)
        return err

    def test_rank1_small_N(self):
        err = self._run_equivalence(N=10, K=1)
        assert err < 1e-13, f"Direct/reduced error {err:.3e} exceeds 1e-13"

    def test_rank2_medium_N(self):
        err = self._run_equivalence(N=50, K=2)
        assert err < 1e-13, f"Direct/reduced error {err:.3e} exceeds 1e-13"

    def test_rank3_with_drift(self):
        err = self._run_equivalence(N=20, K=3, mu=-0.05, T=1.0, dt=5e-3)
        assert err < 1e-13, f"Direct/reduced error {err:.3e} exceeds 1e-13"

    def test_rank1_with_imbalance(self):
        err = self._run_equivalence(N=15, K=1, r=0.05, T=0.5, dt=1e-2)
        assert err < 1e-12, f"Direct/reduced error {err:.3e} exceeds 1e-12"

    def test_rank2_with_imbalance(self):
        err = self._run_equivalence(N=20, K=2, r=0.03, T=0.5, dt=1e-2)
        assert err < 1e-12, f"Direct/reduced error {err:.3e} exceeds 1e-12"

    def test_equivalence_with_quantile_network(self):
        """Test equivalence using a quantile-matched network."""
        from src.sampling import build_quantile_matched_network

        p = np.array([0.5, 0.5])
        m_g = np.array([0.30, 0.20])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        K = 1

        N = 100
        x0, A, B, group_ids = build_quantile_matched_network(N, p, m_g, sigma_g, A_g, B_g)
        E = build_factorised_exposure(A, B)
        _, states_direct = simulate_direct(x0, E, mu=0.0, r=0.0, T=1.0, dt=5e-3,
                                           loss_fn=loss_hard)
        _, states_reduced, _ = simulate_reduced(x0, A, B, mu=0.0, r=0.0, T=1.0, dt=5e-3,
                                                loss_fn=loss_hard)
        err = sup_vector_error_inf(states_direct, states_reduced)
        assert err < 1e-12, f"Quantile-matched direct/reduced error {err:.3e}"


# ---------------------------------------------------------------------------
# Population ODE tests
# ---------------------------------------------------------------------------

class TestGaussianMixtureODE:
    def test_zero_beta_zero_time(self):
        """At t=0, β=0, ODE rhs = Σ_g p_g b_{g,k} Φ(-m_g/σ_g)."""
        p = np.array([1.0])
        m_g = np.array([0.5])
        sigma_g = np.array([0.2])
        A_g = np.array([[1.0]])
        B_g = np.array([[1.0]])
        mu = 0.0
        beta = np.zeros(1)
        from scipy.stats import norm
        expected = float(norm.cdf(-0.5 / 0.2))
        rhs = gaussian_mixture_ode_rhs(beta, 0.0, p, m_g, sigma_g, A_g, B_g, mu)
        np.testing.assert_allclose(rhs[0], expected, atol=1e-14)

    def test_two_groups_rhs_shape(self):
        p = np.array([0.5, 0.5])
        m_g = np.array([0.3, 0.2])
        sigma_g = np.array([0.15, 0.15])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        beta = np.array([0.1])
        rhs = gaussian_mixture_ode_rhs(beta, 0.5, p, m_g, sigma_g, A_g, B_g, 0.0)
        assert rhs.shape == (1,)

    def test_population_ode_initial_zero(self):
        p = np.array([0.5, 0.5])
        m_g = np.array([0.5, 0.4])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        times, betas = solve_population_ode(p, m_g, sigma_g, A_g, B_g, 0.0, 2.0, 5e-3)
        assert betas[0, 0] == 0.0

    def test_population_ode_monotone(self):
        """β_k(t) should be non-decreasing since drift is non-negative."""
        p = np.array([0.5, 0.5])
        m_g = np.array([0.5, 0.4])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        times, betas = solve_population_ode(p, m_g, sigma_g, A_g, B_g, 0.0, 2.0, 5e-3)
        diffs = np.diff(betas[:, 0])
        assert np.all(diffs >= -1e-12), "Population feedback not monotone"

    def test_population_threshold_fraction_in_01(self):
        p = np.array([0.5, 0.5])
        m_g = np.array([0.5, 0.4])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        times, betas = solve_population_ode(p, m_g, sigma_g, A_g, B_g, 0.0, 2.0, 1e-2)
        h = population_threshold_fraction(betas, times, p, m_g, sigma_g, A_g, 0.0)
        assert np.all(h >= 0.0) and np.all(h <= 1.0)
