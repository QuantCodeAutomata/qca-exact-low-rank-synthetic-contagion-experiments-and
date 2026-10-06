"""
tests/test_metrics.py — Unit tests for src/metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    sup_scalar_error,
    sup_vector_error_inf,
    sup_vector_error_mean,
    sup_feedback_error,
    threshold_fraction_path,
    group_threshold_fractions,
    wasserstein1_empirical,
    loglog_slope,
    svd_truncate,
    frobenius_relative_error,
)


class TestSupScalarError:
    def test_identical_paths(self):
        a = np.linspace(0, 1, 100)
        assert sup_scalar_error(a, a) == 0.0

    def test_known_error(self):
        a = np.array([0.0, 0.5, 1.0])
        b = np.array([0.1, 0.5, 0.8])
        assert abs(sup_scalar_error(a, b) - 0.2) < 1e-14

    def test_symmetric(self):
        a = np.random.randn(50)
        b = np.random.randn(50)
        assert abs(sup_scalar_error(a, b) - sup_scalar_error(b, a)) < 1e-14


class TestSupVectorErrors:
    def test_identical(self):
        states = np.random.randn(10, 5)
        assert sup_vector_error_inf(states, states) == 0.0
        assert sup_vector_error_mean(states, states) == 0.0

    def test_inf_vs_mean(self):
        states_a = np.zeros((5, 4))
        states_b = np.ones((5, 4))
        assert sup_vector_error_inf(states_a, states_b) == 1.0
        assert sup_vector_error_mean(states_a, states_b) == 1.0

    def test_partial_error(self):
        states_a = np.zeros((3, 4))
        states_b = states_a.copy()
        states_b[2, 0] = 2.0  # one large error
        assert sup_vector_error_inf(states_a, states_b) == 2.0
        assert sup_vector_error_mean(states_a, states_b) == 0.5  # 2/4 at t=2


class TestThresholdFractionPath:
    def test_all_positive(self):
        states = np.ones((5, 10))
        h = threshold_fraction_path(states)
        np.testing.assert_array_equal(h, np.zeros(5))

    def test_all_negative(self):
        states = -np.ones((5, 10))
        h = threshold_fraction_path(states)
        np.testing.assert_array_equal(h, np.ones(5))

    def test_mixed(self):
        states = np.array([[-1.0, 0.0, 0.5, 0.1]])
        h = threshold_fraction_path(states)
        np.testing.assert_allclose(h, [0.5])

    def test_zero_is_distressed(self):
        states = np.array([[0.0, 0.1]])
        h = threshold_fraction_path(states)
        np.testing.assert_allclose(h, [0.5])


class TestGroupThresholdFractions:
    def test_two_groups(self):
        states = np.array([[
            -1.0, -0.5, 0.3, 0.1,  # group 0: -1, -0.5 → h0=0.5; group 1: 0.3, 0.1 → h1=0
        ]])
        group_ids = np.array([0, 0, 1, 1])
        h_g = group_threshold_fractions(states, group_ids, G=2)
        np.testing.assert_allclose(h_g[0, 0], 1.0)  # both negative in group 0
        np.testing.assert_allclose(h_g[0, 1], 0.0)  # both positive in group 1

    def test_shape(self):
        states = np.ones((10, 6))
        states[:, :3] = -1.0
        group_ids = np.array([0, 0, 0, 1, 1, 1])
        h_g = group_threshold_fractions(states, group_ids, G=2)
        assert h_g.shape == (10, 2)


class TestWasserstein1:
    def test_identical_samples(self):
        a = np.array([0.1, 0.3, 0.5, 0.7])
        w1 = wasserstein1_empirical(a, a)
        assert w1 == pytest.approx(0.0, abs=1e-10)

    def test_shifted_samples(self):
        """W_1 between N(0,1) and N(δ,1) samples ≈ δ for large N."""
        rng = np.random.default_rng(0)
        delta = 0.5
        n = 10000
        a = rng.standard_normal(n)
        b = rng.standard_normal(n) + delta
        w1 = wasserstein1_empirical(a, b)
        assert abs(w1 - delta) < 0.05, f"W1={w1:.4f}, expected ≈ {delta}"

    def test_nonnegative(self):
        a = np.random.randn(50)
        b = np.random.randn(50)
        assert wasserstein1_empirical(a, b) >= 0.0


class TestLoglogSlope:
    def test_slope_one(self):
        """log(y) = log(x) → slope 1."""
        x = np.array([0.1, 0.01, 0.001])
        y = x.copy()
        _, gamma = loglog_slope(x, y)
        assert abs(gamma - 1.0) < 1e-10

    def test_slope_half(self):
        """log(y) = 0.5 log(x) → slope 0.5."""
        x = np.array([1e-2, 1e-3, 1e-4, 1e-5])
        y = np.sqrt(x)
        _, gamma = loglog_slope(x, y)
        assert abs(gamma - 0.5) < 1e-10


class TestSVDTruncate:
    def test_rank1_outer_product(self):
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 1.0, 1.0])
        E = np.outer(u, v)
        E_approx = svd_truncate(E, rank=1)
        np.testing.assert_allclose(E_approx, E, atol=1e-13)

    def test_frobenius_decreases_with_rank(self):
        rng = np.random.default_rng(42)
        E = rng.standard_normal((6, 6))
        errors = [frobenius_relative_error(E, svd_truncate(E, k)) for k in [1, 2, 3, 4, 5, 6]]
        assert all(errors[i] >= errors[i + 1] - 1e-14 for i in range(len(errors) - 1))

    def test_full_rank_is_exact(self):
        rng = np.random.default_rng(0)
        E = rng.standard_normal((5, 5))
        E_approx = svd_truncate(E, rank=5)
        np.testing.assert_allclose(E_approx, E, atol=1e-13)

    def test_frobenius_zero_for_identical(self):
        E = np.eye(4)
        E_approx = svd_truncate(E, rank=4)
        assert frobenius_relative_error(E, E_approx) < 1e-14

    def test_frobenius_one_for_zero_approx(self):
        E = np.eye(4)
        E_approx = np.zeros((4, 4))
        assert abs(frobenius_relative_error(E, E_approx) - 1.0) < 1e-14
