"""
tests/test_sampling.py — Unit tests for src/sampling.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sampling import (
    allocate_group_counts,
    quantile_initial_conditions,
    build_quantile_matched_network,
    iid_sample_network,
)


class TestAllocateGroupCounts:
    def test_sum_equals_N(self):
        N = 100
        p = np.array([0.3, 0.3, 0.4])
        n_g = allocate_group_counts(N, p)
        assert n_g.sum() == N

    def test_exact_fractions(self):
        N = 10
        p = np.array([0.5, 0.5])
        n_g = allocate_group_counts(N, p)
        assert n_g.sum() == N
        np.testing.assert_array_equal(n_g, [5, 5])

    def test_all_positive(self):
        N = 400
        p = np.array([0.5, 0.5])
        n_g = allocate_group_counts(N, p)
        assert np.all(n_g > 0)

    def test_probabilities_not_sum_one_raises(self):
        with pytest.raises(AssertionError):
            allocate_group_counts(100, np.array([0.4, 0.4]))

    def test_three_groups(self):
        N = 11
        p = np.array([1 / 3, 1 / 3, 1 / 3])
        n_g = allocate_group_counts(N, p)
        assert n_g.sum() == N

    def test_large_N(self):
        N = 1600
        p = np.array([0.5, 0.5])
        n_g = allocate_group_counts(N, p)
        assert n_g.sum() == N
        np.testing.assert_array_equal(n_g, [800, 800])


class TestQuantileInitialConditions:
    def test_shape(self):
        n_g = np.array([5, 3])
        m_g = np.array([0.5, 0.3])
        sigma_g = np.array([0.1, 0.15])
        x0 = quantile_initial_conditions(n_g, m_g, sigma_g)
        assert x0.shape == (8,)

    def test_group_means_approximate(self):
        """Quantile-matched samples have approximately correct group means."""
        n_g = np.array([200, 200])
        m_g = np.array([0.5, 0.3])
        sigma_g = np.array([0.1, 0.1])
        x0 = quantile_initial_conditions(n_g, m_g, sigma_g)
        # Group 0: first 200 samples
        assert abs(x0[:200].mean() - 0.5) < 0.01
        assert abs(x0[200:].mean() - 0.3) < 0.01

    def test_sorted_within_group(self):
        """Quantile-matched samples should be sorted within each group."""
        n_g = np.array([10, 10])
        m_g = np.array([0.5, 0.3])
        sigma_g = np.array([0.1, 0.1])
        x0 = quantile_initial_conditions(n_g, m_g, sigma_g)
        assert np.all(np.diff(x0[:10]) >= 0), "Group 0 not sorted"
        assert np.all(np.diff(x0[10:]) >= 0), "Group 1 not sorted"


class TestBuildQuantileMatchedNetwork:
    def test_output_shapes(self):
        N = 50
        p = np.array([0.5, 0.5])
        m_g = np.array([0.5, 0.4])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        x0, A, B, group_ids = build_quantile_matched_network(N, p, m_g, sigma_g, A_g, B_g)
        assert x0.shape == (N,)
        assert A.shape == (N, 1)
        assert B.shape == (N, 1)
        assert group_ids.shape == (N,)

    def test_group_loadings_correct(self):
        N = 20
        p = np.array([0.5, 0.5])
        m_g = np.array([0.5, 0.4])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        x0, A, B, group_ids = build_quantile_matched_network(N, p, m_g, sigma_g, A_g, B_g)
        # Group 0: A should all be 2.0
        g0_mask = group_ids == 0
        np.testing.assert_allclose(A[g0_mask, 0], 2.0 * np.ones(g0_mask.sum()))
        # Group 1: A should all be 0.6
        g1_mask = group_ids == 1
        np.testing.assert_allclose(A[g1_mask, 0], 0.6 * np.ones(g1_mask.sum()))

    def test_group_ids_valid(self):
        N = 100
        p = np.array([0.5, 0.5])
        m_g = np.array([0.5, 0.4])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        _, _, _, group_ids = build_quantile_matched_network(N, p, m_g, sigma_g, A_g, B_g)
        assert set(group_ids).issubset({0, 1})

    def test_N_400(self):
        N = 400
        p = np.array([0.5, 0.5])
        m_g = np.array([0.50, 0.45])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        x0, A, B, group_ids = build_quantile_matched_network(N, p, m_g, sigma_g, A_g, B_g)
        assert x0.shape == (N,)
        assert A.shape == (N, 1)


class TestIIDSampleNetwork:
    def test_output_shapes(self):
        N_source = 100
        p = np.array([0.5, 0.5])
        m_g = np.array([0.5, 0.4])
        sigma_g = np.array([0.18, 0.18])
        A_g = np.array([[2.0], [0.6]])
        B_g = np.array([[0.6], [2.0]])
        x0_base, A_base, B_base, ids_base = build_quantile_matched_network(
            N_source, p, m_g, sigma_g, A_g, B_g
        )
        N_sample = 50
        x0, A, B, ids = iid_sample_network(
            N_sample, x0_base, A_base, B_base, ids_base, rng=np.random.default_rng(42)
        )
        assert x0.shape == (N_sample,)
        assert A.shape == (N_sample, 1)
        assert B.shape == (N_sample, 1)
