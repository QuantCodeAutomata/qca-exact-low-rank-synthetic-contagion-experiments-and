"""
tests/test_model.py — Unit tests for src/model.py

Tests cover loss functions, direct solver correctness,
and mathematical properties of the finite-N dynamics.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    loss_hard,
    loss_positive_ramp,
    loss_negative_ramp,
    simulate_direct,
    threshold_fraction,
    mean_smoothed_distress,
)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLossHard:
    def test_values_below_zero(self):
        x = np.array([-1.0, -0.5, -0.01])
        np.testing.assert_array_equal(loss_hard(x), np.ones(3))

    def test_zero_is_distressed(self):
        assert loss_hard(np.array([0.0]))[0] == 1.0

    def test_positive_values(self):
        x = np.array([0.001, 0.5, 10.0])
        np.testing.assert_array_equal(loss_hard(x), np.zeros(3))

    def test_mixed(self):
        x = np.array([-1.0, 0.0, 0.1])
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_equal(loss_hard(x), expected)

    def test_output_range(self):
        x = np.random.uniform(-5, 5, 100)
        result = loss_hard(x)
        assert np.all((result == 0.0) | (result == 1.0))


class TestLossPositiveRamp:
    def test_below_zero(self):
        x = np.array([-2.0, -0.5])
        result = loss_positive_ramp(x, eps=0.1)
        np.testing.assert_array_equal(result, np.ones(2))

    def test_zero_is_one(self):
        assert loss_positive_ramp(np.array([0.0]), eps=0.1)[0] == 1.0

    def test_above_eps(self):
        x = np.array([0.15, 1.0])
        result = loss_positive_ramp(x, eps=0.1)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_inside_ramp(self):
        eps = 0.1
        x = np.array([0.05])
        result = loss_positive_ramp(x, eps=eps)
        np.testing.assert_allclose(result, np.array([0.5]), atol=1e-14)

    def test_range_in_01(self):
        x = np.linspace(-1.0, 1.0, 200)
        result = loss_positive_ramp(x, eps=0.1)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_monotone_decreasing(self):
        x = np.linspace(-0.5, 0.5, 100)
        result = loss_positive_ramp(x, eps=0.2)
        diffs = np.diff(result)
        assert np.all(diffs <= 1e-14)

    def test_converges_to_hard_as_eps_shrinks(self):
        x = np.array([0.001])
        for eps in [0.1, 0.01, 0.001]:
            val = loss_positive_ramp(x, eps=eps)
            if eps <= x[0]:
                assert val[0] == 0.0

    def test_invalid_eps(self):
        with pytest.raises(AssertionError):
            loss_positive_ramp(np.array([0.0]), eps=-0.1)


class TestLossNegativeRamp:
    def test_positive_values(self):
        x = np.array([0.1, 1.0])
        result = loss_negative_ramp(x, eps=0.1)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_large_negative(self):
        x = np.array([-1.0, -2.0])
        result = loss_negative_ramp(x, eps=0.1)
        np.testing.assert_array_equal(result, np.ones(2))

    def test_zero(self):
        result = loss_negative_ramp(np.array([0.0]), eps=0.05)
        assert result[0] == 0.0

    def test_inside_ramp(self):
        eps = 0.1
        x = np.array([-0.05])
        result = loss_negative_ramp(x, eps=eps)
        np.testing.assert_allclose(result, [0.5], atol=1e-14)

    def test_range_in_01(self):
        x = np.linspace(-1.0, 1.0, 200)
        result = loss_negative_ramp(x, eps=0.1)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_negative_eps_raises(self):
        with pytest.raises(AssertionError):
            loss_negative_ramp(np.array([0.0]), eps=-0.1)


# ---------------------------------------------------------------------------
# Direct solver tests
# ---------------------------------------------------------------------------

class TestSimulateDirect:
    def _trivial_setup(self):
        """Zero exposure, drift=0 → states remain at x0."""
        N = 4
        x0 = np.array([0.5, 0.3, -0.1, 0.7])
        E = np.zeros((N, N))
        return x0, E

    def test_zero_exposure_no_drift(self):
        x0, E = self._trivial_setup()
        times, states = simulate_direct(x0, E, mu=0.0, r=0.0, T=1.0, dt=0.1,
                                        loss_fn=loss_hard)
        # With no exposure and no drift, all rows of states must equal x0
        expected = np.tile(x0, (len(times), 1))
        np.testing.assert_allclose(states, expected, atol=1e-14)

    def test_output_shapes(self):
        N = 5
        x0 = np.random.randn(N)
        E = np.random.rand(N, N) * 0.1
        n_steps = 10
        T = 0.1
        dt = T / n_steps
        times, states = simulate_direct(x0, E, mu=0.0, r=0.0, T=T, dt=dt,
                                        loss_fn=loss_hard)
        assert times.shape == (n_steps + 1,)
        assert states.shape == (n_steps + 1, N)

    def test_initial_state_is_x0(self):
        x0 = np.array([0.2, -0.3, 0.5])
        E = np.eye(3) * 0.1
        _, states = simulate_direct(x0, E, mu=0.0, r=0.0, T=0.5, dt=0.05,
                                    loss_fn=loss_hard)
        np.testing.assert_array_equal(states[0], x0)

    def test_times_start_and_end(self):
        T = 2.0
        dt = 0.1
        x0 = np.ones(3)
        E = np.zeros((3, 3))
        times, _ = simulate_direct(x0, E, mu=0.0, r=0.0, T=T, dt=dt,
                                   loss_fn=loss_hard)
        assert abs(times[0] - 0.0) < 1e-14
        assert abs(times[-1] - T) < 1e-10

    def test_pure_drift(self):
        """With zero exposure and nonzero drift, states = x0 + μt."""
        x0 = np.array([0.1, 0.2])
        E = np.zeros((2, 2))
        mu = -0.05
        T = 1.0
        dt = 0.01
        times, states = simulate_direct(x0, E, mu=mu, r=0.0, T=T, dt=dt,
                                        loss_fn=loss_hard)
        expected = x0[None, :] + mu * times[:, None]
        np.testing.assert_allclose(states, expected, atol=1e-12)

    def test_contagion_reduces_states(self):
        """With all states initially distressed and positive exposures,
        cumulative distress should reduce states over time."""
        N = 4
        x0 = -np.ones(N)    # All distressed
        E = np.full((N, N), 0.5)
        _, states = simulate_direct(x0, E, mu=0.0, r=0.0, T=1.0, dt=0.1,
                                    loss_fn=loss_hard)
        # States should move more negative (more distressed) with time
        assert np.all(states[-1] <= states[0])


class TestThresholdFraction:
    def test_all_distressed(self):
        states = np.array([[-1.0, -0.5, -0.1]])
        result = threshold_fraction(states)
        assert result[0] == 1.0

    def test_none_distressed(self):
        states = np.array([[0.1, 0.5, 1.0]])
        result = threshold_fraction(states)
        assert result[0] == 0.0

    def test_half_distressed(self):
        states = np.array([[-1.0, 0.0, 0.1, 0.5]])
        result = threshold_fraction(states)
        np.testing.assert_allclose(result[0], 2 / 4)

    def test_2d_path(self):
        states = np.array([
            [-1.0, 0.5],   # 0.5 distressed
            [-1.0, -0.5],  # 1.0 distressed
        ])
        result = threshold_fraction(states)
        np.testing.assert_allclose(result, [0.5, 1.0])
