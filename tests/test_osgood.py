"""
tests/test_osgood.py — Unit tests for src/osgood.py

Tests for F functions, averaged drifts, and known exact solutions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.osgood import (
    F_linear,
    F_log_osgood,
    F_sqrt,
    averaged_drift_linear,
    averaged_drift_log,
    averaged_drift_sqrt,
    averaged_drift_sqrt_negative,
)


# ---------------------------------------------------------------------------
# F function tests
# ---------------------------------------------------------------------------

class TestFunctions:
    def test_linear_at_zero(self):
        assert F_linear(0.0) == 0.0

    def test_linear_positive(self):
        assert F_linear(2.5) == 2.5

    def test_log_osgood_at_zero(self):
        """F(0) = 0 by continuity."""
        np.testing.assert_allclose(F_log_osgood(np.array([0.0])), [0.0])

    def test_log_osgood_at_one(self):
        """F(1) = 1 * log(e/1) = 1.0."""
        np.testing.assert_allclose(F_log_osgood(np.array([1.0])), [1.0], atol=1e-14)

    def test_log_osgood_at_half(self):
        """F(0.5) = 0.5 * log(e/0.5) = 0.5 * (1 + log(2))."""
        expected = 0.5 * np.log(np.e / 0.5)
        np.testing.assert_allclose(
            F_log_osgood(np.array([0.5]))[0], expected, atol=1e-14
        )

    def test_sqrt_at_zero(self):
        np.testing.assert_allclose(F_sqrt(np.array([0.0])), [0.0])

    def test_sqrt_at_one(self):
        np.testing.assert_allclose(F_sqrt(np.array([1.0])), [1.0])

    def test_sqrt_at_quarter(self):
        np.testing.assert_allclose(F_sqrt(np.array([0.25])), [0.5])

    def test_sqrt_negative_clamped(self):
        """sqrt(-x) → 0 for x < 0."""
        result = F_sqrt(np.array([-1.0, -0.5]))
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_sqrt_known_solution(self):
        """β^+(t) = t^2/4 satisfies β' = sqrt(β): d/dt(t^2/4) = t/2 = sqrt(t^2/4)."""
        t_vals = np.linspace(0.01, 1.0, 50)
        beta_plus = t_vals ** 2 / 4
        derivative = t_vals / 2
        computed = F_sqrt(beta_plus)
        np.testing.assert_allclose(computed, derivative, atol=1e-14)


# ---------------------------------------------------------------------------
# Averaged drift tests
# ---------------------------------------------------------------------------

class TestAveragedDriftLinear:
    def test_zero_beta(self):
        eps = 0.1
        result = averaged_drift_linear(np.array([0.0]), eps)
        np.testing.assert_allclose(result, [eps / 2], atol=1e-14)

    def test_nonzero_beta(self):
        eps = 0.2
        beta = np.array([0.5])
        result = averaged_drift_linear(beta, eps)
        np.testing.assert_allclose(result, [0.5 + eps / 2], atol=1e-14)

    def test_matches_integral(self):
        """Compare to numerical ∫_0^ε (β+s)/ε ds."""
        eps = 0.15
        beta = np.array([0.3])
        from scipy.integrate import quad
        integral_val, _ = quad(lambda s: (beta[0] + s), 0, eps)
        expected = integral_val / eps
        result = averaged_drift_linear(beta, eps)
        np.testing.assert_allclose(result[0], expected, atol=1e-14)


class TestAveragedDriftLog:
    def test_matches_numerical_integral(self):
        """Compare closed-form to numerical quadrature."""
        from scipy.integrate import quad

        eps = 0.1
        for beta_val in [0.01, 0.1, 0.5, 1.0]:
            integral_val, _ = quad(
                lambda s: (beta_val + s) * np.log(np.e / (beta_val + s)),
                0, eps,
            )
            expected = integral_val / eps
            result = averaged_drift_log(np.array([beta_val]), eps)
            np.testing.assert_allclose(
                result[0], expected, rtol=1e-8, atol=1e-10,
                err_msg=f"Failed at beta={beta_val}"
            )

    def test_zero_beta(self):
        """Averaged drift at β=0 should be (1/ε) ∫_0^ε s log(e/s) ds."""
        from scipy.integrate import quad
        eps = 0.05
        integral_val, _ = quad(lambda s: s * np.log(np.e / s), 1e-14, eps)
        expected = integral_val / eps
        result = averaged_drift_log(np.array([0.0]), eps)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)


class TestAveragedDriftSqrt:
    def test_matches_closed_form(self):
        """(2/(3ε))[(β+ε)^{3/2} - β^{3/2}]."""
        eps = 0.1
        beta = np.array([0.5])
        expected = (2.0 / (3.0 * eps)) * ((0.5 + eps) ** 1.5 - 0.5 ** 1.5)
        result = averaged_drift_sqrt(beta, eps)
        np.testing.assert_allclose(result[0], expected, atol=1e-14)

    def test_matches_numerical_integral(self):
        from scipy.integrate import quad
        eps = 0.1
        for beta_val in [0.0, 0.1, 0.5, 1.0]:
            integral_val, _ = quad(lambda s: np.sqrt(beta_val + s), 0, eps)
            expected = integral_val / eps
            result = averaged_drift_sqrt(np.array([beta_val]), eps)
            np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_positive_output(self):
        result = averaged_drift_sqrt(np.array([0.0, 0.1, 1.0]), eps=0.1)
        assert np.all(result >= 0.0)


class TestAveragedDriftSqrtNegative:
    def test_at_zero_beta(self):
        """At β=0, negative ramp drift = (1/ε) ∫_{-ε}^0 sqrt(max(0,s)) ds = 0."""
        result = averaged_drift_sqrt_negative(np.array([0.0]), eps=0.1)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-14)

    def test_nonnegative_output(self):
        betas = np.linspace(-1.0, 2.0, 50)
        result = averaged_drift_sqrt_negative(betas, eps=0.1)
        assert np.all(result >= 0.0)

    def test_minimal_fixed_point_at_zero(self):
        """β=0 is a fixed point of the negative ramp ODE for F_sqrt."""
        result = averaged_drift_sqrt_negative(np.array([0.0]), eps=0.05)
        assert result[0] == pytest.approx(0.0, abs=1e-14)
