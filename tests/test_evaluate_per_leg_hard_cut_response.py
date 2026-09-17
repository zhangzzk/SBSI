import numpy as np
import pytest

from scripts.evaluate_per_leg_hard_cut_response import (
    bootstrap_relative_bias,
    self_response,
)


def test_self_response_collapses_to_the_trace_half():
    values = np.array(
        [
            [0.8, 0.01, -0.02, 0.6],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(self_response(values), [0.7, 1.0])


def test_bootstrap_relative_bias_recovers_an_exact_ratio():
    measured = np.full((8, 3), 0.5)
    model = np.full((8, 3), 0.5) * np.array([1.0, 0.99, 1.02])
    summary = bootstrap_relative_bias(model, measured, resamples=200)
    np.testing.assert_allclose(summary["m_percent"], [0.0, -1.0, 2.0], atol=1.0e-9)
    # Every case is identical, so resampling cannot move the ratio.
    np.testing.assert_allclose(
        summary["m_standard_error_percentage_points"], np.zeros(3), atol=1.0e-9
    )


def test_bootstrap_relative_bias_reports_case_scatter():
    rng = np.random.default_rng(11)
    measured = rng.normal(0.5, 0.01, size=(40, 2))
    model = measured * 1.01 + rng.normal(0.0, 0.004, size=(40, 2))
    summary = bootstrap_relative_bias(model, measured, resamples=2000)
    assert np.all(summary["m_standard_error_percentage_points"] > 0.0)
    low = summary["m_ci95_percent"][:, 0]
    high = summary["m_ci95_percent"][:, 1]
    assert np.all(low < summary["m_percent"])
    assert np.all(summary["m_percent"] < high)


def test_bootstrap_relative_bias_rejects_degenerate_input():
    with pytest.raises(ValueError):
        bootstrap_relative_bias(np.ones((1, 2)), np.ones((1, 2)), resamples=10)
    with pytest.raises(ValueError):
        bootstrap_relative_bias(np.ones((4, 2)), np.ones((4, 3)), resamples=10)
