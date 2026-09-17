import numpy as np
import pytest

from sbsi.flow_guard_response import make_guard_band_cuts, make_guard_cuts
from scripts.evaluate_per_leg_hard_cut_response import (
    bootstrap_relative_bias,
    headline_cut,
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


def test_headline_cut_prefers_the_deployment_cut():
    cuts = make_guard_cuts((2.8, 3.0, 3.2), (25.6, 25.8, 26.0))
    assert cuts[headline_cut(cuts)]["name"] == "radius_gt_3_magnitude_lt_25.8"


def test_headline_cut_falls_back_for_a_band_bank():
    # A band profile has no cumulative deployment cut.  The headline must
    # still resolve, because it is used both for progress and for the final
    # summary line after the results file is written.
    cuts = make_guard_band_cuts((2.6, 3.0, 3.5))
    index = headline_cut(cuts)
    assert 0 <= index < len(cuts)
    assert cuts[index]["name"] == "radius_gt_3.5"

    limited = make_guard_band_cuts((2.6, 3.0, 3.5), magnitude_max=25.8)
    assert limited[headline_cut(limited)]["name"] == "radius_gt_3.5_magnitude_lt_25.8"
