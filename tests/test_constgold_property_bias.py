"""Regression tests for the property-binned ConstGold response diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from sbsi.fixed_g0_domain import FLOW_FEATURES
from plots.constgold_property_bias import (
    assign_bins,
    binned_sufficient,
    collapse_bins,
    display_coordinate,
    histogram_edges,
    make_case_weights,
    population_mask,
    property_values,
    quantile_profile,
    summarize_branch,
)


def test_quantile_profile_populates_equal_count_bins():
    profile = quantile_profile(np.arange(80, dtype=float), 8)
    assert len(profile["centers"]) == 8
    assert profile["counts"].tolist() == [10] * 8
    assert np.all(np.diff(profile["edges"]) > 0)


@pytest.mark.parametrize("scale", ["linear", "log", "symlog"])
def test_histogram_edges_are_uniform_in_display_coordinate(scale):
    profile = {
        "centers": np.array([1.0, 2.0, 4.0, 8.0]),
        "edges": np.array([0.5, 1.5, 3.0, 6.0, 12.0]),
        "scale": scale,
    }
    edges, linthresh = histogram_edges(profile, bins=12)
    transformed = display_coordinate(edges, scale, linthresh=linthresh)
    assert len(edges) == 13
    assert np.all(np.diff(edges) > 0)
    np.testing.assert_allclose(np.diff(transformed), np.diff(transformed)[0])


def test_assign_bins_uses_half_open_internal_edges_and_keeps_endpoint():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    assert assign_bins(np.array([0.0, 0.999, 1.0, 2.0, 3.0]), edges).tolist() == [
        0,
        0,
        1,
        2,
        2,
    ]


def test_binned_sufficient_ignores_nonfinite_zero_weight_rows():
    values = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0], [5.0, 6.0]])
    weights = np.array([1.0, 0.0, 2.0, 1.0])
    assignment = np.array([0, 0, 1, 1])
    denominator, numerator = binned_sufficient(values, weights, assignment, 2)
    np.testing.assert_allclose(denominator, [1.0, 3.0])
    np.testing.assert_allclose(numerator, [[1.0, 2.0], [11.0, 14.0]])


def planted_branch_stats(cases=4, bins=2):
    result = {}
    for source, response in (("measured", 0.51), ("model", 0.50)):
        result[source] = {}
        for direction, sign in (("plus", 1.0), ("minus", -1.0)):
            denominator = np.full((cases, bins), 100.0)
            mean = sign * 0.02 * response
            numerator = np.zeros((cases, bins, 2))
            numerator[..., 0] = denominator * mean
            result[source][direction] = {
                "denominator": denominator,
                "numerator": numerator,
            }
    return result


def test_summarize_branch_recovers_planted_two_percent_bias():
    stats = planted_branch_stats()
    weights = make_case_weights(replicates=32, cases=4, seed=9)
    summary = summarize_branch(stats, weights, h=0.02)
    np.testing.assert_allclose(summary["measured_response"][:, 0], 0.51)
    np.testing.assert_allclose(summary["model_response"][:, 0], 0.50)
    np.testing.assert_allclose(
        summary["measured_response_standard_error"], 0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        summary["model_response_standard_error"], 0.0, atol=2e-14
    )
    np.testing.assert_allclose(summary["m_percent"], 2.0)
    np.testing.assert_allclose(
        summary["m_standard_error_percentage_points"], 0.0, atol=2e-14
    )


def test_collapsing_bins_preserves_all_sufficient_statistics():
    stats = planted_branch_stats(cases=3, bins=4)
    collapsed = collapse_bins(stats)
    for source in ("measured", "model"):
        for direction in ("plus", "minus"):
            assert collapsed[source][direction]["denominator"].shape == (3, 1)
            np.testing.assert_allclose(
                collapsed[source][direction]["denominator"], 400.0
            )
            np.testing.assert_allclose(
                collapsed[source][direction]["numerator"],
                stats[source][direction]["numerator"].sum(axis=1, keepdims=True),
            )


def test_property_values_use_fixed_g0_targets_and_context():
    context = np.zeros((2, len(FLOW_FEATURES)), dtype=float)
    context[:, FLOW_FEATURES.index("r_input_p")] = [25.0, 26.0]
    context[:, FLOW_FEATURES.index("circularized_Re_input_p")] = [0.3, 0.6]
    context[:, FLOW_FEATURES.index("nbr_flux_max")] = [1.0, 2.0]
    target = np.zeros((2, 4), dtype=float)
    target[:, 2] = [3.5, 5.0]
    target[:, 3] = 10.0 ** ((30.0 - np.array([24.5, 25.5])) / 2.5)
    values = property_values(context, target, np.array([0.1, 0.2]))
    np.testing.assert_allclose(values["magnitude_boost"], [0.5, 0.5])
    np.testing.assert_allclose(values["g0_radius"], [0.7, 1.0])
    np.testing.assert_allclose(values["blend_response_abs"], [0.1, 0.2])


def test_property_values_reject_nonphysical_crowding():
    context = np.ones((1, len(FLOW_FEATURES)), dtype=float)
    context[:, FLOW_FEATURES.index("nbr_flux_max")] = -1.0
    target = np.ones((1, 4), dtype=float)
    with pytest.raises(ValueError, match="crowding"):
        property_values(context, target, np.array([0.1]))


def test_population_mask_applies_strict_true_magnitude_boundary():
    context = np.zeros((4, len(FLOW_FEATURES)), dtype=float)
    context[:, FLOW_FEATURES.index("r_input_p")] = [25.9, 26.0, 26.1, 24.0]
    assert population_mask(context, None).tolist() == [True, True, True, True]
    assert population_mask(context, 26.0).tolist() == [True, False, False, True]


def test_population_mask_rejects_empty_or_nonfinite_boundary():
    context = np.zeros((2, len(FLOW_FEATURES)), dtype=float)
    context[:, FLOW_FEATURES.index("r_input_p")] = [26.0, 27.0]
    with pytest.raises(ValueError, match="removes every"):
        population_mask(context, 26.0)
    with pytest.raises(ValueError, match="must be finite"):
        population_mask(context, np.inf)
