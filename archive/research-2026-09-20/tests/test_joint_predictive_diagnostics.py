import numpy as np
import pytest

from scripts.joint_predictive_diagnostics import (
    HISTOGRAM_SHAPE, empty_statistics, output_statistics, summarize_statistics,
)
from scripts.evaluate_constgold_truth_parent import output_cuts


def test_draw_weights_preserve_parent_mass_and_ignore_unusable_rows():
    values = np.array([[.1, .2, 4., 100.], [np.nan, np.nan, -1., np.nan]])
    observed = output_statistics(values, [1., 0.], output_cuts(values))
    draws = np.repeat(values[:, None, :], 8, axis=1)
    model = output_statistics(draws, np.broadcast_to(np.array([[1.], [0.]])/8, (2, 8)).copy(), output_cuts(draws))
    for field in observed:
        np.testing.assert_allclose(observed[field], model[field], atol=1e-12)
    assert observed["histogram"].sum() == 1


def test_joint_histogram_detects_dependence_with_identical_marginals():
    positive = np.array([[-.5, -.5, 4., 100.], [.5, .5, 4., 100.]])
    negative = positive.copy()
    negative[:, 1] *= -1
    a = output_statistics(positive, np.ones(2), output_cuts(positive))
    b = output_statistics(negative, np.ones(2), output_cuts(negative))
    np.testing.assert_allclose(a["first"], b["first"])
    for axis in range(4):
        drop = tuple(i for i in range(4) if i != axis)
        np.testing.assert_array_equal(a["histogram"].reshape(HISTOGRAM_SHAPE).sum(drop),
                                      b["histogram"].reshape(HISTOGRAM_SHAPE).sum(drop))
    assert np.abs(a["histogram"] - b["histogram"]).sum()/4 == 1
    assert a["second"][0, 0, 1] == .5
    assert b["second"][0, 0, 1] == -.5


def test_predictive_cuts_act_on_each_draw_not_on_observed_membership():
    values = np.array([[[.1, 0., 2., 100.], [.2, 0., 4., 100.]]])
    stats = output_statistics(values, np.full((1, 2), .5), output_cuts(values))
    np.testing.assert_array_equal(stats["mass"], [1., .5, .5])
    np.testing.assert_allclose(stats["first"][:, 0], [.15, .1, .1])


def test_histogram_retains_extreme_outputs_and_strict_radius_boundary():
    values = np.array([[-1.1, .2, .01, 1e-8], [1.1, .3, 100., 1e8], [.1, .1, 3., 100.]])
    stats = output_statistics(values, np.ones(3), output_cuts(values))
    histogram = stats["histogram"].reshape(HISTOGRAM_SHAPE)
    assert histogram.sum() == 3
    assert histogram[0].sum() == histogram[-1].sum() == 1
    assert histogram[:, :, 2].sum() == 1  # radius=3 belongs to <=3 bin


@pytest.mark.parametrize("weights", [[-1.], [np.nan]])
def test_invalid_weights_rejected(weights):
    values = np.array([[0., 0., 4., 100.]])
    with pytest.raises(ValueError, match="nonnegative"):
        output_statistics(values, weights, output_cuts(values))


def test_positive_weight_invalid_measurement_rejected():
    values = np.array([[np.nan, 0., 4., 100.]])
    with pytest.raises(ValueError, match="finite"):
        output_statistics(values, [1.], output_cuts(values))


def test_empty_usable_batch_returns_zero_statistics():
    values = np.array([[np.nan, np.nan, -1., np.nan]])
    stats = output_statistics(values, [0.], output_cuts(values))
    for value in stats.values():
        assert not np.any(value)


def test_joint_disk_violation_is_reported_without_discarding_model_mass():
    values = np.array([[.8, .8, 4., 100.]])
    stats = output_statistics(values, [1.], output_cuts(values))
    assert stats["outside_disk_mass"] == 1
    assert stats["histogram"].sum() == stats["mass"][0] == 1


def test_nonpositive_photometry_remains_in_mass_and_invalid_histogram_bin():
    values = np.array([[.1, .2, 4., 100.], [.9, .1, 4., 0.], [.3, .4, 0., 100.], [.2, .1, -1., 100.]])
    stats = output_statistics(values, np.ones(4), output_cuts(values))
    np.testing.assert_array_equal(stats["mass"], [4, 1, 1])
    np.testing.assert_array_equal(stats["moment_mass"], [1, 1, 1])
    assert stats["invalid_photometry_mass"] == 3
    assert stats["histogram"].sum()+stats["invalid_photometry_mass"] == 4
    assert stats["zero_radius_mass"] == stats["zero_flux_mass"] == stats["negative_photometry_mass"] == 1
    np.testing.assert_allclose(stats["first"][:, :2], [[.1, .2]]*3)


def test_invalid_bin_is_included_in_histogram_tv_and_log_moment_denominator():
    measured, generated = empty_statistics(2, 3), empty_statistics(2, 3)
    for i in range(2):
        values = np.array([[.1, .2, 4., 100.], [.1, .2, 4., 100.]])
        for destination, invalid in ((measured, False), (generated, True)):
            current = values.copy()
            if invalid:
                current[1, 3] = 0
            stats = output_statistics(current, np.ones(2), output_cuts(current))
            for field in destination:
                destination[field][i] = stats[field]
    result = summarize_statistics(measured, generated, [[2, 0], [1, 1], [0, 2]])
    assert result["joint_histogram"]["total_variation"] == .5
    assert result["joint_histogram"]["generated"].sum() == .5
    assert result["joint_histogram"]["invalid_photometry_bin_probability"]["generated"] == .5
    np.testing.assert_allclose(result["mean"]["generated_minus_observed"], 0)
    np.testing.assert_array_equal(result["undefined_log_moment_fraction"]["generated"], [.5, 0, 0])


def test_scene_bootstrap_identity_and_known_shift():
    measured, generated = empty_statistics(2, 3), empty_statistics(2, 3)
    for i in range(2):
        values = np.array([[.1*i, .1, 4., 100.], [.1*i, -.1, 5., 100.]])
        for destination, shift in ((measured, 0.), (generated, .05)):
            shifted = values.copy()
            shifted[:, 0] += shift
            stats = output_statistics(shifted, np.ones(2), output_cuts(shifted))
            for field in destination:
                destination[field][i] = stats[field]
    weights = np.array([[2, 0], [1, 1], [0, 2]])
    identity = summarize_statistics(measured, measured, weights)
    assert identity["joint_histogram"]["total_variation"] == 0
    result = summarize_statistics(measured, generated, weights)
    np.testing.assert_allclose(result["mean"]["generated_minus_observed"][:, 0], .05)
    np.testing.assert_allclose(result["mean"]["paired_case_bootstrap_se"], 0, atol=1e-16)
    np.testing.assert_allclose(result["covariance"]["generated_minus_observed"], 0, atol=1e-14)
