import numpy as np

from scripts.audit_common_optimizer_pilot import paired_marginal_summary


def test_per_leg_marginals_keep_different_success_populations():
    old = np.array([[.1, 0], [.2, 0], [np.nan, np.nan], [.4, 0]])
    new = np.array([[.1, 0], [np.nan, np.nan], [.3, 0], [.5, 0]])
    result = paired_marginal_summary(old, new, np.ones(4, dtype=bool))
    assert result['original_only'] == result['candidate_only'] == 1
    assert result['original_selected_usable'] == result['candidate_selected_usable'] == 3
    np.testing.assert_allclose(result['original_moments'][0], .7/3)
    np.testing.assert_allclose(result['candidate_moments'][0], .9/3)
    np.testing.assert_allclose(result['moment_difference'][0], .2/3)
    assert result['moment_difference'][0] != .05  # common-success-only difference


def test_identical_measurements_cancel_paired_error_exactly():
    shapes = np.array([[.1, -.2], [.3, .1], [np.nan, np.nan], [.4, -.3]])
    result = paired_marginal_summary(shapes, shapes, np.array([True, False, True, True]))
    np.testing.assert_array_equal(result['moment_difference'], 0)
    np.testing.assert_array_equal(result['moment_difference_paired_object_se'], 0)
    assert result['selected_mass_difference'] == result['selected_mass_paired_object_se'] == 0
