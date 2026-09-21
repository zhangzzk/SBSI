import numpy as np
import pytest

from scripts.audit_scene_classifier_calibration import geometry_error, case_summary


def test_zero_mean_probability_error_can_hide_geometry_coupling():
    geometry = np.zeros((2, 5))
    geometry[0, 0], geometry[1, 1] = 1., 1.
    old = geometry_error(geometry, [.5, .5], [.5, .5], [0, 1], [0, 1])
    assert old['residual_sum'] == 0.
    np.testing.assert_array_equal(old['geometry_residual_sum']/old['n'], [.25, -.25, 0, 0, 0])
    corrected = geometry_error(geometry, [0, 1], [0, 1], [0, 1], [0, 1])
    np.testing.assert_array_equal(corrected['geometry_residual_sum'], 0.)


def test_pooled_case_ratio_and_single_case_uncertainty():
    report = case_summary([2, 6], [-1., -3.], np.array([[1., 1.], [2., 0.], [0., 2.]]))
    assert report['estimate'] == -.5
    assert report['case_bootstrap_se'] == 0.
    assert case_summary([2], [-1.], np.ones((1, 1)))['case_bootstrap_se'] is None


def test_reject_wrong_geometry_shape():
    with pytest.raises(ValueError, match='aligned finite'):
        geometry_error(np.zeros((2, 4)), [.5, .5], [.5, .5], [0, 1], [0, 1])
