import numpy as np
import pytest

from scripts.audit_flow_shape_symmetry import ACTIONS, orbit_statistics, sign_action


def test_actions_are_involutions_and_preserve_scalar_physics():
    context = np.arange(24, dtype=float).reshape(3, 8)
    target = np.arange(12, dtype=float).reshape(3, 4)
    for action in ACTIONS:
        rotated_x, rotated_y = sign_action(context, target, action)
        np.testing.assert_array_equal(rotated_x[:, :2], context[:, :2]*action)
        np.testing.assert_array_equal(rotated_y[:, :2], target[:, :2]*action)
        np.testing.assert_array_equal(rotated_x[:, 2:], context[:, 2:])
        np.testing.assert_array_equal(rotated_y[:, 2:], target[:, 2:])
        recovered_x, recovered_y = sign_action(rotated_x, rotated_y, action)
        np.testing.assert_array_equal(recovered_x, context)
        np.testing.assert_array_equal(recovered_y, target)


def test_symmetric_density_has_zero_orbit_diagnostic():
    values = np.repeat(np.array([[-1000.], [1000.]]), 4, axis=1)
    stats = orbit_statistics(values)
    assert stats['orbit_minus_identity_nll'] == 0
    np.testing.assert_array_equal(stats['action_rms_log_density_delta'], 0)


def test_density_mixture_averages_probabilities_not_log_densities():
    probabilities = np.array([[1., 2., 3., 6.]])
    stats = orbit_statistics(np.log(probabilities))
    assert stats['orbit_average_nll'] == pytest.approx(-np.log(3.))
    assert stats['orbit_minus_identity_nll'] != pytest.approx(-np.log(probabilities).mean())


def test_nonfinite_orbit_scores_are_rejected():
    with pytest.raises(ValueError, match='finite'):
        orbit_statistics([[0., np.nan, 0., 0.]])
