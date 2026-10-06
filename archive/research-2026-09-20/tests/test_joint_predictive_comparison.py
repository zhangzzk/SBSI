import numpy as np

from scripts.compare_joint_predictive_results import integrate_moments, residual_modes


def test_pooled_means_use_counts_not_equal_scene_weighting():
    raw = {'mass': [[10.], [90.]], 'moment_mass': [[10.], [90.]],
           'first': [[[10., 20.]], [[270., 360.]]]}
    actual = integrate_moments(raw, np.ones((1, 2)))
    np.testing.assert_allclose(actual['mean'], [[[2.8, 3.8]]])


def test_even_offset_does_not_disappear_into_small_response_error():
    data = {source: {} for source in ('observed', 'generated')}
    for source in data:
        for leg, sign in [('plus', 1), ('minus', -1)]:
            shape = sign*.02*.7 + (.003+sign*.02*.01 if source == 'generated' else 0)
            data[source][leg] = {'mass': [[10.], [90.]], 'moment_mass': [[10.], [90.]],
                                 'first': [[[10*shape, 0.]], [[90*shape, 0.]]]}
    result = residual_modes(data, np.ones((1, 2)), .02)
    np.testing.assert_allclose(result['mean/even'], [[[.003, 0.]]])
    np.testing.assert_allclose(result['mean/odd_derivative'], [[[.01, 0.]]])
    np.testing.assert_allclose(result['selection_fraction/even'], 0.)
