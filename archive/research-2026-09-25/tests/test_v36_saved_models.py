import numpy as np

from scripts.compare_v36_saved_models import normalized_response
from scripts.audit_v36_saved_rows import response_bin_contributions


def test_per_leg_normalization_is_invariant_to_independent_mass_scaling():
    x = np.array([[100., .8, .4], [80., -.64, -.32]])
    np.testing.assert_allclose(normalized_response(x), [.4, .2])
    np.testing.assert_allclose(normalized_response(x*np.array([[3.], [7.]])), [.4, .2])


def test_raw_sum_response_can_fake_bias_when_only_counts_differ():
    x = np.array([[100., .8, .4], [100., -.8, -.4]])
    y = .99*x
    np.testing.assert_allclose(normalized_response(x), normalized_response(y))
    assert abs((x[0,1]-x[1,1])/(y[0,1]-y[1,1])-1) > .01


def test_bin_allocation_closes_with_different_leg_masses():
    sums = np.array([[[100., 1., .2], [10., -.1, .3]], [[80., -.4, .1], [20., -.3, -.1]]])
    np.testing.assert_allclose(response_bin_contributions(sums).sum(0), normalized_response(sums.sum(1)))
    batched = np.stack((sums, sums*2), axis=1)
    np.testing.assert_allclose(response_bin_contributions(batched).sum(1), normalized_response(batched.sum(2)))
