import numpy as np
import pytest
from _script_loader import load_script_module


M = load_script_module('v39_exact_tail.py')


def test_tail_importance_expectation_covers_atoms_with_zero_center_likelihood():
    # Atom 3 has no centre contribution, but contributes at another shear.
    center = np.array([9., 2., 1., 0., 0.])
    prior = np.array([.1, .2, .3, .4, 0.])
    logs = np.full(5, -np.inf)
    logs[center > 0] = np.log(center[center > 0])
    d = M.build_tail_distribution(logs, prior, [0], delta=.1)
    expected_q = np.array([0., .9*2/3+.1*2/9, .9/3+.1/3, .1*4/9, 0.])
    q = np.diff(d.cdf, prepend=0.)
    np.testing.assert_allclose(q, expected_q, atol=1.e-15)
    np.testing.assert_allclose(np.exp(d.log_center_mass), 3.)
    other_shear = np.array([8., 1., 4., 7., 0.])
    support = np.flatnonzero(q)
    mids = (d.cdf[support]+np.r_[0., d.cdf[:-1]][support])/2
    ids, probabilities = d.inverse(mids)
    np.testing.assert_array_equal(ids, support)
    # Exact expectation over all one-draw outcomes is an independent oracle.
    estimate = other_shear[0] + np.sum(q[ids]*other_shear[ids]/probabilities)
    np.testing.assert_allclose(estimate, other_shear.sum(), atol=1.e-14)
    assert q[0] == q[4] == 0 and q[3] > 0


def test_uniform_prefix_and_object_identity_are_preserved():
    d = M.build_tail_distribution(np.log([1., 2., 3., 4.]), [.1, .2, .3, .4], [3], delta=.1)
    ids, probs = M.draw_tail(d, n_draws=64, seed=8701, object_id=1958)
    small = M.draw_tail(d, n_draws=16, seed=8701, object_id=1958)
    np.testing.assert_array_equal(small[0], ids[:16])
    np.testing.assert_array_equal(small[1], probs[:16])
    uniforms = np.random.default_rng(np.random.SeedSequence([8701, 1958])).random((64, 3))[:, 1]
    np.testing.assert_array_equal(ids, np.searchsorted(d.cdf, uniforms, side='right'))
    assert not np.any(ids == 3)


def test_extreme_log_scale_and_zero_mass_bins_do_not_break_support():
    logs = np.array([-np.inf, -1000., -1002., -np.inf, -1100.])
    d = M.build_tail_distribution(logs, [0., .1, .2, .3, .4], [1], delta=.1)
    ids, probability = d.inverse([0., np.nextafter(1., 0.)])
    np.testing.assert_array_equal(ids, [2, 4])
    assert np.all(probability > 0) and np.isfinite(d.log_center_mass)
    widths = np.diff(d.cdf, prepend=0.)
    assert np.all(widths[[2, 3, 4]] > 0) and np.all(widths[[0, 1]] == 0)


@pytest.mark.parametrize('candidates,delta', [([0, 0], .1), ([3], .1), ([0], 0), ([0], 1)])
def test_rejects_invalid_support_or_defensive_fraction(candidates, delta):
    with pytest.raises(ValueError):
        M.build_tail_distribution(np.log([1., 2., 3.]), [.2, .3, .5], candidates, delta=delta)
