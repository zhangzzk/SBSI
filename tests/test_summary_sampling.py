"""Known posterior, mixed prior and rejection bookkeeping checks."""
import numpy as np
import pytest

from sbsi.summary_sampling import prior_metropolis


def test_normal_posterior_and_evidence_ratio_from_fresh_prior_proposals():
    rng = np.random.default_rng(621)
    x = rng.normal(size=(32, 4000))
    y, sigma = .8, .7
    likelihood = -.5*((x-y)/sigma)**2
    history, accepted = prior_metropolis(likelihood, rng.uniform(1e-12, 1, (32, 3999)))
    samples = np.take_along_axis(x, history, axis=1)[:, 1000:]
    mean = y/(1+sigma**2)
    variance = sigma**2/(1+sigma**2)
    for statistic, expected in ((samples, mean), ((samples-mean)**2, variance)):
        batch_means = statistic.mean(1)
        assert abs(batch_means.mean()-expected) < 6*batch_means.std(ddof=1)/np.sqrt(len(batch_means))
    y2 = .9
    ratios = np.exp(-.5*((samples-y2)/sigma)**2+.5*((samples-y)/sigma)**2).mean(1)
    expected_ratio = np.exp(-.5*(y2*y2-y*y)/(1+sigma*sigma))
    assert abs(ratios.mean()-expected_ratio) < 6*ratios.std(ddof=1)/np.sqrt(len(ratios))
    assert .1 < accepted.mean() < .9


def test_point_mass_prior_is_preserved_without_a_learned_density():
    rng = np.random.default_rng(774)
    states = rng.random((32, 4000)) < .2
    history, _ = prior_metropolis(np.where(states, np.log(4.), 0.), rng.uniform(1e-12, 1, (32, 3999)))
    probability = np.take_along_axis(states, history, axis=1)[:, 1000:].mean(1)
    # Prior P(1)=.2; K(1)=4, K(0)=1 => posterior P(1)=.5.
    assert abs(probability.mean()-.5) < 6*probability.std(ddof=1)/np.sqrt(len(probability))


def test_zero_density_initials_escape_and_rejected_states_are_recorded():
    history, accepted = prior_metropolis([[-np.inf, -np.inf, 0., -100., 1.]], [[.5, .5, .5, .5]])
    np.testing.assert_array_equal(history, [[0, 0, 2, 2, 4]])
    np.testing.assert_array_equal(accepted, [[False, True, False, True]])
    for values, uniforms in (([[np.nan, 0]], [[.5]]), ([[np.inf, 0]], [[.5]]),
                             ([[0, 1]], [[0]]), ([[0, 1]], [[1]])):
        with pytest.raises(ValueError):
            prior_metropolis(values, uniforms)
