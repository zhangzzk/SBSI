import numpy as np

from sbsi.particle_diagnostics import combine_populations, empirical_second_moment


def test_population_combination_recovers_unnormalized_mixture_weights():
    # Equally likely proposal outcomes each represent a different region.
    # Their fhat values alone give .5, while the target region probability is .9.
    result = combine_populations(np.log([1., 9., 1., 9.]), [[0.], [1.], [0.], [1.]])
    np.testing.assert_allclose(result['mean'], [.9])
    np.testing.assert_allclose(np.exp(result['log_mean_normalizer']), 5.)
    shifted = combine_populations(np.log([1., 9., 1., 9.])+500., [[0.], [1.], [0.], [1.]])
    np.testing.assert_allclose(result['mean'], shifted['mean'])
    np.testing.assert_allclose(result['covariance'], shifted['covariance'])


def test_extinct_populations_count_as_zero_normalizer_and_shared_errors_cancel():
    result = combine_populations([0., -np.inf, 0., -np.inf], [[1., 2.], [np.nan, np.nan],
                                                              [3., 6.], [np.nan, np.nan]])
    np.testing.assert_allclose(np.exp(result['log_mean_normalizer']), .5)
    np.testing.assert_allclose(result['mean'], [2., 4.])
    contrast = np.array([-2., 1.])
    np.testing.assert_allclose(contrast@result['covariance']@contrast, 0., atol=1e-15)
    assert result['extinct_populations'] == 2


def test_ratio_influence_matches_independent_population_bootstrap():
    rng = np.random.default_rng(790)
    logz = rng.normal(0., .2, 256)
    f = np.column_stack([logz+rng.normal(0., .5, 256), logz**2])
    result = combine_populations(logz, f)
    indices = rng.integers(0, len(logz), (2000, len(logz)))
    z = np.exp(logz[indices])
    boot = (z[:, :, None]*f[indices]).sum(1)/z.sum(1)[:, None]
    np.testing.assert_allclose(result['covariance'], np.cov(boot.T), rtol=.1, atol=2e-5)


def test_empirical_second_moment_matches_correlated_particle_duplicates():
    particles = np.array([[-2., 1.], [-2., 1.], [2., 3.], [2., 3.]])
    result = empirical_second_moment(particles.mean(0), particles.var(0, ddof=1), len(particles))
    np.testing.assert_allclose(result, (particles**2).mean(0), rtol=0, atol=1e-15)
    old = particles.var(0, ddof=1)+particles.mean(0)**2
    assert np.all(old > result)


def test_all_four_population_contributions_retained_including_two_extinctions():
    result = combine_populations([np.log(2.), -np.inf, np.log(6.), -np.inf],
                                 [[1., 2.], [np.nan, np.nan], [3., 5.], [np.nan, np.nan]])
    np.testing.assert_allclose(result['mean'], [2.5, 4.25])
    np.testing.assert_allclose(result['weights'], [.25, 0., .75, 0.])
    np.testing.assert_allclose(np.exp(result['log_mean_normalizer']), 2.)
    assert result['independent_populations'] == 4 and result['extinct_populations'] == 2
    # Delta-method covariance from the four unnormalized influence vectors,
    # including both zero-mass vectors in the sample covariance denominator.
    influence = np.array([[-1.5, -2.25], [0., 0.], [1.5, 2.25], [0., 0.]])
    np.testing.assert_allclose(result['covariance'], influence.T@influence/12)
