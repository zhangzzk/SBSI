"""Measure/support checks for the optional analytic prior; no GalSBI import."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import beta, betaprime, norm

from sbsi.galsbi_prior import ExactMemo, GalSBIContinuousPrior, upper_gamma


def prior_fixture():
    p = dict(galaxy_types=['red', 'blue'], lum_fct_parametrization='truncated_logexp',
             template_coeff_sampler='dirichlet_alpha_mode', logr50_sampling_method='sdss_fit',
             ellipticity_sampling_method='beta_mode_red_blue', sersic_sampling_method='blue_red_betaprime',
             reference_band='i', lum_fct_filter_band='B', lum_fct_m_max=-10.,
             gals_mag_min=14., gals_mag_max=28., sersic_n_min=.2, sersic_n_max=4., ell_beta_emax=.9)
    for t in p['galaxy_types']:
        p.update({f'lum_fct_m_star_{t}_intcpt': -20., f'lum_fct_m_star_{t}_slope': -1.,
                  f'lum_fct_phi_star_{t}_amp': .01, f'lum_fct_phi_star_{t}_exp': -.2,
                  f'lum_fct_z_const_{t}': 2., f'lum_fct_alpha_{t}': -1.2,
                  f'template_coeff_weight_{t}': np.ones(5), f'template_coeff_z1_{t}': 3.,
                  f'template_coeff_alpha0_{t}_std': .1, f'template_coeff_alpha1_{t}_std': .08,
                  f'logr50_sdss_fit_M0_{t}': -20., f'logr50_sdss_fit_sigma1_{t}': .5,
                  f'logr50_sdss_fit_sigma2_{t}': .3, f'logr50_alpha_{t}': -.5,
                  f'sersic_betaprime_{t}_mode': 2., f'sersic_betaprime_{t}_size': 5.,
                  f'sersic_betaprime_{t}_mode_alpha': .6,
                  f'ell_beta_ab_sum_{t}': 7., f'ell_beta_mode_{t}': .3})
        for index in range(5):
            p[f'template_coeff_alpha0_{t}_{index}'] = .2
            p[f'template_coeff_alpha1_{t}_{index}'] = .2
    p.update(logr50_sdss_fit_a_red=.5, logr50_sdss_fit_b_red=-3.,
             logr50_sdss_fit_alpha_blue=.3, logr50_sdss_fit_beta_blue=.6,
             logr50_sdss_fit_gamma_blue=-1.)

    def photo(z, M, c):
        return dict(m_r=M+40, m_i=M+40, M_i=M)

    return GalSBIContinuousPrior(p, photo, lambda z: z*z,
                                lambda z: np.zeros_like(z)-10., {'red': 3., 'blue': 3.}, grid_size=1025)


def test_negative_schechter_gamma_matches_direct_integral():
    for shape in (-.2, 0., .7):
        for x in (.01, 1., 5.):
            expected = quad(lambda t: t**(shape-1)*np.exp(-t), x, np.inf, epsabs=1e-10)[0]
            np.testing.assert_allclose(upper_gamma(shape, x), expected, rtol=2e-9)


def test_native_density_has_size_jacobian_and_conditional_truncation():
    prior = prior_fixture()
    d = dict(type=np.array([0, 0]), z=np.array([.2, 2.]), M_B=np.array([-20., -20.]),
             c=np.full((2, 5), .2), R_kpc=np.array([1., 3.]), n=np.array([2., 2.]),
             a=np.array([.3, .3]), psi=np.array([0., 1.]))
    total, parts = prior.log_intensity(d, return_factors=True)
    assert np.isfinite(total).all()
    # Numerically integrate the native R density, rather than rechecking its formula.
    mean, sigma = prior.size_parameters(d['z'], d['M_B'], 'red')
    for k in range(2):
        integral = quad(lambda R: np.exp(norm.logpdf(np.log(R), mean[k], sigma[k])-np.log(R)),
                        0, np.inf)[0]
        np.testing.assert_allclose(integral, 1, rtol=1e-8)
    a, b = prior.sersic_parameters(d['z'], 'red')
    correction = parts['sersic']-betaprime.logpdf(d['n'], a, b)
    assert abs(correction[0]-correction[1]) > .1  # It cannot be a global constant.
    for k in range(2):
        integral = quad(lambda n: np.exp(betaprime.logpdf(n, a[k], b)+correction[k]), .2, 4.)[0]
        np.testing.assert_allclose(integral, 1, rtol=1e-8)
    ac = prior.sed_alpha(d['z'], 'red')
    expected_at_center = gammaln(ac.sum(1))-gammaln(ac).sum(1)+((ac-1)*np.log(.2)).sum(1)
    np.testing.assert_allclose(parts['sed'], expected_at_center)
    aa, bb = prior.shape_parameters('red')
    # Cartesian p(e1,e2) adds 1/a; angular integration must recover unit probability.
    integral = quad(lambda a: beta.pdf(a, aa, bb, scale=.9)/(2*np.pi*a)*2*np.pi*a, 0, .9)[0]
    np.testing.assert_allclose(integral, 1, rtol=1e-8)
    for key, bad in (('n', 5.), ('a', 1.), ('type', 9), ('z', 4.), ('M_B', -5.)):
        shifted = {k: v.copy() for k, v in d.items()}
        shifted[key][0] = bad
        assert prior.log_intensity(shifted)[0] == -np.inf


def test_continuous_sampling_and_normalization_contract():
    prior = prior_fixture()
    sample, report = prior.sample(4096, 18)
    assert 0 < report['n_retained'] < report['n_proposals']
    assert report['maximum_magnitude_inverse_cdf_error'] < 1e-5
    assert len(np.unique(sample['z'])) == len(sample['z'])
    logp = prior.log_prob(sample, surface_density=report['surface_density_sr'])
    assert np.isfinite(logp).all()
    # Surface density is an explicit constant, not inferred from evaluation points.
    np.testing.assert_allclose(prior.log_prob(sample, surface_density=2*report['surface_density_sr']),
                               logp-np.log(2))
    # A tiny draw can contain only one galaxy type.
    _, small = prior.sample(1, 18)
    assert small['n_proposals'] == 1
    with pytest.raises(ValueError, match='positive integer'):
        prior.sample(0, 18)


def test_exact_memo_returns_function_values_and_reuses_them():
    calls = []
    memo = ExactMemo(lambda z: calls.append(len(z)) or np.sqrt(z), limit=4)
    z = np.array([[.5, 2.], [.5, 3.]])
    assert np.array_equal(memo(z), np.sqrt(z))
    assert np.array_equal(memo(z[::-1]), np.sqrt(z[::-1])) and calls == [3]
    memo(np.array([4., 5.]))
    memo(np.array([6.]))  # beyond the limit: memory restarts with this call
    assert calls == [3, 2, 1] and np.array_equal(memo.keys, [6.])


def test_window_rejected_proposals_do_not_shift_retained_draws():
    """Skipping work for window-rejected rows leaves every retained row bit-identical."""
    narrow, wide = prior_fixture(), prior_fixture()
    wide.p.update(gals_mag_min=-np.inf, gals_mag_max=np.inf)
    a, _ = narrow.sample(4000, 7)
    b, meta = wide.sample(4000, 7)
    assert meta['n_retained'] == 4000 > len(a['z']) > 0
    for key, value in a.items():
        assert np.array_equal(value, b[key][a['proposal_index']]), key
