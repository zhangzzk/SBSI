"""Numerical-generator measure tests; no likelihood, GPU, or production change."""
import copy

import numpy as np
import pytest
from scipy.integrate import quad

from sbsi.galsbi_prior import GalSBIContinuousPrior, upper_gamma
from sbsi.numerical_native_intensity import (
    luminosity_table_coverage, numerical_luminosity_log_density,
    numerical_native_log_intensity, numerical_redshift_log_density,
)
from sbsi.scene_exchange import numerical_over_analytic
from test_galsbi_prior import prior_fixture


@pytest.fixture
def prior():
    base = prior_fixture()
    p = dict(base.p)
    p['lum_fct_phi_star_blue_amp'] *= 3
    p['lum_fct_alpha_blue'] = .3
    # Coarsen deliberately to resolve interpolation-edge support in tests.
    return GalSBIContinuousPrior(p, base.photometry, base.volume,
        lambda z: np.zeros_like(z)-18.373, {'red': 3., 'blue': 2.3},
        grid_size=257, magnitude_step=.1)


def row(kind=0, z=.731, magnitude=-20.):
    return dict(type=np.array([kind]), z=np.array([z]), M_B=np.array([magnitude]),
        c=np.full((1, 5), .2), R_kpc=np.array([1.]), n=np.array([2.]),
        a=np.array([.3]), psi=np.array([1.]))


def logmax_at(prior, name, z):
    ms, _, alpha = prior.luminosity_parameters(np.asarray(z), name)
    cap = np.minimum(prior.magnitude_cap(np.asarray(z)), prior.p['lum_fct_m_max'])
    return ms, np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))


def test_generated_rows_and_shared_support_agree_with_existing_helper(prior):
    data, metadata = prior.sample(6000, 20261021)
    before = {k: v.copy() for k, v in data.items()}
    logq, qparts = numerical_native_log_intensity(prior, data, return_factors=True)
    logp, pparts = prior.log_intensity(data, return_factors=True)
    common = np.isfinite(logp)
    assert metadata['n_retained'] > 5000
    assert np.isfinite(logq).all()
    assert common.sum() > 5000
    np.testing.assert_allclose(logq[common]-logp[common], numerical_over_analytic(prior, data)[common], atol=2e-13, rtol=0)
    for name in ['sed', 'size', 'sersic', 'shape']:
        np.testing.assert_array_equal(qparts[name][common], pparts[name][common])
    for key in before:
        np.testing.assert_array_equal(data[key], before[key])


@pytest.mark.parametrize('kind', [0, 1])
def test_interpolation_edge_sliver_is_supported_beyond_analytic_cap(prior, kind):
    d = row(kind)
    name = prior.types[kind]
    ms, logmax = logmax_at(prior, name, d['z'])
    tab = prior.tables[name]
    numerical_edge = ms-np.interp(logmax, tab['log_g'], tab['inverse_y'])
    analytic_edge = prior.magnitude_cap(d['z'])
    assert (numerical_edge > analytic_edge+1e-8).all()
    d['M_B'] = (numerical_edge+analytic_edge)/2
    assert np.isfinite(numerical_native_log_intensity(prior, d)).all()
    assert np.isneginf(prior.log_intensity(d)).all()
    assert np.isneginf(numerical_over_analytic(prior, d)).all()
    d['M_B'] = numerical_edge+1e-7
    assert np.isneginf(numerical_native_log_intensity(prior, d)).all()


@pytest.mark.parametrize('kind', [0, 1])
def test_redshift_density_normalizes_with_absolute_type_rate(prior, kind):
    name, tab = prior.types[kind], prior.tables[prior.types[kind]]
    middle = (tab['z'][1:]+tab['z'][:-1])/2
    density = np.exp(numerical_redshift_log_density(prior, name, middle))
    np.testing.assert_allclose(np.dot(density, np.diff(tab['z'])), 1., atol=2e-15, rtol=0)
    np.testing.assert_allclose(np.dot(tab['rate']*density, np.diff(tab['z'])), tab['rate'], rtol=2e-15)
    outside = np.array([-1., 0., tab['z'][-1]+1e-8, np.inf, np.nan])
    assert np.isneginf(numerical_redshift_log_density(prior, name, outside)).all()
    # Independent derivative of the exact forward map at interior uniforms.
    u = np.array([.11, .43, .83])
    z = np.interp(u, tab['cdf'], tab['z'])
    h = 1e-8
    derivative = (np.interp(u+h, tab['cdf'], tab['z'])-np.interp(u-h, tab['cdf'], tab['z']))/(2*h)
    np.testing.assert_allclose(np.exp(numerical_redshift_log_density(prior, name, z)), 1/derivative, rtol=3e-8)


def integrate_luminosity(prior, name, z):
    tab = prior.tables[name]
    ms, logmax = logmax_at(prior, name, z)
    low = max(tab['log_g'][0], logmax+np.log(np.finfo(float).tiny))
    high = min(tab['log_g'][-1], logmax)
    bounds = np.sort(ms-np.interp([low, high], tab['log_g'], tab['inverse_y']))
    knots = ms-tab['inverse_y']
    points = sorted(knots[(knots > bounds[0]) & (knots < bounds[1])])
    integral, error = quad(lambda M: float(np.exp(numerical_luminosity_log_density(prior, name, z, M))),
        *bounds, points=points, epsabs=2e-10, epsrel=2e-10, limit=len(points)+100)
    assert error < 1e-8
    return integral


@pytest.mark.parametrize('kind,z', [(0, .1), (0, 2.5), (1, .1), (1, 2.)])
def test_conditional_inverse_cdf_normalization_and_forward_jacobian(prior, kind, z):
    name = prior.types[kind]
    coverage = luminosity_table_coverage(prior, name, z)
    np.testing.assert_allclose(integrate_luminosity(prior, name, z), coverage['continuous_mass'], atol=2e-9, rtol=0)
    assert coverage['continuous_mass'] > 1-1e-14
    tab = prior.tables[name]
    ms, logmax = logmax_at(prior, name, z)
    u = np.array([.13, .47, .89])
    draw = lambda u: ms-np.interp(logmax+np.log(u), tab['log_g'], tab['inverse_y'])
    h = 1e-8
    derivative = (draw(u+h)-draw(u-h))/(2*h)
    np.testing.assert_allclose(np.exp(numerical_luminosity_log_density(prior, name, z, draw(u))),
                               1/np.abs(derivative), rtol=3e-7)


def test_finite_table_failed_mass_is_not_renormalized(prior):
    name, z = 'red', .731
    _, logmax = logmax_at(prior, name, z)
    tab = prior.tables[name]
    selected = (tab['log_g'] > logmax+np.log(.1)) & (tab['log_g'] < logmax+np.log(.9))
    tab['log_g'], tab['inverse_y'] = tab['log_g'][selected], tab['inverse_y'][selected]
    coverage = luminosity_table_coverage(prior, name, z)
    assert coverage['lower_failed_draw_mass'] > .1
    assert coverage['upper_failed_draw_mass'] > .1
    assert .5 < coverage['continuous_mass'] < .8
    np.testing.assert_allclose(integrate_luminosity(prior, name, z), coverage['continuous_mass'], atol=2e-10)
    np.testing.assert_allclose(sum(coverage[k] for k in ['continuous_mass','clamp_atom_mass','lower_failed_draw_mass','upper_failed_draw_mass']), 1., atol=2e-16)


def test_covered_clamp_atom_is_explicit_and_not_a_lebesgue_density(prior):
    name, z = 'red', .731
    ms, logmax = logmax_at(prior, name, z)
    tiny = np.finfo(float).tiny
    prior.tables[name]['log_g'] = np.array([logmax+np.log(tiny)-2., logmax+1.])
    prior.tables[name]['inverse_y'] = np.array([8., -10.])
    coverage = luminosity_table_coverage(prior, name, z)
    assert coverage['clamp_atom_mass'] == tiny
    assert coverage['lower_failed_draw_mass'] == coverage['upper_failed_draw_mass'] == 0
    assert np.isfinite(coverage['clamp_magnitude'])
    # Brighter than the clamp point is outside the continuous part's support.
    assert np.isneginf(numerical_luminosity_log_density(prior, name, z, coverage['clamp_magnitude']-.01))


def test_zero_redshift_cdf_slope_has_zero_density(prior):
    tab = prior.tables['red']
    tab['cdf'][30:34] = tab['cdf'][30]
    z = (tab['z'][31]+tab['z'][32])/2
    assert np.isneginf(numerical_redshift_log_density(prior, 'red', z))
    d = row(z=z)
    assert np.isneginf(numerical_native_log_intensity(prior, d)).all()
    midpoint = (tab['z'][:-1]+tab['z'][1:])/2
    np.testing.assert_allclose(np.dot(np.exp(numerical_redshift_log_density(prior, 'red', midpoint)),np.diff(tab['z'])), 1., atol=2e-15)


@pytest.mark.parametrize('boundary,inside', [(14., True), (28., True),
    (np.nextafter(14., -np.inf), False), (np.nextafter(28., np.inf), False)])
def test_intrinsic_i_cut_is_inclusive_and_derived_fields_are_ignored(prior, boundary, inside):
    prior.photometry = lambda z, M, c: dict(m_r=M+40, m_i=np.full_like(M, boundary), M_i=M)
    d = row()
    d['m_i'] = np.array([99.])
    result = numerical_native_log_intensity(prior, d)
    assert bool(np.isfinite(result[0])) == inside


def test_absolute_rates_no_post_cut_or_parent_normalization(prior):
    d = row()
    before = numerical_native_log_intensity(prior, d)
    prior.parent_rate *= 7
    for tab in prior.tables.values():
        tab['rate'] *= 7
    after = numerical_native_log_intensity(prior, d)
    np.testing.assert_allclose(after-before, np.log(7), atol=2e-15)
    # Narrowing the population window does not renormalize retained rows.
    prior.p.update(gals_mag_min=19.9, gals_mag_max=20.1)
    np.testing.assert_array_equal(numerical_native_log_intensity(prior, d), after)


@pytest.mark.parametrize('key,value', [('type', .5), ('type', 9), ('z', 0.), ('z', 3.1),
    ('z', np.nan), ('M_B', np.inf), ('R_kpc', 0.), ('n', 5.), ('a', .9), ('psi', 2*np.pi)])
def test_invalid_native_rows_untouched_and_rejected_before_photometry(prior, key, value):
    d = row()
    d[key] = np.array([value])
    before = copy.deepcopy(d)
    prior.photometry = lambda *args: pytest.fail('invalid rows must not reach photometry')
    assert np.isneginf(numerical_native_log_intensity(prior, d)).all()
    for key in before:
        np.testing.assert_array_equal(d[key], before[key])


def test_simplex_invalid_shape_empty_and_permutation_contract(prior):
    d = row()
    d['c'][0, 0] = -.1
    assert np.isneginf(numerical_native_log_intensity(prior, d)).all()
    d = row()
    d['c'][0, 0] += .01
    assert np.isneginf(numerical_native_log_intensity(prior, d)).all()
    d = row()
    d['n'] = np.array([1., 2.])
    with pytest.raises(ValueError, match='aligned'):
        numerical_native_log_intensity(prior, d)
    data, _ = prior.sample(100, 127)
    expected = numerical_native_log_intensity(prior, data)
    np.testing.assert_array_equal(numerical_native_log_intensity(prior, {k:v[::-1] for k,v in data.items()}), expected[::-1])
    assert numerical_native_log_intensity(prior, {k:v[:0] for k,v in data.items()}).shape == (0,)


def test_generator_outside_table_still_raises_without_retry(prior, monkeypatch):
    delegate = np.random.default_rng(919)
    class ForcedEndpoint:
        calls = 0
        def __getattr__(self, key):
            return getattr(delegate, key)
        def random(self, n):
            self.calls += 1
            return np.full(n, .5 if self.calls == 1 else 0.)
    rng = ForcedEndpoint()
    for tab in prior.tables.values():
        # Deliberately exclude the lower half of the table, including U=0 clamp.
        middle = len(tab['log_g'])//2
        tab['log_g'], tab['inverse_y'] = tab['log_g'][middle:], tab['inverse_y'][middle:]
    monkeypatch.setattr(np.random, 'default_rng', lambda seed: rng)
    with pytest.raises(RuntimeError, match='outside table support'):
        prior.sample(16, 919)
    assert rng.calls == 2
