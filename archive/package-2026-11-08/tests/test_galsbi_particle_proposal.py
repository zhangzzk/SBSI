"""Normalized, evaluable particle seeds preserve the full coordinate support."""
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import f, norm, multivariate_normal, multivariate_t, t

from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates


CONDITIONS = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224)
PARAMETERS = dict(sersic_n_min=.2, sersic_n_max=5., ell_beta_emax=.98)


def proposal(**kwargs):
    return MeasurementParticleProposal.from_measurement(
        [.1, -.2, 4., 1000.], CONDITIONS, PARAMETERS, (.02, -.01), **kwargs)


def test_density_equals_finite_mixture_of_normalized_components_and_wraps():
    q = proposal(component_weights=(2., 3.))
    assert q.coordinate_names == GalSBITargetCoordinates.names
    x = q.sample(100, np.random.default_rng(173))
    reference = np.column_stack([
        multivariate_normal.logpdf(x[:, :9], mean=mean, cov=np.diag(scale**2))+np.log(weight)
        for mean, scale, weight in zip(q.means, q.scales, q.weights)])
    np.testing.assert_allclose(q.log_prob(x), logsumexp(reference, axis=1)-np.log(2*np.pi),
                               rtol=0, atol=1e-13)
    x[:, 9] += 2*np.pi*np.arange(-50, 50)
    np.testing.assert_allclose(q.log_prob(x), logsumexp(reference, axis=1)-np.log(2*np.pi),
                               rtol=0, atol=1e-13)


def test_quadrature_integrates_density_to_one():
    # The non-magnitude coordinates are shared by the two components, allowing
    # a one-dimensional numerical normalization check of the actual log_prob.
    q = proposal()
    fixed = np.r_[q.means[0], .4]
    other = np.arange(9) != 1
    log_others = norm.logpdf(fixed[:9][other], q.means[0, other], q.scales[0, other]).sum()

    def marginal_magnitude(magnitude):
        point = fixed.copy()
        point[1] = magnitude
        return np.exp(q.log_prob(point[None, :])[0]-log_others)*2*np.pi

    total, error = quad(marginal_magnitude, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-11)
    assert abs(total-1) < 1e-10
    assert error < 1e-8
    for mean, scale in zip(q.means[0, other], q.scales[0, other]):
        total, _ = quad(lambda t: norm.pdf(t, mean, scale), -np.inf, np.inf,
                        epsabs=1e-10, epsrel=1e-11)
        assert abs(total-1) < 1e-10


def test_sample_moments_and_uniform_periodic_phase():
    q = proposal(component_weights=(.35, .65))
    x = q.sample(120000, np.random.default_rng(28041))
    mean = np.sum(q.weights[:, None]*q.means, axis=0)
    variance = np.sum(q.weights[:, None]*(q.scales**2+(q.means-mean)**2), axis=0)
    np.testing.assert_allclose((x[:, :9].mean(axis=0)-mean)/np.sqrt(variance/len(x)), 0, atol=4)
    np.testing.assert_allclose(x[:, :9].var(axis=0), variance, rtol=.02)
    assert np.all((x[:, 9] >= 0) & (x[:, 9] < 2*np.pi))
    for harmonic in (1, 2, 3):
        assert abs(np.exp(1j*harmonic*x[:, 9]).mean()) < .009
    assert np.max(np.abs(np.mean((x[:, :9]-mean)*np.cos(x[:, 9, None]), axis=0)
                         /np.sqrt(variance))) < .009


def test_outside_support_draws_are_not_retried_or_modified():
    q = proposal()

    class FixedRandom:
        def choice(self, count, size, p):
            assert count == 2 and size == 2
            return np.array([0, 1])

        def standard_normal(self, shape):
            assert shape == (2, 9)
            return np.full(shape, 30.)

        def uniform(self, low, high, size):
            assert low == 0 and high == 2*np.pi and size == 2
            return np.array([0., 2*np.pi-.001])

    x = q.sample(2, FixedRandom())
    np.testing.assert_array_equal(x[:, :9], q.means+30*q.scales)
    # These redshifts and magnitudes are outside any finite native envelope;
    # their proposal densities remain evaluable and they are never replaced.
    assert np.all(x[:, 0] > 20) and np.all(x[:, 1] > 60)
    assert np.isfinite(q.log_prob(x)).all()


def test_centres_units_metadata_and_input_immutability():
    measurement = np.array([.1, -.2, 4., 1000.])
    old = measurement.copy()
    q = MeasurementParticleProposal.from_measurement(
        measurement, dict(CONDITIONS, psf_fwhm=0.), PARAMETERS, (.02, -.01))
    np.testing.assert_array_equal(measurement, old)
    np.testing.assert_allclose(q.means[:, 1], [22.5, 27.5])
    np.testing.assert_allclose(np.exp(q.means[:, 6]), [.8, .8])
    assert q.metadata['uses_generating_truth'] is False
    assert q.metadata['retry_or_clip_draws'] is False
    json.dumps(q.metadata, allow_nan=False)
    metadata = q.metadata
    metadata['means'][0][1] = -100
    assert q.means[0, 1] == 22.5
    with pytest.raises(ValueError):
        q.means[0, 1] = -100


@pytest.mark.parametrize('measurement', [[0., 0., .1, 100.], [.98, .1, .1, 100.]])
def test_circular_unresolved_and_boundary_measurements_have_regular_proposals(measurement):
    q = MeasurementParticleProposal.from_measurement(
        measurement, CONDITIONS, PARAMETERS, (.02, -.01))
    assert np.isfinite(q.means).all()
    np.testing.assert_allclose(np.exp(q.means[:, 6]), .002)
    assert np.isfinite(q.log_prob(q.sample(100, np.random.default_rng(429)))).all()


@pytest.mark.parametrize('kwargs', [dict(faint_magnitude_shift=0.), dict(coordinate_scales=np.zeros(9)),
                                   dict(component_weights=(0., 1.)), dict(redshift_centres=(0., 1.)),
                                   dict(sersic_fractions=(0., 1.))])
def test_invalid_construction_rejected(kwargs):
    with pytest.raises(ValueError):
        proposal(**kwargs)


def test_empty_samples_and_nonfinite_inputs():
    q = proposal()
    x = q.sample(0, np.random.default_rng(1))
    assert x.shape == (0, 10) and q.log_prob(x).shape == (0,)
    with pytest.raises(ValueError):
        q.sample(-1, np.random.default_rng(1))
    with pytest.raises(ValueError):
        q.log_prob(np.full((1, 10), np.nan))
    with pytest.raises(ValueError):
        MeasurementParticleProposal.from_measurement(
            [0., 0., 4., -1000.], CONDITIONS, PARAMETERS, (.02, -.01))


@pytest.mark.parametrize('degrees', [3.5, 5., 30.])
def test_student_density_matches_scipy_multivariate_mixture(degrees):
    q = proposal(component_family='student_t', student_df=degrees,
                 component_weights=(.3, .7))
    x = q.sample(200, np.random.default_rng(9281))
    reference = np.column_stack([
        multivariate_t.logpdf(x[:, :9], loc=mean, shape=np.diag(scale**2), df=degrees)+np.log(weight)
        for mean, scale, weight in zip(q.means, q.scales, q.weights)])
    np.testing.assert_allclose(q.log_prob(x), logsumexp(reference, axis=1)-np.log(2*np.pi),
                               rtol=0, atol=2e-13)
    wrapped = x.copy()
    wrapped[:, 9] += 14*np.pi
    np.testing.assert_array_equal(q.log_prob(wrapped), q.log_prob(x))


def test_student_finite_mixture_quadrature_normalization():
    # Integrating a nine-dimensional Student-t over magnitude leaves an
    # eight-dimensional Student-t marginal, not eight independent t factors.
    q = proposal(component_family='student_t', student_df=5.)
    fixed = np.r_[q.means[0], .4]
    other = np.arange(9) != 1
    log_marginal = multivariate_t.logpdf(fixed[:9][other], loc=q.means[0, other],
                                        shape=np.diag(q.scales[0, other]**2), df=5.)

    def conditional(magnitude):
        point = fixed.copy()
        point[1] = magnitude
        return np.exp(q.log_prob(point[None, :])[0]-log_marginal)*2*np.pi

    total, error = quad(conditional, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-11)
    assert abs(total-1) < 1e-10
    assert error < 1e-8


def test_student_radial_and_marginal_tails_match_common_scale_law():
    # Identical components remove label ambiguity. The radial F law detects
    # an incorrect implementation with nine independent Student-t draws.
    q = MeasurementParticleProposal(np.zeros((2, 9)), np.ones(9), [.5, .5],
                                    component_family='student_t', student_df=5.)
    size = 160000
    x = q.sample(size, np.random.default_rng(9282))
    radial = np.sum(x[:, :9]**2, axis=1)/9
    for probability in (.5, .9, .99):
        fraction = np.mean(radial <= f.ppf(probability, 9, 5))
        assert abs(fraction-probability) < 5*np.sqrt(probability*(1-probability)/size)
    expected_tail = 2*t.sf(5., df=5.)
    measured_tail = np.mean(np.abs(x[:, 0]) > 5.)
    assert abs(measured_tail-expected_tail) < 5*np.sqrt(expected_tail*(1-expected_tail)/size)
    assert measured_tail > 1000*(2*norm.sf(5.))
    np.testing.assert_allclose(x[:, :9].var(axis=0), 5/3, rtol=.04)
    assert abs(np.exp(1j*x[:, 9]).mean()) < .009
    assert q.metadata['student_df'] == 5.
    assert 'shape' in q.metadata['scale_semantics']
    assert q.sample(0, np.random.default_rng(1)).shape == (0, 10)


@pytest.mark.parametrize('count', [0, 1, 137])
def test_disabled_student_preserves_v1_gaussian_samples_density_and_rng(count):
    q = proposal()
    explicit = proposal(component_family='gaussian', student_df=17.)
    rng, legacy_rng, explicit_rng = (np.random.default_rng(9283) for _ in range(3))
    # The exact v1 expressions are intentionally frozen here as a numerical
    # replay contract, including their random-number consumption order.
    choices = legacy_rng.choice(len(q.weights), size=count, p=q.weights)
    legacy = np.empty((count, 10))
    legacy[:, :9] = q.means[choices]+q.scales[choices]*legacy_rng.standard_normal((count, 9))
    legacy[:, 9] = legacy_rng.uniform(0., 2*np.pi, count)
    actual, configured = q.sample(count, rng), explicit.sample(count, explicit_rng)
    np.testing.assert_array_equal(actual, legacy)
    np.testing.assert_array_equal(configured, legacy)
    assert rng.bit_generator.state == legacy_rng.bit_generator.state == explicit_rng.bit_generator.state
    standardized = (legacy[:, None, :9]-q.means[None, :, :])/q.scales[None, :, :]
    terms = (-.5*np.sum(standardized**2, axis=2)-np.log(q.scales).sum(axis=1)
             -4.5*np.log(2*np.pi)+q.log_weights)
    legacy_log_prob = logsumexp(terms, axis=1)-np.log(2*np.pi)
    np.testing.assert_array_equal(q.log_prob(legacy), legacy_log_prob)
    np.testing.assert_array_equal(explicit.log_prob(legacy), legacy_log_prob)


def test_student_log_density_remains_finite_when_radius_square_overflows():
    q = proposal(component_family='student_t')
    points = np.r_[q.means[0], 0.][None, :].repeat(2, axis=0)
    points[:, 2] = [1e200, -1e200]
    density = q.log_prob(points)
    assert np.isfinite(density).all()
    expected_slope = -(q.student_df+9)*np.log(2.)
    twice = points.copy()
    twice[:, 2] *= 2
    np.testing.assert_allclose(q.log_prob(twice)-density, expected_slope, rtol=0, atol=2e-12)


@pytest.mark.parametrize('kwargs', [dict(component_family='independent_t'), dict(student_df=0.),
                                   dict(student_df=-1.), dict(student_df=np.inf), dict(student_df=np.nan)])
def test_invalid_student_family_or_degrees_rejected(kwargs):
    with pytest.raises(ValueError):
        proposal(**kwargs)
