"""Normalized relocation law and full native-coordinate detailed balance."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from sbsi.galsbi_particle_target import bridge_log_density
from sbsi.galsbi_scene_prior import GalSBISceneBatch
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.scene_exchange import (draw_relocation_positions, numerical_over_analytic,
    propose_native_exchange, relocation_position_log_density)
from test_galsbi_prior import prior_fixture


class ScriptedRNG:
    def __init__(self, choices=(), arrays=()):
        self.choices, self.arrays = iter(choices), iter(arrays)
        self.random_calls = 0

    def choice(self, n, p):
        value = next(self.choices)
        assert 0 <= value < n and p[value] > 0
        return value

    def random(self, count):
        self.random_calls += 1
        value = np.asarray(next(self.arrays), float)
        assert value.shape == (count,) and np.all((value >= 0) & (value < 1))
        return value


def fixture_scene(counts=(2,)):
    prior = prior_fixture()
    native, _ = prior.sample(128, 718)
    n, count = len(counts), sum(counts)
    targets = {k: v[:n].copy() for k, v in native.items() if k != 'proposal_index'}
    neighbours = {k: v[n:n+count].copy() for k, v in native.items() if k != 'proposal_index'}
    neighbours['type'][:] = 1
    offsets = np.column_stack((np.arange(count)*.3+.2, np.arange(count)*.2+.3))
    scene = GalSBISceneBatch(targets, neighbours, np.r_[0, np.cumsum(counts)], offsets)
    coordinates = GalSBITargetCoordinates(prior, lambda z: np.full_like(z, 1000.))
    x = coordinates.encode(targets)
    lp, scene.targets, probabilities = coordinates.log_mixture_prior(x)
    return prior, coordinates, scene, x, lp, probabilities


def independent_position_density(position, scale, aperture, weight):
    radius = np.linalg.norm(position)
    if radius > aperture:
        return 0.
    normalization = 1-np.exp(-aperture**2/(2*scale**2))
    return (weight*np.exp(-radius**2/(2*scale**2))/(2*np.pi*scale**2*normalization)
            +(1-weight)/(np.pi*aperture**2))


@pytest.mark.parametrize('scale,aperture,weight', [(.4, .6, .9), (.1, 10., .9),
                                                    (.4, .6, 0.), (.4, .6, 1.)])
def test_relocation_cartesian_density_integrates_to_one(scale, aperture, weight):
    def radial_density(radius):
        density = relocation_position_log_density([[radius, 0.]], scale=scale,
            aperture_arcsec=aperture, central_weight=weight)[0]
        return 2*np.pi*radius*np.exp(density)
    result = quad(radial_density, 0., aperture, epsabs=1e-11, epsrel=1e-11)[0]
    assert result == pytest.approx(1., abs=2e-11)
    for radius in (0., .03*aperture, .4*aperture, .9*aperture):
        actual = np.exp(relocation_position_log_density([[radius, 0.]], scale=scale,
            aperture_arcsec=aperture, central_weight=weight)[0])
        expected = independent_position_density([radius, 0.], scale, aperture, weight)
        assert actual == pytest.approx(expected, rel=2e-13, abs=1e-300)
    assert np.isneginf(relocation_position_log_density([[aperture*1.01, 0.]], scale=scale,
        aperture_arcsec=aperture, central_weight=weight)[0])


@pytest.mark.parametrize('scale,aperture', [(.4, .6), (.1, 10.), (10., .1)])
def test_relocation_draws_match_independent_radial_cdf_and_uniform_angle(scale, aperture):
    count, weight = 80000, .9
    points = draw_relocation_positions(count, np.random.default_rng(921), scale=scale,
        aperture_arcsec=aperture, central_weight=weight)
    radius = np.linalg.norm(points, axis=1)
    assert (radius <= aperture).all()
    for r in np.r_[np.linspace(.01, .99, 15)*aperture, min(scale, aperture/2)]:
        gaussian = -np.expm1(-r*r/(2*scale*scale))/-np.expm1(-aperture*aperture/(2*scale*scale))
        expected = weight*gaussian+(1-weight)*(r/aperture)**2
        assert abs(np.mean(radius <= r)-expected) < .008
    unit = points/radius[:, None]
    assert np.max(np.abs(unit.mean(0))) < .01
    assert np.max(np.abs(np.square(unit).mean(0)-.5)) < .01


def test_relocation_inverse_cdf_uses_three_uniform_draws_without_retry():
    rng = ScriptedRNG(arrays=([.1, .95], [.2, .8], [.3, .6]))
    points = draw_relocation_positions(2, rng, scale=.4, aperture_arcsec=.6, central_weight=.9)
    expected_radius = np.array([.4*np.sqrt(-2*np.log(1-.2*(1-np.exp(-.6**2/(2*.4**2))))),
                                .6*np.sqrt(.8)])
    expected = expected_radius[:, None]*np.column_stack((np.cos(2*np.pi*np.array([.3,.6])),
                                                        np.sin(2*np.pi*np.array([.3,.6]))))
    np.testing.assert_allclose(points, expected, rtol=0, atol=2e-16)
    assert rng.random_calls == 3


@pytest.mark.parametrize('beta', [0., .001, .4, 1.])
@pytest.mark.parametrize('auxiliary_type', [0, 1])
def test_joint_exchange_relocation_full_probability_flux_balance(beta, auxiliary_type):
    prior, coordinates, old, x, lp, probabilities = fixture_scene()
    old.offset_arcsec[:] = [[3., 4.], [.3, .4]]
    scale, aperture, weight, selection_scale = .1, 10., .9, 1.
    rng = ScriptedRNG([0, auxiliary_type], ([.1], [.25], [.3]))
    new, correction, active = propose_native_exchange(prior, old, probabilities, rng,
        radius_scale=selection_scale, relocation_scale=scale,
        aperture_arcsec=aperture, central_weight=weight)
    y = coordinates.encode(new.targets)
    lp_new, _, probability_new = coordinates.log_mixture_prior(y)
    old_j = coordinates.decode(x, np.array([auxiliary_type]))[1]
    new_j = coordinates.decode(y, np.array([1]))[1]
    log_env = lambda s: (prior.log_intensity(s.neighbours)
        +numerical_over_analytic(prior, s.neighbours)).sum()-len(s.neighbours['z'])*np.log(np.pi*aperture**2)
    log_q0 = lambda v: -.5*np.square((v[:, :9]-x[:, :9]-.3)/2).sum(1)-9*np.log(2*np.sqrt(2*np.pi))-np.log(2*np.pi)
    def log_kernel(s, v):
        crowding = np.exp(-np.square(s.offset_arcsec).sum(1)).sum()
        return -.08*(v[:, 1]-19-.7*crowding)**2+.2*np.cos(v[:, 9])
    def selection(s):
        logw = -.5*np.square(s.offset_arcsec).sum(1)/selection_scale**2
        return logw[0]-logsumexp(logw)
    q_position_old = np.log(independent_position_density(old.offset_arcsec[0], scale, aperture, weight))
    q_position_new = np.log(independent_position_density(new.offset_arcsec[0], scale, aperture, weight))
    q_forward = np.log(probabilities[:, auxiliary_type])+selection(old)+q_position_new
    q_reverse = np.log(probability_new[:, 1])+selection(new)+q_position_old
    reduced_old = bridge_log_density(lp, log_kernel(old, x), log_q0(x), beta)
    reduced_new = bridge_log_density(lp_new, log_kernel(new, y), log_q0(y), beta)
    log_pi_old, log_pi_new = reduced_old+log_env(old), reduced_new+log_env(new)
    actual_ratio = reduced_new-reduced_old+correction-(lp_new-lp)
    independent_ratio = log_pi_new-log_pi_old+q_reverse-q_forward+old_j-new_j
    np.testing.assert_allclose(actual_ratio, independent_ratio, rtol=0, atol=2e-11)
    np.testing.assert_allclose(log_pi_old+q_forward+np.minimum(0, actual_ratio),
        log_pi_new+q_reverse+np.minimum(0, -actual_ratio)+old_j-new_j, rtol=0, atol=2e-11)
    assert active.tolist() == [True]
    assert abs(selection(new)-selection(old)) > 10  # Discriminate missing index correction.
    assert abs(q_position_old-q_position_new) > 5  # Discriminate missing relocation density.
    # Force the reverse uniform component to restore the original radius 5 and angle.
    reverse_rng = ScriptedRNG([0, 1], ([.95], [.25], [np.arctan2(4., 3.)/(2*np.pi)]))
    restored, reverse_correction, _ = propose_native_exchange(prior, new, probability_new,
        reverse_rng, relocation_scale=scale, aperture_arcsec=aperture, central_weight=weight)
    np.testing.assert_allclose(correction+reverse_correction, 0., rtol=0, atol=1e-12)
    np.testing.assert_allclose(restored.offset_arcsec, old.offset_arcsec, rtol=0, atol=1e-15)
    for key in old.targets:
        # Target type is an auxiliary variable, so compare the physical marks.
        if key != 'type':
            np.testing.assert_array_equal(restored.targets[key], old.targets[key])
        np.testing.assert_array_equal(restored.neighbours[key], old.neighbours[key])


def test_relocation_keeps_counts_order_unselected_rows_and_rejected_payloads():
    prior, _, old, _, _, probability = fixture_scene((2, 0, 1))
    old_offsets = old.offset_arcsec.copy()
    rng = ScriptedRNG([1, 0, 0, 1], ([.1, .95], [.2, .4], [.3, .6]))
    proposed, ratio, active = propose_native_exchange(prior, old, probability, rng,
        relocation_scale=.1, aperture_arcsec=10.)
    np.testing.assert_array_equal(proposed.indptr, old.indptr)
    np.testing.assert_array_equal(proposed.counts, [2, 0, 1])
    np.testing.assert_array_equal(old.offset_arcsec, old_offsets)
    np.testing.assert_array_equal(proposed.offset_arcsec[0], old_offsets[0])
    assert not np.array_equal(proposed.offset_arcsec[1:], old_offsets[1:])
    assert active.tolist() == [True, False, True] and ratio[1] == 0
    for key in old.neighbours:
        np.testing.assert_array_equal(proposed.neighbours[key][0], old.neighbours[key][0])
    retained = old.retained(proposed, np.array([True, False, False]))
    for row in range(3):
        expected = (proposed if row == 0 else old).take(np.array([row]))
        actual = retained.take(np.array([row]))
        np.testing.assert_array_equal(actual.offset_arcsec, expected.offset_arcsec)
        for name in ('targets', 'neighbours'):
            for key in getattr(expected, name):
                np.testing.assert_array_equal(getattr(actual, name)[key], getattr(expected, name)[key])


def legacy_exchange(prior, scene, probability, rng, radius_scale):
    """Frozen pre-relocation algorithm, retained solely for RNG/bitwise replay."""
    targets = {k: v.copy() for k, v in scene.targets.items()}
    neighbours = {k: v.copy() for k, v in scene.neighbours.items()}
    active = scene.counts > 0
    rows = np.flatnonzero(active)
    selected = np.empty(len(rows), dtype=int)
    for j, row in enumerate(rows):
        start, stop = scene.indptr[row:row+2]
        logw = -.5*np.sum(scene.offset_arcsec[start:stop]**2, axis=1)/radius_scale**2
        w = np.exp(logw-logw.max())
        selected[j] = start+rng.choice(stop-start, p=w/w.sum())
    logratio = np.zeros(len(active))
    if len(rows):
        old = {k: v[rows].copy() for k, v in targets.items()}
        old['type'] = np.array([rng.choice(len(prior.types), p=probability[row]) for row in rows])
        new = {k: v[selected].copy() for k, v in neighbours.items()}
        old_correction = numerical_over_analytic(prior, old)
        new_correction = numerical_over_analytic(prior, new)
        valid = np.isfinite(old_correction) & np.isfinite(new_correction)
        logratio[rows] = -np.inf
        logratio[rows[valid]] = old_correction[valid]-new_correction[valid]
        for key in targets:
            targets[key][rows] = new[key]
            neighbours[key][selected] = old[key]
    return GalSBISceneBatch(targets, neighbours, scene.indptr, scene.offset_arcsec), logratio, active


@pytest.mark.parametrize('seed,radius_scale', [(31, .3), (32, 1.), (33, 3.)])
def test_default_exchange_is_bitwise_legacy_including_rng_state(seed, radius_scale):
    prior, _, scene, _, _, probability = fixture_scene((0, 1, 2, 3, 1))
    rng_old, rng_new = np.random.default_rng(seed), np.random.default_rng(seed)
    expected, expected_ratio, expected_active = legacy_exchange(prior, scene, probability, rng_old, radius_scale)
    actual, ratio, active = propose_native_exchange(prior, scene, probability, rng_new,
                                                   radius_scale=radius_scale)
    assert rng_old.bit_generator.state == rng_new.bit_generator.state
    np.testing.assert_array_equal(ratio, expected_ratio)
    np.testing.assert_array_equal(active, expected_active)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.offset_arcsec, expected.offset_arcsec)
    for name in ('targets', 'neighbours'):
        for key in getattr(expected, name):
            np.testing.assert_array_equal(getattr(actual, name)[key], getattr(expected, name)[key])


@pytest.mark.parametrize('kwargs', [dict(relocation_scale=0., aperture_arcsec=10.),
    dict(relocation_scale=.1), dict(relocation_scale=.1, aperture_arcsec=-1.),
    dict(relocation_scale=.1, aperture_arcsec=10., central_weight=1.01)])
def test_invalid_relocation_settings_are_rejected(kwargs):
    prior, _, scene, _, _, probability = fixture_scene()
    with pytest.raises(ValueError, match='relocation'):
        propose_native_exchange(prior, scene, probability, np.random.default_rng(1), **kwargs)


def test_relocation_rejects_invalid_original_spatial_support():
    prior, _, scene, _, _, probability = fixture_scene()
    scene.offset_arcsec[0] = [11., 0.]
    with pytest.raises(ValueError, match='inside the native aperture'):
        propose_native_exchange(prior, scene, probability, np.random.default_rng(1),
                                relocation_scale=.1, aperture_arcsec=10.)
