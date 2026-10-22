"""Forward-law and complete bridge-flux checks; no trained models or data."""
import copy

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import beta as beta_distribution, betaprime, dirichlet, lognorm

import sbsi.galsbi_particle_target as particle_module
from sbsi.galsbi_particle_target import bridge_log_density
from sbsi.galsbi_prior import GalSBIContinuousPrior, upper_gamma
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.numerical_native_intensity import numerical_native_log_intensity
from sbsi.scene_exchange import propose_native_exchange
from test_galsbi_particle_target import ToyTarget, ToyLikelihood, PeriodicGaussian
from test_galsbi_prior import prior_fixture
from test_scene_exchange_relocation import independent_position_density, legacy_exchange


def assert_bits(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def numerical_fixture():
    base = prior_fixture()
    p = dict(base.p)
    p['lum_fct_phi_star_blue_amp'] *= 3
    p['lum_fct_alpha_blue'] = .3
    return GalSBIContinuousPrior(p, base.photometry, base.volume,
        lambda z: np.full_like(z, -18.373), {'red': 3., 'blue': 2.3},
        grid_size=257, magnitude_step=.1)


def marks(prior):
    data = dict(type=np.array([0, 1, 0]), z=np.array([.731, .91, 1.21]),
        M_B=np.array([-20., -21., -19.]), c=np.array([[.2]*5, [.1, .2, .3, .1, .3], [.1, .1, .2, .4, .2]]),
        R_kpc=np.array([1., 2., .8]), n=np.array([2., 1.5, 2.5]), a=np.array([.3, .2, .4]),
        psi=np.array([1., 2., 3.]))
    data.update(prior.photometry(data['z'], data['M_B'], data['c']))
    return data


def numerical_edges(prior, z):
    edges = []
    for name in prior.types:
        ms, _, alpha = prior.luminosity_parameters(np.asarray([z]), name)
        cap = min(float(prior.magnitude_cap(np.asarray([z]))[0]), prior.p['lum_fct_m_max'])
        logmax = np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))
        table = prior.tables[name]
        edges.append(float((ms-np.interp(logmax, table['log_g'], table['inverse_y']))[0]))
    return np.asarray(edges)


def coordinates(prior, **kwargs):
    return GalSBITargetCoordinates(prior, lambda z: np.full_like(z, 1000.), **kwargs)


def forward_native_log_density(prior, data, kind):
    """Differentiate generator inverse maps; use independent SciPy conditionals."""
    name, z, M, c = prior.types[kind], data['z'][0], data['M_B'][0], data['c']
    table = prior.tables[name]
    uz = np.interp(z, table['z'], table['cdf'])
    h = 1e-8
    dz = (np.interp(uz+h, table['cdf'], table['z'])-np.interp(uz-h, table['cdf'], table['z']))/(2*h)
    ms, _, alpha = prior.luminosity_parameters(np.asarray([z]), name)
    cap = min(prior.magnitude_cap(np.asarray([z]))[0], prior.p['lum_fct_m_max'])
    logmax = np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))[0]
    logu = np.interp(ms[0]-M, table['inverse_y'][::-1], table['log_g'][::-1])-logmax
    u = np.exp(logu)
    # Stay inside this inverse-table segment, including near the faint edge.
    h = min(1e-8, u*1e-5, (1-u)*.1)
    draw = lambda v: ms[0]-np.interp(logmax+np.log(v), table['log_g'], table['inverse_y'])
    dm = (draw(u+h)-draw(u-h))/(2*h)
    photo = prior.photometry(np.asarray([z]), np.asarray([M]), c)
    mean, sigma = prior.size_parameters(np.asarray([z]), photo['M_i'], name)
    a, b = prior.sersic_parameters(np.asarray([z]), name)
    truncation = betaprime.cdf(prior.p['sersic_n_max'], a, b)-betaprime.cdf(prior.p['sersic_n_min'], a, b)
    aa, bb = prior.shape_parameters(name)
    return float(np.log(table['rate'])-np.log(dz)-np.log(abs(dm))
        +dirichlet.logpdf(c[0], prior.sed_alpha(np.asarray([z]), name)[0])
        +lognorm.logpdf(data['R_kpc'][0], sigma[0], scale=np.exp(mean[0]))
        +betaprime.logpdf(data['n'][0], a[0], b)-np.log(truncation[0])
        +beta_distribution.logpdf(data['a'][0], aa, bb, scale=prior.p['ell_beta_emax'])-np.log(2*np.pi))


def test_numerical_type_sum_matches_forward_law_and_absolute_abundances():
    prior = numerical_fixture()
    coord = coordinates(prior, target_prior='numerical')
    native = {k: v[:1] for k, v in marks(prior).items()}
    x = coord.encode(native)
    data, jacobian = coord.decode(x, np.array([0]))
    terms = np.array([forward_native_log_density(prior, data, kind) for kind in range(2)])
    total, _, probability = coord.log_mixture_prior(x)
    np.testing.assert_allclose(total, logsumexp(terms)+jacobian, atol=3e-6, rtol=0)
    np.testing.assert_allclose(probability[0], np.exp(terms-logsumexp(terms)), atol=3e-7, rtol=0)
    assert abs(probability[0, 0]-.5) > .1
    for kind in range(2):
        conditional, _ = coord.log_prior(x, np.array([kind]))
        np.testing.assert_allclose(conditional, terms[kind]+jacobian, atol=3e-6, rtol=0)
    for table in prior.tables.values():
        table['rate'] *= 7.
    changed, _, changed_probability = coord.log_mixture_prior(x)
    np.testing.assert_allclose(changed-total, np.log(7), atol=1e-14, rtol=0)
    np.testing.assert_allclose(changed_probability, probability, atol=2e-15, rtol=0)


def test_full_numerical_support_including_type_only_interpolation_slivers():
    prior = numerical_fixture()
    numerical = coordinates(prior, target_prior='numerical')
    analytic = coordinates(prior)
    native = {k: v[:1].copy() for k, v in marks(prior).items()}
    cap = prior.magnitude_cap(native['z'])[0]
    edges = numerical_edges(prior, native['z'][0])
    assert np.min(edges) > cap+1e-8 and np.ptp(edges) > 1e-8
    for magnitude, expected_types in [((cap+min(edges))/2, 2), (np.mean(edges), 1)]:
        native['M_B'][:] = magnitude
        x = numerical.encode(native)
        lp, data, probability = numerical.log_mixture_prior(x)
        assert np.isfinite(lp).all() and np.isneginf(analytic.log_mixture_prior(x)[0]).all()
        terms = np.column_stack([numerical_native_log_intensity(prior, data | {'type': np.array([kind])})
                                 for kind in range(2)])
        assert np.isfinite(terms).sum() == expected_types
        np.testing.assert_allclose(probability, np.exp(terms-logsumexp(terms, axis=1)[:, None]), rtol=0, atol=0)


class Prescribed:
    def __init__(self, choices, *, arrays=(), uniform=.5):
        self.choices, self.arrays = iter(choices), iter(arrays)
        self.choice_calls = self.random_calls = self.uniform_calls = 0
        self.u = uniform

    def choice(self, count, p):
        self.choice_calls += 1
        value = next(self.choices)
        assert 0 <= value < count and p[value] > 0
        return value

    def random(self, count):
        self.random_calls += 1
        value = np.asarray(next(self.arrays))
        assert value.shape == (count,)
        return value

    def uniform(self, low, high, count):
        self.uniform_calls += 1
        return np.full(count, self.u)


def target_fixture(*, numerical_only=False, target_prior='numerical'):
    galaxy = numerical_fixture()
    coord = coordinates(galaxy, target_prior=target_prior)
    data = marks(galaxy)
    if numerical_only:
        cap = galaxy.magnitude_cap(data['z'][:1])[0]
        data['M_B'][0] = (cap+min(numerical_edges(galaxy, data['z'][0])))/2
        data.update(galaxy.photometry(data['z'], data['M_B'], data['c']))
    scene = GalSBISceneBatch({k: v[:1].copy() for k, v in data.items()},
                            {k: v[1:].copy() for k, v in data.items()}, [0, 2], [[3., 4.], [.3, .4]])
    x = coord.encode(scene.targets)
    observed = pd.DataFrame([[.1, .2, 3., 2.]], columns=['e1', 'e2', 'radius', 'flux'])
    target = ToyTarget(GalSBIScenePrior(galaxy, aperture_arcsec=10.), coord, ToyLikelihood(), None,
                      coord.angular_distance, {}, observed, [PeriodicGaussian(x[0])], np.array([0]), (.02, .001))
    return target, target.evaluate(x, scene)


@pytest.mark.parametrize('temperature', [0., .001, .4, 1.])
@pytest.mark.parametrize('kind', [0, 1])
@pytest.mark.parametrize('relocation', [False, True])
@pytest.mark.parametrize('numerical_only', [False, True])
def test_actual_mutation_satisfies_full_forward_reverse_flux(monkeypatch, temperature, kind, relocation, numerical_only):
    target, old = target_fixture(numerical_only=numerical_only)
    prior, coord = target.prior.galaxy, target.coordinates
    arrays = ([.1], [.25], [.3]) if relocation else ()
    parameters = dict(relocation_scale=.1, aperture_arcsec=10.) if relocation else {}
    new_scene, correction, active = propose_native_exchange(prior, old.scene, old.type_probability,
        Prescribed([0, kind], arrays=arrays), target_prior='numerical', **parameters)
    new = target.evaluate(coord.encode(new_scene.targets), new_scene)
    old_j = coord.decode(old.x, np.array([kind]))[1]
    new_j = coord.decode(new.x, np.array([1]))[1]
    log_env = lambda scene: numerical_native_log_intensity(prior, scene.neighbours).sum()-2*np.log(np.pi*100)
    selection = lambda scene: (-.5*np.square(scene.offset_arcsec).sum(1))[0]-logsumexp(-.5*np.square(scene.offset_arcsec).sum(1))
    position_old = position_new = 0.
    if relocation:
        position_old = np.log(independent_position_density(old.scene.offset_arcsec[0], .1, 10., .9))
        position_new = np.log(independent_position_density(new.scene.offset_arcsec[0], .1, 10., .9))
        assert abs(selection(new.scene)-selection(old.scene)) > 10
        assert abs(position_old-position_new) > 5
    q_forward = np.log(old.type_probability[:, kind])+selection(old.scene)+position_new
    q_reverse = np.log(new.type_probability[:, 1])+selection(new.scene)+position_old
    pi_old = log_env(old.scene)+bridge_log_density(old.log_prior, old.log_kernel, old.log_proposal, temperature)
    pi_new = log_env(new.scene)+bridge_log_density(new.log_prior, new.log_kernel, new.log_proposal, temperature)
    expected = pi_new-pi_old+q_reverse-q_forward+old_j-new_j
    captured = []
    monkeypatch.setattr(particle_module, 'metropolis_accept',
                        lambda a, b, u: captured.append(b-a) or np.ones(len(a), dtype=bool))
    rng = Prescribed([0, kind], arrays=arrays)
    moved, accepted, moved_active = target.mutate(old, temperature, rng,
        block='exchange_relocate' if relocation else 'exchange')
    assert active.all() and accepted.all() and moved_active.all()
    assert rng.choice_calls == 2 and rng.random_calls == (3 if relocation else 0) and rng.uniform_calls == 1
    np.testing.assert_allclose(captured[0], expected, atol=2e-11, rtol=0)
    np.testing.assert_allclose(pi_old+q_forward+np.minimum(0, captured[0]),
        pi_new+q_reverse+np.minimum(0, -captured[0])+old_j-new_j, atol=2e-11, rtol=0)
    np.testing.assert_allclose(moved.potential, new.potential, atol=1e-12, rtol=0)
    np.testing.assert_allclose(correction, selection(new.scene)-selection(old.scene)+position_old-position_new,
                               atol=2e-13, rtol=0)
    if not relocation:
        np.testing.assert_array_equal(correction, [0.])
    if temperature < 1:
        incorrect = temperature*(new.log_kernel-old.log_kernel)+correction
        assert abs(float((expected-incorrect)[0])) > 1e-3
    reverse_arrays = ([.95], [.25], [np.arctan2(4., 3.)/(2*np.pi)]) if relocation else ()
    restored, reverse_correction, _ = propose_native_exchange(prior, new_scene, new.type_probability,
        Prescribed([0, 1], arrays=reverse_arrays), target_prior='numerical', **parameters)
    np.testing.assert_allclose(correction+reverse_correction, 0., atol=2e-12, rtol=0)
    np.testing.assert_allclose(restored.offset_arcsec, old.scene.offset_arcsec, atol=1e-15, rtol=0)
    for key in old.scene.targets:
        if key != 'type':
            np.testing.assert_array_equal(restored.targets[key], old.scene.targets[key])
        np.testing.assert_array_equal(restored.neighbours[key], old.scene.neighbours[key])


@pytest.mark.parametrize('temperature', [0., .4, 1.])
def test_zero_inverse_cdf_slope_gives_zero_target_and_null_exchange(temperature):
    target, old = target_fixture()
    prior = target.prior.galaxy
    z = old.scene.targets['z'][0]
    for table in prior.tables.values():
        j = np.searchsorted(table['z'], z, side='right')-1
        table['cdf'][j+1] = table['cdf'][j]
    state = target.evaluate(old.x, old.scene)
    assert np.isneginf(state.log_prior).all() and np.isneginf(state.log_kernel).all()
    np.testing.assert_array_equal(state.type_probability, [[0., 0.]])
    assert np.isfinite(coordinates(prior).log_mixture_prior(old.x)[0]).all()
    rng = Prescribed([])
    moved, accepted, active = target.mutate(state, temperature, rng, block='exchange')
    assert not accepted.any() and not active.any() and rng.choice_calls == 0 and rng.uniform_calls == 1
    np.testing.assert_array_equal(moved.x, state.x)


def test_numerical_zero_neighbour_rejects_without_redraw():
    target, old = target_fixture()
    old.scene.neighbours['M_B'][0] = 0.
    rng = Prescribed([0, 0])
    _, correction, active = propose_native_exchange(target.prior.galaxy, old.scene, old.type_probability,
                                                     rng, target_prior='numerical')
    assert active.all() and np.isneginf(correction).all()
    assert rng.choice_calls == 2


@pytest.mark.parametrize('offset', [-1e-7, 1e-7])
def test_target_uses_lower_uniform_support_without_analytic_mask(offset):
    # Fine tables cover the clamp; the coarse sliver fixture intentionally has
    # a failed-draw tail before that point for one type.
    prior = prior_fixture()
    coord = coordinates(prior, target_prior='numerical')
    native = {k: v[:1] for k, v in marks(prior).items()}
    original = coord.encode(native)
    z = native['z']
    prior.p['gals_mag_min'] = 0.
    # Both types are checked independently because their inverse tables differ.
    for kind, name in enumerate(prior.types):
        ms, _, alpha = prior.luminosity_parameters(z, name)
        cap = np.minimum(prior.magnitude_cap(z), prior.p['lum_fct_m_max'])
        logmax = np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))
        table = prior.tables[name]
        logu = np.log(np.finfo(float).tiny)+offset
        assert table['log_g'][0] < logmax[0]+logu < table['log_g'][-1]
        magnitude = ms-np.interp(logmax+logu, table['log_g'], table['inverse_y'])
        x = original.copy()
        x[:, 1] = magnitude+40.
        analytic, _ = coordinates(prior).log_prior(x, np.array([kind]))
        numerical, _ = coord.log_prior(x, np.array([kind]))
        assert np.isfinite(analytic).all()
        assert np.isfinite(numerical).all() == (offset > 0)
        assert np.isneginf(numerical).all() == (offset < 0)


def test_empty_environment_under_numerical_law_is_inactive_without_random_draws():
    target, old = target_fixture()
    empty = GalSBISceneBatch(old.scene.targets, {k: v[:0] for k, v in old.scene.neighbours.items()},
                            [0, 0], np.empty((0, 2)))
    rng = Prescribed([])
    _, correction, active = propose_native_exchange(target.prior.galaxy, empty, old.type_probability,
                                                     rng, target_prior='numerical')
    assert not active.any() and correction[0] == 0. and rng.choice_calls == rng.random_calls == 0


def test_native_prior_safe_placeholders_and_target_identity_are_preserved():
    target, old = target_fixture()
    x = np.repeat(old.x, 3, axis=0)
    x[1:, 0] = [1000., -1000.]
    lp, data, probabilities = target.native_prior(x)
    assert np.isfinite(lp[0]) and np.isneginf(lp[1:]).all()
    assert all(np.isfinite(value).all() for value in data.values())
    np.testing.assert_array_equal(probabilities[1:], np.zeros((2, 2)))
    assert target.coordinates.target_prior == 'numerical'
    with pytest.raises(AttributeError):
        target.coordinates.target_prior = 'analytic'
    with pytest.raises(ValueError, match='target_prior'):
        coordinates(target.prior.galaxy, target_prior='other')
    with pytest.raises(ValueError, match='target_prior'):
        propose_native_exchange(target.prior.galaxy, old.scene, old.type_probability,
                                Prescribed([]), target_prior='other')


def test_default_analytic_coordinates_are_bitwise_original_type_sum():
    prior = numerical_fixture()
    for coord in (coordinates(prior), coordinates(prior, target_prior='analytic')):
        x = coord.encode(marks(prior))
        data, jac = coord.decode(x, np.zeros(len(x), dtype=int))
        terms = []
        for kind in range(2):
            data['type'] = np.full(len(x), kind, dtype=int)
            terms.append(prior.log_intensity(data))
        terms = np.column_stack(terms)
        total = logsumexp(terms, axis=1)
        probabilities = np.exp(terms-total[:, None])
        data['type'] = np.zeros(len(x), dtype=int)
        actual, actual_data, actual_probability = coord.log_mixture_prior(x)
        assert_bits(actual, total+jac)
        assert_bits(actual_probability, probabilities)
        for key in data:
            assert_bits(actual_data[key], data[key])


@pytest.mark.parametrize('seed', [31, 32, 33])
def test_analytic_exchange_still_replays_original_arithmetic_and_rng(seed):
    target, old = target_fixture(target_prior='analytic')
    expected_rng, actual_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    expected, expected_ratio, expected_active = legacy_exchange(target.prior.galaxy, old.scene,
        old.type_probability, expected_rng, 1.)
    actual, ratio, active = propose_native_exchange(target.prior.galaxy, old.scene, old.type_probability,
                                                  actual_rng, target_prior='analytic')
    assert expected_rng.bit_generator.state == actual_rng.bit_generator.state
    for a, b in ((ratio, expected_ratio), (active, expected_active),
                 (actual.indptr, expected.indptr), (actual.offset_arcsec, expected.offset_arcsec)):
        assert_bits(a, b)
    for name in ('targets', 'neighbours'):
        for key in getattr(expected, name):
            assert_bits(getattr(actual, name)[key], getattr(expected, name)[key])


def test_default_and_explicit_analytic_mutation_paths_have_identical_rng_and_payloads():
    default, old = target_fixture(target_prior='analytic')
    default.coordinates = coordinates(default.prior.galaxy)
    explicit = copy.deepcopy(default)
    explicit.coordinates = coordinates(explicit.prior.galaxy, target_prior='analytic')
    other = copy.deepcopy(old)
    left, right = np.random.default_rng(410), np.random.default_rng(410)
    for block in ('exchange', 'exchange_relocate', 'physical', 'nuisance', 'sed', 'environment', 'positions', 'independence'):
        old, accepted, active = default.mutate(old, .4, left, block=block)
        other, other_accepted, other_active = explicit.mutate(other, .4, right, block=block)
        assert left.bit_generator.state == right.bit_generator.state
        for key in ('x', 'log_prior', 'log_kernel', 'log_proposal', 'type_probability', 'summaries'):
            assert_bits(getattr(old, key), getattr(other, key))
        for name in ('targets', 'neighbours'):
            for key in getattr(old.scene, name):
                assert_bits(getattr(old.scene, name)[key], getattr(other.scene, name)[key])
        assert_bits(old.scene.indptr, other.scene.indptr)
        assert_bits(old.scene.offset_arcsec, other.scene.offset_arcsec)
        assert_bits(accepted, other_accepted)
        assert_bits(active, other_active)
    assert default.audit == explicit.audit
