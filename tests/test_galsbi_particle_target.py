"""Independent bridge-measure and bookkeeping audit without trained models."""
import numpy as np
import pandas as pd
import pytest

import sbsi.galsbi_particle_target as particle_module
from sbsi.galsbi_particle_target import GalSBIParticleTarget, bridge_log_density
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import SummaryBatch, native_shapes
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.scene_exchange import numerical_over_analytic, propose_native_exchange
from test_galsbi_prior import prior_fixture


class PrescribedRNG:
    """Prescribe a proposal and inspect the actual MH threshold, not MC noise."""
    def __init__(self, choices=(), uniform=.5, normal=0.):
        self.choices = iter(choices)
        self.u, self.normal_value = uniform, normal

    def choice(self, n, p):
        selected = next(self.choices)
        assert 0 <= selected < n and p[selected] > 0
        return selected

    def uniform(self, low, high, size):
        assert low < self.u < high
        return np.full(size, self.u)

    def normal(self, *, size):
        return np.broadcast_to(self.normal_value, size).copy()

    def random(self):
        return .5

    def integers(self, *args, **kwargs):
        return 183


class PeriodicGaussian:
    """Normalized nine-dimensional Gaussian times a uniform phase circle."""
    def __init__(self, mean, sigma=2.):
        self.mean, self.sigma = np.asarray(mean)[:9], sigma

    def log_prob(self, x):
        return (-.5*np.square((x[:, :9]-self.mean)/self.sigma).sum(1)
                -9*np.log(self.sigma*np.sqrt(2*np.pi))-np.log(2*np.pi))

    def sample(self, n, rng):
        x = np.column_stack((self.mean+self.sigma*rng.normal(size=(n, 9)),
                             rng.uniform(0, 2*np.pi, n)))
        return x


class ToyLikelihood:
    """Nonconstant target/environment coupling, independent of hidden type."""
    def log_kernel(self, summary, observed, g1, g2):
        v = summary.values
        return (-.08*np.square(v[:, 3]-20-.4*v[:, 5])-.12*v[:, 2]
                +.2*v[:, 0]-.15*v[:, 1]+g1*v[:, 8]+g2*v[:, 4])


class ToyTarget(GalSBIParticleTarget):
    def summarize(self, scene, rows, *, full_response=False):
        shape = native_shapes(scene.targets, self.angular_distance)
        crowding = np.bincount(scene.owner, weights=np.exp(-.1*scene.neighbours['m_r']),
                               minlength=len(scene.counts))
        # Nonzero response coefficient constructed from physical inputs only.
        response = shape['semimajor']*(1+crowding)
        values = np.column_stack((shape['e1'], shape['e2'], scene.targets['n'],
            scene.targets['m_r'], shape['circularized'], crowding,
            scene.counts, np.zeros(len(scene.counts)), response))
        return SummaryBatch(values, self.auxiliary[rows])


def setup_target(n=1, *, shear=(.02, .001)):
    galaxy = prior_fixture()
    prior = GalSBIScenePrior(galaxy, aperture_arcsec=10.)
    coordinates = GalSBITargetCoordinates(galaxy, lambda z: np.full_like(z, 1000.))
    native, _ = galaxy.sample(64, 71)
    targets = {k: v[:n].copy() for k, v in native.items() if k != 'proposal_index'}
    neighbours = {k: v[n:2*n].copy() for k, v in native.items() if k != 'proposal_index'}
    neighbours['type'][:] = 1
    scene = GalSBISceneBatch(targets, neighbours, np.arange(n+1),
                            np.tile([.2, .4], (n, 1)))
    x = coordinates.encode(targets)
    observed = pd.DataFrame([[.1, .2, 3., 2.]], columns=['e1', 'e2', 'radius', 'flux'])
    target = ToyTarget(prior, coordinates, ToyLikelihood(), None, coordinates.angular_distance,
        {}, observed, [PeriodicGaussian(x[0])], np.zeros(n, dtype=int), shear)
    state = target.evaluate(x, scene)
    return target, state


@pytest.mark.parametrize('beta', [0., .001, .4, 1.])
def test_bridge_endpoints_and_zero_target_support(beta):
    lp = np.array([-.7, -np.inf, -.8])
    lk = np.array([-1.3, -.9, -np.inf])
    lq = np.array([-2.7, -2.1, -3.2])
    result = bridge_log_density(lp, lk, lq, beta)
    if beta == 0:
        np.testing.assert_array_equal(result, lq)
    else:
        assert np.isneginf(result[1:]).all()
        assert result[0] == pytest.approx(beta*(-2.)+(1-beta)*(-2.7))
    assert not np.isnan(result).any()


@pytest.mark.parametrize('beta', [0., .001, .4, 1.])
@pytest.mark.parametrize('auxiliary_type', [0, 1])
def test_native_exchange_detailed_balance_for_tempered_marginal_types(monkeypatch, beta, auxiliary_type):
    """Include native qEnv, conditional type proposals and coordinate Jacobian.

    The auxiliary type law is p(t|u), even when the marginal target is tempered.
    Tempering that proposal correction along with the target would fail here.
    """
    target, old = setup_target()
    new_scene, native_ratio, _ = propose_native_exchange(target.prior.galaxy, old.scene,
        old.type_probability, PrescribedRNG([0, auxiliary_type]))
    new = target.evaluate(target.coordinates.encode(new_scene.targets), new_scene)
    native = target.prior.galaxy
    logenv_old = native.log_intensity(old.scene.neighbours)+numerical_over_analytic(native, old.scene.neighbours)
    logenv_new = native.log_intensity(new.scene.neighbours)+numerical_over_analytic(native, new.scene.neighbours)
    old_j = target.coordinates.decode(old.x, np.array([auxiliary_type]))[1]
    new_j = target.coordinates.decode(new.x, old.scene.neighbours['type'])[1]
    log_forward_proposal = np.log(old.type_probability[:, auxiliary_type])
    log_reverse_proposal = np.log(new.type_probability[:, 1])
    log_pi_old = logenv_old+bridge_log_density(old.log_prior, old.log_kernel, old.log_proposal, beta)
    log_pi_new = logenv_new+bridge_log_density(new.log_prior, new.log_kernel, new.log_proposal, beta)
    expected_ratio = (log_pi_new-log_pi_old+log_reverse_proposal-log_forward_proposal
                      +old_j-new_j)
    observed_ratio = []

    def capture(old_density, new_density, uniforms):
        observed_ratio.append(new_density-old_density)
        return np.ones(len(old_density), dtype=bool)

    monkeypatch.setattr(particle_module, 'metropolis_accept', capture)
    moved, accepted, active = target.mutate(old, beta,
        PrescribedRNG([0, auxiliary_type]), block='exchange')
    assert accepted.all() and active.all()
    np.testing.assert_allclose(observed_ratio[0], expected_ratio, rtol=0, atol=1e-11)
    forward_flux = log_pi_old+log_forward_proposal+np.minimum(0, observed_ratio[0])
    reverse_flux = log_pi_new+log_reverse_proposal+np.minimum(0, -observed_ratio[0])+old_j-new_j
    np.testing.assert_allclose(forward_flux, reverse_flux, rtol=0, atol=1e-11)
    np.testing.assert_allclose(moved.potential, new.potential, rtol=0, atol=1e-12)
    assert abs(float((old_j-new_j)[0])) > .1
    if beta < 1:
        # Ensure this fixture discriminates the tempting, incorrect beta*R1.
        incorrect = beta*(native_ratio+new.log_kernel-old.log_kernel)
        assert abs(float((expected_ratio-incorrect)[0])) > 1e-3


@pytest.mark.parametrize('beta', [.001, .4, 1.])
def test_environment_kernel_retains_untempered_environment_prior(monkeypatch, beta):
    target, old = setup_target()
    altered = {k: v.copy() for k, v in old.scene.neighbours.items()}
    altered['M_B'] -= 1.
    altered.update(target.prior.galaxy.photometry(altered['z'], altered['M_B'], altered['c']))
    proposed_scene = GalSBISceneBatch(old.scene.targets, altered, old.scene.indptr,
                                     old.scene.offset_arcsec)
    new = target.evaluate(old.x, proposed_scene)
    monkeypatch.setattr(target.prior, 'refresh', lambda scene, fraction, seed: proposed_scene)
    ratio = []
    monkeypatch.setattr(particle_module, 'metropolis_accept',
        lambda old, new, u: ratio.append(new-old) or np.ones(len(old), dtype=bool))
    target.mutate(old, beta, PrescribedRNG(), block='environment')
    native = target.prior.galaxy
    q_old = native.log_intensity(old.scene.neighbours)+numerical_over_analytic(native, old.scene.neighbours)
    q_new = native.log_intensity(new.scene.neighbours)+numerical_over_analytic(native, new.scene.neighbours)
    old_log_pi = q_old+bridge_log_density(old.log_prior, old.log_kernel, old.log_proposal, beta)
    new_log_pi = q_new+bridge_log_density(new.log_prior, new.log_kernel, new.log_proposal, beta)
    # An independent qEnv draw has Q(new|old)=qEnv(new); the qEnv terms cancel
    # completely, rather than receiving a temperature exponent.
    expected = new_log_pi-old_log_pi+q_old-q_new
    np.testing.assert_allclose(ratio[0], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ratio[0], beta*(new.log_kernel-old.log_kernel), rtol=0, atol=1e-12)
    np.testing.assert_allclose(old_log_pi+q_new+np.minimum(0, ratio[0]),
                               new_log_pi+q_old+np.minimum(0, -ratio[0]), atol=1e-12)


def test_nuisance_cache_replays_and_newly_valid_rows_are_evaluated():
    target, state = setup_target(2)
    shifted = state.x.copy()
    shifted[:, 0] -= .1
    shifted[:, 2:6] += .07
    reused = target.evaluate(shifted, state.scene, reuse=state)
    actual = target.evaluate(shifted, state.scene)
    np.testing.assert_allclose(reused.log_kernel, actual.log_kernel, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reused.summaries, actual.summaries, rtol=0, atol=1e-12)
    invalid_x = state.x.copy()
    invalid_x[1, 0] = np.log(4.)
    invalid = target.evaluate(invalid_x, state.scene)
    assert invalid.log_prior[1] == invalid.log_kernel[1] == -np.inf
    before = target.audit['likelihood_rows']
    recovered = target.evaluate(state.x, invalid.scene, reuse=invalid)
    assert target.audit['likelihood_rows'] == before+1
    np.testing.assert_allclose(recovered.log_kernel, state.log_kernel, rtol=0, atol=1e-12)
    np.testing.assert_allclose(recovered.summaries, state.summaries, rtol=0, atol=1e-12)


@pytest.mark.parametrize('beta', [0., .001, .4, 1.])
def test_exchange_is_null_outside_analytic_support(beta):
    target, state = setup_target(2)
    x = state.x.copy()
    x[1, 0] = np.log(4.)
    invalid = target.evaluate(x, state.scene)
    updated, accepted, active = target.mutate(invalid, beta,
        PrescribedRNG([0, 0]), block='exchange')
    assert active.tolist() == [True, False]
    assert not accepted[1]
    np.testing.assert_array_equal(updated.x[1], invalid.x[1])
    np.testing.assert_array_equal(updated.scene.neighbours['z'][1:], invalid.scene.neighbours['z'][1:])
    one = invalid.take(np.array([1]))
    target.owner = np.zeros(1, dtype=int)
    target.observed = target.observed.iloc[:1]
    target.auxiliary = target.auxiliary[:1]
    updated, accepted, active = target.mutate(one, beta, PrescribedRNG(), block='exchange')
    assert not accepted.any() and not active.any()
    assert updated.log_prior[0] == -np.inf


def test_initialization_keeps_offsupport_q0_draws_without_retry(monkeypatch):
    target, state = setup_target(3)
    x = state.x.copy()
    x[1, 0] = np.log(4.)
    calls = []
    def draw(n, rng):
        calls.append(n)
        return x.copy()
    monkeypatch.setattr(target.proposals[0], 'sample', draw)
    initial = target.initialize(np.random.default_rng(741))
    assert calls == [3]
    np.testing.assert_array_equal(initial.x, x)
    assert initial.potential[1] == -np.inf
    assert np.isfinite(initial.potential[[0, 2]]).all()


def test_offsupport_redshift_never_reaches_photometry_or_cosmology(monkeypatch):
    target, state = setup_target(3)
    original_photo = target.prior.galaxy.photometry
    def strict_photo(z, m, c):
        assert np.all((z > 0) & (z <= 3.))
        return original_photo(z, m, c)
    def strict_distance(z):
        assert np.all((z > 0) & (z <= 3.))
        return np.full_like(z, 1000.)
    monkeypatch.setattr(target.prior.galaxy, 'photometry', strict_photo)
    target.coordinates.angular_distance = strict_distance
    target.angular_distance = strict_distance
    x = state.x.copy()
    x[:, 0] = [np.log(4.), 1000., -1000.]
    expected_q = target.log_proposal(x)
    initial_calls = []
    def draw(n, rng):
        initial_calls.append(n)
        return x.copy()
    monkeypatch.setattr(target.proposals[0], 'sample', draw)
    initial = target.initialize(np.random.default_rng(744))
    assert initial_calls == [3]
    np.testing.assert_array_equal(initial.x, x)
    np.testing.assert_array_equal(initial.log_proposal, expected_q)
    np.testing.assert_array_equal(initial.type_probability, np.zeros((3, 2)))
    assert np.isneginf(initial.log_prior).all() and np.isneginf(initial.log_kernel).all()
    for values in initial.scene.targets.values():
        assert np.isfinite(values).all()
    recovered = target.evaluate(state.x, initial.scene, reuse=initial)
    assert np.isfinite(recovered.log_kernel).all()


def test_cache_replay_detects_finite_vs_zero_likelihood_mismatch():
    target, state = setup_target()
    state.log_kernel[0] = -np.inf  # Deliberately corrupt an already evaluated cache.
    with pytest.raises(ValueError, match='changes likelihood support'):
        target.mutate(state, .4, PrescribedRNG(normal=0.), block='nuisance', check_reuse=True)


@pytest.mark.parametrize('beta', [0., .4, 1.])
def test_sed_is_null_outside_native_redshift_domain(monkeypatch, beta):
    target, state = setup_target(2)
    original_sed_alpha = target.prior.galaxy.sed_alpha
    def strict_sed_alpha(z, kind):
        assert np.all((z > 0) & (z <= 3.))
        return original_sed_alpha(z, kind)
    monkeypatch.setattr(target.prior.galaxy, 'sed_alpha', strict_sed_alpha)
    x = state.x.copy()
    x[1, 0] = 1000.
    initial = target.evaluate(x, state.scene)
    moved, accepted, active = target.mutate(initial, beta, np.random.default_rng(814), block='sed')
    assert active.tolist() == [True, False] and not accepted[1]
    np.testing.assert_array_equal(moved.x[1], initial.x[1])


@pytest.mark.parametrize('beta', [.001, .4, 1.])
def test_sed_proposal_hastings_is_not_tempered(monkeypatch, beta):
    target, old = setup_target()
    x_new, correction = target.coordinates.propose_sed(old.x, np.random.default_rng(815))
    new = target.evaluate(x_new, old.scene)
    ratio = []
    monkeypatch.setattr(particle_module, 'metropolis_accept',
        lambda old, new, u: ratio.append(new-old) or np.ones(len(old), dtype=bool))
    target.mutate(old, beta, np.random.default_rng(815), block='sed')
    expected = (bridge_log_density(new.log_prior, new.log_kernel, new.log_proposal, beta)
                -bridge_log_density(old.log_prior, old.log_kernel, old.log_proposal, beta)
                +correction)
    np.testing.assert_allclose(ratio[0], expected, rtol=0, atol=1e-12)
    if beta < 1:
        assert abs(float(((1-beta)*correction)[0])) > 1e-3


@pytest.mark.parametrize('beta', [.001, .4, 1.])
def test_independence_proposal_uses_untempered_q0_hastings(monkeypatch, beta):
    target, old = setup_target()
    x_new = old.x.copy()
    x_new[:, 1] += .3
    x_new[:, 2] -= .4
    new = target.evaluate(x_new, old.scene)
    monkeypatch.setattr(target.proposals[0], 'sample', lambda n, rng: x_new.copy())
    ratio = []
    monkeypatch.setattr(particle_module, 'metropolis_accept',
        lambda old, new, u: ratio.append(new-old) or np.ones(len(old), dtype=bool))
    target.mutate(old, beta, PrescribedRNG(), block='independence')
    np.testing.assert_allclose(ratio[0], beta*(new.potential-old.potential), rtol=0, atol=1e-12)


def test_phase_wrapping_and_rejected_payload_retention(monkeypatch):
    target, old = setup_target(2)
    old.x[:, 9] = [2*np.pi-.01, .01]
    old = target.evaluate(old.x, old.scene)
    monkeypatch.setattr(particle_module, 'metropolis_accept', lambda old, new, u: np.array([True, False]))
    moved, accepted, _ = target.mutate(old, 1.,
        PrescribedRNG(normal=[0, 0, 0, 0, 1]), block='physical')
    assert moved.x[0, 9] == pytest.approx(.19)
    np.testing.assert_array_equal(moved.x[1], old.x[1])
    assert moved.log_kernel[1] == old.log_kernel[1]
    np.testing.assert_array_equal(moved.summaries[1], old.summaries[1])
    # The q0 density is defined on the circle and invariant to 2pi shifts.
    wrapped = old.x.copy()
    wrapped[:, 9] += 6*np.pi
    np.testing.assert_array_equal(target.log_proposal(old.x), target.log_proposal(wrapped))


def test_resampling_duplicates_every_payload_without_aliases():
    target, state = setup_target(3)
    rows = np.array([2, 0, 2])
    copied = state.take(rows)
    for key in ('x', 'log_prior', 'log_kernel', 'log_proposal', 'type_probability', 'summaries'):
        np.testing.assert_array_equal(getattr(copied, key), getattr(state, key)[rows])
        assert not np.shares_memory(getattr(copied, key), getattr(state, key))
    for field in ('targets', 'neighbours'):
        for key, values in getattr(state.scene, field).items():
            np.testing.assert_array_equal(getattr(copied.scene, field)[key], values[rows])
            assert not np.shares_memory(getattr(copied.scene, field)[key], values)
    copied.scene.neighbours['z'][0] = -1
    assert copied.scene.neighbours['z'][2] > 0 and state.scene.neighbours['z'][2] > 0
