"""Deterministic prior reuse against full uncached native evaluations."""
from copy import deepcopy
from dataclasses import fields
from types import MethodType

import numpy as np
import pytest
from scipy.special import logsumexp

from test_galsbi_particle_target import setup_target
from test_informed_environment import fixture


def assert_bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.tobytes() == right.tobytes()


def assert_state_bits(left, right):
    for name in ('x', 'log_prior', 'log_kernel', 'log_proposal',
                 'type_probability', 'summaries'):
        assert_bits(getattr(left, name), getattr(right, name))
    for name in ('indptr', 'offset_arcsec'):
        assert_bits(getattr(left.scene, name), getattr(right.scene, name))
    for name in ('targets', 'neighbours'):
        assert getattr(left.scene, name).keys() == getattr(right.scene, name).keys()
        for key, value in getattr(left.scene, name).items():
            assert_bits(value, getattr(right.scene, name)[key])
    if left.environment_payload is None:
        assert right.environment_payload is None
    else:
        for field in fields(left.environment_payload):
            assert_bits(getattr(left.environment_payload, field.name),
                        getattr(right.environment_payload, field.name))


def uncached_mixture(coordinates, x):
    """Original explicit type sum using only the public native-density API."""
    data, jacobian = coordinates.decode(x, np.zeros(len(x), dtype=int))
    terms = []
    for kind in range(len(coordinates.prior.types)):
        data['type'] = np.full(len(x), kind, dtype=int)
        terms.append(coordinates.prior.log_intensity(data))
    terms = np.column_stack(terms)
    total = logsumexp(terms, axis=1)
    probabilities = np.zeros_like(terms)
    valid = np.isfinite(total)
    probabilities[valid] = np.exp(terms[valid]-total[valid, None])
    data['type'] = np.zeros(len(x), dtype=int)
    return total+jacobian, data, probabilities


def force_uncached(target):
    target.coordinates.log_mixture_prior = MethodType(uncached_mixture, target.coordinates)
    evaluate_prepared = target._evaluate_prepared

    def full(self, x, scene, native, *, log_proposal=None, **kwargs):
        return evaluate_prepared(x, scene, self.native_prior(x), **kwargs)

    target._evaluate_prepared = MethodType(full, target)


def test_fresh_photometry_reuse_matches_public_density_at_support_boundaries():
    target, state = setup_target()
    prior, coordinates = target.prior.galaxy, target.coordinates
    prior.z_max['red'] = 1.
    calls = []

    def photo(z, magnitude, sed):
        calls.append(len(z))
        return dict(m_r=magnitude+40+.3*z+3*sed[:, 0],
                    m_i=magnitude+40+.2*z+2*sed[:, 1],
                    M_i=magnitude+.1*sed[:, 2])

    prior.photometry = photo
    x = np.tile(state.x, (8, 1))
    x[:, 0] = np.log([.2, .7, 1.5, 2.8, 3., 3.01, .4, .4])
    x[:, 1] = [20., 14., 28., 29., 20., 20., 20., 20.]
    x[-2, 8] = -1000.  # finite coordinates but zero native shape support
    x[-1, 2] = -1000.  # finite coordinates but zero native SED support
    expected = uncached_mixture(coordinates, x)
    old_calls = len(calls)
    calls.clear()
    actual = coordinates.log_mixture_prior(x)
    assert len(calls) == 2 < old_calls
    assert_bits(actual[0], expected[0])
    assert_bits(actual[2], expected[2])
    assert np.isfinite(actual[0]).any() and np.isneginf(actual[0]).any()
    assert np.any((actual[2][:, 0] == 0) & (actual[2][:, 1] == 1))
    for key in expected[1]:
        assert_bits(actual[1][key], expected[1][key])


def test_public_prior_does_not_trust_derived_photometry_or_bypass_override(monkeypatch):
    target, state = setup_target(3)
    prior = target.prior.galaxy
    native = deepcopy(state.scene.targets)
    expected, factors = prior.log_intensity(native, return_factors=True)
    native.update(m_r=np.full(3, np.nan), m_i=np.full(3, 1000.), M_i=np.full(3, -1000.))
    actual, actual_factors = prior.log_intensity(native, return_factors=True)
    assert_bits(actual, expected)
    for key in factors:
        assert_bits(actual_factors[key], factors[key])
    original = prior.log_intensity
    calls = []

    def custom(data, **kwargs):
        calls.append(1)
        return original(data, **kwargs)-.3

    monkeypatch.setattr(prior, 'log_intensity', custom)
    expected = uncached_mixture(target.coordinates, state.x)
    calls.clear()
    actual = target.coordinates.log_mixture_prior(state.x)
    assert len(calls) == 2
    assert_bits(actual[0], expected[0])
    assert_bits(actual[2], expected[2])


@pytest.mark.parametrize('informed', [False, True])
@pytest.mark.parametrize('block', ['environment', 'positions', 'independence'])
@pytest.mark.parametrize('beta', [.2, 1.])
def test_transition_reuse_preserves_every_payload_rng_and_audit(informed, block, beta):
    target, state, proposal = fixture(4)
    if informed:
        target.informed_environment = proposal
        state = target.evaluate(state.x, state.scene)
    else:
        x = state.x.copy()
        x[0, 0] = np.log(4.)  # complete zero-target row with safe native placeholders
        state = target.evaluate(x, state.scene)
    baseline, old = deepcopy(target), deepcopy(state)
    force_uncached(baseline)
    before = deepcopy(state)
    rng, reference_rng = np.random.default_rng(318), np.random.default_rng(318)
    actual, decisions, active = target.mutate(state, beta, rng, block=block)
    expected, expected_decisions, expected_active = baseline.mutate(
        old, beta, reference_rng, block=block)
    assert_state_bits(actual, expected)
    assert_state_bits(state, before)
    assert_bits(decisions, expected_decisions)
    assert_bits(active, expected_active)
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert target.audit == baseline.audit
    for key in ('x', 'log_prior', 'log_kernel', 'log_proposal', 'type_probability', 'summaries'):
        assert not np.shares_memory(getattr(actual, key), getattr(state, key))
    for key in state.scene.targets:
        assert not np.shares_memory(actual.scene.targets[key], state.scene.targets[key])


@pytest.mark.parametrize('informed', [False, True])
def test_initialization_reuses_native_decode_with_zero_target_draws(informed, monkeypatch):
    target, state, proposal = fixture(4)
    if informed:
        target.informed_environment = proposal
    raw = state.x.copy()
    raw[0, 0] = np.log(4.)

    def draw(count, rng):
        assert count == len(raw)
        rng.normal(size=count)
        return raw.copy()

    monkeypatch.setattr(target.proposals[0], 'sample', draw)
    baseline = deepcopy(target)
    force_uncached(baseline)
    calls = []
    original = target.native_prior
    monkeypatch.setattr(target, 'native_prior', lambda x: calls.append(1) or original(x))
    rng, reference_rng = np.random.default_rng(71), np.random.default_rng(71)
    actual, expected = target.initialize(rng), baseline.initialize(reference_rng)
    assert calls == [1]
    assert np.isneginf(actual.log_prior[0]) and np.isneginf(actual.log_kernel[0])
    assert_state_bits(actual, expected)
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert target.audit == baseline.audit


@pytest.mark.parametrize('block', ['environment', 'positions'])
def test_unchanged_target_reuse_avoids_native_calls_and_rejects_nonfinite_x(block, monkeypatch):
    target, state = setup_target(3)
    monkeypatch.setattr(target, 'native_prior', lambda x: pytest.fail('unchanged target prior recomputed'))
    monkeypatch.setattr(target, 'log_proposal', lambda x: pytest.fail('unchanged proposal density recomputed'))
    target.mutate(state, .4, np.random.default_rng(7), block=block)
    state.x[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite ten-coordinate'):
        target.mutate(state, .4, np.random.default_rng(7), block=block)


def test_independence_proposal_density_computed_once(monkeypatch):
    target, state = setup_target(3)
    calls, original = [], target.log_proposal
    monkeypatch.setattr(target, 'log_proposal', lambda x: calls.append(1) or original(x))
    target.mutate(state, .4, np.random.default_rng(2), block='independence')
    assert calls == [1]


@pytest.mark.parametrize('informed', [False, True])
def test_mixed_transitions_and_duplicate_ancestors_replay_exactly(informed):
    target, state, proposal = fixture(4)
    if informed:
        target.informed_environment = proposal
        state = target.evaluate(state.x, state.scene)
    baseline, old = deepcopy(target), deepcopy(state)
    force_uncached(baseline)
    rng, reference_rng = np.random.default_rng(92), np.random.default_rng(92)
    blocks = ['physical', 'nuisance', 'environment', 'positions',
              'sed', 'exchange', 'independence', 'physical', 'environment', 'positions']
    for step, block in enumerate(blocks):
        if step == 5:
            # Duplicated ancestry must carry independent, current target caches.
            state, old = state.take(np.array([2, 0, 2, 1])), old.take(np.array([2, 0, 2, 1]))
        state, accepted, active = target.mutate(state, .3+.07*step, rng, block=block, check_reuse=True)
        old, expected_accepted, expected_active = baseline.mutate(
            old, .3+.07*step, reference_rng, block=block, check_reuse=True)
        assert_state_bits(state, old)
        assert_bits(accepted, expected_accepted)
        assert_bits(active, expected_active)
        assert target.audit == baseline.audit
        assert rng.bit_generator.state == reference_rng.bit_generator.state
