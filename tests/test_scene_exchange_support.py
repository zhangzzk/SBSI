"""A finite analytic mark cannot be swapped into a zero-density environment."""
import numpy as np
import pytest

from sbsi.galsbi_prior import upper_gamma
from sbsi.numerical_native_intensity import numerical_native_log_intensity
from sbsi.scene_exchange import numerical_over_analytic
from test_informed_environment import fixture, assert_state_equal, assert_payload_equal


def tail_state(informed=False, logu=None):
    target, state, law = fixture(1)
    galaxy = target.prior.galaxy
    galaxy.p['gals_mag_min'] = 0.
    z = state.scene.targets['z']
    ms, _, alpha = galaxy.luminosity_parameters(z, 'red')
    cap = np.minimum(galaxy.magnitude_cap(z), galaxy.p['lum_fct_m_max'])
    logmax = np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))
    table = galaxy.tables['red']
    logtiny = np.log(np.finfo(float).tiny)
    assert table['log_g'][0]-logmax[0] < logtiny
    if logu is None:
        logu = (table['log_g'][0]-logmax[0]+logtiny)/2
    magnitude = ms-np.interp(logmax+logu, table['log_g'], table['inverse_y'])
    x = state.x.copy()
    x[:, 1] = magnitude+40.
    if informed:
        target.informed_environment = law
    return target, target.evaluate(x, state.scene)


@pytest.mark.parametrize('block', ['exchange', 'exchange_relocate'])
@pytest.mark.parametrize('beta', [0., .001, .4, 1.])
def test_default_exchange_rejects_continuous_zero_neighbour(beta, block):
    target, state = tail_state()
    assert np.isfinite(state.log_prior).all()
    assert np.isfinite(state.log_kernel).all()
    correction, audit = numerical_over_analytic(target.prior.galaxy,
        state.scene.targets, return_support_audit=True)
    # The old rule admitted this state; the numerical generator cannot draw it.
    assert audit['legacy_finite'].all() and audit['below_uniform_clamp'].all()
    assert np.isneginf(correction).all()
    assert np.isneginf(numerical_native_log_intensity(
        target.prior.galaxy, state.scene.targets)).all()
    moved, accepted, active = target.mutate(state, beta,
        np.random.default_rng(2), block=block)
    assert active.all() and not accepted.any()
    assert_state_equal(moved, state)


@pytest.mark.parametrize('block', ['exchange', 'exchange_relocate'])
@pytest.mark.parametrize('beta', [.001, .4, 1.])
def test_informed_exchange_retains_every_payload_on_support_rejection(beta, block):
    target, state = tail_state(informed=True)
    moved, accepted, active = target.mutate(state, beta,
        np.random.default_rng(2), block=block)
    assert active.all() and not accepted.any()
    assert_state_equal(moved, state)
    assert_payload_equal(moved.environment_payload, state.environment_payload)


def test_continuous_density_agrees_on_each_side_of_clamp_boundary():
    logtiny = np.log(np.finfo(float).tiny)
    for offset in (-1e-7, 1e-7):
        target, state = tail_state(logu=logtiny+offset)
        ratio, audit = numerical_over_analytic(target.prior.galaxy,
            state.scene.targets, return_support_audit=True)
        q = numerical_native_log_intensity(target.prior.galaxy, state.scene.targets)
        assert audit['legacy_finite'].all()
        assert audit['below_uniform_clamp'].all() == (offset < 0)
        assert np.isfinite(q).all() == (offset > 0)
        assert np.isfinite(ratio).all() == (offset > 0)
        if offset > 0:
            np.testing.assert_allclose(target.prior.galaxy.log_intensity(
                state.scene.targets)+ratio, q, rtol=0, atol=2e-12)


def test_support_diagnostic_preserves_ordinary_values():
    target, state, _ = fixture(8)
    ordinary = numerical_over_analytic(target.prior.galaxy, state.scene.targets)
    diagnosed, audit = numerical_over_analytic(target.prior.galaxy,
        state.scene.targets, return_support_audit=True)
    np.testing.assert_array_equal(ordinary, diagnosed)
    assert audit['legacy_finite'].all() and not audit['below_uniform_clamp'].any()
