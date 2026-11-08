"""Caller-level checks that relocation affects the MH density and retained state."""
import numpy as np
import pytest

import sbsi.galsbi_particle_target as particle_module
from sbsi.galsbi_particle_target import bridge_log_density
from sbsi.galsbi_scene_prior import GalSBISceneBatch
from sbsi.scene_exchange import propose_native_exchange
from test_galsbi_particle_target import setup_target


def position_sensitive_target(monkeypatch, count):
    target, state = setup_target(count)
    original = target.summarize
    def summarize(scene, rows, *, full_response=False):
        summary = original(scene, rows, full_response=full_response)
        position_feature = np.bincount(scene.owner,
            weights=np.exp(-np.square(scene.offset_arcsec).sum(1)),
            minlength=len(scene.counts))
        summary.values[:, 5] += position_feature
        summary.values[:, 8] += .4*position_feature
        return summary
    monkeypatch.setattr(target, 'summarize', summarize)
    return target, target.evaluate(state.x, state.scene)


def standalone_proposal(target, state, rows, seed):
    scene, native_correction, active = propose_native_exchange(target.prior.galaxy,
        state.scene.take(rows), state.type_probability[rows], np.random.default_rng(seed),
        radius_scale=1., relocation_scale=.1,
        aperture_arcsec=target.prior.aperture_arcsec, central_weight=.9)
    # Explicitly copy each CSR segment so the oracle does not share the
    # implementation's flattened boolean scatter operation.
    targets = {k: v.copy() for k, v in state.scene.targets.items()}
    neighbours = {k: v.copy() for k, v in state.scene.neighbours.items()}
    offsets, x = state.scene.offset_arcsec.copy(), state.x.copy()
    for source, destination in enumerate(rows):
        old_slice = slice(*state.scene.indptr[destination:destination+2])
        new_slice = slice(*scene.indptr[source:source+2])
        for key in targets:
            targets[key][destination] = scene.targets[key][source]
            neighbours[key][old_slice] = scene.neighbours[key][new_slice]
        offsets[old_slice] = scene.offset_arcsec[new_slice]
    x[rows] = target.coordinates.encode(scene.targets)
    full_scene = GalSBISceneBatch(targets, neighbours, state.scene.indptr, offsets)
    return target.evaluate(x, full_scene), native_correction, active


@pytest.mark.parametrize('beta', [0., .001, .4, 1.])
@pytest.mark.parametrize('decision', [(True, False), (False, True)])
def test_exchange_relocation_caller_uses_new_positions_in_ratio_and_retained_payload(monkeypatch, beta, decision):
    target, old = position_sensitive_target(monkeypatch, 2)
    old_offsets = old.scene.offset_arcsec.copy()
    seed = 9712
    proposed, native_correction, expected_active = standalone_proposal(target, old, np.arange(2), seed)
    old_density = bridge_log_density(old.log_prior, old.log_kernel, old.log_proposal, beta)
    new_density = bridge_log_density(proposed.log_prior, proposed.log_kernel, proposed.log_proposal, beta)
    expected_ratio = new_density-old_density+native_correction-(proposed.log_prior-old.log_prior)
    decision = np.asarray(decision, bool)
    captured = []
    def inspect_metropolis(old_value, new_value, uniforms):
        captured.append(new_value-old_value)
        return decision.copy()
    monkeypatch.setattr(particle_module, 'metropolis_accept', inspect_metropolis)
    updated, accepted, active = target.mutate(old, beta, np.random.default_rng(seed),
                                            block='exchange_relocate')
    np.testing.assert_allclose(captured[0], expected_ratio, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(active, expected_active)
    np.testing.assert_array_equal(accepted, decision)
    np.testing.assert_array_equal(old.scene.offset_arcsec, old_offsets)
    for field in ('x', 'log_prior', 'log_kernel', 'log_proposal', 'type_probability', 'summaries'):
        before, after = getattr(old, field), getattr(proposed, field)
        use = decision.reshape(decision.shape+(1,)*(before.ndim-1))
        np.testing.assert_allclose(getattr(updated, field), np.where(use, after, before), rtol=0, atol=1e-12)
    for row in range(2):
        expected = (proposed if decision[row] else old).scene.take(np.array([row]))
        actual = updated.scene.take(np.array([row]))
        np.testing.assert_array_equal(actual.offset_arcsec, expected.offset_arcsec)
        for field in ('targets', 'neighbours'):
            for key in getattr(expected, field):
                np.testing.assert_array_equal(getattr(actual, field)[key], getattr(expected, field)[key])
    replay = target.evaluate(updated.x, updated.scene)
    np.testing.assert_allclose(updated.log_kernel, replay.log_kernel, rtol=0, atol=1e-12)
    np.testing.assert_allclose(updated.summaries, replay.summaries, rtol=0, atol=1e-12)
    # Holding the old positions while swapping marks must change the fixture's
    # actual likelihood, so this catches a caller that discards the relocation.
    discarded = GalSBISceneBatch(proposed.scene.targets, proposed.scene.neighbours,
                                 proposed.scene.indptr, old_offsets)
    stale = target.evaluate(proposed.x, discarded)
    assert np.min(np.abs(stale.log_kernel-proposed.log_kernel)) > 1e-5


@pytest.mark.parametrize('beta', [0., .4, 1.])
def test_relocation_subset_scatter_skips_offsupport_particles_without_shifting_rows(monkeypatch, beta):
    target, initial = position_sensitive_target(monkeypatch, 3)
    x = initial.x.copy()
    x[1, 0] = np.log(4.)
    old = target.evaluate(x, initial.scene)
    rows, seed = np.array([0, 2]), 9713
    proposed, native_correction, _ = standalone_proposal(target, old, rows, seed)
    captured = []
    def inspect_metropolis(old_value, new_value, uniforms):
        captured.append(new_value[rows]-old_value[rows])
        return np.array([True, True, False])
    monkeypatch.setattr(particle_module, 'metropolis_accept', inspect_metropolis)
    updated, accepted, active = target.mutate(old, beta, np.random.default_rng(seed),
                                            block='exchange_relocate')
    expected_ratio = (bridge_log_density(proposed.log_prior[rows], proposed.log_kernel[rows], proposed.log_proposal[rows], beta)
        -bridge_log_density(old.log_prior[rows], old.log_kernel[rows], old.log_proposal[rows], beta)
        +native_correction-(proposed.log_prior[rows]-old.log_prior[rows]))
    np.testing.assert_allclose(captured[0], expected_ratio, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(active, [True, False, True])
    np.testing.assert_array_equal(accepted, [True, False, False])
    for row in range(3):
        expected = proposed if row == 0 else old
        np.testing.assert_array_equal(updated.x[row], expected.x[row])
        np.testing.assert_array_equal(updated.scene.offset_arcsec[row], expected.scene.offset_arcsec[row])
        assert updated.log_kernel[row] == expected.log_kernel[row]
    assert updated.log_prior[1] == -np.inf
