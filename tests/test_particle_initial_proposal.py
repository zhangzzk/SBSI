"""Frozen initial-law normalization, provenance, fitting, and bridge integration."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import i0e, i1e, logsumexp
from scipy.stats import multivariate_t, vonmises

from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
from sbsi.particle_initial_proposal import (FIT_RULE, NATIVE_TARGET_SOURCES,
    FrozenParticleProposal, digest, fit_initial_proposal, fit_initial_proposal_from_runs,
    phase_moment_parameters, target_identity)
from test_galsbi_particle_proposal import proposal
from test_galsbi_particle_target import setup_target, PrescribedRNG
from test_particle_extinction import driver


def identity_protocol():
    return dict(observations=[11], fixed_shear=[0., 0.],
        reference_protocol_sha256='a'*64, observation_sha256='b'*64,
        density_batch_size=4096, bridge='test bridge', selection='none',
        implementation_sha256={f'/snapshot/{name}':'c'*64 for name in NATIVE_TARGET_SOURCES})


def fitted_fixture():
    rng = np.random.default_rng(333)
    runs = [rng.normal(size=(1, 32, 10)) for _ in range(4)]
    for i, x in enumerate(runs):
        x[..., :9] += i
        x[..., 9] = (.2+i+np.linspace(-.1, .1, 32)) % (2*np.pi)
    metadata = dict(target_identity(identity_protocol()), pilot_seeds=[1, 2, 3, 4])
    return fit_initial_proposal(runs, [proposal()], metadata), runs


@pytest.mark.parametrize('family', ['gaussian', 'student_t'])
def test_disabled_phase_preserves_samples_density_and_rng_bitwise(family):
    original = proposal(component_family=family)
    explicit = MeasurementParticleProposal(original.means, original.scales, original.weights,
        component_family=family, phase_centres=[123., -10.], phase_concentrations=[0., 0.])
    a, b = np.random.default_rng(53), np.random.default_rng(53)
    x, y = original.sample(100, a), explicit.sample(100, b)
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(original.log_prob(x), explicit.log_prob(x))
    assert a.bit_generator.state == b.bit_generator.state
    assert explicit.metadata['phase'] == 'independent uniform periodic'


def test_phase_density_is_inside_mixture_and_matches_scipy_with_wrapping():
    base = proposal()
    q = MeasurementParticleProposal(base.means, base.scales, [.2, .8],
        component_family='student_t', phase_centres=[0., 1.3], phase_concentrations=[0., 1e4])
    x = q.sample(500, np.random.default_rng(54))
    reference = np.column_stack([np.log(w)+multivariate_t.logpdf(x[:, :9], m, np.diag(s*s), df=5)
        +vonmises.logpdf(x[:, 9], k, loc=mu)
        for m, s, w, mu, k in zip(q.means, q.scales, q.weights, q.phase_centres, q.phase_concentrations)])
    expected = logsumexp(reference, axis=1)
    np.testing.assert_allclose(q.log_prob(x), expected, rtol=0, atol=4e-12)
    x[:, 9] += 20*np.pi
    np.testing.assert_allclose(q.log_prob(x), expected, rtol=0, atol=1e-10)
    # At a fixed real point, integrating the phase must recover the complete
    # multivariate-t mixture marginal, not a separately mixed phase density.
    fixed = x[0].copy()
    real_marginal = np.exp(logsumexp([np.log(w)+multivariate_t.logpdf(
        fixed[:9], m, np.diag(s*s), df=5) for m, s, w in zip(q.means, q.scales, q.weights)]))
    def ratio(phi):
        fixed[9] = phi
        return np.exp(q.log_prob(fixed[None])[0])/real_marginal
    total, error = quad(ratio, 0, 2*np.pi, points=[1.2, 1.3, 1.4], epsabs=1e-10)
    assert total == pytest.approx(1., abs=1e-10) and error < 1e-8


def test_sample_circular_moments_and_component_real_phase_dependence():
    base = proposal()
    q = MeasurementParticleProposal(base.means, base.scales, [.3, .7],
        component_family='student_t', phase_centres=[.2, 2.], phase_concentrations=[0., 8.])
    x = q.sample(120000, np.random.default_rng(55))
    a = i1e(q.phase_concentrations)/i0e(q.phase_concentrations)*np.exp(1j*q.phase_centres)
    expected = q.weights@a
    assert abs(np.exp(1j*x[:, 9]).mean()-expected) < .009
    # Cross moment distinguishes conditional phase from an independent draw
    # of a second component label for the circular part.
    observed_cross = (x[:, 1]*np.exp(1j*x[:, 9])).mean()
    expected_cross = q.weights@(q.means[:, 1]*a)
    assert abs(observed_cross-expected_cross) < .25
    assert abs(expected_cross-(q.weights@q.means[:, 1])*expected) > .5
    assert np.all((x[:, 9] >= 0) & (x[:, 9] < 2*np.pi))


@pytest.mark.parametrize('kwargs', [dict(phase_centres=[0.]), dict(phase_concentrations=[-1., 0.]),
    dict(phase_centres=[np.nan, 0.]), dict(phase_concentrations=[0., np.inf])])
def test_bad_phase_configuration_rejected(kwargs):
    base = proposal()
    with pytest.raises(ValueError, match='phase centres'):
        MeasurementParticleProposal(base.means, base.scales, base.weights, **kwargs)


def test_fixed_rule_preserves_broad_t_and_separate_pilot_modes():
    fitted, runs = fitted_fixture()
    q = fitted.proposals[0]
    base = proposal()
    assert q.component_family == 'student_t' and q.student_df == 5
    assert q.metadata['broad_measurement_proposal']['component_family'] == 'student_t'
    np.testing.assert_array_equal(q.means[:2], base.means)
    np.testing.assert_array_equal(q.scales[:2], base.scales)
    np.testing.assert_allclose(q.weights, [.1, .1, .2, .2, .2, .2], rtol=0, atol=1e-15)
    for i, run in enumerate(runs):
        np.testing.assert_array_equal(q.means[i+2], run[0, :, :9].mean(0))
        expected = np.sqrt(.6)*np.maximum(1.5*np.sqrt(run[0, :, :9].var(0)), .05*base.scales.max(0))
        np.testing.assert_allclose(q.scales[i+2], expected, rtol=1e-15)
        resultant = abs(np.exp(1j*run[0, :, 9]).mean())
        assert i1e(q.phase_concentrations[i+2])/i0e(q.phase_concentrations[i+2]) == pytest.approx(resultant**2.25)
    assert fitted.metadata['fit_rule'] == FIT_RULE
    assert fitted.metadata['uses_scores_or_normalizers'] is False
    assert np.isfinite(q.log_prob(np.array([[1e100]*9+[2.]]))).all()


def test_constant_and_uniform_pilot_phase_and_real_scale_floor():
    fitted, runs = fitted_fixture()
    for run in runs:
        run[:] = 0.
    runs[1][..., 9] = np.linspace(0, 2*np.pi, 32, endpoint=False)
    new = fit_initial_proposal(runs, [proposal()], fitted.metadata)
    q = new.proposals[0]
    # Exactly constant real coordinates force every pilot scale onto its
    # floor. The floor is converted to t shape after the empirical-SD max.
    expected_floor = np.sqrt(3/5)*.05*proposal().scales.max(0)
    np.testing.assert_allclose(q.scales[2:], np.tile(expected_floor, (4, 1)))
    assert np.all(q.scales[2:] < .05*proposal().scales.max(0))
    assert q.phase_concentrations.tolist() == [0., 0., 10000., 0., 10000., 10000.]
    centre = phase_moment_parameters([2*np.pi-1e-3, 1e-3])[0]
    assert np.angle(np.exp(1j*centre)) == pytest.approx(0., abs=1e-12)


def test_artifact_roundtrip_exact_and_rejects_seed_or_target_leakage(tmp_path):
    fitted, _ = fitted_fixture()
    path = tmp_path/'initial.json'; fitted.save(path)
    loaded = FrozenParticleProposal.load(path)
    x = fitted.proposals[0].sample(20, np.random.default_rng(2))
    np.testing.assert_array_equal(loaded.proposals[0].log_prob(x), fitted.proposals[0].log_prob(x))
    assert loaded.metadata == fitted.metadata
    loaded.validate_production(6, identity_protocol())
    with pytest.raises(FileExistsError): fitted.save(path)
    with pytest.raises(ValueError, match='pilot seed'): loaded.validate_production(2, identity_protocol())
    for key, value in [('observations', [12]), ('fixed_shear', [.01, 0.]),
                       ('observation_sha256', 'd'*64), ('density_batch_size', 4)]:
        changed = identity_protocol(); changed[key] = value
        with pytest.raises(ValueError, match='identity differs'): loaded.validate_production(6, changed)
    changed = identity_protocol()
    changed['implementation_sha256']['/snapshot/galsbi_summary.py'] = 'e'*64
    with pytest.raises(ValueError, match='native_target'): loaded.validate_production(6, changed)
    changed = identity_protocol(); changed['implementation_sha256']['/new/galsbi_particle_target.py'] = 'e'*64
    loaded.validate_production(6, changed)  # Explicit documented mutation-only wrapper exception.
    bad = json.loads(path.read_text()); bad['proposals'][0]['component_family'] = 'gaussian'
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Student-t'): FrozenParticleProposal.load(path)


def pilot_receipts(tmp_path):
    fitted, runs = fitted_fixture()
    paths = []
    for i, run in enumerate(runs):
        path = tmp_path/f'run{i}'; path.mkdir(); paths.append(path)
        protocol = dict(identity_protocol(), seed=i+1, particles=32, stages=2, moves=2,
            betas=[0., .5, 1.], scale=1., blocks=['physical'], algorithm='fixture',
            proposals=[proposal().metadata])
        np.savez(path/'particles.npz', coordinates=run)
        (path/'protocol.json').write_text(json.dumps(protocol))
        result = dict(status='completed', observations=[11], particles=32,
            final_stage=dict(stage=2, beta=1.), log_normalizer=[-5.],
            protocol_sha256=digest(path/'protocol.json'), particles_sha256=digest(path/'particles.npz'))
        (path/'result.json').write_text(json.dumps(result))
    return paths


def test_receipt_fit_uses_coordinates_only_records_sources_and_converts_gaussian_broad(tmp_path):
    paths = pilot_receipts(tmp_path)
    fitted = fit_initial_proposal_from_runs(paths)
    q = fitted.proposals[0]
    assert q.component_family == q.metadata['broad_measurement_proposal']['component_family'] == 'student_t'
    assert q.student_df == 5
    assert len(fitted.metadata['pilot_runs']) == 4
    for source, path in zip(fitted.metadata['pilot_runs'], paths):
        assert source['protocol_sha256'] == digest(path/'protocol.json')
        assert source['result_sha256'] == digest(path/'result.json')
        assert source['implementation_sha256']['galsbi_summary.py'] == 'c'*64
    # Normalizers do not affect the fitted law; scores are absent altogether.
    result = json.loads((paths[0]/'result.json').read_text()); result['log_normalizer'] = [1e9]
    (paths[0]/'result.json').write_text(json.dumps(result))
    changed = fit_initial_proposal_from_runs(paths)
    assert changed.proposals[0].metadata == q.metadata


@pytest.mark.parametrize('problem', ['hash', 'seed', 'extinction', 'protocol', 'duplicate_path'])
def test_bad_or_leaking_pilot_receipts_rejected(tmp_path, problem):
    paths = pilot_receipts(tmp_path)
    protocol_path, result_path = paths[1]/'protocol.json', paths[1]/'result.json'
    protocol, result = json.loads(protocol_path.read_text()), json.loads(result_path.read_text())
    if problem == 'hash': result['particles_sha256'] = 'f'*64
    elif problem == 'seed': protocol['seed'] = 1
    elif problem == 'extinction': result['status'] = 'completed_with_extinction'
    elif problem == 'protocol': protocol['blocks'] = ['independence']
    else: paths[1] = paths[0]
    protocol_path.write_text(json.dumps(protocol)); result['protocol_sha256'] = digest(protocol_path)
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError): fit_initial_proposal_from_runs(paths)


@pytest.mark.parametrize('beta', [0., .4, 1.])
def test_nonuniform_phase_enters_bridge_and_independence_hastings(monkeypatch, beta):
    import sbsi.galsbi_particle_target as target_module
    target, state = setup_target()
    q = MeasurementParticleProposal(np.tile(state.x[0, :9], (2, 1)), np.ones(9), [.2, .8],
        component_family='student_t', phase_centres=[0., .7], phase_concentrations=[0., 8.])
    target.proposals = [q]
    old = target.evaluate(state.x, state.scene)
    point = old.x.copy(); point[:, 9] += .13
    proposed = target.evaluate(point, old.scene)
    np.testing.assert_array_equal(old.log_proposal, q.log_prob(old.x))
    monkeypatch.setattr(q, 'sample', lambda n, rng: point.copy())
    ratios = []
    monkeypatch.setattr(target_module, 'metropolis_accept',
        lambda old, new, u: ratios.append(new-old) or np.ones(len(old), bool))
    target.mutate(old, beta, PrescribedRNG(), block='independence')
    expected = beta*(proposed.log_prior+proposed.log_kernel-old.log_prior-old.log_kernel
                     +old.log_proposal-proposed.log_proposal)
    np.testing.assert_allclose(ratios[0], expected, atol=1e-12, rtol=0)
    assert abs(float((proposed.log_proposal-old.log_proposal)[0])) > .01


@pytest.mark.parametrize('option', ['--proposal-family', '--proposal-df', '--faint-shift'])
def test_driver_rejects_silent_override_before_loading_models(monkeypatch, driver, option):
    value = 'student_t' if option == '--proposal-family' else '5'
    monkeypatch.setattr(sys, 'argv', ['driver', '--reference', 'missing', '--output', 'missing',
        '--seed', '99', '--initial-proposal', 'missing.json', option, value])
    with pytest.raises(SystemExit) as error: driver.main()
    assert error.value.code == 2
