"""Periodic identity, fixed independent regression, and whole-run diagnostics."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import vonmises

from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import scene_model_inputs
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.phase_control import (FrozenPhaseControl, digest, fit_frozen_phase_control,
    load_phase_diagnostic, phase_stein_control, rotate_summary_phase, summarize_phase_populations)
from sbsi.summary_gradients import SummaryGradients
from test_galsbi_prior import prior_fixture
from test_galsbi_summary import marks
from test_summary_gradients import fixture as gradient_fixture


def metadata():
    return dict(observations=[11], pilot_seeds=[1, 2, 3, 4],
                target_identity=dict(model='analytic von Mises', shear=[0., 0.], precision='continuous'))


def periodic_values(phase, kappa=4., mu=.7, radius=.4):
    phase = np.asarray(phase)
    values = np.zeros((len(phase), 9))
    values[:, :2] = radius*np.column_stack((np.cos(phase), np.sin(phase)))
    gradient = np.zeros((len(phase), 8))
    gradient[:, :2] = kappa/radius*np.array([np.cos(mu), np.sin(mu)])
    control = phase_stein_control(values, gradient)
    return values, gradient, control


def fit_fixture():
    scores, controls = [], []
    for seed in metadata()['pilot_seeds']:
        phase = np.random.default_rng(seed).vonmises(.7, 4., 50+seed)
        _, _, control = periodic_values(phase)
        controls.append(control[None])
        scores.append((-control[:, None]*np.array([1., 2.]))[None])
    return fit_frozen_phase_control(scores, controls, metadata())


@pytest.mark.parametrize('mu', [0., .7, 2*np.pi-1e-3])
@pytest.mark.parametrize('kappa', [0., 2., 30.])
def test_periodic_stein_identity_integrates_to_zero_with_nonuniform_target(mu, kappa):
    def integrand(phase):
        _, _, control = periodic_values([phase], mu=mu, kappa=kappa)
        return control[0]*vonmises.pdf(phase, kappa, loc=mu)
    value, error = quad(integrand, 0., 2*np.pi, epsabs=1e-10, points=[mu])
    assert abs(value) < 1e-9 and error < 1e-8


def test_frozen_independent_translation_score_has_zero_residual_and_preserves_raw():
    frozen = fit_fixture()
    np.testing.assert_allclose(frozen.coefficients, [[-1., -2.]], atol=1e-14)
    phase = np.random.default_rng(83).vonmises(.7, 4., 1000)
    _, _, control = periodic_values(phase)
    scores = (-control[:, None]*[1., 2.])[None]
    expected = scores.copy()
    result = frozen.apply(scores, control[None], seed=83,
        observations=[11], target_identity=metadata()['target_identity'])
    np.testing.assert_allclose(result['corrected_scores'], 0., atol=2e-14)
    np.testing.assert_array_equal(result['raw_scores'], expected)
    np.testing.assert_array_equal(scores, expected)
    result['raw_scores'][:] = 0
    np.testing.assert_array_equal(scores, expected)
    assert not frozen.coefficients.flags.writeable
    assert frozen.metadata['pilot_particles'] == [51, 52, 53, 54]


def test_no_empirical_recentering_and_nonzero_expectation_preserved():
    rng = np.random.default_rng(84)
    t = [rng.normal(.5, 1., size=(1, 30)) for _ in range(4)]
    s = [np.array([3., -7.])+a[:, :, None]*[2., -4.] for a in t]
    frozen = fit_frozen_phase_control(s, t, metadata())
    production_t = np.array([[1., 2., 3.]])
    production_s = np.array([3., -7.])+production_t[:, :, None]*[2., -4.]
    out = frozen.apply(production_s, production_t, seed=83,
        observations=[11], target_identity=metadata()['target_identity'])
    np.testing.assert_allclose(out['corrected_scores'], np.tile([3., -7.], (1, 3, 1)), atol=1e-14)
    assert not np.allclose(out['corrected_scores'].mean(1), production_s.mean(1))


def test_constant_control_disables_coefficient_without_false_zero_mean_assumption():
    controls = [np.full((1, 4), 7.) for _ in range(4)]
    scores = [np.arange(8.).reshape(1, 4, 2) for _ in range(4)]
    frozen = fit_frozen_phase_control(scores, controls, metadata())
    np.testing.assert_array_equal(frozen.coefficients, [[0., 0.]])
    assert frozen.metadata['zero_variance'] == [True]
    assert frozen.metadata['pilot_control_mean'] == [7.]


def test_artifact_provenance_roundtrip_seed_guard_and_no_overwrite(tmp_path):
    frozen = fit_fixture(); path = tmp_path/'control.json'; frozen.save(path)
    loaded = FrozenPhaseControl.load(path)
    np.testing.assert_array_equal(loaded.coefficients, frozen.coefficients)
    assert loaded.metadata == frozen.metadata and len(loaded.metadata['pilot_array_sha256']) == 4
    changed = loaded.metadata; changed['pilot_seeds'][0] = 99
    assert loaded.metadata['pilot_seeds'][0] == 1
    with pytest.raises(FileExistsError): frozen.save(path)
    with pytest.raises(ValueError, match='overlaps'): loaded.validate_production(2, [11], metadata()['target_identity'])
    with pytest.raises(ValueError, match='observations'): loaded.validate_production(99, [12], metadata()['target_identity'])
    with pytest.raises(ValueError, match='identity'): loaded.validate_production(99, [11], {'precision':'different'})
    bad = metadata(); bad['pilot_seeds'] = [1, 1, 3, 4]
    with pytest.raises(ValueError, match='distinct'): FrozenPhaseControl([[1., 1.]], bad)


def test_phase_wrap_chain_rule_and_zero_shape():
    values, gradient, control = periodic_values([0., 1e-9, 2*np.pi-1e-9, 3.])
    for angle in (-4*np.pi, 0., 2*np.pi):
        moved = rotate_summary_phase(values, angle)
        np.testing.assert_allclose(phase_stein_control(moved, gradient), control, atol=1e-14)
        np.testing.assert_array_equal(moved[:, 2:], values[:, 2:])
    values[0, :2] = 0
    assert phase_stein_control(values, gradient)[0] == 0
    with pytest.raises(ValueError, match='beta=1'): phase_stein_control(values, gradient, beta=.9)


@pytest.mark.parametrize('shear', [[0., 0.], [.02, -.01]])
def test_real_tensor_context_phase_derivative_matches_periodic_finite_difference(shear):
    flow, detector, values, observed = gradient_fixture()
    evaluator = SummaryGradients(flow, detector, batch_size=3)
    density, gradient = evaluator.evaluate(values, observed, shear, wrt='context')
    control = phase_stein_control(values, gradient)
    step = 1e-5
    up = evaluator.evaluate(rotate_summary_phase(values, step), observed, shear)[0]
    down = evaluator.evaluate(rotate_summary_phase(values, -step), observed, shear)[0]
    np.testing.assert_allclose(control, (up-down)/(2*step), rtol=2e-6, atol=2e-8)
    wrapped = evaluator.evaluate(rotate_summary_phase(values, 2*np.pi), observed, shear)[0]
    np.testing.assert_allclose(density, wrapped, rtol=0, atol=1e-12)


def test_native_target_prior_jacobian_crowding_and_response_features_are_phase_invariant():
    prior = GalSBIScenePrior(SimpleNamespace(), aperture_arcsec=10.)
    targets, neighbours = marks([.2, .5]), marks([.3, .4])
    scenes = GalSBISceneBatch(targets, neighbours, [0, 1, 2], [[.2, .4], [.8, -.1]])
    conditions = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224, pixel_rms=.312)
    def inputs(s):
        return scene_model_inputs(s, prior, lambda z: np.full(len(z), 1000.), conditions,
                                  response_radius_arcsec=10., response_neighbours=20)
    before = inputs(scenes)
    rotated = deepcopy(targets); rotated['psi'] += .4
    after = inputs(GalSBISceneBatch(rotated, neighbours, scenes.indptr, scenes.offset_arcsec))
    np.testing.assert_array_equal(before[0].iloc[:, 2:], after[0].iloc[:, 2:])
    np.testing.assert_array_equal(before[1], after[1]); np.testing.assert_array_equal(before[2], after[2])
    assert not np.allclose(before[0].iloc[:, :2], after[0].iloc[:, :2])
    galaxy = prior_fixture(); native, _ = galaxy.sample(64, 19)
    coordinates = GalSBITargetCoordinates(galaxy, lambda z: np.full(len(z), 1000.))
    x = np.repeat(coordinates.encode({k:v[:1] for k,v in native.items()})[:1], 4, axis=0)
    x[:, 9] = [0., .4, 2*np.pi-1e-9, 2*np.pi+.4]
    logp, _, _ = coordinates.log_mixture_prior(x)
    assert np.isfinite(logp).all()
    np.testing.assert_array_equal(logp, np.full(4, logp[0]))


def test_independent_population_mcse_and_precision_reweighting_use_whole_run_normalizers():
    controls = [np.array([[1., 2.]]), np.array([[-3., -1., 2.]])]
    scores = [np.stack((t, 2*t), axis=2) for t in controls]
    delta = [np.array([[0., 0.]]), np.log(np.array([[1., 4., 2.]]))]
    logz = np.log([[1.], [5.]])
    row = summarize_phase_populations(logz, controls, scores, delta, np.array([[1., 2.]]))[0]
    raw = row['raw_float32_sampled']; weighted = row['precision_reweighted_float64']
    raw_means = np.array([t.mean() for t in controls])
    expected_raw = np.average(raw_means, weights=[1., 5.])
    assert raw['mean'][0] == pytest.approx(expected_raw)
    influence = np.array([1., 5.])/3*(raw_means-expected_raw)
    assert raw['mcse'][0] == pytest.approx(np.sqrt(np.sum(influence**2)/2))
    z64 = np.array([1., 5.*7/3])
    expected = (1.*1.5+5.*np.mean(np.array([-3., -1., 2.])*[1., 4., 2.]))/z64.sum()
    assert weighted['mean'][0] == pytest.approx(expected)
    np.testing.assert_allclose(weighted['weights'], z64/z64.sum())
    np.testing.assert_allclose(weighted['mean'][-2:], 0., atol=1e-15)
    assert weighted['control_standardized_residual'] == pytest.approx(weighted['mean'][0]/weighted['mcse'][0])


def test_log_precision_reweighting_is_stable_and_constant_delta_leaves_control_mean_unchanged():
    controls = [np.array([[float(i), float(i+1)]]) for i in range(4)]
    scores = [np.stack((t, t), axis=2) for t in controls]
    delta = [np.full_like(t, 1000.) for t in controls]
    row = summarize_phase_populations(np.zeros((4, 1)), controls, scores, delta)[0]
    a, b = row['raw_float32_sampled'], row['precision_reweighted_float64']
    np.testing.assert_allclose(a['mean'], b['mean'], atol=1e-12)
    np.testing.assert_allclose(a['mcse'], b['mcse'], atol=1e-12)
    assert b['log_mean_normalizer']-a['log_mean_normalizer'] == pytest.approx(1000.)


def test_zero_corrected_variance_does_not_hide_nonstationary_stein_residual():
    controls = [np.full((1, 4), i+1.) for i in range(4)]
    scores = [-np.stack((t, 2*t), axis=2) for t in controls]
    row = summarize_phase_populations(np.zeros((4, 1)), controls, scores,
        [np.zeros_like(t) for t in controls], np.array([[-1., -2.]]))[0]['raw_float32_sampled']
    np.testing.assert_allclose(row['mean'][-2:], 0.)
    assert row['mean'][0] == pytest.approx(2.5)
    assert row['control_standardized_residual'] > 3.


def write_diagnostic(tmp_path, *, passed=True):
    path = tmp_path/'diagnostic.json'; source = tmp_path/'source.json'; source.write_text('{}')
    np.savez(path.with_suffix('.npz'), raw_scores=np.ones((1, 3, 2)),
             phase_control=np.ones((1, 3)), delta_log_kernel=np.zeros((1, 3)))
    report = dict(schema='sbsi-phase-diagnostic-v1', numerical_checks_pass=passed,
        arrays_sha256=digest(path.with_suffix('.npz')), source_hashes={str(source):digest(source)}, observations=[11])
    path.write_text(json.dumps(report))
    return path


def test_diagnostic_loader_rejects_failed_checks_or_mutated_input(tmp_path):
    path = write_diagnostic(tmp_path)
    _, arrays = load_phase_diagnostic(path)
    assert arrays['phase_control'].shape == (1, 3)
    (tmp_path/'source.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='input receipts'): load_phase_diagnostic(path)
    path = write_diagnostic(tmp_path, passed=False)
    with pytest.raises(ValueError, match='successfully replayed'): load_phase_diagnostic(path)


def load_script(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def diagnostic_group(root, label, seeds):
    paths = []
    for i, seed in enumerate(seeds):
        folder = root/f'{label}_{i}'; folder.mkdir(parents=True)
        path = write_diagnostic(folder)
        report = json.loads(path.read_text())
        report.update(run=str(folder), seed=seed, target_identity=metadata()['target_identity'],
                      log_normalizer=[0.], stride=1)
        path.write_text(json.dumps(report)); paths.append(path)
    return paths


def test_fit_and_report_cli_keep_pilot_and_evaluation_seeds_separate(tmp_path, monkeypatch):
    fitter = load_script('fit_particle_phase_control')
    reporter = load_script('report_particle_phase_control')
    pilots = diagnostic_group(tmp_path, 'dev_v1', [1, 2, 3, 4])
    coefficients = tmp_path/'frozen.json'
    monkeypatch.setattr(sys, 'argv', ['fit', '--diagnostics', *map(str, pilots), '--output', str(coefficients)])
    fitter.main()
    assert FrozenPhaseControl.load(coefficients).metadata['pilot_seeds'] == [1, 2, 3, 4]
    evaluation = diagnostic_group(tmp_path, 'dev_v5', list(range(10, 18)))
    output = tmp_path/'report.json'
    argv = ['report', '--diagnostics', *map(str, evaluation), '--coefficients', str(coefficients), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv); reporter.main()
    report = json.loads(output.read_text())
    assert report['raw_score_and_normalizer_gates_unchanged'] is True and report['accepted_sampler'] is False
    assert report['moment_columns'] == ['T', 'raw_g1', 'raw_g2', 'corrected_g1', 'corrected_g2']
    row = report['observations'][0]['raw_float32_sampled']
    assert row['independent_populations'] == 8 and row['control_nonzero_with_zero_estimated_mcse']
    assert row['control_standardized_residual'] is None
    changed = json.loads(evaluation[0].read_text()); changed['seed'] = 2
    evaluation[0].write_text(json.dumps(changed))
    argv[-1] = str(tmp_path/'invalid.json'); monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='overlaps'): reporter.main()
    monkeypatch.setattr(sys, 'argv', ['fit', '--diagnostics', *map(str, evaluation[:4]), '--output', str(tmp_path/'bad_fit.json')])
    with pytest.raises(ValueError, match='dev_v1'): fitter.main()


def test_saved_state_validator_rejects_extinction_and_preserves_fixed_stride(tmp_path):
    diagnostic = load_script('check_particle_phase_control')
    protocol = dict(observations=[11], particles=4, stages=3)
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    np.savez(tmp_path/'particles.npz', summaries=np.ones((1, 4, 9)), scores=np.ones((1, 4, 2)),
             log_kernel=np.ones((1, 4)))
    receipt = dict(status='completed', observations=[11], particles=4, final_stage=dict(stage=3, beta=1.),
        log_normalizer=[0.], protocol_sha256=digest(tmp_path/'protocol.json'), particles_sha256=digest(tmp_path/'particles.npz'))
    (tmp_path/'result.json').write_text(json.dumps(receipt))
    _, _, arrays = diagnostic.validate_saved_run(tmp_path, stride=2)
    assert arrays['summaries'].shape == (1, 2, 9)
    receipt['status'] = 'completed_with_extinction'
    (tmp_path/'result.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='no dropping'): diagnostic.validate_saved_run(tmp_path)


@pytest.mark.parametrize('bad_gradient', [False, True])
def test_full_diagnostic_retains_delta_raw_and_corrected_and_blocks_failed_replay(tmp_path, monkeypatch, bad_gradient):
    diagnostic = load_script('check_particle_phase_control')
    root = Path(diagnostic.__file__).resolve().parents[1]
    run, reference = tmp_path/'run', tmp_path/'reference'; run.mkdir(); reference.mkdir()
    observed = pd.DataFrame([[.1, .2, 3., 4.]], columns=list('wxyz'))
    observed.to_parquet(reference/'observations.parquet')
    (reference/'protocol.json').write_text('{}')
    values, gradient, control = periodic_values([.2, .7, 2.2, 4.], mu=0.)
    raw = -control[:, None]*[1., 2.]
    density = 10*values[:, 0]
    np.savez(run/'particles.npz', summaries=values[None], scores=raw[None], log_kernel=density[None])
    protocol = dict(observations=[11], particles=4, stages=3, seed=99, reference=str(reference),
        fixed_shear=[0., 0.], density_batch_size=4, selection='none',
        reference_protocol_sha256=digest(reference/'protocol.json'),
        observation_sha256=digest(reference/'observations.parquet'),
        implementation_sha256={str(root/('configs' if n.endswith('.json') else 'sbsi')/n):
            digest(root/('configs' if n.endswith('.json') else 'sbsi')/n) for n in diagnostic.NATIVE_SOURCES})
    # Reference IDs index the complete table, not its diagnostic subset.
    observed = pd.concat([observed]*12, ignore_index=True)
    observed.to_parquet(reference/'observations.parquet')
    protocol['observation_sha256'] = digest(reference/'observations.parquet')
    (run/'protocol.json').write_text(json.dumps(protocol))
    receipt = dict(status='completed', observations=[11], particles=4, log_normalizer=[0.],
        final_stage=dict(stage=3, beta=1.), protocol_sha256=digest(run/'protocol.json'),
        particles_sha256=digest(run/'particles.npz'))
    (run/'result.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(diagnostic, 'V39_LIKE', SimpleNamespace(validate=lambda:None, flow_checkpoints=['fixture']))
    monkeypatch.setattr(diagnostic, 'load_measurement_model', lambda *a, **kw: SimpleNamespace(
        model=torch.nn.Linear(1, 1), target_transform=SimpleNamespace(target_names=list('wxyz'))))
    monkeypatch.setattr(diagnostic, 'load_detection_classifier', lambda *a, **kw: SimpleNamespace(model=torch.nn.Linear(1, 1)))
    class Likelihood:
        def __init__(self, *args, **kwargs): pass
        def log_kernel(self, summary, observed, *shear): return 10*summary.values[:, 0]
    class Gradients:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, v, observed, shear, *, wrt):
            if wrt == 'context':
                result = np.zeros((len(v), 8)); result[:, 0] = 20 if bad_gradient else 10
            else:
                result = (10*v[:, 1, None])*[1., 2.]
            return 10*v[:, 0], result
    monkeypatch.setattr(diagnostic, 'SummaryLikelihood', Likelihood)
    monkeypatch.setattr(diagnostic, 'SummaryGradients', Gradients)
    output = tmp_path/'first.json'
    argv = ['diagnostic', '--run', str(run), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv); diagnostic.main()
    report = json.loads(output.read_text())
    assert report['numerical_checks_pass'] is (not bad_gradient)
    assert report['exact_zero_mean_for_float32_target_established'] is False
    with np.load(output.with_suffix('.npz')) as arrays:
        np.testing.assert_array_equal(arrays['delta_log_kernel'], np.zeros((1, 4)))
        np.testing.assert_array_equal(arrays['saved_raw_scores'], raw[None])
        np.testing.assert_allclose(arrays['raw_scores'], raw[None], atol=1e-14)
    fitted = fit_frozen_phase_control([raw[None]]*4, [control[None]]*4,
        dict(metadata(), target_identity=report['target_identity']))
    coefficient_path = tmp_path/'coefficients.json'; fitted.save(coefficient_path)
    output2 = tmp_path/'second.json'
    argv[-1] = str(output2); argv += ['--coefficients', str(coefficient_path)]
    monkeypatch.setattr(sys, 'argv', argv); diagnostic.main()
    report2 = json.loads(output2.read_text())
    with np.load(output2.with_suffix('.npz')) as arrays:
        if bad_gradient:
            assert 'corrected_scores' not in arrays and 'correction_blocked' in report2
        else:
            np.testing.assert_allclose(arrays['corrected_scores'], 0., atol=1e-14)
    assert digest(run/'particles.npz') == receipt['particles_sha256']
    assert digest(run/'protocol.json') == receipt['protocol_sha256']
