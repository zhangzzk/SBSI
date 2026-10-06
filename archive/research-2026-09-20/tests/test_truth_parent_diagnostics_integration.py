"""Synthetic CPU integration of the truth-parent evaluator; no real catalogue scan."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("zero_flux", [False, True])
def test_staged_main_no_g0_and_diagnostics_do_not_change_response(tmp_path, monkeypatch, zero_flux):
    from scripts import evaluate_constgold_truth_parent as module
    ids = np.array([1, 2])
    context = np.array([[.1, .05, 1., 25., 1., 0., 0., 0.],
                        [-.1, -.05, 1., 24., 1., 0., 0., 0.]])
    reports = {}
    for case in (200, 201):
        parent = tmp_path / f'case{case:03d}.npz'
        np.savez(parent, input_index=ids, context=context)
        reports[str(case)] = {'sha256': module.sha256(parent)}
    manifest = {'cases': [200, 201], 'case_reports': reports, 'population': 'synthetic truth parent',
                'cohorts': ['truth_parent'], 'g0_diagnostics_available': False}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    dummy = tmp_path/'dummy_model'
    dummy.write_text('synthetic model')
    features = SimpleNamespace(feature_names=list(module.FLOW_FEATURES))
    bundle = SimpleNamespace(model=SimpleNamespace(eval=lambda: None),
        condition_preprocessor=features,
        target_transform=SimpleNamespace(target_names=list(module.FLOW_TARGETS)),
        context_tensor=lambda frame: torch.from_numpy(frame.to_numpy(float)))
    classifier = SimpleNamespace(preprocessor=features,
                                 predict_proba=lambda frame: np.full(len(frame), .8))
    monkeypatch.setattr(module, 'load_measurement_model', lambda *a, **k: bundle)
    monkeypatch.setattr(module, 'load_selection_model_ensemble', lambda *a, **k: classifier)
    monkeypatch.setattr(module, 'load_lookup', lambda *a, **k: {
        case: {'input_index': ids, 'R_blend': np.zeros(2)} for case in (200, 201)})

    def measured(root, case, shear, identities):
        values = np.column_stack((context[:, 0] + .7*shear, context[:, 1],
                                  np.full(2, 4.), np.full(2, 100.)))
        return values, np.ones(2, dtype=bool), {'parent_rows': 2, 'parent_usable': 2}

    def draws(model, contexts, count, seed):
        result = []
        for frame in contexts:
            value = torch.cat((frame[:, :2], torch.full((len(frame), 1), 4.),
                               torch.full((len(frame), 1), 100.)), dim=1)
            sampled = value[:, None, :].repeat(1, count, 1)
            if zero_flux:
                sampled[0, 0, 3] = 0.
            result.append(sampled)
        return result

    monkeypatch.setattr(module, 'measured_leg', measured)
    monkeypatch.setattr(module, 'physical_context_draws', draws)
    real_as_tensor = torch.as_tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return real_as_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, 'as_tensor', cpu_tensor)
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic-unit-test')
    common = ['staged', '--parent-root', str(tmp_path), '--constgold-root', str(tmp_path),
              '--flow', str(dummy), '--classifier', str(dummy), '--rblend', str(dummy),
              '--cases', '200', '201', '--cohorts', 'truth_parent', '--draws', '4', '--h', '.05']
    results = []
    for predictive in (False, True):
        output = tmp_path/f'result_{predictive}.json'
        monkeypatch.setattr(sys, 'argv', common + ['--output', str(output)] +
                            (['--predictive-diagnostics'] if predictive else []))
        module.main()
        results.append(json.loads(output.read_text()))
    assert results[0]['summary'] == results[1]['summary']
    assert results[0]['per_case_sufficient'] == results[1]['per_case_sufficient']
    assert results[1]['cohorts'] == ['truth_parent']
    assert len(results[1]['summary']) == 9
    assert results[0]['predictive_diagnostics'] is None
    check = results[1]['predictive_diagnostics']
    for direction in ('plus', 'minus'):
        for source in ('observed', 'generated'):
            expected = [2., 1.75, 1.75] if zero_flux and source == 'generated' else [2., 2., 2.]
            np.testing.assert_allclose(check['per_case'][source][direction]['mass'], [expected, expected])
            block = check['per_case'][source][direction]
            assert np.asarray(block['histogram']).sum()+np.asarray(block['invalid_photometry_mass']).sum() == 4
        assert check['summary'][direction]['outside_disk_fraction'] == {'observed': 0., 'generated': 0.}
