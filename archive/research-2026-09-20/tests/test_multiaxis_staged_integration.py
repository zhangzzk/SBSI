"""Full-main checks for an explicitly supplied staged evaluator source.

Set SBSI_MULTIAXIS_EVALUATOR while validating the staged
copy. The active evaluator remains frozen while existing GPU jobs use it.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.mark.parametrize('axis', [1, 2])
def test_staged_disk_main_recovers_known_response_on_each_axis(tmp_path, monkeypatch, axis):
    source = os.environ.get('SBSI_MULTIAXIS_EVALUATOR')
    if not source:
        pytest.skip('explicit staged evaluator path required')
    spec = importlib.util.spec_from_file_location('staged_axis_evaluator', Path(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = np.array([1, 2])
    context = np.array([[0., 0., 1., 25., 1., 0., 0., 0.],
                        [0., 0., 1., 24., 1., 0., 0., 0.]])
    reports = {}
    for case in (200, 201):
        path = tmp_path/f'case{case:03d}.npz'
        np.savez(path, input_index=ids, context=context)
        reports[str(case)] = {'sha256': module.sha256(path)}
    manifest = {'cases': [200, 201], 'case_reports': reports, 'population': 'synthetic',
                'cohorts': ['truth_parent'], 'g0_diagnostics_available': False}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    dummy = tmp_path/'dummy'
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
        case: {'input_index': ids, 'R_blend': np.full(2, .2)} for case in (200, 201)})
    candidate = SimpleNamespace(conditioned=False, response_coordinate='disk_velocity', metadata={},
                                 predict_pairs=lambda *a: np.full(2, .2))
    monkeypatch.setattr(module, 'DiskResponseMoment', lambda *a, **k: candidate)
    monkeypatch.setattr(module, 'load_candidate_pairs', lambda *a: (
        pd.DataFrame({'index_input_p': ids}), {}))

    def measured(root, case, h, identities):
        # Closed form for collinear Mobius composition of .5*h and tanh(.2*h).
        value = (.5*h+np.tanh(.2*h))/(1+.5*h*np.tanh(.2*h))
        physical = np.tile([0., 0., 4., 100.], (2, 1))
        physical[:, axis-1] = value
        return physical, np.ones(2, bool), {'parent_rows': 2, 'parent_usable': 2}

    def draws(model, contexts, count, seed):
        return [torch.cat((.5*frame[:, :2], torch.full((2, 1), 4.),
                           torch.full((2, 1), 100.)), dim=1)[:, None].repeat(1, count, 1)
                for frame in contexts]

    monkeypatch.setattr(module, 'measured_leg', measured)
    monkeypatch.setattr(module, 'physical_context_draws', draws)
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic-unit-test')
    common = ['staged', '--parent-root', str(tmp_path), '--constgold-root', str(tmp_path),
              '--flow', str(dummy), '--classifier', str(dummy), '--rblend', str(dummy),
              '--response-candidate', str(dummy), '--pair-root', str(tmp_path),
              '--response-composition', 'mobius', '--device', 'cpu', '--axis', str(axis),
              '--cohorts', 'truth_parent', '--draws', '4', '--h', '.02']
    for single in (False, True):
        output = tmp_path/f'axis{axis}_single{single}.json'
        options = ['--cases', '200', '--operational-single-case'] if single else ['--cases', '200', '201']
        monkeypatch.setattr(sys, 'argv', common+options+['--output', str(output)])
        module.main()
        result = json.loads(output.read_text())
        assert result['shear_axis'] == axis
        assert result['m_diagonal'] == f'R{axis}{axis}'
        assert result['operational_single_case'] is single
        for entry in result['summary'].values():
            np.testing.assert_allclose(entry['m_percent'], 0., atol=1e-11)
            response = np.array(entry['model_response'])
            assert .69 < response[axis-1] < .71
            assert response[2-axis] == 0.
            np.testing.assert_allclose(entry['model_flow_response'][axis-1], .5)
            assert .19 < entry['model_blend_response'][axis-1] < .21
            if single:
                assert entry['m_standard_error_percentage_points'] is None
                assert entry['m_ci95_percent'] == [None, None]
