import copy

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.flow_disk import radial_tanh
from sbsi.measurement_model import TargetStandardizer, build_flow, load_measurement_model
from sbsi.selection_model import TabularPreprocessor
from scripts.train_flow_disk_nll import fit, load_torch, nll_pass


def _fixture():
    torch.manual_seed(731)
    cfg = dict(flow_type='disk_affine', target_dim=4, context_dim=2,
               hidden_dim=8, n_layers=1, n_flows=2, disk_map='radial_tanh',
               disk_shape_means=[0., 0.], disk_shape_scales=[.3, .3],
               disk_coordinate_means=[0., 0.], disk_coordinate_scales=[.6, .6])
    model = build_flow(cfg)
    data = {}
    for name, rows in [('train', 23), ('validation', 11)]:
        c = torch.randn(rows, 2)
        u = torch.randn(rows, 2).double() * .3 + c.double() * .4
        e, _ = radial_tanh(u)
        data[name] = [torch.cat((e / .3, torch.randn(rows, 2).double()), dim=1), c]
    target = TargetStandardizer(['measured_ngmix_g1', 'measured_ngmix_g2', 'mag', 'logr'],
                                 np.zeros(4), np.array([.3, .3, 1., 1.]), dtype='float64')
    preprocessor = TabularPreprocessor.fit(pd.DataFrame({'x': [0., 1.]}), ['x'])
    return model, data, preprocessor, target, cfg


def test_nll_pass_counts_remainder_rows_and_validation_is_exact():
    model, data, *_ = _fixture()
    y, c = data['validation']
    expected = float(-model.log_prob(y, c).mean().detach())
    result = nll_pass(model, y, c, 4)
    assert result['rows'] == 11 and result['optimizer_steps'] == 0
    assert abs(result['nll'] - expected) < 1.e-12
    assert all(p.grad is None for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.e-4, weight_decay=0.)
    train = nll_pass(model, *data['train'], 5, optimizer=optimizer)
    assert train['rows'] == 23 and train['optimizer_steps'] == 5
    assert np.isfinite(train['gradient_norms']).all()


def test_disk_nll_probes_reset_and_optimizer_resume_match_uninterrupted(tmp_path):
    initial, data, condition, target, config = _fixture()
    common = dict(batch_size=5, learning_rates=[1.e-5, 1.e-3], probe_steps=2, seed=912)
    baseline_state = copy.deepcopy(initial.state_dict())
    full = fit(copy.deepcopy(initial), data, condition, target, config, {}, tmp_path / 'full', epochs=2, **common)
    fit(copy.deepcopy(initial), data, condition, target, config, {}, tmp_path / 'resumed', epochs=1, **common)
    resumed = fit(copy.deepcopy(initial), data, condition, target, config, {}, tmp_path / 'resumed', epochs=2,
                  resume=True, **common)
    left = load_torch(tmp_path / 'full' / 'resume.pt')
    right = load_torch(tmp_path / 'resumed' / 'resume.pt')
    for name in left['state_dict']:
        torch.testing.assert_close(left['state_dict'][name], right['state_dict'][name], rtol=0, atol=0)
    for key, state in left['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, right['optimizer']['state'][key][name], rtol=0, atol=0)
    assert full['chosen_lr'] == resumed['chosen_lr']
    for a, b in zip(full['history'], resumed['history']):
        assert a['validation']['nll'] == b['validation']['nll']
        if a['train']:
            assert a['train']['rows'] == 23 and a['train']['optimizer_steps'] == 5
            assert a['train']['gradient_norms'] == b['train']['gradient_norms']
    epoch0 = load_measurement_model(tmp_path / 'full' / 'disk_nll_epoch000.pt')
    for name, value in baseline_state.items():
        torch.testing.assert_close(value, epoch0.model.state_dict()[name], rtol=0, atol=0)
    assert epoch0.metadata['training_phase'] == 'coordinate_changed_initialization'
    assert not full['response_accuracy_validated']
    # A partial future epoch file is replaced from the committed state on resume;
    # changing the recipe is not accepted as an optimizer continuation.
    with pytest.raises(ValueError, match='unchanged data/recipe'):
        fit(copy.deepcopy(initial), data, condition, target, config, {}, tmp_path / 'resumed', epochs=3,
            resume=True, **dict(common, batch_size=4))


def test_nll_failure_does_not_silently_step_on_invalid_shapes():
    model, data, *_ = _fixture()
    bad = data['train'][0].clone()
    bad[0, :2] = torch.tensor([4., 0.])  # physical e=(1.2,0)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.e-4)
    with pytest.raises(FloatingPointError, match='nonfinite NLL'):
        nll_pass(model, bad, data['train'][1], len(bad), optimizer=optimizer)
    for name, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
