"""Cell regression/covariance, unbiased loss gradients, and real training integration."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.flow_grid_response import fit_grid_response, independent_grid_score, GridResponsePopulation
from sbsi.flow_response_target import estimate_response_matrix


def synthetic():
    rng = np.random.default_rng(31)
    cases = np.repeat(np.arange(16), 48)
    bins = np.tile(np.repeat(np.arange(2), 24), 16)
    angle = rng.uniform(-np.pi, np.pi, len(cases))
    gamma = .05*np.column_stack([np.cos(angle), np.sin(angle)])
    response = np.array([[.9,.04],[-.03,.8]])
    shifts = rng.normal(0, .08, (16,2,2))
    delta = np.einsum("nij,nj->ni", response+shifts[cases], gamma)
    delta += rng.normal(0,.002, delta.shape)
    return delta, gamma, bins, cases


def test_full_covariance_matches_brute_case_jackknife_and_fixed_design():
    delta, gamma, bins, cases = synthetic()
    result = fit_grid_response(delta, gamma, bins, cases, 2)
    for b in range(2):
        keep = bins == b
        direct = estimate_response_matrix(delta[keep], gamma[keep])
        np.testing.assert_allclose(result['target'][b], direct.ravel(), atol=1e-12)
        # Averaging per-pair contributions recovers the *same* finite WLS matrix.
        contributions = delta[keep,:,None]*result['design'][keep,None,:]
        np.testing.assert_allclose(contributions.mean(0), direct, atol=1e-12)
        leave = np.array([estimate_response_matrix(delta[keep & (cases != c)],
                         gamma[keep & (cases != c)]).ravel() for c in range(16)])
        centered = leave-leave.mean(0)
        expected = 15/16*centered.T@centered
        np.testing.assert_allclose(result['covariance'][b], expected, rtol=1e-11, atol=1e-15)
        np.testing.assert_allclose(result['precision'][b]@expected, np.eye(4), atol=1e-11)
        assert np.max(np.abs(expected-np.diag(np.diag(expected)))) > 1e-6


def test_covariance_rejects_unidentified_cells_and_directions():
    delta,gamma,bins,cases = synthetic()
    with pytest.raises(ValueError,match='every grid cell'):
        fit_grid_response(delta,gamma,bins,cases,3)
    gamma[:,1] = 0
    with pytest.raises(ValueError,match='shear design'):
        fit_grid_response(delta,gamma,bins,cases,2)
    with pytest.raises(ValueError,match='eight independent'):
        fit_grid_response(delta,gamma,bins,np.zeros_like(cases),2)


def test_independent_score_has_exact_mean_gradient_without_variance_penalty():
    theta = torch.tensor(.6,dtype=torch.double,requires_grad=True)
    spread = torch.tensor(3.,dtype=torch.double,requires_grad=True)
    direction = torch.tensor([[1.,2.,-.5,.3]],dtype=torch.double)
    target = torch.tensor([[.1,.1,.2,-.2]],dtype=torch.double)
    precision = torch.tensor([[[2.,.1,0,0],[.1,1.,0,0],[0,0,3.,.2],[0,0,.2,2.]]],dtype=torch.double)
    scores = [independent_grid_score(theta*direction+a*spread*direction,
                                    theta*direction+c*spread*direction,target,precision).sum()
              for a,c in itertools.product([-1,1],repeat=2)]
    actual = torch.stack(scores).mean()
    expected = independent_grid_score(theta*direction,theta*direction,target,precision).sum()
    torch.testing.assert_close(actual,expected)
    grad_theta,grad_spread = torch.autograd.grad(actual,(theta,spread),retain_graph=True)
    torch.testing.assert_close(grad_theta,torch.autograd.grad(expected,theta)[0])
    torch.testing.assert_close(grad_spread,torch.zeros_like(grad_spread),atol=1e-12,rtol=0)


def test_preparation_and_one_epoch_preserve_pairing_targets_and_selection(tmp_path,monkeypatch):
    from scripts import train_flow_grid_response as script
    from sbsi.measurement_model import ConditionalAffineFlow, TargetStandardizer, save_measurement_model
    from sbsi.selection_model import TabularPreprocessor
    from sbsi.training import split_group_values
    rng = np.random.default_rng(82)
    features = ['e1_input_p','e2_input_p','r_input_p','Re_input_p','r_blend']
    targets = ['measured_ngmix_g1','measured_ngmix_g2','measured_mag_auto','measured_log_flux_radius']
    rows = []
    for case in range(40):
        response = np.eye(2)+rng.normal(0,.1,(2,2))
        for i in range(5):
            angle = rng.uniform(-np.pi,np.pi)
            gamma = .05*np.array([np.cos(angle),np.sin(angle)])
            base = dict(zip(features,[.1,.2,23.,.8,.4]))
            base.update(_ratio_base_e1=.1,_ratio_base_e2=.2)
            y = rng.normal(size=4)
            for leg in range(2):
                if i == 4 and leg == 1:
                    continue
                row = dict(base,case=case,input_index=i,__catalogue_index=leg,
                           _paired_shear=.05*leg,_grid_gamma1=gamma[0]*leg,_grid_gamma2=gamma[1]*leg)
                out = y.copy()
                out[:2] += leg*(response@gamma)
                row.update(zip(targets,out))
                row['e1_input_p'] += gamma[0]*leg
                row['e2_input_p'] += gamma[1]*leg
                rows.append(row)
    frame = pd.DataFrame(rows)
    condition = TabularPreprocessor.fit(frame,features,add_missing_indicators=True)
    transform = TargetStandardizer.fit(frame,targets)
    config = dict(flow_type='affine',target_dim=4,context_dim=condition.output_dim,
                  hidden_dim=8,n_layers=1,n_flows=2)
    model = ConditionalAffineFlow(**{k:v for k,v in config.items() if k!='flow_type'})
    parent = tmp_path/'parent.pt'
    save_measurement_model(parent,model,condition,transform,config)
    tr,va = split_group_values(range(40),501,.2)
    grid = tmp_path/'grid.npz'
    np.savez(grid,edges_flux=[18.,26.],edges_size=[.37,1.5],edges_crowd=[0.,1.],train_cases=tr,validation_cases=va)
    for name in ('g0','g1'):
        (tmp_path/name).write_text('test input')
    monkeypatch.setattr(script,'load_measurement_data',lambda *a,**k:frame.copy())
    cache = tmp_path/'cache'
    script.main(['prepare','--root',str(cache),'--parent',str(parent),'--grid',str(grid),
                 '--g0',str(tmp_path/'g0'),'--g1',str(tmp_path/'g1')])
    manifest = json.loads((cache/'manifest.json').read_text())
    assert manifest['splits']['train']['unmatched_source'] == 32
    assert manifest['splits']['validation']['unmatched_source'] == 8
    payload = script.load_torch(cache/'train.pt')
    assert len(payload['values']) == 32*9
    assert len(payload['pairs']) == 32*4
    assert not set(payload['pair_keys'][:,0]) & set(va)
    population = GridResponsePopulation(payload['context'],payload,transform.scales[:2])
    before = [p.clone().detach() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
    score = population.sampled_score(model,1,16,4)
    score.backward()
    optimizer.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
    first = population.evaluate(model,4,batch_size=64,seed=12)
    second = population.evaluate(model,4,batch_size=64,seed=12)
    assert first['loss'] == second['loss']
    assert not np.array_equal(first['mean_a'],first['mean_c'])
    output = tmp_path/'trial.pt'
    script.main(['train','--root',str(cache),'--output',str(output),'--weight','1',
                 '--grid-loss-reduction','mean',
                 '--device','cpu','--epochs','1','--batch-size','128','--cells-per-step','1',
                 '--pairs-per-cell','8','--draws','2','--validation-draws','2'])
    history = json.loads(output.with_suffix('.history.json').read_text())
    assert len(history['history']) == 2  # parent baseline and completed epoch
    for row in history['history']:
        assert row['val_total'] == pytest.approx(row['val_nll']+row['val_grid'])
    checkpoint = script.load_torch(output)
    assert checkpoint['model_config']['flow_type'] == 'affine'
    assert checkpoint['metadata']['best_val_objective'] == min(r['val_total'] for r in history['history'])
    assert checkpoint['metadata']['grid_response_weight'] == 1
    assert checkpoint['metadata']['grid_loss_divisor'] == 4  # one cell, four components
    for row in history['history']:
        audit = np.load(tmp_path/f'trial_epoch{row["epoch"]:03d}_validation.npz')
        assert float(audit['loss_sum']) == pytest.approx(float(audit['per_cell'].sum()))
        assert row['val_grid'] == pytest.approx(float(audit['loss_sum'])/4)
    assert checkpoint['metadata']['coupling_weight'] == 0
    assert output.with_suffix('.resume.pt').exists()
