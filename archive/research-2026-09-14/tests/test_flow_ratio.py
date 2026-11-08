"""Analytic ratio identifiability, balanced labels, and training integration."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.flow_ratio import (
    paired_ratio_losses, epoch_nll_ratio, validate_ratio_options, verify_ratio_base_properties,
)


class BinaryDensity(torch.nn.Module):
    def __init__(self, p0=.25, p1=.75):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.logit(torch.tensor([p0, p1], dtype=torch.double)))

    def log_prob(self, x, c):
        logit = self.logits[c[:, 0].long()]
        return -torch.nn.functional.binary_cross_entropy_with_logits(logit, x[:, 0], reduction='none')


def population_pairs():
    # Each leg has four measurements: p0=1/4, p1=3/4. Same base U for both labels.
    x0 = torch.tensor([[0.], [0.], [0.], [1.]], dtype=torch.double)
    x1 = torch.tensor([[0.], [1.], [1.], [1.]], dtype=torch.double)
    c0, c1 = torch.zeros_like(x0), torch.ones_like(x0)
    return x0, c0, x1, c1


def test_population_optimum_balanced_bce_and_no_rng_or_shear_scaling():
    model = BinaryDensity()
    pair = population_pairs()
    rng = torch.random.get_rng_state().clone()
    score = paired_ratio_losses(model, pair).mean()
    # Bayes balanced BCE is H(1/4), not BCE with a logit divided by 0.05.
    assert float(score.detach()) == pytest.approx(-.25*math.log(.25)-.75*math.log(.75))
    grad = torch.autograd.grad(score, model.logits)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    for a, b in [(.25,.25),(.75,.25),(.1,.9)]:
        assert float(paired_ratio_losses(BinaryDensity(a,b),pair).mean().detach()) > float(score.detach())


def test_leg_reversal_and_same_context_jacobian_cancel():
    model = BinaryDensity()
    x0,c0,x1,c1 = population_pairs()
    original = paired_ratio_losses(model,(x0,c0,x1,c1))
    torch.testing.assert_close(original,paired_ratio_losses(model,(x1,c1,x0,c0)))

    class WithJacobian:
        def log_prob(self,x,c):
            return model.log_prob(x,c)+2*x[:,0]+3
    torch.testing.assert_close(original,paired_ratio_losses(WithJacobian(),(x0,c0,x1,c1)))
    same = paired_ratio_losses(model,(x0,c0,x1,c0))
    torch.testing.assert_close(same,torch.full_like(same,math.log(2)))


def test_epoch_pair_denominator_unmatched_nll_rows_and_optimizer():
    model = BinaryDensity(.4,.6)
    pair = population_pairs()
    pairs = [tuple(v[:3] for v in pair),tuple(v[3:] for v in pair)]
    # NLL includes an additional unmatched row, with unequal batches/weights.
    x = torch.cat((pair[0],pair[2],torch.ones(1,1,dtype=torch.double)))
    c = torch.cat((pair[1],pair[3],torch.zeros(1,1,dtype=torch.double)))
    w = torch.arange(1,10,dtype=torch.double)
    loader = [(x[:7],c[:7],w[:7]),(x[7:],c[7:],w[7:])]
    expected_nll = float((-(w*model.log_prob(x,c)).sum()/w.sum()).detach())
    expected_bce = float(paired_ratio_losses(model,pair).mean().detach())
    nll,bce = epoch_nll_ratio(model,loader,pairs,'cpu',1.)
    assert nll == pytest.approx(expected_nll)
    assert bce == pytest.approx(expected_bce)
    # Cycle a shorter pair loader; the combined objective must drive parameters.
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    epoch_nll_ratio(model,loader,[pair],'cpu',1.,optimizer=optimizer)
    final_nll,final_bce = epoch_nll_ratio(model,loader,pairs,'cpu',1.)
    assert final_nll+final_bce < nll+bce


def test_ratio_gradients_reach_real_conditional_flow_without_sampling():
    from sbsi.measurement_model import ConditionalAffineFlow
    model = ConditionalAffineFlow(target_dim=4,context_dim=2,hidden_dim=8,n_layers=1,n_flows=2).double()
    x0,x1 = torch.randn(7,4,dtype=torch.double),torch.randn(7,4,dtype=torch.double)
    c0,c1 = torch.zeros(7,2,dtype=torch.double),torch.ones(7,2,dtype=torch.double)*.05
    model.forward = lambda *args,**kwargs: pytest.fail('ratio loss must not sample')
    loss = paired_ratio_losses(model,(x0,c0,x1,c1)).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.parameters())


def ratio_args(*extra):
    from sbsi.flow_training import parse_args
    return parse_args(['--catalogue','zero.feather','--additional-catalogue','one.feather',
                      '--output','model.pt','--fold-catalogue-shear',
                      '--validation-group-column','case','--ratio-response-weight','1',*extra])


def test_ratio_cli_has_no_grid_requirement_and_rejects_competing_regularizers():
    validate_ratio_options(ratio_args())
    for option in ['--response-weight','--coupling-weight','--distribution-response-weight',
                   '--paired-transport-weight','--response-dispersion-weight','--preserve-weight']:
        with pytest.raises(ValueError,match='single derivative'):
            validate_ratio_options(ratio_args(option,'1'))
    with pytest.raises(ValueError,match='NLL weight 1'):
        validate_ratio_options(ratio_args('--nll-weight','0'))
    with pytest.raises(ValueError,match='finite'):
        validate_ratio_options(ratio_args('--ratio-response-weight','nan'))


def test_ratio_base_u_check_allows_only_folded_shape_motion():
    frame = pd.DataFrame({'e1_input_p':[.1,.15], 'Re_input_p':[.5,.5],
                          '_ratio_base_e1':[.1,.1],'_ratio_base_e2':[.2,.2]})
    pairs=np.array([[0,1]])
    verify_ratio_base_properties(frame,pairs,['e1_input_p','Re_input_p'])
    for column in ['Re_input_p','_ratio_base_e1']:
        bad=frame.copy();bad.loc[1,column]+=.01
        with pytest.raises(ValueError,match='base-U'):
            verify_ratio_base_properties(bad,pairs,['e1_input_p','Re_input_p'])


def test_ratio_cli_trains_and_selects_nll_plus_bce_without_grid_or_sampling(tmp_path, monkeypatch):
    from sbsi import flow_training as training
    rng = np.random.default_rng(501)

    def fake_load(args, conditions, targets, **kwargs):
        rows=[]
        for case in range(5):
            for i in range(4):
                base={name:float(rng.uniform(.2,.8)) for name in conditions}
                base.update(_ratio_base_e1=base['e1_input_p'],_ratio_base_e2=base['e2_input_p'])
                for leg in (0,1):
                    if i==3 and leg==1:
                        continue  # retain an unmatched NLL row in each case
                    row={**base,**dict(zip(targets,rng.normal(size=4))),
                         'case':case,'input_index':i,'__catalogue_index':leg,
                         '_paired_shear':.05*leg,'detected':True}
                    row['e1_input_p']+=.05*leg
                    rows.append(row)
        return pd.DataFrame(rows)

    monkeypatch.setattr(training,'load_measurement_data',fake_load)
    monkeypatch.setattr(training,'complete_flow_context_means',lambda *a,**k:pytest.fail('sampled'))
    monkeypatch.setattr(training,'epoch_response',lambda *a,**k:pytest.fail('old response loss'))
    original=training.epoch_nll_ratio
    def checked_epoch(model,loader,*args,**kwargs):
        tensors=loader.dataset.tensors
        assert len(tensors)==2 or torch.all(tensors[2]==1)  # pairs must not downweight NLL
        return original(model,loader,*args,**kwargs)
    monkeypatch.setattr(training,'epoch_nll_ratio',checked_epoch)
    output=tmp_path/'ratio_swabase.pt'
    training.main(['--catalogue','zero','--additional-catalogue','one','--output',str(output),
                   '--fold-catalogue-shear','--validation-group-column','case',
                   '--ratio-response-weight','1','--device','cpu','--epochs','1',
                   '--batch-size','8','--hidden-dim','8','--n-flows','2','--condition-layers','1',
                   '--target-features','measured_ngmix_g1','measured_ngmix_g2',
                   'measured_mag_auto','measured_log_flux_radius',
                   '--feature-set','g0_meas_crowd_conc_szfl_noz','--num-workers','0'])
    metadata=torch.load(output,weights_only=False)['metadata']
    assert metadata['response_weight']==metadata['coupling_weight']==metadata['distribution_response_weight']==0
    assert metadata['ratio_response_weight']==1
    hist=metadata['history']
    assert metadata['best_val_objective']==pytest.approx(hist['val_nll'][0]+hist['val_ratio_response'][0])
    assert sum(p['unmatched_source'] for p in metadata['paired_transport_populations'].values())==5
