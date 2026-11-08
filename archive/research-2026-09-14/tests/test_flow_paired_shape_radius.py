"""Physical radius, unchanged shape weighting, and selection of their sum."""
import copy
import itertools
import json
import numpy as np
import pytest
import torch

from sbsi.flow_paired_shape import (PairedResponsePopulation, PairedShapePopulation,
    paired_cross_score, paired_shape_backward, summarize_paired_residuals)
from scripts.continue_circularized_flow import continue_fit
from scripts.train_circularized_grid_free import paired_shape_epoch
from scripts.trial_circularized_flow import fit_nll_arm
from test_flow_paired_shape import fixture
from test_flow_physical_disk import config


class Physical(torch.nn.Module):
    target_dim, context_dim = 4, 2
    def __init__(self, radius_unit=1.):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor([.1, .2, 1., 200.], dtype=torch.float64))
        self.radius_unit = radius_unit

    def forward_physical(self, z, context):
        unit = self.offset.new_tensor([1., 1., self.radius_unit, 1.])
        return (context[:, :1]*self.offset + z)*unit, z.new_zeros(len(z))


def population(radius_unit=1.):
    context = torch.arange(8., dtype=torch.float64).reshape(4,2)/2
    delta = np.tile([.05, .1, .25*radius_unit], (2,1))
    return PairedResponsePopulation(context, [[0,1],[2,3]], delta,
                                   np.tile([.03,.04], (2,1)), radius_scale=2*radius_unit)


def test_physical_radius_scale_and_flux_exclusion():
    model, pop = Physical(), population()
    r = pop.residual(model, torch.arange(2), 4, 3)
    torch.testing.assert_close(r, torch.tensor([[1.,2.,7.5],[1.,2.,7.5]], dtype=torch.float64))
    (10*paired_cross_score(r,r).mean()).backward()
    torch.testing.assert_close(model.offset.grad[2], torch.tensor(375., dtype=torch.float64))
    assert model.offset.grad[3] == 0


def test_radius_unit_conversion_preserves_loss_and_gradient():
    pixel, arcsec = Physical(), Physical(radius_unit=.2)
    scores = []
    for model, pop in [(pixel,population()),(arcsec,population(.2))]:
        r = pop.residual(model, torch.arange(2), 4, 3)
        score = paired_cross_score(r,r).mean()
        score.backward(); scores.append(score)
    torch.testing.assert_close(*scores)
    torch.testing.assert_close(pixel.offset.grad, arcsec.offset.grad)


def test_radius_requires_declared_valid_scale_and_delta():
    pop = population()
    for scale in [0., -1., np.inf, np.nan]:
        with pytest.raises(ValueError, match="radius scale"):
            PairedResponsePopulation(pop.context, [[0,1]], [[0.,0.,0.]], [[.05,0.]], radius_scale=scale)
    with pytest.raises(ValueError, match="aligned"):
        PairedShapePopulation(pop.context, [[0,1]], [[0.,0.,0.]], [[.05,0.]])


def test_radius_cross_product_has_no_mc_variance_penalty_and_keeps_shape_coefficient():
    mean = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    scatter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    scores = []
    for x,y in itertools.product((-1.,1.),repeat=2):
        a = torch.stack((mean*0+1,mean*0+2,mean+x*scatter))[None]
        c = torch.stack((mean*0+1,mean*0+2,mean+y*scatter))[None]
        scores.append(paired_cross_score(a,c))
    loss = torch.stack(scores).mean()
    torch.testing.assert_close(loss, (5+mean.square())/4)
    gm, gs = torch.autograd.grad(loss,(mean,scatter))
    torch.testing.assert_close(gm,mean/2); torch.testing.assert_close(gs,torch.zeros_like(gs))
    a = torch.tensor([[1.,2.],[-1.,-2.]])
    torch.testing.assert_close(paired_cross_score(a,a),paired_cross_score(
        torch.cat((a,a[:,:1]*0),1),torch.cat((a,a[:,:1]*0),1)))


def test_shape_radius_chunked_gradient_and_component_accounting():
    model, pop = Physical(), population()
    direct = copy.deepcopy(model)
    seed, count, chunk = 71, 7, 3
    rows = torch.as_tensor(np.random.default_rng(seed).integers(len(pop.pairs),size=count))
    total = 0.
    for block,start in enumerate(range(0,count,chunk)):
        ids = rows[start:start+chunk]
        total += paired_cross_score(pop.residual(direct,ids,4,seed+100000+2*block),
                                    pop.residual(direct,ids,4,seed+100001+2*block)).sum()/count
    (10*total).backward()
    for p in model.parameters(): p.grad = torch.ones_like(p)
    result = paired_shape_backward(model,pop,count,4,chunk_size=chunk,seed=seed,weight=10.)
    torch.testing.assert_close(model.offset.grad, direct.offset.grad+1)
    np.testing.assert_allclose(result['component_losses'].sum(),result['loss'])


def test_validation_reports_true_shape_radius_sum_and_mc_uncertainty():
    values = np.random.default_rng(19).normal(size=(4,7,3))
    result = summarize_paired_residuals(values)
    expected = np.mean([np.sum(a*c,axis=-1)/4 for i,a in enumerate(values)
                        for j,c in enumerate(values) if i!=j],axis=0)
    np.testing.assert_allclose(result['per_pair'],expected)
    np.testing.assert_allclose(result['response_loss'],result['shape_loss']+result['radius_loss'])
    shape = summarize_paired_residuals(values[:,:,:2])
    np.testing.assert_allclose(result['shape_loss'],shape['shape_loss'])
    np.testing.assert_allclose(result['shape_mc_jackknife_se'],shape['mc_jackknife_se'])


def test_checkpoint_selection_uses_shape_plus_radius(tmp_path):
    _, values, shape_pop, pre, target = fixture()
    pop = PairedResponsePopulation(shape_pop.context,shape_pop.pairs.cpu().numpy(),
        np.column_stack((shape_pop.measured_delta.cpu().numpy(),np.ones(12)*.01)),
        np.tile([.03,.04],(12,1)),radius_scale=1.)
    train=(values[:18],pop.context[:18]); val=(values[18:],pop.context[18:])
    first = dict(seed=501,learning_rate=1e-4,epochs=1,rows_per_epoch=13,batch_size=6,early_stopping_patience=20)
    fit_nll_arm(tmp_path/'parent','parent',config(),pre,target,train,val,first,{})
    initial = torch.load(tmp_path/'parent/optimizer.pt',weights_only=False)
    parent = torch.load(tmp_path/'parent/epoch001.pt',weights_only=False)
    recipe=first|dict(max_epoch=3,min_lr=1e-7,diagnostic_interval=1,pairs_per_step=5,
        training_draws=4,response_training_chunk_size=3,training_response_seed=12000,
        response_weight=10.,response_loss_key='response_loss')
    def diagnostic(model,epoch):
        return dict(response_loss=4.-epoch,shape_loss=.1*epoch,radius_loss=4.-1.1*epoch)
    continue_fit(tmp_path/'run','paired',parent,initial,None,train,val,recipe,{},diagnostic,pop,
                 train_epoch=paired_shape_epoch)
    h=json.loads((tmp_path/'run/history.json').read_text())
    assert h['best_epoch']==3
    for r in h['history']:
        assert r['selection_metric']==r['validation']['nll']+10*r['response']['response_loss']
    assert 'shape+radius' in json.loads((tmp_path/'run/selection.json').read_text())['selection']
