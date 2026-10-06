import numpy as np
import pytest
import torch

from sbsi.flow_grid_response import GridResponsePopulation,independent_grid_score
from sbsi.flow_grid_full_cells import full_cell_response,epoch_full_cells,_cell_mean
from sbsi.measurement_model import ConditionalAffineFlow


def fixture():
    torch.manual_seed(14)
    model=ConditionalAffineFlow(target_dim=4,context_dim=2,hidden_dim=8,n_layers=1,n_flows=2)
    with torch.no_grad():
        for layer in model.layers:
            layer.net[-1].weight.normal_(std=.03)
    context=torch.randn(36,2)
    counts=np.array([3,7,8])
    payload=dict(pairs=np.arange(36).reshape(18,2),counts=counts,
                 bin_ids=np.repeat(np.arange(3),counts),
                 design=np.random.default_rng(55).normal(size=(18,2)),
                 target=np.random.default_rng(41).normal(size=(3,4)),
                 precision=np.tile(np.eye(4),(3,1,1)))
    return model,context,GridResponsePopulation(context,payload,[.3,.4])


def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def test_replayed_complete_cell_gradient_matches_direct_graph_with_unequal_cells_and_nll():
    model,context,pop=fixture()
    cells=[2,0]
    means=[torch.stack([_cell_mean(model,pop,b,np.diff(pop.offsets)[b],4,2,81,s)
                         for b in cells]) for s in range(2)]
    score=independent_grid_score(*means,pop.target[cells],pop.precision[cells]).mean()/4
    values=torch.randn(36,4)
    nll=-model.log_prob(values,context).mean()
    (nll+3*score).backward()
    direct=grads(model).clone()
    model.zero_grad(set_to_none=True)
    (-model.log_prob(values,context).mean()).backward()
    result=full_cell_response(model,pop,cells,4,chunk_size=2,seed=81,weight=3,backward=True)
    assert result['loss']==pytest.approx(float(score.detach()))
    assert result['pairs_per_group']==11
    torch.testing.assert_close(grads(model),direct,rtol=1e-5,atol=2e-7)
    assert result['mean_a'].requires_grad is False
    assert not torch.equal(result['mean_a'],result['mean_c'])


def test_epoch_covers_every_nll_row_and_pair_and_updates_only_after_complete_cells(monkeypatch):
    model,context,pop=fixture()
    values=torch.randn(36,4)
    pair_calls=[]
    original=pop.contributions
    def record(model,indices,draws,seed=None):
        pair_calls.append(indices.cpu().numpy())
        return original(model,indices,draws,seed)
    monkeypatch.setattr(pop,'contributions',record)
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)
    before=[p.detach().clone() for p in model.parameters()]
    result=epoch_full_cells(model,values,context,pop,optimizer,weight=1.,cells_per_step=2,
                            draws=2,response_chunk_size=2,nll_chunk_size=5,seed=90)
    assert result['nll_rows']==36
    assert result['response_pairs_per_group']==18
    assert result['cells']==3
    assert result['optimizer_steps']==2
    # Two independent latent groups, each first evaluated and then replayed.
    np.testing.assert_array_equal(np.bincount(np.concatenate(pair_calls),minlength=18),np.full(18,4))
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))


def test_cell_validation_and_no_gradient_side_effect_without_backward():
    model,context,pop=fixture()
    with pytest.raises(ValueError,match='distinct valid'):
        full_cell_response(model,pop,[0,0],4)
    with pytest.raises(ValueError,match='distinct valid'):
        full_cell_response(model,pop,[0],3)
    result=full_cell_response(model,pop,[0,1,2],4,chunk_size=3)
    assert all(p.grad is None for p in model.parameters())
    assert result['pairs_per_group']==18


def test_replay_rejects_stochastic_training_layers():
    model,context,pop=fixture()
    model.add_module('unexpected_dropout',torch.nn.Dropout(.2))
    with pytest.raises(ValueError,match='deterministic'):
        full_cell_response(model,pop,[0],4)
