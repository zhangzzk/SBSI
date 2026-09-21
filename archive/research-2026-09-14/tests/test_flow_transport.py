import pytest
import torch
import numpy as np
import pandas as pd

from sbsi.flow_transport import (
    paired_flow_transport, matched_transport_indices, PairedTransportBatches,
    paired_transport_loss,
)
from sbsi.measurement_model import ConditionalAffineFlow, ConditionalMeanFlow
from sbsi.training import GPUBatches


@pytest.mark.parametrize("with_mean", [False, True])
def test_paired_transport_identity_reversibility_and_gradients(with_mean):
    torch.manual_seed(7)
    cls = ConditionalMeanFlow if with_mean else ConditionalAffineFlow
    kw = dict(target_dim=4, context_dim=3, hidden_dim=12, n_layers=1, n_flows=2)
    if with_mean:
        kw.update(mean_hidden=8, flow_drop_indices=[0])
    model = cls(**kw).double()
    # Nonidentity conditional coupling, including context-dependent scales.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.03 * torch.randn_like(parameter))
    x = torch.randn(9, 4, dtype=torch.double)
    c0 = torch.randn(9, 3, dtype=torch.double)
    c1 = c0 + .1 * torch.randn_like(c0)
    same = paired_flow_transport(model, x, c0, c0)
    moved = paired_flow_transport(model, x, c0, c1)
    back = paired_flow_transport(model, moved, c1, c0)
    torch.testing.assert_close(same, x, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(back, x, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(moved, x)
    (moved-x).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())


def test_paired_transport_rejects_misaligned_context():
    model = ConditionalAffineFlow(target_dim=4, context_dim=3, hidden_dim=8, n_flows=2)
    with pytest.raises(ValueError, match="align"):
        paired_flow_transport(model, torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(1, 3))


def test_pairing_uses_case_and_identity_and_retains_unmatched_nll_rows():
    f = pd.DataFrame({"case":[2,1,1,2,1], "input_index":[5,5,5,5,9],
                      "__catalogue_index":[1,0,1,0,0]},index=[9,8,7,6,5])
    pairs, report = matched_transport_indices(f)
    assert set(map(tuple,pairs)) == {(1,2),(3,0)}
    assert report["unmatched_source"] == 1
    assert report["unmatched_destination"] == 0
    x, c = torch.arange(20).reshape(5,4).float(), torch.arange(15).reshape(5,3).float()
    source = GPUBatches([x,c],2,False,"cpu")
    batch = next(iter(PairedTransportBatches(source,pairs,False)))
    torch.testing.assert_close(batch[0],x[pairs[:,0]])
    torch.testing.assert_close(batch[2],x[pairs[:,1]])
    assert source.n == 5
    with pytest.raises(pd.errors.MergeError):
        matched_transport_indices(pd.concat([f,f.iloc[[0]]]))


def test_paired_loss_recovers_known_response_without_fitting_a_calibration_factor():
    model = ConditionalMeanFlow(target_dim=4,context_dim=1,hidden_dim=8,n_flows=2,
                                mean_hidden=0,flow_drop_indices=[0]).double()
    # A plain translation is the exactly solvable special case.
    with torch.no_grad():
        model.mean_net.weight.zero_()
        model.mean_net.bias.zero_()
    x0=torch.randn(32,4,dtype=torch.double)
    c0=torch.zeros(32,1,dtype=torch.double)
    c1=torch.full((32,1),.05,dtype=torch.double)
    slope=torch.tensor([.9,.1,-.03,.2],dtype=torch.double)
    x1=x0+.05*slope
    optimizer=torch.optim.SGD(model.mean_net.parameters(),lr=.5)
    first=float(paired_transport_loss(model,(x0,c0,x1,c1),.05).detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss=paired_transport_loss(model,(x0,c0,x1,c1),.05)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < first*1e-8
    np.testing.assert_allclose(model.mean_net.weight.detach().numpy()[:,0],slope.numpy(),atol=2e-5)


def test_warm_start_ignores_training_annotations_but_not_architecture():
    from sbsi.flow_training import _architecture_config
    config={"flow_type":"affine","n_flows":10}
    annotated={**config,"response_difference":"central","response_error":"absolute",
               "response_components":"matrix"}
    assert _architecture_config(annotated) == config
    assert "response_difference" in annotated
    assert _architecture_config({**annotated,"n_flows":9}) != config


def test_response_epoch_includes_pair_loss_and_validates_each_pair_once():
    from sbsi.flow_training import epoch_response
    torch.manual_seed(51)
    model=ConditionalAffineFlow(target_dim=4,context_dim=2,hidden_dim=8,n_flows=2)
    x=torch.randn(4,4); c=torch.zeros(4,2)
    cp=c+.05; cm=c-.05
    batch=(x,c,torch.ones(4),c,cp,cp,cm,cm,torch.zeros(4,dtype=torch.long))
    pairs=[(x,c,x+.05,cp),(x[:1],c[:1],x[:1]+.1,cp[:1])]
    # Validation must average the two pair batches by rows, without cycling
    # the first pair batch when the NLL loader is longer.
    expected=(4*paired_transport_loss(model,pairs[0],.05)+paired_transport_loss(model,pairs[1],.05))/5
    args=(model,[batch]*3,torch.device('cpu'),[1.,1.],.05,torch.zeros((1,4)),0.)
    result=epoch_response(*args,response_difference='central',response_components='matrix',
                          response_estimator='complete_flow',response_draws=2,
                          paired_loader=pairs,paired_weight=1.)
    assert result[-1]['paired_transport'] == pytest.approx(float(expected.detach()))
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)
    before=[p.detach().clone() for p in model.parameters()]
    epoch_response(*args,response_difference='central',response_components='matrix',
                   response_estimator='complete_flow',response_draws=2,nll_weight=0.,
                   paired_loader=pairs,paired_weight=1.,optimizer=optimizer)
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
