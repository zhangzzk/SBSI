import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from sbsi.flow_grid_caps import validation_pair_subset
from sbsi.flow_grid_response import fit_grid_response, independent_grid_score
from sbsi.flow_grid_accumulation import response_plan,sampled_response_backward,accumulated_epoch
from test_flow_grid_accumulation import fixture,prepared_fixture


def test_uneven_response_budget_has_exact_direct_equal_cell_gradient():
    model,context,pop=fixture()
    cells,local,groups=response_plan(pop,3,7,42,pairs_per_group=19)
    counts=torch.bincount(local).double()
    assert sorted(counts.tolist())==[6,6,7]
    means=[]
    for stream in range(2):
        total=torch.zeros((3,4),dtype=torch.float64)
        for block,start in enumerate(range(0,19,5)):
            v=pop.contributions(model,groups[stream][start:start+5],4,seed=42+100000+2*block+stream)
            total=total.index_add(0,local[start:start+5],v)
        means.append(total/counts[:,None])
    loss=independent_grid_score(*means,pop.target[cells],pop.precision[cells]).mean()/4
    loss.backward();expected=[p.grad.clone() for p in model.parameters()];model.zero_grad()
    result=sampled_response_backward(model,pop,3,7,4,chunk_size=5,seed=42,weight=1.,pairs_per_group=19)
    assert result['pairs_per_group']==19
    assert result['loss']==pytest.approx(float(loss.detach()))
    for p,g in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=1e-5,atol=2e-7)


def test_capped_epoch_unique_nll_rows_redraw_and_exact_pair_budget():
    model,context,pop=fixture();values=torch.randn(36,4);values[:,0]=torch.arange(36)/36
    original=model.log_prob;seen=[]
    def capture(values,context):
        seen.extend(torch.round(values[:,0]*36).long().tolist());return original(values,context)
    model.log_prob=capture;opt=torch.optim.SGD(model.parameters(),lr=1e-4)
    options=dict(weight=.1,mode='sampled',nll_batch_size=13,nll_chunk_size=5,cells_per_step=3,pairs_per_cell=7,draws=2,response_chunk_size=5,max_nll_rows=23,response_pairs_per_epoch=19)
    a=accumulated_epoch(model,values,context,pop,opt,seed=42,**options);first=seen.copy();seen.clear()
    b=accumulated_epoch(model,values,context,pop,opt,seed=43,**options)
    assert len(first)==len(set(first))==23 and len(seen)==len(set(seen))==23
    assert set(first)!=set(seen)
    for result in [a,b]:assert result['nll_rows']==23 and result['response_pairs_per_group']==19 and result['optimizer_steps']==2


def test_validation_subset_refits_same_raw_pairs_preserves_precision_and_is_fixed():
    rng=np.random.default_rng(31);n=2000
    cells=np.repeat([0,1],n//2);cases=np.tile(np.arange(20),n//20)
    gamma=rng.normal(size=(n,2))*.02;delta=.8*gamma+rng.normal(size=(n,2))*.01
    shapes=np.zeros((n*2,2));shapes[1::2]=delta;pairs=np.arange(2*n).reshape(n,2)
    stats=fit_grid_response(delta,gamma,cells,cases,2)
    payload=dict(pairs=pairs,gamma=gamma,bin_ids=cells,pair_keys=np.column_stack([cases,np.arange(n)]),**{k:stats[k] for k in ['counts','target','design','precision']})
    before=copy.deepcopy(payload);result,report=validation_pair_subset(payload,shapes,1000,123)
    again,report2=validation_pair_subset(payload,shapes,1000,123)
    idx=report['selected_pair_indices'];assert len(np.unique(idx))==1000
    expected=fit_grid_response(delta[idx],gamma[idx],cells[idx],cases[idx],2)
    for k in ['target','design','counts']:np.testing.assert_array_equal(result[k],expected[k])
    np.testing.assert_array_equal(result['precision'],before['precision'])
    for k in payload:np.testing.assert_array_equal(payload[k],before[k])
    for k in report:np.testing.assert_array_equal(report[k],report2[k])


def test_capped_trainer_resume_preserves_subset_and_exact_optimizer(tmp_path):
    from scripts.train_flow_grid_accumulated import main
    from scripts.train_flow_grid_response import load_torch
    root=tmp_path/'prepared';prepared_fixture(root)
    opts=['--root',str(root),'--weight','.5','--mode','sampled','--device','cpu','--disable-nll-guard',
          '--nll-batch-size','13','--nll-chunk-size','5','--cells-per-step','3','--pairs-per-cell','7',
          '--draws','2','--response-chunk-size','5','--validation-draws','2','--validation-batch-size','5',
          '--max-train-nll-rows','23','--response-pairs-per-epoch','19','--validation-nll-rows','17']
    a=tmp_path/'a.pt';b=tmp_path/'b.pt'
    main(opts+['--output',str(a),'--epochs','1']);main(opts+['--output',str(a),'--epochs','2','--resume'])
    main(opts+['--output',str(b),'--epochs','2'])
    x,y=load_torch(a.with_suffix('.resume.pt')),load_torch(b.with_suffix('.resume.pt'))
    for k in x['state_dict']:torch.testing.assert_close(x['state_dict'][k],y['state_dict'][k],rtol=0,atol=0)
    for key,value in x['optimizer']['state'].items():
        for k,v in value.items():torch.testing.assert_close(v,y['optimizer']['state'][key][k],rtol=0,atol=0)
    with np.load(a.with_suffix('.validation_subset.npz')) as f,np.load(b.with_suffix('.validation_subset.npz')) as g:
        np.testing.assert_array_equal(f['selected_nll_indices'],g['selected_nll_indices'])
    assert x['history'][-1]['nll_rows']==23 and x['history'][-1]['response_pairs_per_group']==19
