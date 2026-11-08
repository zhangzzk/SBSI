import copy
import json

import numpy as np
import pytest
import torch

from sbsi.flow_grid_accumulation import (
    response_plan,sampled_response_backward,accumulated_epoch,summarize_validation_means,
    evaluate_validation_groups,
)
from sbsi.flow_grid_response import GridResponsePopulation,independent_grid_score
from sbsi.measurement_model import ConditionalAffineFlow


def fixture():
    torch.manual_seed(732)
    model = ConditionalAffineFlow(target_dim=4,context_dim=2,hidden_dim=8,n_layers=1,n_flows=2)
    with torch.no_grad():
        for layer in model.layers:
            layer.net[-1].weight.normal_(std=.03)
    context = torch.randn(36,2)
    counts = np.array([3,7,8])
    payload = dict(pairs=np.arange(36).reshape(18,2),counts=counts,
                   bin_ids=np.repeat(np.arange(3),counts),
                   design=np.random.default_rng(91).normal(size=(18,2)),
                   target=np.random.default_rng(92).normal(size=(3,4)),
                   precision=np.tile(np.eye(4),(3,1,1)))
    pop = GridResponsePopulation(context,payload,[.3,.4])
    return model,context,pop


def test_sampled_replay_gradient_matches_direct_graph_and_preserves_nll_gradient():
    model,context,pop = fixture()
    # A mildly asymmetric P also checks the transpose in the C coefficient.
    pop.precision[:,0,1] = .01
    cells,local,groups = response_plan(pop,2,7,42)
    means = []
    for stream,rows in enumerate(groups):
        total = torch.zeros((2,4),dtype=torch.float64)
        for block,start in enumerate(range(0,len(rows),5)):
            values = pop.contributions(model,rows[start:start+5],4,seed=42+100000+2*block+stream)
            total = total.index_add(0,local[start:start+5],values)
        means.append(total/7)
    score = independent_grid_score(*means,pop.target[cells],pop.precision[cells]).mean()/4
    values = torch.randn(36,4)
    (-model.log_prob(values,context).mean()+3*score).backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    (-model.log_prob(values,context).mean()).backward()
    actual = sampled_response_backward(model,pop,2,7,4,chunk_size=5,seed=42,weight=3.)
    assert actual['loss'] == pytest.approx(float(score.detach()))
    for p,g in zip(model.parameters(),expected):
        torch.testing.assert_close(p.grad,g,rtol=1e-5,atol=2e-7)
    assert not torch.equal(groups[0],groups[1])
    assert not torch.equal(actual['mean_a'],actual['mean_c'])


def test_accumulated_nll_matches_macro_batches_and_visits_all_rows_once(monkeypatch):
    model,context,pop = fixture()
    direct = copy.deepcopy(model)
    values = torch.randn(36,4)
    values[:,0] = torch.arange(36)/36
    seen = []
    original = model.log_prob
    def record(values,context):
        seen.extend(torch.round(values[:,0]*36).long().tolist())
        return original(values,context)
    monkeypatch.setattr(model,'log_prob',record)
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    expected_optimizer = torch.optim.SGD(direct.parameters(),lr=1e-3)
    generator = torch.Generator().manual_seed(92)
    order = torch.randperm(36,generator=generator)
    for start in range(0,36,13):
        rows = order[start:start+13]
        expected_optimizer.zero_grad(set_to_none=True)
        (-direct.log_prob(values[rows],context[rows]).mean()).backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(),5.)
        expected_optimizer.step()
    result = accumulated_epoch(model,values,context,pop,optimizer,weight=0.,mode='sampled',
        nll_batch_size=13,nll_chunk_size=5,cells_per_step=2,pairs_per_cell=7,draws=2,
        response_chunk_size=5,seed=92)
    assert sorted(seen) == list(range(36))
    assert result['optimizer_steps'] == 3
    assert result['response_pairs_per_group'] == 3*2*7
    for p,expected in zip(model.parameters(),direct.parameters()):
        torch.testing.assert_close(p,expected,atol=2e-7,rtol=1e-6)


def test_multi_group_validation_equals_all_distinct_cross_products():
    rng = np.random.default_rng(854)
    means,target = rng.normal(size=(6,3,4)),rng.normal(size=(3,4))
    precision = np.tile(np.diag([1,2,3,4]),(3,1,1))
    result = summarize_validation_means(means,target,precision)
    def direct(groups):
        scores = [.5*np.einsum('bi,bij,bj->b',groups[a]-target,precision,groups[c]-target)/4
                  for a in range(len(groups)) for c in range(len(groups)) if a != c]
        return np.mean(scores,axis=0)
    np.testing.assert_allclose(result['per_cell'],direct(means),atol=1e-14)
    deleted = [direct(np.delete(means,k,axis=0)).mean() for k in range(6)]
    np.testing.assert_allclose(result['mc_deleted_scores'],deleted,atol=1e-14)
    assert result['mean_quadratic']-result['integration_quadratic'] == pytest.approx(result['loss'])
    assert result['mc_jackknife_se'] > 0
    with pytest.raises(ValueError,match='four finite'):
        summarize_validation_means(means[:2],target,precision)


def test_tail_observation_preserves_all_group_means_and_draw_counts():
    model,context,pop = fixture()
    ordinary = evaluate_validation_groups(model,pop,4,4,5,91)
    observed = evaluate_validation_groups(model,pop,4,4,5,91,shape_means=[.1,-.1])
    np.testing.assert_array_equal(observed['group_means'],ordinary['group_means'])
    assert observed['loss'] == ordinary['loss']
    assert observed['tail_draw_count'] == 18*4*4*2  # pairs, draws, groups, shear legs
    assert np.all(np.diff(observed['tail_exceedance_counts']) <= 0)
    assert observed['max_shape_amplitude'] > 0
    assert observed['shape_support_counts'][0] == sum(observed['shape_support_counts'][1:3])
    assert observed['max_shape_squared_radius'] == pytest.approx(observed['max_shape_amplitude']**2)


def test_literal_disk_boundary_is_visible_independent_of_sqrt_rounding():
    from sbsi.flow_grid_accumulation import _TailObservedPopulation
    _,_,population = fixture()
    observer = _TailObservedPopulation(population,[0.,0.])
    eps = np.finfo(np.float64).eps
    shapes = torch.tensor([[0.,0.], [1.,0.], [1.,2**-26],
                           [1+64*eps,0.], [np.nextafter(1.,0.),0.]],dtype=torch.float64)
    original = shapes.clone()
    assert float(shapes[2].square().sum()) > 1
    # CPU kernels may round this half-ulp boundary to either neighbour.
    assert float(shapes[2].square().sum().sqrt()) in (1., 1.+eps)
    observer.observe_shapes(shapes)
    np.testing.assert_array_equal(observer.support_counts,[3,2,1,1])
    assert observer.counts[0] in (1,2)  # near-boundary sqrt rounding is platform dependent
    np.testing.assert_array_equal(observer.counts[1:],[0,0,0])
    assert observer.draw_count == 5
    torch.testing.assert_close(shapes,original,rtol=0,atol=0)
    with pytest.raises(ValueError,match='nonfinite physical shape'):
        observer.observe_shapes(torch.tensor([[float('inf'),0.]]))


def prepared_fixture(root):
    from scripts.train_flow_grid_response import DEFINITION,digest
    from sbsi.measurement_model import TargetStandardizer,save_measurement_model
    from sbsi.selection_model import TabularPreprocessor
    root.mkdir()
    model,context,pop = fixture()
    transform = TargetStandardizer(
        ['measured_ngmix_g1','measured_ngmix_g2','measured_mag_auto','measured_log_flux_radius'],
        np.zeros(4,dtype=np.float32),np.array([.3,.4,1,1],dtype=np.float32))
    condition = TabularPreprocessor(['x','y'],np.zeros(2),np.zeros(2),np.ones(2),False,())
    config = dict(flow_type='affine',context_dim=2,target_dim=4,hidden_dim=8,n_layers=1,n_flows=2)
    parent = root/'parent.pt'
    save_measurement_model(str(parent),model,condition,transform,config,metadata={'response_weight':0.})
    payload = dict(pairs=pop.pairs.numpy(),bin_ids=pop.bin_ids.numpy(),counts=np.diff(pop.offsets),
                   design=pop.design.numpy(),target=pop.target.numpy(),precision=pop.precision.numpy())
    for split in ('train','validation'):
        torch.save(dict(payload,values=torch.randn(36,4),context=context if split=='train' else torch.randn_like(context)),root/f'{split}.pt')
    manifest = dict(version=1,definition=DEFINITION,parent=str(parent),parent_sha256=digest(parent),
                    n_cells=3,splits={s:{'nll_rows':36,'used_pairs':18} for s in ('train','validation')},
                    files={f'{s}.pt':digest(root/f'{s}.pt') for s in ('train','validation')})
    (root/'manifest.json').write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize('mode',['sampled','full'])
def test_trainer_checkpoints_and_resume_match_uninterrupted_training(tmp_path,mode):
    from scripts.train_flow_grid_accumulated import main
    from scripts.train_flow_grid_response import load_torch
    root = tmp_path/'prepared'
    manifest = prepared_fixture(root)
    options = ['--root',str(root),'--weight','1','--mode',mode,'--device','cpu',
               '--nll-batch-size','13','--nll-chunk-size','5','--cells-per-step','2',
               '--pairs-per-cell','7','--response-chunk-size','5','--draws','2',
               '--validation-draws','2','--validation-groups','4','--validation-batch-size','5']
    resumed,uninterrupted = tmp_path/'resumed.pt',tmp_path/'uninterrupted.pt'
    main(options+['--output',str(resumed),'--epochs','1'])
    with pytest.raises(FileExistsError,match='overwrite'):
        main(options+['--output',str(resumed),'--epochs','1'])
    main(options+['--output',str(resumed),'--epochs','2','--resume'])
    main(options+['--output',str(uninterrupted),'--epochs','2'])
    a,b = [load_torch(tmp_path/f'{name}_epoch002.pt') for name in ('resumed','uninterrupted')]
    for key in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
    assert a['metadata']['parent_sha256'] == manifest['parent_sha256']
    assert a['metadata']['grid_loss_divisor'] == 12
    assert a['metadata']['response_weight'] == 0
    history = json.loads(resumed.with_suffix('.history.json').read_text())
    assert len(history['history']) == 3
    assert history['history'][1]['nll_rows'] == 36
    assert history['history'][1]['response_pairs_per_group'] == (18 if mode=='full' else 42)
    assert history['history'][1]['optimizer_steps'] == (2 if mode=='full' else 3)
    for record in history['history']:
        assert record['val_total'] == pytest.approx(record['val_nll']+record['val_grid'])
        assert record['val_grid_mc_jackknife_se'] >= 0
        assert (tmp_path/f'resumed_epoch{record["epoch"]:03d}.pt').exists()
    best = load_torch(resumed)
    assert best['metadata']['epoch'] == history['best_epoch']
    assert history['history'][history['best_epoch']]['checkpoint_eligible']
    changed = [*options]
    changed[changed.index('--weight')+1] = '10'
    with pytest.raises(ValueError,match='resume options changed'):
        main(changed+['--output',str(resumed),'--epochs','3','--resume'])


@pytest.mark.parametrize('failure',['nll','tail'])
def test_quality_guard_rejects_improving_score_preserves_parent_and_stops(tmp_path,monkeypatch,failure):
    from scripts import train_flow_grid_accumulated as script
    root = tmp_path/'prepared'
    prepared_fixture(root)
    nlls = iter([0.,.2 if failure=='nll' else 0.])
    monkeypatch.setattr(script,'epoch_nll',lambda *a,**k:next(nlls))
    original = script.evaluate_validation_groups
    calls = []
    def evaluate(*args,**kwargs):
        result = original(*args,**kwargs)
        result['loss'] = 0. if calls else 10.  # candidate wins the unconstrained score
        result['tail_exceedance_counts'] = np.array([30,25,0,0] if calls and failure=='tail' else [3,1,0,0])
        calls.append(True)
        return result
    monkeypatch.setattr(script,'evaluate_validation_groups',evaluate)
    output = tmp_path/'trial.pt'
    options = ['--root',str(root),'--output',str(output),'--weight','1','--mode','sampled',
               '--device','cpu','--epochs','3','--nll-batch-size','36','--nll-chunk-size','18',
               '--cells-per-step','1','--pairs-per-cell','4','--response-chunk-size','4',
               '--draws','2','--validation-draws','2','--validation-batch-size','18']
    script.main(options)
    history = json.loads(output.with_suffix('.history.json').read_text())
    assert len(history['history']) == 2
    assert history['stopped_on_guard']
    assert history['best_epoch'] == 0
    assert not history['history'][1]['checkpoint_eligible']
    assert (tmp_path/'trial_epoch001.pt').exists()  # failed candidate retained for diagnosis
    assert not (tmp_path/'trial_epoch002.pt').exists()
    with pytest.raises(ValueError,match='guard-stopped'):
        script.main(options+['--resume'])


def test_disabled_nll_guard_uses_combined_objective_and_patience(tmp_path,monkeypatch):
    from scripts import train_flow_grid_accumulated as script
    root=tmp_path/'prepared'
    prepared_fixture(root)
    nlls=iter([0.,.2,.3,.4])
    losses=iter([10.,2.,3.,4.])
    monkeypatch.setattr(script,'epoch_nll',lambda *a,**k:next(nlls))
    original=script.evaluate_validation_groups
    def evaluate(*a,**k):
        result=original(*a,**k)
        result['loss']=next(losses)
        result['tail_exceedance_counts']=np.array([0,0,0,0])
        return result
    monkeypatch.setattr(script,'evaluate_validation_groups',evaluate)
    output=tmp_path/'trial.pt'
    script.main(['--root',str(root),'--output',str(output),'--weight','.5',
        '--mode','sampled','--device','cpu','--epochs','8','--patience','2',
        '--disable-nll-guard','--nll-batch-size','36','--nll-chunk-size','18',
        '--cells-per-step','1','--pairs-per-cell','4','--response-chunk-size','4',
        '--draws','2','--validation-draws','2','--validation-batch-size','18'])
    result=json.loads(output.with_suffix('.history.json').read_text())
    assert result['best_epoch']==1
    assert result['stopped_on_patience'] and not result['stopped_on_guard']
    assert len(result['history'])==4
    for record in result['history'][1:]:
        assert not record['nll_guard_pass'] and not record['nll_guard_enforced']
        assert record['checkpoint_eligible']
        assert record['val_total']==pytest.approx(record['val_nll']+.5*record['val_grid'])
