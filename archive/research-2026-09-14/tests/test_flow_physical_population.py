import numpy as np
import torch
from sbsi.measurement_model import build_flow
from sbsi.flow_physical_population import PhysicalResponsePopulation,evaluate_physical_groups
from sbsi.flow_grid_accumulation import response_plan,sampled_response_backward
from sbsi.flow_grid_response import independent_grid_score
from test_flow_physical_disk import config


def fixture():
 torch.manual_seed(23);m=build_flow(config());context=torch.randn(20,2)
 rng=np.random.default_rng(6)
 stats=dict(target=rng.normal(size=(2,8)),precision=np.tile(np.eye(8),(2,1,1)),counts=np.array([5,5]),shape_design=rng.normal(size=(10,2)),scalar_design=rng.normal(size=(10,2)))
 return m,PhysicalResponsePopulation(context,np.arange(20).reshape(10,2),np.repeat([0,1],5),stats)


def six_component_fixture(weights=(1,1,1,1,.2,.2)):
 torch.manual_seed(23);m=build_flow(config());context=torch.randn(20,2)
 rng=np.random.default_rng(6);base=np.diag(np.arange(1,7,dtype=float))
 stats=dict(target=rng.normal(size=(2,6)),precision=np.tile(base,(2,1,1)),
            component_weights=np.asarray(weights),response_component_indices=np.arange(6),
            counts=np.array([5,5]),shape_design=rng.normal(size=(10,2)),
            scalar_design=rng.normal(size=(10,2)))
 return m,PhysicalResponsePopulation(context,np.arange(20).reshape(10,2),np.repeat([0,1],5),stats),base


def test_six_component_population_drops_flux_and_symmetrically_weights_precision():
 m,p,base=six_component_fixture();rows=torch.arange(3)
 delta=torch.tensor([[1.,2.,3.,400.],[5.,6.,7.,800.],[9.,10.,11.,1200.]],dtype=torch.float64)
 contribution=p.from_difference(delta,rows)
 assert contribution.shape==(3,6)
 assert p.all_from_difference(delta,rows).shape==(3,8)
 torch.testing.assert_close(p.all_from_difference(delta,rows)[:,6:],
                            (delta[:,3,None]*p.design[rows,1]).reshape(-1,2))
 expected_radius=(delta[:,2,None]*p.design[rows,1]).reshape(-1,2)
 torch.testing.assert_close(contribution[:,4:],expected_radius)
 expected=np.sqrt(np.array([1,1,1,1,.2,.2]))
 np.testing.assert_allclose(p.precision[0].numpy(),expected[:,None]*base*expected[None,:])
 np.testing.assert_allclose(p.base_precision[0].numpy(),base)


def test_shape_only_loss_has_no_scalar_output_gradient():
 m,old=fixture()
 stats=dict(target=old.target.numpy()[:,:4],precision=old.precision.numpy()[:,:4,:4],
            counts=np.array([5,5]),shape_design=old.design[:,0].numpy(),
            scalar_design=old.design[:,1].numpy())
 p=PhysicalResponsePopulation(old.context,old.pairs.numpy(),old.bin_ids.numpy(),stats)
 delta=torch.tensor([[1.,2.,3.,400.],[5.,6.,7.,800.]],dtype=torch.float64,requires_grad=True)
 contribution=p.from_difference(delta,torch.arange(2))
 assert contribution.shape==(2,4)
 contribution.square().sum().backward()
 torch.testing.assert_close(delta.grad[:,2:],torch.zeros_like(delta.grad[:,2:]),rtol=0,atol=0)
 assert torch.any(delta.grad[:,:2]!=0)


def test_six_component_replay_matches_direct_gradient():
 m,p,_=six_component_fixture();cells,local,groups=response_plan(p,2,3,51);means=[]
 for stream,rows in enumerate(groups):
  total=torch.zeros((2,6),dtype=torch.float64)
  for block,start in enumerate(range(0,len(rows),2)):
   v=p.contributions(m,rows[start:start+2],4,seed=51+100000+2*block+stream)
   total=total.index_add(0,local[start:start+2],v)
  means.append(total/3)
 score=independent_grid_score(*means,p.target[cells],p.precision[cells]).mean()/6
 score.backward();grads=[x.grad.clone() for x in m.parameters()];m.zero_grad(set_to_none=True)
 result=sampled_response_backward(m,p,2,3,4,chunk_size=2,seed=51)
 np.testing.assert_allclose(result['loss'],float(score.detach()))
 for a,b in zip(m.parameters(),grads):torch.testing.assert_close(a.grad,b,rtol=2e-5,atol=2e-5)


def test_eight_component_replay_matches_direct_gradient():
 m,p=fixture();cells,local,groups=response_plan(p,2,3,51);means=[]
 for stream,rows in enumerate(groups):
  total=torch.zeros((2,8),dtype=torch.float64)
  for block,start in enumerate(range(0,len(rows),2)):
   v=p.contributions(m,rows[start:start+2],4,seed=51+100000+2*block+stream)
   total=total.index_add(0,local[start:start+2],v)
  means.append(total/3)
 score=independent_grid_score(*means,p.target[cells],p.precision[cells]).mean()/8
 score.backward();grads=[x.grad.clone() for x in m.parameters()];m.zero_grad(set_to_none=True)
 result=sampled_response_backward(m,p,2,3,4,chunk_size=2,seed=51)
 np.testing.assert_allclose(result['loss'],float(score.detach()))
 for a,b in zip(m.parameters(),grads):torch.testing.assert_close(a.grad,b,rtol=2e-5,atol=2e-5)


def test_physical_validation_counts_and_eight_component_normalization():
 m,p=fixture();r=evaluate_physical_groups(m,p,draws=4,groups=4,batch_size=3,seed=123)
 assert r['draw_count']==10*4*4*2
 assert np.all(r['nonpositive_scalar_draws']==0)
 means=r['group_means'];target=p.target.numpy();prec=p.precision.numpy()
 expected=np.mean([.5*np.einsum('bi,bij,bj->b',a-target,prec,c-target).mean()/8 for i,a in enumerate(means) for j,c in enumerate(means) if i!=j])
 np.testing.assert_allclose(r['loss'],expected)


def test_physical_accumulated_epoch_and_optimizer_serialization(tmp_path):
 from sbsi.flow_grid_accumulation import accumulated_epoch
 from sbsi.flow_grid_full_cells import full_cell_response
 m,p=fixture()
 with torch.no_grad():values=m(torch.randn(20,4),p.context)[0]
 opt=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=0.)
 result=accumulated_epoch(m,values,p.context,p,opt,weight=.1,mode='sampled',nll_batch_size=7,nll_chunk_size=3,cells_per_step=2,pairs_per_cell=3,draws=4,response_chunk_size=2,seed=51,max_nll_rows=17,response_pairs_per_epoch=20)
 assert result['nll_rows']==17 and result['response_pairs_per_group']==20 and result['optimizer_steps']==3
 saved=tmp_path/'state.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),saved)
 clone,_=fixture();other=torch.optim.AdamW(clone.parameters(),lr=1e-5,weight_decay=0.)
 state=torch.load(saved,weights_only=False);clone.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
 for mm,oo in [(m,opt),(clone,other)]:
  oo.zero_grad(set_to_none=True)
  full_cell_response(mm,p,[0,1],4,chunk_size=3,seed=60,weight=.1,backward=True)
  oo.step()
 for a,b in zip(m.parameters(),clone.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
