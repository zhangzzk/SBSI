"""Standalone flow response localized in fixed truth cells."""
import os,sys,json,time,importlib.util,hashlib
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
D=Path('/project/ls-gruen/users/zekang.zhang');F=D/'sbsi_flow_f8_20260924';sys.path.insert(0,str(F))
import pandas as pd
import torch
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.measurement_model import load_measurement_model
from sbsi.flow_guard_response import physical_context_draws,torch_guard_weights
O=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('guard',Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/scripts/run_galsbi_guard_strength.py'))
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
trainer=guard.frozen();torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
cases=json.loads((guard.DOMAIN/'manifest.json').read_text())['split']['validation_cases']
start=time.monotonic();data=trainer.base.load_split(guard.DOMAIN,cases)
path=guard.SOURCE/'guard/selected.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='e3e38f7f6aaf85669b892daf4f362767f7fc92e0e5685035bcdc063e6b989adc'
bundle=load_measurement_model(path,device='cuda');bundle.model.eval()
context=bundle.context_tensor(pd.DataFrame(data['context_raw'],columns=FLOW_FEATURES))
pairs=torch.as_tensor(data['pairs'],device='cuda');n=len(pairs)
case_of={c:i for i,c in enumerate(cases)};ci=torch.as_tensor([case_of[int(c)] for c in data['pair_case']],device='cuda')
g=torch.as_tensor(data['gamma'],dtype=torch.float64,device='cuda')
measured=[torch.as_tensor(data[k],dtype=torch.float64,device='cuda') for k in ['pair_zero_target','pair_shear_target']]
cut=({'name':'target','radius_min':3.,'magnitude_max':25.},)
def weight(v):
 soft=torch_guard_weights(v,cut,radius_softness=trainer.RADIUS_SOFTNESS,magnitude_softness=trainer.MAGNITUDE_SOFTNESS)[...,0]
 hard=((v[...,2]>3)&(v[...,3]>100)).double()
 return torch.stack([soft,hard],-1)
# case; kind soft/hard; branch measured/generated-cut/observed-cut; shape component; shear component
base=torch.zeros((40,2,3),dtype=torch.float64,device='cuda') # sum w0, sum w0*e01,e02
normal=torch.zeros((40,2,2),dtype=torch.float64,device='cuda')
count=torch.zeros(40,dtype=torch.float64,device='cuda')
cross_e=torch.zeros((40,2,3,2,2),dtype=torch.float64,device='cuda')
cross_w=torch.zeros((40,2,3,2),dtype=torch.float64,device='cuda')
raw=data['context_raw'][data['pairs'][:,0]]
cell=(np.digitize(raw[:,4],[.15,.25,.4,.7])*4+np.digitize(raw[:,3],[24,25.5,27.5]))*5+np.digitize(raw[:,6],[.1,.3,1,3])
cci=ci*100+torch.as_tensor(cell,device='cuda')
cell_base=torch.zeros((4000,2,3),dtype=torch.float64,device='cuda')
cell_cross_e=torch.zeros((4000,2,3,2,2),dtype=torch.float64,device='cuda')
cell_cross_w=torch.zeros((4000,2,3,2),dtype=torch.float64,device='cuda')
print('START',n,'pairs',torch.cuda.get_device_name(0),'load_seconds',time.monotonic()-start,flush=True)
with torch.no_grad():
 for block,lo in enumerate(range(0,n,2048)):
  hi=min(n,lo+2048);pc=ci[lo:hi];gg=g[lo:hi]
  v0,v1=[v[lo:hi] for v in measured];ws=[weight(v0),weight(v1)]
  base.index_add_(0,pc,torch.cat([ws[0][...,None],ws[0][...,None]*v0[:,None,:2]],-1))
  normal.index_add_(0,pc,gg[:,:,None]*gg[:,None,:]);count.index_add_(0,pc,torch.ones(len(pc),dtype=torch.float64,device='cuda'))
  generated=physical_context_draws(bundle.model,[context[pairs[lo:hi,j]] for j in [0,1]],16,614000000001+block)
  leg_shapes=[];leg_weights=[]
  for v,w,draw in zip([v0,v1],ws,generated):
   dw=weight(draw);mean=draw[...,:2].mean(1)
   shape=torch.stack([w[...,None]*v[:,None,:2],(dw[...,None]*draw[:,:,None,:2]).mean(1),w[...,None]*mean[:,None,:]],2)
   weights=torch.stack([w,dw.mean(1),w],2)
   leg_shapes.append(shape);leg_weights.append(weights)
  cross_e.index_add_(0,pc,(leg_shapes[1]-leg_shapes[0])[...,None]*gg[:,None,None,None,:])
  cross_w.index_add_(0,pc,(leg_weights[1]-leg_weights[0])[...,None]*gg[:,None,None,:])
  cell_base.index_add_(0,cci[lo:hi],torch.cat([ws[0][...,None],ws[0][...,None]*v0[:,None,:2]],-1))
  cell_cross_e.index_add_(0,cci[lo:hi],(leg_shapes[1]-leg_shapes[0])[...,None]*gg[:,None,None,None,:])
  cell_cross_w.index_add_(0,cci[lo:hi],(leg_weights[1]-leg_weights[0])[...,None]*gg[:,None,None,:])
  if block%400==0:print('block',block,'seconds',time.monotonic()-start,flush=True)
arrays={k:v.cpu().numpy() for k,v in [('base',base),('normal',normal),('count',count),('cross_e',cross_e),('cross_w',cross_w)]}
np.savez(O/'flow_sums.npz',**arrays,cases=cases)
cell_arrays={k:v.cpu().numpy().reshape(40,100,*v.shape[1:]) for k,v in [('base',cell_base),('cross_e',cell_cross_e),('cross_w',cell_cross_w)]}
for k,v in cell_arrays.items():np.testing.assert_allclose(v.sum(1),arrays[k],atol=1e-7,rtol=1e-10)
np.savez(O/'flow_cell_sums.npz',**cell_arrays,cases=cases)
# Paired whole-case bootstrap, recomputing nuisance terms from each resample.
w=np.vstack([np.ones(40),np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)])
a={k:np.einsum('bc,c...->b...',w,v) for k,v in arrays.items()}
mass=a['base'][...,0]/a['count'][:,None];mean=a['base'][...,1:]/a['base'][...,0,None]
centered=a['cross_e']-mean[:,:,None,:,None]*a['cross_w'][...,None,:]
R=np.einsum('bksoi,bij->bksoj',centered,np.linalg.inv(a['normal']))/mass[:,:,None,None,None]
half=np.trace(R,axis1=-2,axis2=-1)/2
old=json.loads((O.parent/'absolute_response_errors_20261104/report.json').read_text())['rows']['flow_validation']
np.testing.assert_allclose(half[0,0,:2],[old['simulation'],old['model']],atol=2e-10,rtol=0)
def entry(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
out={'cases':cases,'pairs':n,'draws':16,'definition':'half-trace forward influence response; data baseline mass/mean shared by model and simulation','bootstrap':'10000 whole-case paired resamples, seed20260925; mass, mean and shear normal recomputed','cuts':{}}
for k,name in enumerate(['soft','hard']):
 out['cuts'][name]={'simulation':entry(half[:,k,0]),'model_generated_selection':entry(half[:,k,1]),'model_true_selection':entry(half[:,k,2]),'error_generated_selection':entry(half[:,k,0]-half[:,k,1]),'error_true_selection':entry(half[:,k,0]-half[:,k,2]),'change_in_error_true_minus_generated':entry(half[:,k,1]-half[:,k,2]),'m_true_selection_percent':entry(100*(half[:,k,0]/half[:,k,2]-1))}
out['replay']='Original soft-cut simulation and model point values agree within2e-10.'
out['seconds']=time.monotonic()-start
(O/'flow_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['cuts'],indent=2),flush=True);print('COMPLETE',out['seconds'],flush=True)
