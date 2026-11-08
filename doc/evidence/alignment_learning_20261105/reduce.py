"""Case bootstrap and interpretable profile contrasts for frozen-flow probe."""
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_learning_20261105')
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
    return v

def ent(v):return {'value':v[0],'case_se':np.std(v[1:],axis=0,ddof=1)}

out={'definition':'Baseline intrinsic Re[0.15,0.25), matched usable pairs. Alignment dot/power, model minus simulation. 10000 case resamples; models fixed. NLL and adopted are paired within case. Training subset is first40 pre-recorded randomized training IDs.', 'families':{}}
for fi,family in enumerate(['fs2','galsbi']):
    protocol=json.loads((ROOT/family/'protocol.json').read_text());assert (ROOT/family/'completed.json').exists()
    result={}
    for si,split in enumerate(['train','validation']):
        profiles=[];densities=[];receipts=[]
        for case in protocol['cases'][split]:
            file=ROOT/family/f'{split}_case{case:03d}.npz'
            with np.load(file) as z:
                v=z['profile'];profiles.append(np.stack([v[0],v[1:3].mean(0),v[3:5].mean(0)]));densities.append(z['density'])
            receipts.append(json.loads(file.with_suffix('.json').read_text()))
        x=np.array(profiles);densities=np.array(densities)
        w=np.vstack([np.ones(len(x)),np.random.default_rng(202611052000+10*fi+si).multinomial(len(x),np.full(len(x),1/len(x)),10000)])
        groups={}
        for gi,group in enumerate(protocol['groups']):
            d=x[:,:,:,gi] # case, branch, mag, radius, column
            totals=d.sum(0)
            with np.errstate(divide='ignore',invalid='ignore'):
                point={'alignment':totals[...,2]/totals[...,1],
                       'negative_fraction':totals[...,3]/totals[...,0],
                       'negative_alignment_contribution':totals[...,4]/totals[...,1],
                       'positive_alignment_contribution':totals[...,5]/totals[...,1],
                       'perpendicular_alignment':totals[...,6]/totals[...,1],
                       'count':totals[...,0]}
            # Key per-bin bootstrap contrasts, plus integrated sides of cut.
            bands={}
            for label,indices in [('2.7-2.8',[8]),('2.9-3.0',[10]),('3.0-3.1',[11]),('3.1-3.2',[12]),('2.4-3.0',list(range(5,11))),('3.0-3.4',list(range(11,15)))]:
                dd=d[:,:,:,indices,:].sum(3)
                t=np.einsum('nc,cbmk->nbmk',w,dd)
                with np.errstate(divide='ignore',invalid='ignore'):
                    a=t[...,2]/t[...,1];neg=t[...,3]/t[...,0]
                bands[label]={'alignment':ent(a),'nll_minus_sim':ent(a[:,1]-a[:,0]),
                    'adopted_minus_sim':ent(a[:,2]-a[:,0]),'adopted_minus_nll':ent(a[:,2]-a[:,1]),
                    'negative_fraction':ent(neg),'counts':t[0,...,0]}
            groups[group]={'point':point,'bands':bands}
        den=np.einsum('nc,ckmrj->nkmrj',w,densities)
        with np.errstate(divide='ignore',invalid='ignore'):
            mean=den[...,1]/den[...,0];rot=den[...,3]/den[...,0]
        result[split]={'groups':groups,'density':{'mean_logp':ent(mean),'rotation_logp_change':ent(rot)},
              'small_pairs':sum(r['small_pairs'] for r in receipts),'all_pairs':sum(r['all_pairs'] for r in receipts),
              'matching':[r['matching'][0] for r in receipts]}
        # Exact point decomposition into within-cell alignment and composition.
        # Use symmetric weights; expose unsupported power rather than impute it.
        cells=x[:,:,:,20:36].sum(0)
        decompositions={}
        for model in [1,2]:
            ps,pm=cells[0,...,1],cells[model,...,1]
            ds,dm=cells[0,...,2],cells[model,...,2]
            support=(ps>0)&(pm>0)
            ps=ps*support;pm=pm*support
            qs=np.divide(ps,ps.sum(1,keepdims=True),out=np.zeros_like(ps),where=ps.sum(1,keepdims=True)>0)
            qm=np.divide(pm,pm.sum(1,keepdims=True),out=np.zeros_like(pm),where=pm.sum(1,keepdims=True)>0)
            aa=np.divide(ds,ps,out=np.zeros_like(ds),where=ps>0)
            bb=np.divide(dm,pm,out=np.zeros_like(dm),where=pm>0)
            within=((qs+qm)/2*(bb-aa)).sum(1)
            composition=((aa+bb)/2*(qm-qs)).sum(1)
            delta=(qm*bb-qs*aa).sum(1)
            np.testing.assert_allclose(within+composition,delta,atol=1e-12)
            decompositions[str(model)]={'within_cell':within,'composition':composition,'common_support_delta':delta,
                'excluded_sim_power':(cells[0,...,1]*(~support)).sum(1),
                'excluded_model_power':(cells[model,...,1]*(~support)).sum(1)}
        result[split]['sersic_magnitude_decomposition']=decompositions
        print(family,split,'small pairs',result[split]['small_pairs'],flush=True)
        print(clean(groups['all']['bands']['2.7-2.8']),flush=True)
    out['families'][family]=result
(ROOT/'report.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n')
print('Saved report.json',flush=True)
