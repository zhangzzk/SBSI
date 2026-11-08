"""Compare fixed own/opposite-group responses on the same four images."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106')
spec=importlib.util.spec_from_file_location('contrasts',HERE.parent/'constant_split_20261105/contrasts.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def ent(v,n):return {'value':float(v[0]),'case_se':float(v[1:].std(ddof=1)) if n>1 else None}

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true')
    p.add_argument('--arm',choices=['baseline','nine_weak']);args=p.parse_args()
    arms=[args.arm] if args.arm else ['baseline','nine_weak']
    cases=[1000] if args.pilot else [1000,1040,1080,1120];n=len(cases)
    w=np.vstack([np.ones(n),np.random.default_rng(202611069301).multinomial(n,np.full(n,1/n),10000)])
    out={'cases':cases,'definitions':'DeltaR=simulation-model. Own group includes same-group neighbours. Opposite group fixes target own shear. Global selected denominators; contributions add exactly. Four development cases; fixed-model case SEs.','arms':{},'paired_changes':{}}
    with np.load(HERE.parent/'constant_split_20261105/sums.npz') as old:
        loc=[list(old['cases']).index(case) for case in cases]
        measured_reference=old['sim'][loc].sum(-2)
        measured_edges=old['common_U_edges'][loc]
    edge_states=[(0,2),(1,3),(0,1),(2,3),(0,3)]
    raw={};reference=None;bootstrap_opposite=[]
    for arm in arms:
        ss=[];mm=[]
        for case in cases:
            with np.load(ROOT/'fourstate'/arm/f'case{case}.npz') as z:
                ss.append(z['sim']);mm.append(z['model']);cuts=list(z['cuts'])
        ss=np.array(ss);mm=np.array(mm)
        np.testing.assert_allclose(ss,measured_reference,rtol=1e-12,atol=1e-7)
        if reference is None:reference=ss
        else:np.testing.assert_array_equal(ss,reference)
        modes={};arm_opposite=[]
        for label,si,mi in [('predicted_U',0,0),('actual_U',0,1),('common_U',1,2)]:
            rows={}
            for k,cut in enumerate(cuts):
                s=np.einsum('bc,csgq->bsgq',w,ss[:,si,:,k]);m=np.einsum('bc,csgq->bsgq',w,mm[:,mi,:,k])
                ds=s[...,0].sum(-1);dm=m[...,0].sum(-1)
                rs=c.contrasts(s[...,1]/ds[...,None]);rf=c.contrasts(m[...,1]/dm[...,None]);re=c.contrasts(m[...,3]/dm[...,None])
                result={}
                for key in rs:
                    rm=rf[key]+re[key];delta=rs[key]-rm;raw[arm,label,cut,key]=delta
                    result[key]={'simulation':ent(rs[key],n),'model':ent(rm,n),'error':ent(delta,n),'model_flow':ent(rf[key],n),'model_emulator':ent(re[key],n)}
                group={}
                opposite_parts=[]
                if label=='common_U':
                    ed=np.einsum('bc,cegq->begq',w,measured_edges[:,k])
                    parts=np.zeros((len(w),5,2,2))
                    means=s[...,1]/s[...,0]
                    for edge,(a,b) in enumerate(edge_states):
                        da=s[:,a,:,0];db=s[:,b,:,0];dbar=.5*(da+db)
                        parts[:,edge,:,0]=ed[:,edge,:,0]/dbar
                        parts[:,edge,:,1]=(ed[:,edge,:,1]-.5*(means[:,a]+means[:,b])*(da-db))/dbar
                        np.testing.assert_allclose(parts[:,edge].sum(-1),means[:,a]-means[:,b],atol=2e-12,rtol=0)
                for g in [0,1]:
                    group[str(g)]={}
                    for name,coef in [('own_group',c.own),('opposite_group',c.opposite)]:
                        sm=(s[:,:,g,1]/s[:,:,g,0])@coef[:,g]
                        fl=(m[:,:,g,1]/m[:,:,g,0])@coef[:,g];em=(m[:,:,g,3]/m[:,:,g,0])@coef[:,g]
                        if label=='common_U' and name=='opposite_group':np.testing.assert_allclose(fl,0,atol=2e-12,rtol=0)
                        group[str(g)][name]={'simulation':ent(sm,n),'model':ent(fl+em,n),'error':ent(sm-fl-em,n),'model_flow':ent(fl,n),'model_emulator':ent(em,n)}
                        if label=='common_U' and name=='opposite_group':
                            ee=[2,3] if g==0 else [0,1]
                            terms=parts[:,ee,g].sum(1)/.08
                            np.testing.assert_allclose(terms.sum(1),sm,atol=2e-12,rtol=0)
                            group[str(g)][name].update(simulation_shape_term=ent(terms[:,0],n),
                                simulation_selection_term=ent(terms[:,1],n),shape_error=ent(terms[:,0]-em,n))
                            opposite_parts.append(np.stack([sm,em,terms[:,0],terms[:,1],terms[:,0]-em,sm-em],-1))
                if label=='common_U':
                    avg=.5*(opposite_parts[0]+opposite_parts[1])
                    arm_opposite.append(avg)
                    result['equal_group_opposite']={name:ent(avg[:,i],n) for i,name in enumerate(
                        ['simulation','model','simulation_shape_term','simulation_selection_term','shape_error','total_error'])}
                    result['equal_group_opposite']['definition']='Equal average of separately normalized A/B opposite-group responses; not a globally normalized coherent contribution.'
                result['within_group']=group;rows[cut]=result
            modes[label]=rows
        out['arms'][arm]=modes
        bootstrap_opposite.append(arm_opposite)
    if len(arms)==2:
        for label in out['arms']['baseline']:
            out['paired_changes'][label]={cut:{key:ent(raw['nine_weak',label,cut,key]-raw['baseline',label,cut,key],n) for key in ['own_group','opposite_group','coherent_total']} for cut in cuts}
    destination=ROOT/('fourstate'+('_'+args.arm if args.arm else '')+('_pilot' if args.pilot else '')+'_report.json')
    destination.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    np.savez(destination.with_suffix('.npz'),values=np.array(bootstrap_opposite),arms=arms,
        quantities=['simulation','model','simulation_shape_term','simulation_selection_term','shape_error','total_error'],
        cases=cases,cuts=cuts,seed=202611069301)
    print('DONE',n,flush=True)

if __name__=='__main__':main()
