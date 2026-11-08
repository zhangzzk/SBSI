"""Measure which size-fit branches account for the existing two-seed discrepancy."""
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pilot',H/'report.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
BRANCH=['both_T_positive','both_T_nonpositive','T_sign_changed','unverified_replay']
def main():
    assert os.environ.get('SLURM_JOB_ID')
    merged=sys.argv[1:]==['merged'];assert merged or not sys.argv[1:]
    old=r.load(H/'paired_diagnostics.json')
    weights=[];vals=[];types=[];receipts=[];fits=Counter();flags=Counter();details=[];params=[]
    largest_diff=0.
    for case in r.CASES:
        p=r.OUT/(f'fit_trace_merged{case}.json' if merged else f'fit_trace{case}.json');d=r.load(p);sf=r.OUT/f'sample_{case}.feather';df=pd.read_feather(sf)
        assert d['sample_sha256']==r.sha(sf) and d['case']==case and d['replicates']==[0,1]
        assert len(d['rows'])==len(df)
        receipts.append(dict(case=case,sha256=r.sha(p),script=d['source_sha256'],supplement=d.get('supplement_sha256'),first_pass_sha256=d.get('first_trace_sha256')))
        for ident,w in zip(df.index_input,df.weight):
            shapes=np.zeros((2,4,2)); cls=np.full(4,-1,int); measurements=[]
            for corner,reps in enumerate(d['rows'][str(ident)]):
                if reps is None:measurements.append(None);continue
                use=[rep is not None and np.dot(rep['g'],rep['g'])<1 for rep in reps]
                assert use[0]==use[1] # Established on the original target sample, tested again here.
                if not use[0]:measurements.append(None);continue
                t=np.array([rep['T'] for rep in reps])
                a=0 if (t>0).all() else (1 if (t<=0).all() else 2)
                largest_diff=max(largest_diff,*[rep.get('max_abs_g_difference',0.) for rep in reps])
                if not all(rep.get('g_exact',d['original_g_exact']) for rep in reps):a=3
                cls[corner]=a
                fits[BRANCH[a]]+=1
                for h,rep in enumerate(reps):
                    shapes[h,corner]=[1,rep['g'][0]]
                    if rep.get('g_exact',d['original_g_exact']):flags[str(int(rep['flags']))]+=1
                m=dict(corner=old['corners'][corner],exact=(a!=3),T=(t.tolist() if a!=3 else None),g1=shapes[:,corner,1].tolist(),chi2per=[rep.get('chi2per') for rep in reps])
                measurements.append(m)
                if a==2:params.append([abs(m['g1'][1]-m['g1'][0]), abs(m['chi2per'][1]-m['chi2per'][0]) if None not in m['chi2per'] else None])
            details.append((case,int(ident),measurements))
            weights.append(w);vals.append(shapes);types.append(cls)
    x=np.array(vals); w=np.array(weights); label=np.array(types)
    np.testing.assert_array_equal(x[...,0].sum(0),old['seed_diagnostic']['corner_selected_counts'])
    counts=w@x[:,0,:,0]
    v=w[:,None]*(x[:,1,:,1]-x[:,0,:,1])/counts
    by_obj=v@r.ROLES.T
    np.testing.assert_allclose(by_obj.sum(0),old['seed_diagnostic']['repeat1_minus_repeat0'],atol=1e-13,rtol=0)
    out=dict(source_report_sha256=r.sha(H/'paired_diagnostics.json'),receipts=receipts,selected_pair_counts=fits,max_abs_replayed_g_difference=largest_diff,exact_selected_fit_flags=flags,
             roles=old['roles'],role_shift=by_obj.sum(0).tolist(),by_branch={},examples={})
    for idx,name in enumerate(BRANCH):
        by=np.where(label==idx,v,0)@r.ROLES.T
        out['by_branch'][name]=dict(delta_R11=by.sum(0).tolist(),
            paired_variance_share=((by*by_obj).sum(0)/np.square(by_obj).sum(0)).tolist())
    if params:
        out['T_flip_abs_g1_difference_quantiles_10_50_90']=np.quantile([a for a,_ in params],[.1,.5,.9]).tolist()
        finite=[b for _,b in params if b is not None]
        if finite:out['T_flip_abs_chi2per_difference_quantiles_10_50_90']=np.quantile(finite,[.1,.5,.9]).tolist()
    for j,role in enumerate(r.ROLES):
        i=int(np.argmax(abs(by_obj[:,j])));case,ident,m=details[i]
        out['examples'][old['roles'][j]]=dict(case=case,input_index=ident,influence=by_obj[i,j],fits=m)
    np.testing.assert_allclose(np.sum([z['delta_R11'] for z in out['by_branch'].values()],0),by_obj.sum(0),atol=2e-13)
    np.testing.assert_allclose(np.sum([z['paired_variance_share'] for z in out['by_branch'].values()],0),1,atol=2e-13)
    dest=H/('fit_trace_report.json' if merged else 'fit_trace_first_pass.json');tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
    print('BRANCH ATTRIBUTION',dict(fits),out['by_branch'],flush=True)
if __name__=='__main__':main()
