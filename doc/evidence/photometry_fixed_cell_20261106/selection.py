"""Selection mismatch versus scene sensitivity at fixed reference photometry."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_fixed_cell_20261106')
spec=importlib.util.spec_from_file_location('reference_bins',HERE.parent/'photometry_response_bins_20261106/report.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
EDGES=np.array([-np.inf,0,.025,.05,.1,.2,.4,.8,np.inf])

def reduce_rows(labels,pg,pm,score,n):
    return np.stack([np.bincount(labels,weights=v,minlength=n) for v in
                     [np.ones(len(pg)),pg,pm,pg*score,pm*score,score]],-1)

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    cases=[1000] if args.pilot else [1000,1040,1080,1120];n=len(cases);rates=[];intrinsic=[];crowding=[]
    for case in cases:
        receipt=json.loads((ROOT/f'case{case}.json').read_text());assert receipt['target_moment_identity_passed']
        with np.load(ROOT/f'targets{case}.npz') as z:
            moment=z['moments'];context=z['context'];score=z['reference_coefficient'].sum(-1)
            pg=moment[0,:,:,0].sum(0)/2;pm=moment[1,:,:,0].sum(0)/2
        assert np.isfinite(score).all() and np.all((pg>=0)&(pg<=1)) and np.all((pm>=0)&(pm<=1))
        refbin=np.searchsorted(EDGES[1:-1],score,side='right')
        rates.append(reduce_rows(refbin,pg,pm,score,8))
        # Raw flow context: e1,e2,n,mag,circularized Re,close050,close100,close200.
        cell=(np.digitize(context[:,4],[.15,.25,.4,.7])*4+np.digitize(context[:,3],[23,25,27]))*3+np.digitize(context[:,2],[1,2.5])
        intrinsic.append(reduce_rows(cell,pg,pm,score,60))
        cell=cell*5+np.digitize(context[:,5],[.05,.2,.7,1.5])
        crowding.append(reduce_rows(cell,pg,pm,score,300))
    end=10001;w=np.vstack([np.ones(n),np.random.default_rng(202611069301).multinomial(n,np.full(n,1/n),10000),np.eye(n)])
    if n>1:w=np.vstack([w,np.ones((n,n))-np.eye(n)])
    unique,inverse=np.unique(w,axis=0,return_inverse=True)
    def ent(x):
        boot=x[1:end];jk=x[end+n:];finite=np.isfinite(boot).all()
        return {'value':float(x[0]) if np.isfinite(x[0]) else None,
                'case_se':float(boot.std(ddof=1)) if n>1 and finite else None,
                'jackknife_se':float(np.sqrt((n-1)/n*np.square(jk-jk.mean()).sum())) if n>1 and np.isfinite(jk).all() else None,
                'finite_bootstrap_replicates':int(np.isfinite(boot).sum())}
    def divide(a,d):return np.divide(a,d,out=np.full_like(a,np.nan),where=d>0)
    rate=np.einsum('wc,cbq->wbq',unique,np.array(rates))[inverse]
    total=rate.sum(1);rows=[]
    for k in range(8):
        x=rate[:,k]
        row={'bin':[str(EDGES[k]),str(EDGES[k+1])],
             'population_fraction':ent(x[:,0]/total[:,0]),
             'generated_pass_rate':ent(divide(x[:,1],x[:,0])),
             'measured_pass_rate':ent(divide(x[:,2],x[:,0])),
             'pass_rate_error':ent(divide(x[:,2]-x[:,1],x[:,0])),
             'generated_selected_fraction':ent(x[:,1]/total[:,1]),
             'measured_selected_fraction':ent(x[:,2]/total[:,2]),
             'selected_mean_coefficient_contribution_change':ent(x[:,4]/total[:,2]-x[:,3]/total[:,1])}
        rows.append(row)
    selected_change=total[:,4]/total[:,2]-total[:,3]/total[:,1]
    controls={}
    for label,raw in [('reference_coefficient_bins',rates),('intrinsic_cells',intrinsic),('intrinsic_and_close050_cells',crowding)]:
        t=np.einsum('wc,cbq->wbq',unique,np.array(raw));v=np.zeros((len(unique),2,1,t.shape[1],3))
        v[:,0,0,:,0]=t[...,1];v[:,0,0,:,2]=t[...,3];v[:,1,0,:,0]=t[...,2];v[:,1,0,:,2]=t[...,4]
        z={key:x[inverse] for key,x in b.decompose(v,2).items()}
        np.testing.assert_allclose(z['diagonal_change'],selected_change,rtol=0,atol=2e-12)
        controls[label]={key:ent(x) for key,x in z.items()}
    out={'cases':cases,'definition':'Reference scene coefficient at radius3.5/magnitude24.5, all modelled neighbours. Not a measured or calibrated R11.',
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bootstrap_seed':202611069301,
         'generated_pass_rate':ent(total[:,1]/total[:,0]),'measured_pass_rate':ent(total[:,2]/total[:,0]),
         'selected_reference_coefficient_change':ent(selected_change),'bins':rows,'controls':controls,
         'limitations':['Common-all-four actual usability; this does not test the usability classifier.',
                        'Coarse intrinsic/crowding cells do not hold every flow input exactly fixed.',
                        'No conditional-density or omitted-feature causal conclusion follows from association alone.']}
    name='selection_pilot' if args.pilot else 'selection'
    (ROOT/(name+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');(HERE/(name+'.json')).write_bytes((ROOT/(name+'.json')).read_bytes())
    def fmt(x):return 'undefined' if x['value'] is None else f"{x['value']:+.6f}"+(f" ± {x['case_se']:.6f}" if x['case_se'] is not None else '')
    lines=['# Selection versus scene sensitivity','',out['definition'],f'Cases: {cases}.','',
           '| Reference coefficient | Generated pass rate | Measured pass rate | Measured minus generated |',
           '|---|---:|---:|---:|']
    for row in rows:lines.append('| '+', '.join(row['bin'])+' | '+' | '.join(fmt(row[q]) for q in ['generated_pass_rate','measured_pass_rate','pass_rate_error'])+' |')
    lines+=['','Selected-mean reference coefficient change: '+fmt(out['selected_reference_coefficient_change'])+'.','',
            '| Control partition | Distribution | Within-cell | Exclusive support |','|---|---:|---:|---:|']
    for key,row in controls.items():lines.append('| '+key+' | '+' | '.join(fmt(row[q]) for q in ['distribution','within_bin','exclusive'])+' |')
    lines+=['']+['- '+s for s in out['limitations']]
    (HERE/(name+'.md')).write_text('\n'.join(lines)+'\n')
    print('DONE: reference selection moments and all decomposition identities passed',flush=True)
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
