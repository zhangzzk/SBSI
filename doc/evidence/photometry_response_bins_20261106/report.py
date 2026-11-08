"""Exact distribution/conditional-mean decomposition of response histograms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_response_bins_20261106')
PARENT=ROOT.parent/'photometry_selection_factorial_20261106'

def decompose(t,q):
    # weights, branch, edge, bin, quantity
    count=t[...,0];numerator=t[...,q]/(.04 if q==1 else 1.)
    mass=count.sum(-1);p=count/mass[...,None]
    mu=np.divide(numerator,count,out=np.zeros_like(numerator),where=count>0)
    shared=(count[:,0]>0)&(count[:,1]>0)
    distribution=(((p[:,1]-p[:,0])*.5*(mu[:,1]+mu[:,0]))*shared).sum(-1).mean(-1)
    within_bin=(((mu[:,1]-mu[:,0])*.5*(p[:,1]+p[:,0]))*shared).sum(-1).mean(-1)
    exclusive=((p[:,1]*mu[:,1]-p[:,0]*mu[:,0])*(~shared)).sum(-1).mean(-1)
    means=(numerator.sum(-1)/mass).mean(-1)
    difference=means[:,1]-means[:,0]
    np.testing.assert_allclose(distribution+within_bin+exclusive,difference,rtol=0,atol=2e-12)
    return {'generated':means[:,0],'measured':means[:,1],'diagonal_change':difference,
            'distribution':distribution,'within_bin':within_bin,'exclusive':exclusive,
            'shared_generated_mass':(p[:,0]*shared).sum(-1).mean(-1),
            'shared_measured_mass':(p[:,1]*shared).sum(-1).mean(-1)}

def coarsen(h,stride,include_size):
    h=np.add.reduceat(h,np.arange(0,h.shape[-3],stride),axis=-3)
    h=np.add.reduceat(h,np.arange(0,h.shape[-2],stride),axis=-2)
    if not include_size:h=h.sum(3)
    return h.reshape(*h.shape[:3],-1,3)

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    cases=[1000] if args.pilot else [1000,1040,1080,1120];n=len(cases);hist=[];receipts=[]
    for case in cases:
        receipts.append(json.loads((ROOT/f'case{case}.json').read_text()));assert receipts[-1]['factorial_replay_passed']
        with np.load(ROOT/f'case{case}.npz') as z:hist.append(z['sums'])
    hist=np.array(hist);end=10001
    weights=np.vstack([np.ones(n),np.random.default_rng(202611069301).multinomial(n,np.full(n,1/n),10000),np.eye(n)])
    if n>1:weights=np.vstack([weights,np.ones((n,n))-np.eye(n)])
    # Four-case resampling repeats a small finite set of weight vectors.
    # Evaluate each exactly once, then restore all original ordered resamples.
    unique_weights,inverse=np.unique(weights,axis=0,return_inverse=True)
    def ent(x):
        jk=x[end+n:]
        return {'value':float(x[0]),'case_se':float(x[1:end].std(ddof=1)) if n>1 else None,
                'jackknife_se':float(np.sqrt((n-1)/n*np.square(jk-jk.mean()).sum())) if n>1 else None}
    parent_name='pilot_report.npz' if args.pilot else 'report.npz'
    with np.load(PARENT/parent_name) as z:
        np.testing.assert_array_equal(z['cases'],cases);assert int(z['seed'])==202611069301
        k=list(z['cuts']).index('mag25_radius_gt0.60');reference=z['response'][k]
    results={};arrays={}
    for stride,label in [(1,'fine'),(2,'medium'),(4,'coarse')]:
        for include_size in [False,True]:
            key=label+('_with_intrinsic_size' if include_size else '')
            h=coarsen(hist,stride,include_size);collected={1:{},2:{}}
            for lo in range(0,len(unique_weights),128):
                t=np.einsum('wc,cbepq->wbepq',unique_weights[lo:lo+128],h)
                for q in [1,2]:
                    for name,x in decompose(t,q).items():collected[q].setdefault(name,[]).append(x)
            result={}
            for q,name in [(1,'finite_shape'),(2,'scalar_coefficient')]:
                vals={term:np.concatenate(parts)[inverse] for term,parts in collected[q].items()}
                if q==1:
                    np.testing.assert_allclose(vals['generated'][:end],reference[:,0,0,0],rtol=0,atol=2e-11)
                    np.testing.assert_allclose(vals['measured'][:end],reference[:,1,1,0],rtol=0,atol=2e-11)
                result[name]={term:ent(x) for term,x in vals.items()}
                arrays[key+'_'+name]=np.column_stack(list(vals.values()))
            results[key]=result
    out={'cases':cases,'definition':'Measured minus generated conditional-model response. Same common-U four-edge normalization as parent factorial.',
         'bootstrap_seed':202611069301,'results':results,'receipts':receipts,'report_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'limitations':['Descriptive bin decomposition, not a unique causal attribution.',
                        'Within-bin contrasts can include differing intrinsic properties and neighbour environments.',
                        'Development scenes at fixed checkpoints; conditional SE excludes model-training uncertainty.',
                        'Rare endpoint cut-crossers retain their original weights and all bin support.']}
    name='pilot_report' if args.pilot else 'report'
    (ROOT/(name+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (HERE/(name+'.json')).write_bytes((ROOT/(name+'.json')).read_bytes())
    np.savez(ROOT/(name+'.npz'),**arrays,cases=cases)
    def fmt(x):return f"{x['value']:+.6f}"+(f" ± {x['case_se']:.6f}" if n>1 else ' (one scene)')
    lines=['# Photometry-bin response decomposition','',out['definition'],f'Cases: {cases}.','',
           '| Bins | Distribution term | Within-bin response term | Exclusive support | Total change |',
           '|---|---:|---:|---:|---:|']
    for key,x in results.items():
        row=x['finite_shape'];lines.append('| '+key+' | '+' | '.join(fmt(row[t]) for t in ['distribution','within_bin','exclusive','diagonal_change'])+' |')
    lines+=['','All terms use finite conditional-flow shape transport. Scalar-coefficient decomposition is also preserved in JSON.','']
    lines+=['- '+s for s in out['limitations']]
    (HERE/('pilot_results.md' if args.pilot else 'results.md')).write_text('\n'.join(lines)+'\n')
    print('DONE: every histogram decomposition and parent bootstrap replay passed',flush=True)
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
