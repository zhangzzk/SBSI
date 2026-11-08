import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/selection_information_20261106')
CASES=[1000,1040,1080,1120]
EDGES=np.array([-np.inf,0,.025,.05,.1,.2,.4,.8,np.inf])
def main():
    assert os.environ.get('SLURM_JOB_ID')
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    folds=[0] if args.pilot else list(range(4));cases=[CASES[i] for i in folds];sums=[];bins=[];receipts=[]
    for i in folds:
        receipt=json.loads((ROOT/f'fold{i}/completed.json').read_text());assert receipt['reload_exact']
        assert receipt['test_case']==CASES[i] and receipt['validation_case']==CASES[(i+1)%4]
        assert len(set(receipt['training_cases']+[receipt['validation_case'],receipt['test_case']]))==4
        receipts.append(receipt)
        with np.load(ROOT/f'fold{i}/predictions.npz') as z:
            p=z['probabilities'];margin=z['margins'];y=z['target'];score=z['score'];ids=z['input_index'];arms=list(z['arms'])
        with np.load(ROOT/f'case{CASES[i]}.npz') as z:
            for key,value in [('input_index',ids),('target',y),('score',score),('flow',p[0])]:np.testing.assert_array_equal(value,z[key])
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        # Baseline flow probabilities use64 draws and can be exactly0/1.
        # Do not invent a log-loss clipping rule or compare its Monte Carlo
        # boundary noise with deterministic selection predictors.
        moments=np.stack([np.full(4,len(y)),np.full(4,y.sum()),np.full(4,(y*score).sum()),p.sum(1),
                          (p*score).sum(1),np.square(p-y).sum(1),
                          np.r_[0.,(np.logaddexp(0,margin[1:])-y*margin[1:]).sum(1)]],-1)
        sums.append(moments);label=np.searchsorted(EDGES[1:-1],score,side='right')
        bins.append(np.array([np.stack([np.bincount(label,minlength=8),np.bincount(label,weights=y,minlength=8),
                         np.bincount(label,weights=pp,minlength=8)],-1) for pp in p]))
    n=len(cases);end=10001;w=np.vstack([np.ones(n),np.random.default_rng(202611069301).multinomial(n,np.full(n,1/n),10000),np.eye(n)])
    if n>1:w=np.vstack([w,np.ones((n,n))-np.eye(n)])
    t=np.einsum('wc,caq->waq',w,np.array(sums));bt=np.einsum('wc,cabq->wabq',w,np.array(bins))
    def ent(x):
        jk=x[end+n:]
        return {'value':float(x[0]),'case_se':float(x[1:end].std(ddof=1)) if n>1 else None,
                'jackknife_se':float(np.sqrt((n-1)/n*np.square(jk-jk.mean()).sum())) if n>1 else None}
    quantities={'squared_error':t[...,5]/t[...,0],'soft_logloss':t[...,6]/t[...,0],
                'predicted_pass_rate':t[...,3]/t[...,0],'measured_pass_rate':t[...,1]/t[...,0],
                'pass_rate_error':(t[...,1]-t[...,3])/t[...,0],
                'reference_coefficient_error':t[...,2]/t[...,1]-t[...,4]/t[...,3]}
    results={}
    for i,arm in enumerate(arms):
        row={q:(None if arm=='flow' and q=='soft_logloss' else ent(value[:,i])) for q,value in quantities.items()}
        row['pass_error_by_score_bin']=[]
        for k in range(8):
            d=bt[:,i,k,0]
            row['pass_error_by_score_bin'].append(ent((bt[:,i,k,1]-bt[:,i,k,2])/d) if np.all(d>0) else None)
        results[arm]=row
    previous_name='selection_pilot.json' if args.pilot else 'selection.json'
    previous=json.loads((HERE.parent/'photometry_fixed_cell_20261106'/previous_name).read_text())
    for current,old in [('predicted_pass_rate','generated_pass_rate'),
                        ('measured_pass_rate','measured_pass_rate'),
                        ('reference_coefficient_error','selected_reference_coefficient_change')]:
        np.testing.assert_allclose(results['flow'][current]['value'],previous[old]['value'],rtol=0,atol=2e-12)
        if n>1:
            np.testing.assert_allclose(results['flow'][current]['case_se'],previous[old]['case_se'],rtol=0,atol=2e-12)
    paired={}
    for base in [1,2]:
        paired[arms[3]+'_minus_'+arms[base]]={q:ent(value[:,3]-value[:,base]) for q,value in quantities.items()}
    out={'test_cases':cases,'definition':'Disjoint-case diagnostic selection prediction; errors are measured minus predicted.',
         'arms':results,'paired_changes':paired,'folds':receipts,'bootstrap_seed':202611069301,
         'previous_selection_baseline_replay_passed':True,
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'limitations':['These are selection predictors, not replacement flows or calibrated responses.',
                        'The target conditions on actual usability in all four legs; conclusions do not automatically transfer to per-leg usability or the unconditional population.',
                        'Flow log loss omitted because its64-draw pass probabilities can be exactly0/1; squared error includes Monte Carlo integration variance.',
                        'Bootstrap fixes the fitted nuisance models; four development scenes cannot establish independent acceptance.',
                        'Additional-feature gain is predictive evidence, not a proof of information-theoretic insufficiency.']}
    name='pilot_report' if args.pilot else 'report'
    (ROOT/(name+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');(HERE/(name+'.json')).write_bytes((ROOT/(name+'.json')).read_bytes())
    np.savez(ROOT/(name+'.npz'),**quantities,cases=cases,arms=arms)
    def fmt(x):return 'not compared' if x is None else f"{x['value']:+.6f}"+(f" ± {x['case_se']:.6f}" if n>1 else ' (one test scene)')
    lines=['# Selection information diagnostic','',out['definition'],f'Test cases: {cases}.','',
           '| Predictor | Held-out log loss | Squared error | Reference coefficient error |',
           '|---|---:|---:|---:|']
    for arm,row in results.items():lines.append('| '+arm+' | '+' | '.join(fmt(row[q]) for q in ['soft_logloss','squared_error','reference_coefficient_error'])+' |')
    lines+=['']+['- '+s for s in out['limitations']]
    (HERE/('pilot_results.md' if args.pilot else 'results.md')).write_text('\n'.join(lines)+'\n')
    print('DONE: disjoint split, untouched test labels and flow baselines verified',flush=True)
    print(json.dumps(out['arms'],indent=2),flush=True)
if __name__=='__main__':main()
