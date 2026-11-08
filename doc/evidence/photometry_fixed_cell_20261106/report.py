"""Paired case contrasts for original and fixed-cell emulator coefficients."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_fixed_cell_20261106')
PREVIOUS=ROOT.parent/'photometry_response_bins_20261106'
spec=importlib.util.spec_from_file_location('bins_report',HERE.parent/'photometry_response_bins_20261106/report.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    cases=[1000] if args.pilot else [1000,1040,1080,1120];n=len(cases);hist=[];receipts=[]
    for case in cases:
        receipts.append(json.loads((ROOT/f'case{case}.json').read_text()));assert receipts[-1]['original_histogram_replay_passed']
        with np.load(ROOT/f'case{case}.npz') as z:hist.append(z['sums'])
    hist=np.array(hist);end=10001
    w=np.vstack([np.ones(n),np.random.default_rng(202611069301).multinomial(n,np.full(n,1/n),10000),np.eye(n)])
    if n>1:w=np.vstack([w,np.ones((n,n))-np.eye(n)])
    unique,inverse=np.unique(w,axis=0,return_inverse=True)
    def ent(x):
        jk=x[end+n:]
        return {'value':float(x[0]),'case_se':float(x[1:end].std(ddof=1)) if n>1 else None,
                'jackknife_se':float(np.sqrt((n-1)/n*np.square(jk-jk.mean()).sum())) if n>1 else None}
    name='pilot_report' if args.pilot else 'report';results={};arrays={}
    with np.load(PREVIOUS/(name+'.npz')) as old:
        np.testing.assert_array_equal(old['cases'],cases)
        for stride,label in [(1,'fine'),(2,'medium'),(4,'coarse')]:
            for include_size in [False,True]:
                key=label+('_with_intrinsic_size' if include_size else '')
                h=b.coarsen(hist,stride,include_size);t=np.einsum('wc,cbepq->wbepq',unique,h)
                vals={}
                for q,arm in [(1,'original'),(2,'fixed_photometry')]:
                    v=t.copy();v[...,2]=t[...,q]
                    vals[arm]={term:x[inverse] for term,x in b.decompose(v,2).items()}
                terms=list(vals['original'])
                reference=old[key+'_scalar_coefficient']
                np.testing.assert_allclose(np.column_stack(list(vals['original'].values())),reference,rtol=0,atol=2e-11)
                difference={term:vals['original'][term]-vals['fixed_photometry'][term] for term in terms}
                result={arm:{term:ent(x) for term,x in v.items()} for arm,v in vals.items()}
                result['original_minus_fixed']={term:ent(x) for term,x in difference.items()}
                result['per_case']={str(case):{arm:{term:float(x[end+i]) for term,x in v.items()}
                    for arm,v in vals.items()} for i,case in enumerate(cases)}
                results[key]=result
                for arm,v in vals.items():arrays[key+'_'+arm]=np.column_stack(list(v.values()))
    out={'cases':cases,'bootstrap_seed':202611069301,'results':results,'receipts':receipts,
         'report_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'definition':'Measured minus generated selection-weighted scalar opposite-neighbour coefficient; original or fixed fine-cell photometry.',
         'limitations':['Fixed-cell inputs are diagnostic substitutions, not corrected predictions.',
                        'Fine bins hold photometry exactly fixed; coarse aggregates contain multiple fixed fine-cell values.',
                        'Truth-context differences include target properties and neighbour properties.',
                        'Fixed-model development-scene uncertainties; no training uncertainty or independent acceptance.']}
    (ROOT/(name+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (HERE/(name+'.json')).write_bytes((ROOT/(name+'.json')).read_bytes())
    np.savez(ROOT/(name+'.npz'),**arrays,cases=cases,quantities=terms)
    def fmt(x):return f"{x['value']:+.6f}"+(f" ± {x['case_se']:.6f}" if n>1 else ' (one scene)')
    lines=['# Fixed-photometry response audit','',out['definition'],f'Cases: {cases}.','',
           '| Fine-bin term | Original photometry | Exactly fixed photometry | Original minus fixed |',
           '|---|---:|---:|---:|']
    fine=results['fine']
    for term in ['distribution','within_bin','exclusive','diagonal_change']:
        lines.append('| '+term+' | '+' | '.join(fmt(fine[arm][term]) for arm in ['original','fixed_photometry','original_minus_fixed'])+' |')
    lines+=['','| Scene | Original total contrast | Fixed total contrast | Original within-bin | Fixed within-bin |',
            '|---|---:|---:|---:|---:|']
    for case,row in fine['per_case'].items():
        lines.append('| '+case+' | '+' | '.join(f"{row[arm][term]:+.6f}" for term in ['diagonal_change','within_bin'] for arm in ['original','fixed_photometry'])+' |')
    lines+=['']+['- '+s for s in out['limitations']]
    (HERE/('pilot_results.md' if args.pilot else 'results.md')).write_text('\n'.join(lines)+'\n')
    print('DONE: original coefficient replays in every resample; all decompositions pass',flush=True)
    print(json.dumps(results['fine'],indent=2),flush=True)

if __name__=='__main__':main()
