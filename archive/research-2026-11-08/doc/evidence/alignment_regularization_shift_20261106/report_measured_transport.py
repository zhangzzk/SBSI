"""Paired scene bootstrap of measured-anchor transport and selection terms."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106')
NAMES=['simulation_shape','simulation_selection','simulation_total','anchor_plus','anchor_minus',
       'anchored_model_shape','linear_model_shape','shape_error','total_error',
       'anchor_plus_minus_difference','finite_minus_linear']

def ent(x,n):
    return {'value':float(x[0]),'case_se':float(x[1:].std(ddof=1)) if n>1 else None}

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--conditional',choices=['baseline','nine_weak']);args=parser.parse_args()
    cases=[1000] if args.pilot else [1000,1040,1080,1120];n=len(cases)
    sim=[];edges=[];pred=[];receipts=[]
    for case in cases:
        directory=ROOT/'conditional_transport'/args.conditional if args.conditional else ROOT/'measured_transport'
        base=directory/f'case{case}'
        receipts.append(json.loads(base.with_suffix('.json').read_text()))
        assert receipts[-1]['measured_replay_passed']
        with np.load(base.with_suffix('.npz')) as z:
            assert int(z['case'])==case
            sim.append(z['sim']);edges.append(z['edges']);pred.append(z['pred']);cuts=list(z['cuts'])
    sim=np.array(sim);edges=np.array(edges);pred=np.array(pred)
    w=np.vstack([np.ones(n),np.random.default_rng(202611069301).multinomial(n,np.full(n,1/n),10000)])
    out={'cases':cases,'definition':'Common-all-four usability. Exact measured edge shape/selection terms versus unchanged emulator anchored to measured endpoints. Equal group average uses separately normalized A/B responses; it is not the coherent total.',
         'limitations':['Conditional diagnostic uses observed endpoint information; not a generative prediction.',
                        'Four development scenes; fixed-model case SE. Anchor averaging is reported alongside each endpoint.',
                        'Selection and photometry population differ from generated-selection evaluation.'],
         'bootstrap_seed':202611069301,'cuts':{},'receipts':receipts}
    out['input_mode']=('Conditional flow shapes at measured photometry: '+args.conditional
                       if args.conditional else 'Measured shape and photometry')
    if args.conditional:
        out['definition']='Common-all-four usability; conditional flow shape draws at measured endpoint photometry. Equal average of within-group opposite responses. Conditional diagnostic, not coherent total.'
    boot=[]
    edge_states=[(0,2),(1,3),(0,1),(2,3)]
    for k,cut in enumerate(cuts):
        ss=np.einsum('bc,csgpq->bsgpq',w,sim[:,:,k])
        ed=np.einsum('bc,cegpq->begpq',w,edges[:,k]);pr=np.einsum('bc,cegpq->begpq',w,pred[:,k])
        count=ss[...,0].sum(-1);mean=ss[...,1].sum(-1)/count
        groups=[];result={}
        for g in [0,1]:
            raw=np.zeros((len(w),5,len(NAMES)))
            for edge in ([2,3] if g==0 else [0,1]):
                a,b=edge_states[edge];dbar=.5*(count[:,a,g]+count[:,b,g]);fbar=.5*(mean[:,a,g]+mean[:,b,g])
                shape=ed[:,edge,g,:,0]/dbar[:,None]
                selection=(ed[:,edge,g,:,1]-fbar[:,None]*(ss[:,a,g,:,0]-ss[:,b,g,:,0]))/dbar[:,None]
                predicted=pr[:,edge,g]/dbar[:,None,None]
                avg=.5*(predicted[...,0]+predicted[...,1])
                row=np.stack([shape,selection,shape+selection,predicted[...,0],predicted[...,1],avg,
                    predicted[...,2],shape-avg,shape+selection-avg,predicted[...,0]-predicted[...,1],avg-predicted[...,2]],-1)
                np.testing.assert_allclose((shape+selection).sum(-1),mean[:,a,g]-mean[:,b,g],atol=2e-12,rtol=0)
                raw+=row/.08
            groups.append(raw)
            result['AB'[g]]={name:ent(raw[...,i].sum(-1),n) for i,name in enumerate(NAMES)}
            result['AB'[g]]['size_contributions']={name:[ent(raw[:,b,i],n) for b in range(5)] for i,name in enumerate(NAMES)}
        raw=.5*(groups[0]+groups[1]);total=raw.sum(1)
        boot.append(total)
        np.testing.assert_allclose(total[:,7]+total[:,1],total[:,8],atol=2e-12,rtol=0)
        result['equal_group_average']={name:ent(total[:,i],n) for i,name in enumerate(NAMES)}
        result['equal_group_size_contributions']={name:[ent(raw[:,b,i],n) for b in range(5)] for i,name in enumerate(NAMES)}
        out['cuts'][cut]=result
    prefix=('conditional_transport_'+args.conditional) if args.conditional else 'measured_transport'
    dest=ROOT/(prefix+('_pilot' if args.pilot else '')+'_report.json')
    dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    np.savez(dest.with_suffix('.npz'),values=np.array(boot),quantities=NAMES,cases=cases,cuts=cuts,seed=202611069301)
    print('DONE',n,'measured-anchor cases; all shape/selection identities passed',flush=True)
    print(json.dumps(out['cuts']['mag25_radius_gt0.60']['equal_group_average'],indent=2),flush=True)

if __name__=='__main__':main()
