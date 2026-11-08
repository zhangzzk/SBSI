"""Paired whole-scene factorial contrasts, with exact replay controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_selection_factorial_20261106')
PREVIOUS=ROOT.parent/'alignment_regularization_shift_20261106'
EDGES=[(0,2),(1,3),(0,1),(2,3)]

def effects(x):
    gg,gm,mg,mm=x[:,0,0],x[:,0,1],x[:,1,0],x[:,1,1]
    value=.5*((mg-gg)+(mm-gm));membership=.5*((gm-gg)+(mm-mg))
    np.testing.assert_allclose(value+membership,mm-gg,rtol=0,atol=2e-12)
    return {'photometry_first':mg-gg,'photometry_after_membership':mm-gm,
            'membership_first':gm-gg,'membership_after_photometry':mm-mg,
            'symmetric_photometry':value,'symmetric_membership':membership,
            'interaction':mm-mg-gm+gg,'diagonal_change':mm-gg}

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    cases=[1000] if args.pilot else [1000,1040,1080,1120];n=len(cases)
    values=[];sim=[];edges=[];old=[];receipts=[]
    for case in cases:
        receipts.append(json.loads((ROOT/f'case{case}.json').read_text()))
        assert receipts[-1]['generated_replay_passed']
        with np.load(ROOT/f'case{case}.npz') as z:
            values.append(z['sums']);sim.append(z['sim']);edges.append(z['edges']);cuts=list(z['cuts'])
        with np.load(PREVIOUS/'fourstate/nine_weak'/f'case{case}.npz') as z:old.append(z['model'][2])
    values=np.array(values);sim=np.array(sim);edges=np.array(edges);old=np.array(old)
    boot_end=10001
    w=np.vstack([np.ones(n),np.random.default_rng(202611069301).multinomial(n,np.full(n,1/n),10000),np.eye(n)])
    if n>1:w=np.vstack([w,np.ones((n,n))-np.eye(n)])
    def ent(x):
        jk=x[boot_end+n:]
        return {'value':float(x[0]),'case_se':float(x[1:boot_end].std(ddof=1)) if n>1 else None,
                'jackknife_se':float(np.sqrt((n-1)/n*np.square(jk-jk.mean()).sum())) if n>1 and len(jk)==n else None}
    out={'cases':cases,'definition':'Weak nine-cut flow, common actual usability; equal A/B within-group opposite response. Errors=simulation-model; fixed-model case SE.',
         'bootstrap_seed':202611069301,'report_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cuts':{},'receipts':receipts,
         'limitations':['Photometry swaps also condition shape on the substituted values.',
                        'Measured masks on independent flow draws are diagnostics, not corrected predictions.',
                        'Symmetric effects are algebraic model contrasts; interaction and both orders are retained.',
                        'Measured neighbour selection response is separate; these contributions cannot be doubled into coherent full-population errors.']}
    all_responses=[]
    for k,cut in enumerate(cuts):
        # Full-bank size numerators are retained with a common group denominator.
        by_size=np.einsum('bc,cimegp->bimegp',w,values[:,:,:,k,...,0,1])
        t=np.einsum('bc,cimegvq->bimegvq',w,values[:,:,:,k].sum(5))
        ss=np.einsum('bc,csgq->bsgq',w,sim[:,:,k].sum(-2))
        ed=np.einsum('bc,cegq->begq',w,edges[:,k].sum(-2))
        oo=np.einsum('bc,csgq->bsgq',w,old[:,:,k])
        response=np.zeros((len(w),2,2,3));size_response=np.zeros((len(w),2,2,5))
        properties=np.zeros((len(w),2,2,3));selection=np.zeros(len(w));shape=np.zeros(len(w));reference=np.zeros(len(w))
        for edge,(s,u) in enumerate(EDGES):
            g=1 if edge<2 else 0
            mass=t[:,:,:,edge,g,:,0]
            response+=t[:,:,:,edge,g,:,1]/mass/.16
            size_response+=by_size[:,:,:,edge,g]/mass[...,0,None]/.16
            for j,denom in [(2,5),(3,6),(4,0)]:
                properties[...,j-2]+=t[:,:,:,edge,g,0,j]/t[:,:,:,edge,g,0,denom]/4
            dbar=.5*(ss[:,s,g,0]+ss[:,u,g,0]);fbar=.5*(ss[:,s,g,1]/ss[:,s,g,0]+ss[:,u,g,1]/ss[:,u,g,0])
            sh=ed[:,edge,g,0]/dbar;sel=(ed[:,edge,g,1]-fbar*(ss[:,s,g,0]-ss[:,u,g,0]))/dbar
            np.testing.assert_allclose(sh+sel,ss[:,s,g,1]/ss[:,s,g,0]-ss[:,u,g,1]/ss[:,u,g,0],rtol=0,atol=2e-12)
            shape+=sh/.16;selection+=sel/.16
            reference+=(oo[:,s,g,3]/oo[:,s,g,0]-oo[:,u,g,3]/oo[:,u,g,0])/.16
        np.testing.assert_allclose(response[:,0,0,0],reference,rtol=0,atol=2e-11)
        np.testing.assert_allclose(size_response.sum(-1),response[...,0],rtol=0,atol=2e-12)
        if cut=='uncut':np.testing.assert_allclose(response[:,:,0],response[:,:,1],rtol=0,atol=2e-12)
        all_responses.append(response)
        rows={}
        for inp in range(2):
            for mask in range(2):
                key='GM'[inp]+'_inputs_'+'GM'[mask]+'_selection';r=response[:,inp,mask,0]
                rows[key]={'model_shape_response':ent(r),'shape_error':ent(shape-r),'error_including_selection':ent(shape+selection-r),
                           'antithetic_bank_difference':float(response[0,inp,mask,1]-response[0,inp,mask,2]),
                           'mean_log_radius':ent(properties[:,inp,mask,0]),'mean_magnitude':ent(properties[:,inp,mask,1]),
                           'mean_opposite_coefficient':ent(properties[:,inp,mask,2])}
        previous_name='conditional_transport_nine_weak'+('_pilot' if args.pilot else '')+'_report.npz'
        with np.load(PREVIOUS/previous_name) as z:
            np.testing.assert_array_equal(z['cases'],cases);assert int(z['seed'])==202611069301
            earlier=z['values'][list(z['cuts']).index(cut),:,list(z['quantities']).index('anchored_model_shape')]
        result={'simulation_shape':ent(shape),'simulation_selection':ent(selection),'branches':rows,
                'effects':{key:ent(v) for key,v in effects(response[...,0]).items()},
                'MM_change_from_previous_draw_bank':ent(response[:boot_end,1,1,0]-earlier),
                'coefficient_effects':{key:ent(v) for key,v in effects(properties[...,2]).items()},
                'transport_geometry_effects':{key:ent(v) for key,v in effects(response[...,0]-properties[...,2]).items()},
                'bank_effect_differences':{key:float(v[0]-effects(response[...,2])[key][0]) for key,v in effects(response[...,1]).items()},
                'per_case':{str(case):{'shape':float(shape[boot_end+i]),'selection':float(selection[boot_end+i]),
                    'responses':response[boot_end+i,:,:,0].tolist(),
                    'effects':{key:float(v[boot_end+i]) for key,v in effects(response[...,0]).items()}}
                    for i,case in enumerate(cases)},
                'size_contributions':{key:[ent(v[:,b]) for b in range(5)] for key,v in effects(size_response).items()}}
        for key,arr in effects(response[...,0]).items():
            np.testing.assert_allclose(arr,effects(properties[...,2])[key]+effects(response[...,0]-properties[...,2])[key],rtol=0,atol=2e-12)
        out['cuts'][cut]=result
    name='pilot_report' if args.pilot else 'report'
    (ROOT/(name+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    all_responses=np.array(all_responses)
    np.savez(ROOT/(name+'.npz'),response=all_responses[:,:boot_end],case_response=all_responses[:,boot_end:boot_end+n],
             leave_one_case_out_response=all_responses[:,boot_end+n:],cuts=cuts,cases=cases,seed=202611069301)
    (HERE/(name+'.json')).write_bytes((ROOT/(name+'.json')).read_bytes())
    def fmt(d):
        return f"{d['value']:+.6f}"+(f" ± {d['case_se']:.6f}" if d['case_se'] is not None else ' (one scene)')
    target=out['cuts']['mag25_radius_gt0.60']
    lines=['# Photometry and selection factorial','',out['definition'],
           f"Cases: {cases}. Target radius>3pixels, magnitude<25.",'',
           '| Inputs / selection | Model shape response | Shape error | Error including selection |',
           '|---|---:|---:|---:|']
    for key,row in target['branches'].items():
        lines.append('| '+key+' | '+' | '.join(fmt(row[q]) for q in ['model_shape_response','shape_error','error_including_selection'])+' |')
    lines+=['','G=generated; M=measured. Crossed branches are diagnostic substitutions.','',
            '| Change in model response | Value |','|---|---:|']
    for key,value in target['effects'].items():lines.append('| '+key+' | '+fmt(value)+' |')
    lines+=['','Measured selection response: '+fmt(target['simulation_selection'])+'.',
            'M/M change from the earlier draw bank: '+fmt(target['MM_change_from_previous_draw_bank'])+'.','']
    lines+=['## Coefficient and scene controls','',
            'Mean scalar emulator coefficient change: '+fmt(target['coefficient_effects']['diagonal_change'])+'.',
            'Remaining finite shape-transport geometry change: '+fmt(target['transport_geometry_effects']['diagonal_change'])+'.','',
            '| Scene | G/G response | M/M response | Diagonal change | Interaction |',
            '|---|---:|---:|---:|---:|']
    for case,row in target['per_case'].items():
        lines.append(f"| {case} | {row['responses'][0][0]:+.6f} | {row['responses'][1][1]:+.6f} | {row['effects']['diagonal_change']:+.6f} | {row['effects']['interaction']:+.6f} |")
    if n>1:
        lines+=['',f"Delete-one-scene jackknife SEs: diagonal {target['effects']['diagonal_change']['jackknife_se']:.6f}, interaction {target['effects']['interaction']['jackknife_se']:.6f}. These supplement the bootstrap SEs; neither includes model-training uncertainty."]
    lines+=['',f"Antithetic half-bank difference in diagonal change: {target['bank_effect_differences']['diagonal_change']:+.6f}.",'']
    lines+=['- '+s for s in out['limitations']]
    (HERE/('pilot_results.md' if args.pilot else 'results.md')).write_text('\n'.join(lines)+'\n')
    print('DONE',n,'cases; original response replay and all factorial identities passed',flush=True)
    print(json.dumps(out['cuts']['mag25_radius_gt0.60'],indent=2),flush=True)

if __name__=='__main__':main()
