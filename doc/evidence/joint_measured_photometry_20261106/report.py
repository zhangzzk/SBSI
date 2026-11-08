"""Paired full coherent response, using each leg's measured photometry/cuts."""
import argparse,importlib.util,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_measured_photometry_20261106')
s=importlib.util.spec_from_file_location('measured_joint_reducer',HERE.parent/'alignment_axis2_constant_20261105/report.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def main():
    assert os.environ.get('SLURM_JOB_ID')
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['gate','four','full']);a=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text())
    if a.phase=='gate':
        receipt=json.loads((ROOT/'axis1/case1000.json').read_text())
        for key in ['measured_photometry_exact','selected_masses_exact','simulation_replay_passed','half_bank_identity_passed']:assert receipt[key]
        assert set(receipt['conditional_checks'])==set(manifest['arms'])
        with np.load(ROOT/'axis1/case1000.npz') as z:
            sim=z['sim'].sum(-2);model=z['model'].sum(-2);ci=list(z['cuts']).index('mag25_radius_gt0.60')
            values={name:r.estimates(sim[None,:,ci],model[None,i,0,:,ci],np.ones((1,1)),axis=1) for i,name in enumerate(z['arms'])}
        write(ROOT/'pilot_passed.json',receipt)
        print('PILOT_PASSED',flush=True)
        for name,v in values.items():print(name,'single-case m_percent',v['m_percent'].tolist(),'deltaR',v['delta_response'].tolist(),flush=True)
        return
    cases=manifest['pilot_cases'] if a.phase=='four' else manifest['cases'];n=len(cases);end=10001
    def weights(nn):return np.vstack([np.ones(nn),np.random.default_rng(manifest['bootstrap_seed']).multinomial(nn,np.full(nn,1/nn),10000),np.eye(nn),np.ones((nn,nn))-np.eye(nn)])
    def ent(x,nn):
        jk=x[end+nn:]
        return {'value':x[0].tolist(),'case_se':x[1:end].std(0,ddof=1).tolist(),
                'jackknife_se':np.sqrt((nn-1)/nn*np.square(jk-jk.mean(0)).sum(0)).tolist()}
    w=weights(n);names=list(manifest['arms']);cuts=[c[0] for c in manifest['cuts']]
    result={'phase':a.phase,'cases':cases,'definition':'Full coherent response with independent per-leg actual usability, measured photometry and selection. Conditional flow shapes plus original disk transport. DeltaR=simulation-model; m_percent=100(Rsim_jj/Rmodel_jj-1).',
        'uncertainty':'Paired whole-case bootstrap and jackknife; fits and measurement realizations fixed. Development evidence.',
        'axes':{},'generated_comparison_cases_frozen':manifest['generated_available_cases'],
        'interpretation':'Measured photometry changes under coherent shear. This intervention is not an additive causal error budget. Selection changes are already included; do not add the opposite-group selection term.'}
    for axis in [1,2]:
        sims=[];models=[];receipts=[]
        for case in cases:
            receipt=json.loads((ROOT/f'axis{axis}/case{case}.json').read_text())
            assert receipt['simulation_replay_passed'] and receipt['selected_masses_exact']
            receipts.append(receipt)
            with np.load(ROOT/f'axis{axis}/case{case}.npz') as z:
                np.testing.assert_array_equal(z['cuts'],cuts);np.testing.assert_array_equal(z['arms'],names)
                sims.append(z['sim'].sum(-2));models.append(z['model'].sum(-2))
        sim=np.array(sims);model=np.array(models);arms={};samples={};comparisons={}
        for mi,name in enumerate(names):
            rows={}
            for k,cut in enumerate(cuts):
                v=r.estimates(sim[:,:,k],model[:,mi,0,:,k],w,axis=axis);samples[name,cut]=v
                row={key:ent(value,n) for key,value in v.items()}
                row['per_case_m_percent']=v['m_percent'][end:end+n].tolist()
                row['m_percent_ci95']=np.quantile(v['m_percent'][1:end],[.025,.975]).tolist()
                halves=[r.estimates(sim[:,:,k],model[:,mi,b,:,k],w,axis=axis) for b in [1,2]]
                row['half_bank_difference']={key:ent(halves[0][key]-halves[1][key],n) for key in ['m_percent','model_response']}
                rows[cut]=row
            ts=np.stack([samples[name,cut]['m_percent'] for cut in manifest['target_cuts']],-1)
            rows['nine_cut_rms_m_percent']=ent(np.sqrt(np.square(ts).mean(-1)),n);arms[name]=rows
            common=[case for case in cases if case in manifest['generated_available_cases'][name]]
            if len(common)<2:continue
            ix=[cases.index(case) for case in common];ww=weights(len(common));generated=[]
            for case in common:
                with np.load(Path(manifest['generated_roots'][name])/f'axis{axis}/case{case}.npz') as z:
                    np.testing.assert_array_equal(z['cuts'],cuts)
                    np.testing.assert_allclose(z['sim'].sum(-2),sim[cases.index(case)],rtol=1e-12,atol=1e-8)
                    generated.append(z['model'][1].sum(-2))
            generated=np.array(generated);comparison={}
            for k,cut in enumerate(cuts):
                gv=r.estimates(sim[ix,:,k],generated[:,:,k],ww,axis=axis)
                mv=r.estimates(sim[ix,:,k],model[ix,mi,0,:,k],ww,axis=axis)
                comparison[cut]={'generated_photometry_generated_selection':{key:ent(value,len(common)) for key,value in gv.items()},
                    'measured_photometry_measured_selection':{key:ent(value,len(common)) for key,value in mv.items()},
                    'measured_minus_generated':{key:ent(mv[key]-gv[key],len(common)) for key in ['m_percent','delta_response','model_flow_response','model_emulator_response']}}
            comparisons[name]={'cases':common,'usability':'actual in both predictions, independently per leg','cuts':comparison}
        paired={cut:{key:ent(samples['nine_weak',cut][key]-samples['baseline',cut][key],n) for key in ['m_percent','delta_response','model_flow_response','model_emulator_response']} for cut in cuts}
        support={}
        for name in names:
            entries=[leg for receipt in receipts for leg in receipt['representable_disk_boundary_audit'][name]]
            support[name]={key:sum(entry[key] for entry in entries) for key in ['draws','rounded_boundary','selected_draws','selected_rounded_boundary']}
            support[name]['max_squared_radius']=max(entry['max_squared_radius'] for entry in entries)
        result['axes'][str(axis)]={'arms':arms,'paired_nine_minus_baseline':paired,'generated_comparisons':comparisons,
            'representable_disk_boundary_audit':support,
            'selected_counts_by_case_leg':sim[:,:,cuts.index('mag25_radius_gt0.60'),0].tolist(),
            'measurement_matching_by_case':[{'case':case,'legs':receipt['measured']} for case,receipt in zip(cases,receipts)],
            'evaluator_source_sha256':sorted({receipt['source_sha256'] for receipt in receipts})}
    write(ROOT/f'{a.phase}_report.json',result);write(HERE/f'{a.phase}_report.json',result)
    def fmt(d):return f"{d['value']:+.5f} ± {d['case_se']:.5f}"
    lines=['# Full joint response with measured photometry and selection','',f'{n} scenes; actual usability and cuts independently per shear leg.','',
        'Target: radius>3 pixels and magnitude<25. Absolute response error is simulation minus model. m is in percent; its SE is in percentage points.','',
        '| Flow | Delta R11 | m1 (%) | Delta R22 | m2 (%) |','|---|---:|---:|---:|---:|']
    for name in names:
        cells=[]
        for axis in [1,2]:
            row=result['axes'][str(axis)]['arms'][name]['mag25_radius_gt0.60'];d=row['delta_response']
            cells+=[fmt({key:value[axis-1] for key,value in d.items()}),fmt(row['m_percent'])]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','Matched comparison against generated photometry and generated selection (actual usability in both):','',
            '| Flow | Axis | Cases | Generated m (%) | Measured-input m (%) | Paired change (pp) |','|---|---|---:|---:|---:|---:|']
    for axis in [1,2]:
        for name,comparison in result['axes'][str(axis)]['generated_comparisons'].items():
            row=comparison['cuts']['mag25_radius_gt0.60']
            lines.append(f"| {name} | {axis} | {len(comparison['cases'])} | "+' | '.join(fmt(row[key]['m_percent']) for key in ['generated_photometry_generated_selection','measured_photometry_measured_selection','measured_minus_generated'])+' |')
    lines+=['','Full matrices, all nine cuts, per-case values, jackknife errors and 32-draw subbank differences are in the JSON report.',
            result['interpretation'],result['uncertainty']]
    (HERE/f'{a.phase}_results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
