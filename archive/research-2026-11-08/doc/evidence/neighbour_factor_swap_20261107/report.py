"""Paired factorial effects of frozen photometry and conditional-shape changes."""
import importlib.util,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('swap_reporting_common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
r=c.module('swap_response_reduce',HERE.parent/'alignment_axis2_constant_20261105/report.py')
def entry(v,n):
    jack=v[10001:]
    return {'value':v[0].tolist(),'case_se':v[1:10001].std(0,ddof=1).tolist(),
            'jackknife_se':np.sqrt((n-1)/n*np.square(jack-jack.mean(0)).sum(0)).tolist()}
def main():
    assert os.environ.get('SLURM_JOB_ID');cfg=c.manifest();cases=cfg['cases'];n=len(cases)
    w=np.vstack([np.ones(n),np.random.default_rng(cfg['bootstrap_seed']).multinomial(n,np.full(n,1/n),10000),np.ones((n,n))-np.eye(n)])
    assembly=json.loads((c.ROOT/'assembly_completed.json').read_text());cuts=[x[0] for x in cfg['cuts']]
    out={'definition':'Fixed-factor intervention effects; not standalone component errors. DeltaR=simulation-model; m=100(Rsim_jj/Rmodel_jj-1). Four reused development scenes; paired whole-case SEs conditional on fitted models. All modes use generated emulator photometry.',
         'cases':cases,'recipes':cfg['recipes'],'assembly':assembly,'axes':{},'own_probes':{},'audits':[]}
    for name in cfg['arms']:out['own_probes'][name]=json.loads((c.ROOT/name/'probe.json').read_text())
    for axis in [1,2]:
        samples={};arms={};reference=None
        for case in cases:
            a=json.loads((c.ROOT/f'audit_case{case}_axis{axis}.json').read_text())
            assert a['same_photo_selection_exact'] and a['source_sha256']==c.sha(HERE/'evaluate.py')
            out['audits'].append({'case':case,'axis':axis,**a})
        for name in cfg['arms']:
            sim=[];model=[]
            for case in cases:
                path=c.ROOT/'constant'/name/f'axis{axis}'/f'case{case}.npz';d=json.loads(path.with_suffix('.json').read_text())
                assert d['checkpoint_sha256']==assembly['arms'][name]['sha256'] and d['data_sha256']==c.sha(path)
                assert d['source_sha256']==c.sha(HERE/'evaluate.py')
                assert all(d[k] for k in ['simulation_replay_passed','reference_score_replay_passed','true_selection_mass_exact','pooled_emulator_roundoff_check_passed'])
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['cuts'],cuts);sim.append(z['sim'].sum(-2));model.append(z['model'].sum(-2))
            sim=np.array(sim);model=np.array(model)
            if reference is None:reference=sim
            else:np.testing.assert_array_equal(sim,reference)
            arms[name]={}
            for mode,label in enumerate(r.MODES):
                arms[name][label]={}
                for ci,cut in enumerate(cuts):
                    v=r.estimates(sim[:,:,ci],model[:,mode,:,ci],w,axis=axis);samples[name,mode,cut]=v
                    arms[name][label][cut]={key:entry(value,n) for key,value in v.items()}
        factorial={}
        for contrast in cfg['factorial_comparisons']:
            result={}
            for mode,label in enumerate(r.MODES):
                result[label]={}
                for cut in cuts:
                    aa,bb,ba,ab=[samples[contrast[key],mode,cut] for key in ['old','new','photo_only','shape_only']]
                    terms={}
                    for key in ['model_response','model_flow_response','model_emulator_response','delta_response','m_percent']:
                        values={'photo_at_old_shape':ba[key]-aa[key],'shape_at_old_photo':ab[key]-aa[key],
                            'interaction':bb[key]-ba[key]-ab[key]+aa[key],'total':bb[key]-aa[key],
                            'photo_at_new_shape':bb[key]-ab[key],'shape_at_new_photo':bb[key]-ba[key]}
                        np.testing.assert_allclose(values['photo_at_old_shape']+values['shape_at_old_photo']+values['interaction'],values['total'],rtol=1e-10,atol=1e-12)
                        terms[key]={term:entry(value,n) for term,value in values.items()}
                    result[label][cut]=terms
            factorial[contrast['name']]=result
        out['axes'][str(axis)]={'arms':arms,'factorial':factorial}
    c.write(c.ROOT/'report.json',out);c.write(HERE/'report.json',out)
    cut='mag25_radius_gt0.60';mode='true_usability_generated_selection'
    lines=['# Fixed-factor constant-shear comparison','',out['definition'],'',
        'Target: radius>3px, magnitude<25; actual usability and generated selection. m in percent, errors in percentage points.','',
        '| Product | m1 | m2 |','|---|---:|---:|']
    for name in cfg['arms']:
        cells=[]
        for axis in [1,2]:
            v=out['axes'][str(axis)]['arms'][name][mode][cut]['m_percent'];cells.append(f"{v['value']:+.4f} ± {v['case_se']:.4f}")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','Changes in the flow contribution, in absolute response units:','',
        '| Contrast | Axis | Photometry at old shapes | Shapes at old photometry | Interaction | Total |','|---|---|---:|---:|---:|---:|']
    for contrast in cfg['factorial_comparisons']:
        for axis in [1,2]:
            row=out['axes'][str(axis)]['factorial'][contrast['name']][mode][cut]['model_flow_response'];cells=[]
            for key in ['photo_at_old_shape','shape_at_old_photo','interaction','total']:
                v=row[key];cells.append(f"{v['value'][axis-1]:+.6f} ± {v['case_se'][axis-1]:.6f}")
            lines.append(f"| {contrast['name']} | {axis} | "+' | '.join(cells)+' |')
    lines+=['','All cuts, flag modes, emulator/total response effects and jackknife SEs are retained in report.json. No model promotion or new training.']
    (HERE/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
