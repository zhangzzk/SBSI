"""Paired own-data and four-scene joint comparisons at frozen endpoints."""
import importlib.util,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('moment_reporting_common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
r=c.module('moment_response_reduce',HERE.parent/'alignment_axis2_constant_20261105/report.py')
def bootstrap(n,seed):return np.vstack([np.ones(n),np.random.default_rng(seed).multinomial(n,np.full(n,1/n),10000),np.ones((n,n))-np.eye(n)])
def ent(v,n):
    jk=v[10001:];return {'value':v[0].tolist(),'case_se':v[1:10001].std(0,ddof=1).tolist(),'jackknife_se':np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0)).tolist()}
def own_report():
    cfg=c.manifest();out={'definition':'Measured minus generated selected coefficient, not a shear response. Whole-case SEs with fixed fits and sampled rows.','splits':{}}
    for split in ['train','validation']:
        with np.load(c.ROOT/f'own_{split}.npz') as z:raw=z['sums'];den=z['density'];names=list(z['arms']);cases=z['cases'];cuts=list(z['cuts'])
        n=len(cases);w=bootstrap(n,cfg['bootstrap_seed']);rows={}
        for kind,ki in [('hard',0),('soft',1)]:
            rows[kind]={}
            for pop,ss in [('pooled',raw[:,ki].sum(3)),('g0',raw[:,ki,:,:,0]),('g05',raw[:,ki,:,:,1])]:
                arms={};samples={}
                for ai,name in enumerate(names):
                    arms[name]={}
                    for ci,cut in enumerate(cuts):
                        t=w@ss[ai,0,:,ci];v={'score_error':t[:,2]/t[:,1]-t[:,4]/t[:,3],'pass_error_pp':100*(t[:,1]-t[:,3])/t[:,0]}
                        samples[name,cut]=v;arms[name][cut]={key:ent(value,n) for key,value in v.items()}
                        a,b=[w@ss[ai,bi,:,ci] for bi in [1,2]]
                        half_difference=(a[:,2]/a[:,1]-a[:,4]/a[:,3])-(b[:,2]/b[:,1]-b[:,4]/b[:,3])
                        arms[name][cut]['half_bank_score_error_difference']=ent(half_difference,n)
                changes={cut:{key:ent(samples['weighted',cut][key]-samples['control',cut][key],n) for key in ['score_error','pass_error_pp']} for cut in cuts}
                rows[kind][pop]={'arms':arms,'weighted_minus_control':changes}
        density={};density_samples={}
        for ai,name in enumerate(names):
            t=w@den[ai].sum(1);density_samples[name]=t[:,1]/t[:,0]
            density[name]=ent(density_samples[name],n)
        out['splits'][split]={'cases':cases.tolist(),'selections':rows,'joint_nll':density,
            'weighted_minus_control_joint_nll':ent(density_samples['weighted']-density_samples['control'],n)}
    c.write(c.ROOT/'own_report.json',out);c.write(HERE/'own_report.json',out)
    print('OWN_REPORT',json.dumps({split:{arm:out['splits'][split]['selections']['hard']['g0']['arms'][arm]['mag25_radius_gt0.60'] for arm in cfg['arms']} for split in ['train','validation']}),flush=True)
def main():
    assert os.environ.get('SLURM_JOB_ID');own_report();cfg=c.manifest();cases=cfg['cases'];n=len(cases);w=bootstrap(n,cfg['bootstrap_seed']);cuts=[x[0] for x in cfg['cuts']]
    training=json.loads((c.ROOT/'training_completed.json').read_text())
    out={'cases':cases,'definition':'DeltaR=simulation-model; m=100(Rsim_jj/Rmodel_jj-1). Whole-case paired bootstrap/jackknife, fixed fits; four development scenes. All three modes retain generated emulator photometry.','axes':{},'selection_association':{},'own_report':json.loads((c.ROOT/'own_report.json').read_text()),'numerical_receipts':[]}
    out['training_receipts']=training
    out['standalone_probes']={name:json.loads((c.ROOT/name/'probe.json').read_text()) for name in cfg['arms']}
    out['standalone_probes']['initial']=json.loads((c.ROOT/'initial_probe.json').read_text())
    center_samples={}
    for axis in [1,2]:
        arms={};samples={};ref=None
        for name in ['initial','previous_nine']+cfg['arms']:
            folder=(c.c.ROOT/'constant'/('neighbour_nll' if name=='initial' else 'neighbour_nine')/f'axis{axis}') if name in ['initial','previous_nine'] else c.ROOT/'constant'/name/f'axis{axis}'
            ss=[];mm=[];assoc=[]
            for case in cases:
                d=json.loads((folder/f'case{case}.json').read_text());assert d['simulation_replay_passed'] and d['true_selection_mass_exact']
                if name in cfg['arms']:
                    assert d['checkpoint_sha256']==training['arms'][name]['sha256']
                    assert d['source_sha256']==c.sha(HERE/'evaluate.py')
                    assert d['data_sha256']==c.sha(folder/f'case{case}.npz')
                    assert d['reference_score_replay_passed'] and d['pooled_emulator_roundoff_check_passed']
                    out['numerical_receipts'].append({'arm':name,'case':case,'axis':axis,
                        'pooled_emulator_first_block_max_abs':d['pooled_emulator_first_block_max_abs'],
                        'seconds_for_both_arms':d['seconds_for_both_arms']})
                with np.load(folder/f'case{case}.npz') as z:
                    np.testing.assert_array_equal(z['cuts'],cuts);ss.append(z['sim'].sum(-2));mm.append(z['model'].sum(-2));assoc.append(z['association'])
            sim,model=np.array(ss),np.array(mm)
            if ref is None:ref=sim
            else:np.testing.assert_allclose(sim,ref,rtol=1e-12,atol=1e-8)
            arms[name]={}
            for mode in range(3):
                rows={}
                for ci,cut in enumerate(cuts):
                    v=r.estimates(sim[:,:,ci],model[:,mode,:,ci],w,axis=axis);samples[name,mode,cut]=v
                    rows[cut]={key:ent(value,n) for key,value in v.items()}
                rms=np.sqrt(np.stack([samples[name,mode,cut]['m_percent']**2 for cut in cfg['target_cuts']],-1).mean(-1))
                rows['nine_cut_rms_m_percent']=ent(rms,n);arms[name][r.MODES[mode]]=rows
                center_samples[name,mode,axis]=samples[name,mode,'mag25_radius_gt0.60']
            if axis==1:
                t=w@np.array(assoc);out['selection_association'][name]={'score_error':ent(t[:,2]/t[:,1]-t[:,4]/t[:,3],n),'pass_error_pp':ent(100*(t[:,1]-t[:,3])/t[:,0],n)}
        changes={}
        for new,old in [('weighted','control'),('control','previous_nine'),('weighted','initial')]:
            changes[new+'_minus_'+old]={r.MODES[mode]:{cut:{key:ent(samples[new,mode,cut][key]-samples[old,mode,cut][key],n) for key in ['m_percent','delta_response','model_flow_response','model_emulator_response']} for cut in cfg['target_cuts']} for mode in range(3)}
        out['axes'][str(axis)]={'arms':arms,'paired_changes':changes}
    out['paired_axes_center']={}
    for name in ['initial','previous_nine']+cfg['arms']:
        out['paired_axes_center'][name]={}
        for mode in range(3):
            a,b=[center_samples[name,mode,axis] for axis in [1,2]]
            trace_sim=a['simulation_response'][:,0]+b['simulation_response'][:,1]
            trace_model=a['model_response'][:,0]+b['model_response'][:,1]
            out['paired_axes_center'][name][r.MODES[mode]]={
                'm2_minus_m1_pp':ent(b['m_percent']-a['m_percent'],n),
                'trace_m_percent':ent(100*(trace_sim/trace_model-1),n)}
    c.write(c.ROOT/'report.json',out);c.write(HERE/'report.json',out)
    def fmt(x):return f"{x['value']:+.4f} ± {x['case_se']:.4f}"
    lines=['# Nine-cut response plus selected-neighbour moment','',out['definition'],'','Target: radius>3px and magnitude<25. m in percent, SE in percentage points.','',
        '| Endpoint | Predicted usability m1 | Actual usability m1 | Predicted usability m2 | Actual usability m2 |','|---|---:|---:|---:|---:|']
    for name in ['initial','previous_nine']+cfg['arms']:
        values=[out['axes'][str(axis)]['arms'][name][r.MODES[mode]]['mag25_radius_gt0.60']['m_percent'] for axis in [1,2] for mode in [0,1]]
        lines.append('| '+name+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines+=['','Paired effect of the added moment in absolute response units (weighted minus control). Component changes are model contributions, not standalone component errors. Residual change is the negative sum of the two component changes.','',
        '| Axis | Usability / selection | Flow contribution change | Emulator contribution change | Residual change |',
        '|---|---|---:|---:|---:|']
    for axis in [1,2]:
        for mode,label in enumerate(['predicted / generated','actual / generated','actual / measured']):
            row=out['axes'][str(axis)]['paired_changes']['weighted_minus_control'][r.MODES[mode]]['mag25_radius_gt0.60']
            cells=[]
            for key in ['model_flow_response','model_emulator_response','delta_response']:
                v=row[key];cells.append(f"{v['value'][axis-1]:+.6f} ± {v['case_se'][axis-1]:.6f}")
            lines.append(f'| {axis} | {label} | '+' | '.join(cells)+' |')
    lines+=['','All endpoints were fixed before response evaluation. Full matrices, all nine cuts, standalone probes and paired changes are retained in JSON.']
    (HERE/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
