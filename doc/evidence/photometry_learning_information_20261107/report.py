"""Four-way, paired reductions with the original response estimator."""
import importlib.util,json,os
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('factorial_report_setup',H/'common.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
base=s.bound('report')
CONTRASTS={'loss_only':[-1,1,0,0],'input_only':[-1,0,1,0],'loss_with_input':[0,0,-1,1],'input_with_loss':[0,-1,0,1],'both_vs_control':[-1,0,0,1],'interaction':[1,-1,-1,1]}
def combine(values,coefficients):
    assert len(values)==len(coefficients)==4
    return sum(a*b for a,b in zip(coefficients,values))
def own_report():
    base.own_report()
    s.load('factorial_probability_report',H/'probability_report.py').main()
    # Enrich the original own-data reducer with all four factorial contrasts.
    cfg=s.manifest();result=json.loads((s.ROOT/'own_report.json').read_text())
    for split in ['train','validation']:
        with np.load(s.ROOT/f'own_{split}.npz') as z:raw=z['sums'];density=z['density'];cases=z['cases'];cuts=list(z['cuts']);arms=list(z['arms'])
        assert arms==cfg['arms'];n=len(cases);weights=base.bootstrap(n,cfg['bootstrap_seed'])
        result['splits'][split]['factorial']={}
        for kind,ki in [('hard',0),('soft',1)]:
            populations={}
            for population,ss in [('pooled',raw[:,ki].sum(3)),('g0',raw[:,ki,:,:,0]),('g05',raw[:,ki,:,:,1])]:
                contrasts={}
                for label,coefs in CONTRASTS.items():
                    contrasts[label]={}
                    for k,cut in enumerate(cuts):
                        vals=[]
                        for ai in range(4):
                            t=weights@ss[ai,0,:,k];vals.append({'score_error':t[:,2]/t[:,1]-t[:,4]/t[:,3],'pass_error_pp':100*(t[:,1]-t[:,3])/t[:,0]})
                        contrasts[label][cut]={key:base.ent(combine([v[key] for v in vals],coefs),n) for key in vals[0]}
                populations[population]=contrasts
            result['splits'][split]['factorial'][kind]=populations
    s.c.write(s.ROOT/'own_report.json',result);s.c.write(H/'own_report.json',result)
def main():
    assert os.environ.get('SLURM_JOB_ID');s.verify();own_report();cfg=s.manifest();cases=cfg['cases'];arms=cfg['arms'];n=len(cases);weights=base.bootstrap(n,cfg['bootstrap_seed']);cuts=[x[0] for x in cfg['cuts']]
    training=json.loads((s.ROOT/'training_completed.json').read_text())
    out={'cases':cases,'arms':cfg['arm_definition'],'definition':'m=100*(Rsim_jj/Rmodel_jj-1); errors from paired whole-case bootstrap/jackknife with fits fixed. Four reused development scenes; no independent acceptance.','axes':{},'training':training,'initial_checkpoint':cfg['initial_checkpoint'],'initial_replay':json.loads((s.ROOT/'initial_replay.json').read_text()),'contrasts':CONTRASTS}
    central={}
    for axis in [1,2]:
        sample={};arm_reports={};reference=None
        for arm in arms:
            ss=[];mm=[]
            for case in cases:
                f=s.ROOT/'constant'/arm/f'axis{axis}'/f'case{case}.npz';receipt=json.loads(f.with_suffix('.json').read_text())
                assert receipt['checkpoint_sha256']==training['arms'][arm]['sha256'] and receipt['source_sha256']==s.c.sha(H/'evaluate.py') and receipt['data_sha256']==s.c.sha(f)
                assert all(receipt[k] for k in ['simulation_replay_passed','reference_score_replay_passed','true_selection_mass_exact','pooled_emulator_roundoff_check_passed'])
                with np.load(f) as z:np.testing.assert_array_equal(z['cuts'],cuts);ss.append(z['sim'].sum(-2));mm.append(z['model'].sum(-2))
            sim=np.array(ss);model=np.array(mm)
            if reference is None:reference=sim
            else:np.testing.assert_array_equal(sim,reference)
            arm_reports[arm]={}
            for mode,label in enumerate(base.r.MODES):
                entries={}
                for k,cut in enumerate(cuts):
                    v=base.r.estimates(sim[:,:,k],model[:,mode,:,k],weights,axis=axis);sample[arm,mode,cut]=v
                    entries[cut]={key:base.ent(value,n) for key,value in v.items()}
                    entries[cut]['per_scene_m_percent']=base.r.estimates(sim[:,:,k],model[:,mode,:,k],np.eye(n),axis=axis)['m_percent'].tolist()
                rms=np.sqrt(np.stack([sample[arm,mode,cut]['m_percent']**2 for cut in cfg['target_cuts']],-1).mean(-1));sample[arm,mode,'rms']=rms
                entries['nine_cut_rms_m_percent']=base.ent(rms,n);arm_reports[arm][label]=entries
                central[arm,mode,axis]=sample[arm,mode,'mag25_radius_gt0.60']
        contrasts={}
        for label,coefs in CONTRASTS.items():
            modes={}
            for mode,mname in enumerate(base.r.MODES):
                entries={}
                for cut in cfg['target_cuts']:
                    keys=['m_percent','delta_response','model_flow_response','model_emulator_response']
                    entries[cut]={key:base.ent(combine([sample[arm,mode,cut][key] for arm in arms],coefs),n) for key in keys}
                    entries[cut]['absolute_m_change']=base.ent(combine([np.abs(sample[arm,mode,cut]['m_percent']) for arm in arms],coefs),n)
                entries['nine_cut_rms_change']=base.ent(combine([sample[arm,mode,'rms'] for arm in arms],coefs),n);modes[mname]=entries
            contrasts[label]=modes
        out['axes'][str(axis)]={'arms':arm_reports,'paired_changes':contrasts}
    out['paired_axes']={}
    for arm in arms:
        out['paired_axes'][arm]={}
        for mode,label in enumerate(base.r.MODES):
            a,b=[central[arm,mode,ax] for ax in [1,2]]
            trace=100*((a['simulation_response'][:,0]+b['simulation_response'][:,1])/(a['model_response'][:,0]+b['model_response'][:,1])-1)
            out['paired_axes'][arm][label]={'trace_m_percent':base.ent(trace,n),'m2_minus_m1_pp':base.ent(b['m_percent']-a['m_percent'],n)}
    s.c.write(s.ROOT/'report.json',out);s.c.write(H/'report.json',out)
    lines=['# Photometry learning × neighbour information: completed factorial','',out['definition'],'','| Arm | Predicted-usability m1 [%] | Predicted-usability m2 [%] |','|---|---:|---:|']
    def fmt(v):return f"{v['value']:+.5f} ± {v['jackknife_se']:.5f}"
    for arm in arms:lines.append('| '+arm+' | '+' | '.join(fmt(out['axes'][str(ax)]['arms'][arm][base.r.MODES[0]]['mag25_radius_gt0.60']['m_percent']) for ax in [1,2])+' |')
    lines+=['','All nine cuts, both axes, three selection/usability modes, individual scenes, model components, paired interactions and RMS comparisons are in report.json. Conditional probability, factor NLL, and selected shape diagnostics are in probability_report.json; fixed shape weights do not imply fixed selected-shape response.']
    (H/'results.md').write_text('\n'.join(lines)+'\n')
    export=Path(cfg['export_root']);export.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ['rule.md','manifest.json','results.md','report.json','probability_report.json','probability_results.md','own_report.json','verification.json','reducer_verification.json','factorial_verification.json']:
        shutil.copy(H/name,export/name)
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
