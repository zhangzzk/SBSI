"""Report every predeclared endpoint with paired case-level uncertainties."""
import argparse, importlib.util, json, os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neighbour_common_report',HERE/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
r=c.module('neighbour_response_report',HERE.parent/'alignment_axis2_constant_20261105/report.py')
def main():
    assert os.environ.get('SLURM_JOB_ID')
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text())
    if a.pilot:
        task=manifest['tasks'][0];path=c.ROOT/'constant'/task['arm']/f"axis{task['axis']}"/f"case{task['case']}.json"
        d=json.loads(path.read_text());assert d['simulation_replay_passed'] and d['reference_score_replay_passed'] and d['true_selection_mass_exact']
        c.write(c.ROOT/'evaluation_pilot_passed.json',d);print('PILOT_PASSED',d,flush=True);return
    cases=manifest['cases'];n=len(cases);names=manifest['arms'];nc=len(manifest['cuts']);end=10001
    weights=np.vstack([np.ones(n),np.random.default_rng(202611068601).multinomial(n,np.full(n,1/n),10000),np.eye(n),np.ones((n,n))-np.eye(n)])
    def ent(x):
        jk=x[end+n:]
        return {'value':x[0].tolist(),'case_se':np.std(x[1:end],axis=0,ddof=1).tolist(),
                'jackknife_se':np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0)).tolist()}
    result={'cases':cases,'definition':'DeltaR=simulation-model; m=100(Rsim_jj/Rmodel_jj-1). Paired whole-case SE with fixed fits; four development scenes, not acceptance.',
            'axes':{},'training':{},'selection_association':{},'source_sha256':c.sha(__file__)}
    for arm in ['control','neighbour']:
        folder=c.ROOT/arm
        done=json.loads((folder/'completed.json').read_text());assert done['reload_exact']
        result['training'][arm]={'completion':done,'protocol':json.loads((folder/'protocol.json').read_text()),
            'nll_probe':json.loads((folder/'nll_probe.json').read_text()),'nine_probe':json.loads((folder/'nine_probe.json').read_text())}
    for axis in [1,2]:
        results={};samples={};reference=None
        for name in ['baseline']+names:
            directory=c.ROOT.parent/f'alignment_target9_20261106/axis{axis}' if name=='baseline' else c.ROOT/'constant'/name/f'axis{axis}'
            simulations=[];models=[];associations=[]
            for case in cases:
                with np.load(directory/f'case{case}.npz') as z:
                    np.testing.assert_array_equal(z['cuts'],[x[0] for x in manifest['cuts']])
                    simulations.append(z['sim'].sum(-2));models.append(z['model'].sum(-2))
                    if name!='baseline' and axis==1:associations.append(z['association'])
                if name!='baseline':
                    receipt=json.loads((directory/f'case{case}.json').read_text());assert receipt['simulation_replay_passed'] and receipt['reference_score_replay_passed']
            sim=np.array(simulations);model=np.array(models)
            if reference is None:reference=sim
            else:np.testing.assert_allclose(sim,reference,rtol=1e-12,atol=1e-8)
            results[name]={}
            for mode in range(model.shape[1]):
                rows={};ts=[]
                for k,(cut,_,_) in enumerate(manifest['cuts']):
                    v=r.estimates(sim[:,:,k],model[:,mode,:,k],weights,axis=axis);samples[name,mode,cut]=v
                    rows[cut]={key:ent(value) for key,value in v.items()}
                    if cut in manifest['target_cuts']:ts.append(v['m_percent'])
                ts=np.stack(ts,-1)
                rows['nine_cut_rms_m_percent']=ent(np.sqrt(np.square(ts).mean(-1)))
                results[name][r.MODES[mode]]=rows
            if associations:
                t=weights@np.array(associations)
                result['selection_association'][name]={
                    'reference_coefficient_error':ent(t[:,2]/t[:,1]-t[:,4]/t[:,3]),
                    'measured_pass_rate':ent(t[:,1]/t[:,0]),'generated_pass_rate':ent(t[:,3]/t[:,0])}
        changes={}
        for new,old in [('neighbour_nll','control_nll'),('neighbour_nine','control_nine'),('control_nine','control_nll'),('neighbour_nine','neighbour_nll')]:
            changes[new+'_minus_'+old]={r.MODES[mode]:{cut:{key:ent(samples[new,mode,cut][key]-samples[old,mode,cut][key])
                for key in ['delta_response','m_percent','model_flow_response','model_emulator_response']} for cut in manifest['target_cuts']} for mode in range(3)}
        result['axes'][str(axis)]={'arms':results,'paired_changes':changes}
    c.write(c.ROOT/'report.json',result);c.write(HERE/'report.json',result)
    def fmt(x):return f"{x['value']:+.4f} ± {x['case_se']:.4f}"
    lines=['# Neighbour-conditioned flow: four-scene test','',result['definition'],'',
        'Target: measured radius>3px and magnitude<25. Values below are m in percent; errors are percentage points.','',
        '| Flow | Predicted usability m1 | True usability m1 | Predicted usability m2 | True usability m2 |','|---|---:|---:|---:|---:|']
    for name in ['baseline']+names:
        vals=[result['axes'][str(axis)]['arms'][name][r.MODES[mode]]['mag25_radius_gt0.60']['m_percent'] for axis in [1,2] for mode in [0,1]]
        lines.append('| '+name+' | '+' | '.join(fmt(v) for v in vals)+' |')
    lines+=['','Reference-coefficient selection errors below are diagnostics, not calibrated R11:','',
            '| Flow | Measured minus generated selected-mean score |','|---|---:|']
    for name,row in result['selection_association'].items():
        v=row['reference_coefficient_error'];lines.append(f"| {name} | {v['value']:+.6f} ± {v['case_se']:.6f} |")
    lines+=['','All nine cuts, full response components, true-selection rows, paired changes and standalone response probes are retained in report.json.',
            'No checkpoint was chosen by these results. The static new feature does not implement neighbour-induced selection response.']
    (HERE/'results.md').write_text('\n'.join(lines)+'\n');print('DONE_REPORT',flush=True)
if __name__=='__main__':main()
