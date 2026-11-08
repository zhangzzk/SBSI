"""Case-paired selection association, score interventions and likelihood factors."""
import json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_flow_selection_20261106')
PAIRS=[('neighbour_nll','control_nll'),('neighbour_nine','control_nine'),('nine_weak','baseline'),('control_nine','control_nll'),('neighbour_nine','neighbour_nll'),('neighbour_nll','neighbour_nll_mean_score'),('neighbour_nine','neighbour_nine_mean_score')]
def metrics(t):
    n,measured,mscore,generated,gscore=np.moveaxis(t,-1,0)
    return {'measured_pass_rate':measured/n,'generated_pass_rate':generated/n,'pass_error_pp':100*(measured-generated)/n,
        'measured_selected_score':mscore/measured,'generated_selected_score':gscore/generated,'score_error':mscore/measured-gscore/generated}
def main():
    assert os.environ.get('SLURM_JOB_ID')
    manifest=json.loads((HERE/'manifest.json').read_text());result={'definition':'Measured minus generated selected mean reference coefficient; not a shear response. Whole-case paired bootstrap and jackknife with fixed fits. All rows are usable in their original leg. Score-at-mean predictions are input interventions.','splits':{}}
    for si,split in enumerate(manifest['split_order']):
        with np.load(ROOT/f'{split}.npz') as z:
            sums=z['sums'];density=z['density'];names=list(z['arms']);cuts=list(z['cuts']);cases=z['cases'];edges=z['score_edges']
        n=len(cases);nboot=5000;end=nboot+1
        w=np.vstack([np.ones(n),np.random.default_rng(manifest['bootstrap_seed']+si).multinomial(n,np.full(n,1/n),nboot),np.ones((n,n))-np.eye(n)])
        def ent(x):
            jk=x[end:]
            return {'value':float(x[0]),'case_se':float(x[1:end].std(ddof=1)),'jackknife_se':float(np.sqrt((n-1)/n*np.square(jk-jk.mean()).sum()))}
        populations={}
        for pop,ss,dd in [('pooled',sums.sum((3,4)),density.sum(2)),('g0',sums[:,:,:,0].sum(3),density[:,:,0]),('g05',sums[:,:,:,1].sum(3),density[:,:,1])]:
            samples={};arms={};density_samples={}
            for ai,name in enumerate(names):
                rows={}
                for ci,cut in enumerate(cuts):
                    v=metrics(w@ss[ai,0,:,ci]);samples[name,cut]=v
                    rows[cut]={k:ent(x) for k,x in v.items()}
                    a,b=[metrics(w@ss[ai,bi,:,ci]) for bi in [1,2]]
                    rows[cut]['half_bank_score_error_difference']=ent(a['score_error']-b['score_error'])
                den=w@dd[ai];dv={'joint_nll':den[:,1]/den[:,0],'photometry_nll':den[:,2]/den[:,0],'conditional_shape_nll':(den[:,1]-den[:,2])/den[:,0]}
                density_samples[name]=dv;arms[name]={'cuts':rows,'nll':{k:ent(v) for k,v in dv.items()}}
            changes={}
            for new,old in PAIRS:
                changes[new+'_minus_'+old]={'cuts':{cut:{key:ent(samples[new,cut][key]-samples[old,cut][key]) for key in ['score_error','pass_error_pp']} for cut in cuts},
                    'nll':{key:ent(density_samples[new][key]-density_samples[old][key]) for key in density_samples[new]}}
            populations[pop]={'arms':arms,'paired_changes':changes}
        ci=cuts.index('mag25_radius_gt0.60');by_bin=sums[:,0,:,:,:,ci].sum((1,2))
        profiles={name:{k:v.tolist() for k,v in metrics(by_bin[ai]).items()} for ai,name in enumerate(names)}
        result['splits'][split]={'cases':cases.tolist(),'populations':populations,'score_quintile_edges':edges.tolist(),'pooled_score_quintiles_center_cut':profiles}
    for path in [ROOT/'report.json',HERE/'report.json']:path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def fmt(x):return f"{x['value']:+.6f} ± {x['case_se']:.6f}"
    lines=['# Selection association on original flow data','',result['definition'],'',
        'Target: radius>3 pixels and magnitude<25. Each split uses500,000 fixed rows. Errors below are case-bootstrap SEs.','',
        '| Flow | Training g0 score error | Validation g0 score error | Validation pooled score error |','|---|---:|---:|---:|']
    for name in names:
        values=[result['splits'][s]['populations'][p]['arms'][name]['cuts']['mag25_radius_gt0.60']['score_error'] for s,p in [('train','g0'),('validation','g0'),('validation','pooled')]]
        lines.append('| '+name+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines+=['','Validation likelihood change from adding neighbour information (matched training, lower is better):','',
        '| Stage | Joint NLL change | Photometry NLL change | Conditional-shape NLL change |','|---|---:|---:|---:|']
    for stage in ['nll','nine']:
        row=result['splits']['validation']['populations']['pooled']['paired_changes'][f'neighbour_{stage}_minus_control_{stage}']['nll']
        lines.append('| '+stage+' | '+' | '.join(fmt(row[k]) for k in ['joint_nll','photometry_nll','conditional_shape_nll'])+' |')
    lines+=['','The two mean-score arms replace only the extra input with its training mean; they diagnose learned dependence and are not calibrated models.',
        'All nine cuts, original g05 rows, paired changes, quintile profiles and half-bank checks are in report.json.']
    (HERE/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
