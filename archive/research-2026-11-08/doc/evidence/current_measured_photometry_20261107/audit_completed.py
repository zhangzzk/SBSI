"""Independent direct sums and delete-one-scene check; no model draws."""
import hashlib
import json
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def one(s,m):
    # [axis,leg,cut,moment]; pool scenes BEFORE calling this function.
    rs=(s[:,0,:,1:]/s[:,0,:,:1]-s[:,1,:,1:]/s[:,1,:,:1])/.04
    flow=(m[:,0,:,1:3]/m[:,0,:,:1]-m[:,1,:,1:3]/m[:,1,:,:1])/.04
    emu=(m[:,0,:,3:5]/m[:,0,:,:1]-m[:,1,:,3:5]/m[:,1,:,:1])/.04
    rm=flow+emu; delta=(rs-rm).transpose(1,2,0)
    a=100*(rs[0,:,0]/rm[0,:,0]-1);b=100*(rs[1,:,1]/rm[1,:,1]-1)
    return dict(m1_percent=a,m2_percent=b,mean_bias_percent=(a+b)/2,half_difference_pp=(a-b)/2,
        two_axis_rms_percent=np.sqrt((a*a+b*b)/2),delta_response=delta,
        trace_half_response=(delta[:,0,0]+delta[:,1,1])/2,
        diagonal_difference_half_response=(delta[:,0,0]-delta[:,1,1])/2,
        symmetric_cross_response=(delta[:,0,1]+delta[:,1,0])/2,
        antisymmetric_cross_response=(delta[:,0,1]-delta[:,1,0])/2,
        model_flow_response=flow.transpose(1,2,0),model_emulator_response=emu.transpose(1,2,0))

def ent(v):
    v=np.asarray(v);n=len(v)-1;jk=v[1:]
    return dict(value=v[0].tolist(),jackknife_se=np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0)).tolist())

def main():
    cfg=read(H/'manifest.json');ref=read(H/'report.json');cases=cfg['cases'];cuts=[x[0] for x in cfg['cuts']]
    for p,d in read(H/'source_pins.json').items():assert sha(p)==d,p
    sims={};models={};matching=[];paths=[]
    modes=['predicted_usability_generated_selection','true_usability_generated_selection','true_usability_true_selection']
    for axis in [1,2]:
        ss=[];mm=[];gg={a:[] for a in cfg['checkpoints']};history={a:[] for a in ['baseline','nine_weak']}
        for case in cases:
            p=ROOT/f'axis{axis}/case{case}.npz';rec=read(p.with_suffix('.json'));assert sha(p)==rec['data_sha256']
            assert rec['checkpoint_pair']==cfg['checkpoints'] and rec['emulator_sha256']==cfg['emulator_sha256']
            paths.append(str(p));matching.extend(rec['measured'])
            with np.load(p) as z:
                s=z['sim'].sum(-2);m=z['model'][list(z['arms']).index('control')].sum(-2);ss.append(s);mm.append(m)
            genroot=Path(cfg['original_root']) if case in cfg['pilot_cases'] else ROOT.parent/'conditional_cut_replication_20261107'
            for arm in gg:
                p=genroot/'constant'/arm/f'axis{axis}/case{case}.npz';rec=read(p.with_suffix('.json'));assert sha(p)==rec['data_sha256'];paths.append(str(p))
                assert rec['checkpoint_sha256']==cfg['checkpoints'][arm]['sha256']
                with np.load(p) as z:
                    np.testing.assert_allclose(s,z['sim'].sum(-2),rtol=1e-12,atol=1e-8);gg[arm].append(z['model'].sum(-2))
            p=ROOT.parent/'joint_measured_photometry_20261106'/f'axis{axis}/case{case}.npz'
            with np.load(p) as z:
                np.testing.assert_allclose(s,z['sim'].sum(-2),rtol=1e-12,atol=1e-8)
                for arm in history:history[arm].append(z['model'][list(z['arms']).index(arm),0].sum(-2))
        sims[axis]=np.array(ss);models[axis]={'measured_shared':np.array(mm)[:,0], 'half_a':np.array(mm)[:,1], 'half_b':np.array(mm)[:,2]}
        for a,x in gg.items():
            for mi,mode in enumerate(modes):models[axis][a+'_'+mode]=np.array(x)[:,mi]
        for a,x in history.items():models[axis]['historical_'+a]=np.array(x)
    checks=0;max_error=0.;historical={}
    def check(v,expected):
        nonlocal checks,max_error
        got=ent(v)
        for k in ['value','jackknife_se']:
            np.testing.assert_allclose(got[k],expected[k],rtol=1e-8,atol=3e-11)
            checks+=np.size(got[k]);max_error=max(max_error,float(np.max(np.abs(np.asarray(got[k])-expected[k]))))
    for cohort,cr in ref['cohorts'].items():
        ix=[cases.index(c) for c in cr['cases']];subsets=[ix]+[ix[:k]+ix[k+1:] for k in range(len(ix))]
        samples={name:[one(np.array([sims[a][ii].sum(0) for a in [1,2]]),np.array([models[a][name][ii].sum(0) for a in [1,2]])) for ii in subsets] for name in models[1]}
        for ci,cut in enumerate(cuts):
            row=cr['cuts'][cut]
            for name,saved in row['predictions'].items():
                for k,v in saved.items():check([x[k][ci] for x in samples[name]],v)
            for name,saved in row['measured_minus_generated'].items():
                for k,v in saved.items():check([x[k][ci]-y[k][ci] for x,y in zip(samples['measured_shared'],samples[name])],v)
            for k,v in row['measured_half_bank_difference'].items():check([x[k][ci]-y[k][ci] for x,y in zip(samples['half_a'],samples['half_b'])],v)
        rms={name:np.array([np.sqrt(np.mean(x['two_axis_rms_percent'][:9]**2)) for x in v]) for name,v in samples.items()}
        for name,v in cr['joint_nine_cut_rms'].items():check(rms[name],v)
        historical[cohort]={'rms_measured_minus_generated':{name:ent(rms['measured_shared']-v) for name,v in rms.items() if name in cr['joint_nine_cut_rms'] and name!='measured_shared'},'comparisons_to_earlier_models':{}}
        for old in ['historical_baseline','historical_nine_weak']:
            historical[cohort]['comparisons_to_earlier_models'][old]={cut:{k:ent([x[k][ci]-y[k][ci] for x,y in zip(samples['measured_shared'],samples[old])]) for k in ['m1_percent','m2_percent','mean_bias_percent','half_difference_pp','model_flow_response','model_emulator_response']} for ci,cut in enumerate(cuts)}
    result={'scalar_statistics_replayed':checks,'maximum_absolute_difference':max_error,'hashed_current_and_generated_files':len(paths),'source_sha256':sha(__file__),'report_sha256':sha(H/'report.json'),'matching_totals':{k:sum(r[k] for r in matching) for k in matching[0]},'cohorts':historical,'historical_comparison_limit':'Different training branches, not a matched attribution to one loss or feature.'}
    (H/'completed_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS',checks,'scalars, max difference',max_error)
    print(json.dumps(result['cohorts']['all_forty']['rms_measured_minus_generated'],indent=2))
    print('Historical nine-weak comparison',result['cohorts']['all_forty']['comparisons_to_earlier_models']['historical_nine_weak']['mag25_radius_gt0.60'])
    print('Matching totals',result['matching_totals'])
if __name__=='__main__':main()
