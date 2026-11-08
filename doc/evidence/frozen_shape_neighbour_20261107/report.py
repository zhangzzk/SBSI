"""Paired readout from the original estimator; label the historical baseline separately."""
import importlib.util,json,os
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixedshape_setup',H/'common.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
base=s.bound('report')
def own_report():
    base.own_report()
    # Original start and new arms use the same sampled training/validation rows,
    # case order and portable draws, so use paired case weights for start comparisons.
    old=s.ROOT.parent/'neighbour_moment_guard_20261106'
    ranges={};lines=['# Frozen-shape photometry training on its own data','',
        'Measured minus generated mean neighbour score among galaxies passing the hard target cut (unsheared usable rows). Paired whole-case jackknife SEs; fixed fitted models. This is not shear response.','',
        '| Split | Untouched nine-cut start | Frozen control | Frozen + neighbour loss | Added loss minus frozen control | Added loss minus start |','|---|---:|---:|---:|---:|---:|']
    for split in ['train','validation']:
        with np.load(old/f'own_{split}.npz') as z,np.load(s.ROOT/f'own_{split}.npz') as q:
            for key in ['arms','cases','cuts','kinds']:np.testing.assert_array_equal(z[key],q[key])
            # arm, hard/soft, bank, case, leg, cut, sufficient statistics
            x=q['sums'][:,0,0,:,0];y=z['sums'][0,0,0,:,0]
            np.testing.assert_array_equal(x[0,:,:,:3],y[:,:,:3])
            n=x.shape[1];w=base.bootstrap(n,s.manifest()['bootstrap_seed'])
            out={};cols=[]
            for i,cut in enumerate(q['cuts']):
                data=[]
                for v in [y[:,i],x[0,:,i],x[1,:,i]]:
                    t=w@v;data.append(t[:,2]/t[:,1]-t[:,4]/t[:,3])
                start,control,weighted=data
                vv=[*data,weighted-control,weighted-start]
                names=['start','control','weighted','weighted_minus_control','weighted_minus_start']
                out[str(cut)]={name:base.ent(v,n) for name,v in zip(names,vv)}
                if cut=='mag25_radius_gt0.60':
                    cols=[f"{v[0]:+.6f} ± {base.ent(v,n)['jackknife_se']:.6f}" for v in vv]
            assert cols;lines.append('| '+split+' | '+' | '.join(cols)+' |');ranges[split]=out
    s.c.write(H/'own_paired_to_start.json',ranges)
    (H/'own_results.md').write_text('\n'.join(lines)+'\n')

def main():
    base.main()
    r=json.loads((H/'report.json').read_text());pins=json.loads((s.ROOT/'training_completed.json').read_text())
    for z in pins['arms'].values():
        assert z['checks']['reloaded_identity']['fixed_conditional_density_max_abs']==0
    r['standalone_probes']['nine_cut_before_continuation']=r['standalone_probes'].pop('initial')
    r['arm_definition']={'initial':'historical NLL-only reference (constant-shear tables)','previous_nine':'starting nine-cut model, no continuation','control':'frozen shape, photometry training without additional moment','weighted':'frozen shape, photometry training with additional moment'}
    s.c.write(s.ROOT/'report.json',r);s.c.write(H/'report.json',r)
    p=H/'results.md';body=p.read_text().replace('# Nine-cut response plus selected-neighbour moment','# Frozen conditional shape: photometry-only training')
    p.write_text(body+'\nThe initial row is the historical NLL model; previous_nine is the exact training start. Control and weighted are the matched photometry-only continuations.\n')
    cfg=s.manifest();cases=cfg['cases'];cuts=cfg['target_cuts'];w=base.bootstrap(len(cases),cfg['bootstrap_seed'])
    old=s.ROOT.parent/'neighbour_moment_guard_20261106'
    more={}
    for ax in [1,2]:
        parts={}
        for name,folder in [('start',old/'constant/control'),('both_factors',old/'constant/weighted'),('control',s.ROOT/'constant/control'),('weighted',s.ROOT/'constant/weighted')]:
            ss=[];mm=[]
            for case in cases:
                f=folder/f'axis{ax}/case{case}.npz';d=json.loads(f.with_suffix('.json').read_text())
                assert s.c.sha(f)==d['data_sha256']
                with np.load(f) as z:
                    assert list(z['cuts'])[:len(cuts)]==cuts;ss.append(z['sim'].sum(-2));mm.append(z['model'].sum(-2))
            sim=np.array(ss);mod=np.array(mm)
            if name=='start':reference=sim
            else:np.testing.assert_array_equal(reference,sim)
            parts[name]={(m,i):base.r.estimates(sim[:,:,i],mod[:,m,:,i],w,axis=ax) for m in range(3) for i in range(len(cuts))}
        contrasts={}
        for a,b in [('weighted','control'),('weighted','start'),('control','start'),('both_factors','start')]:
            modes={}
            for m,mname in enumerate(base.r.MODES):
                modes[mname]={cut:{k:base.ent(parts[a][m,i][k]-parts[b][m,i][k],len(cases)) for k in ['model_flow_response','model_emulator_response','delta_response','m_percent']} for i,cut in enumerate(cuts)}
            contrasts[f'{a}_minus_{b}']=modes
        more[str(ax)]=contrasts
    s.c.write(H/'paired_to_start.json',more)
