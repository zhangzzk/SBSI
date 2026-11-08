"""Measured-only same-ID cohort check, with exact selected-mean accounting."""
import importlib.util,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import scripts.evaluate_smooth_classifier_joint as ev
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
spec=importlib.util.spec_from_file_location('measured_symmetry_math',H/'audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
spec=importlib.util.spec_from_file_location('measured_symmetry_cuts',H.parent/'alignment_factorized_constant_20261105/helpers.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(1)
    cfg=a.read(H.parent/'current_measured_photometry_20261107/manifest.json');e.CUTS=cfg['cuts'];cuts=[c[0] for c in e.CUTS]
    full=[];common=[];paircommon=[];receipts=[];start=time.monotonic()
    for case in cfg['cases']:
        with np.load(Path(cfg['parent_root'])/f'parent/case{case}.npz') as z:ids=z['input_index']
        vals=[];sel=[];rec=[]
        for axis in [1,2]:
            for shear in [.02,-.02]:
                v,u,r=ev.measured_leg(Path(cfg['sim_roots'][str(axis)]),case,shear,ids)
                vals.append(v);sel.append(e.output_cuts(v)&u[:,None]);rec.append(r)
        same=np.logical_and.reduce(sel);pair=[sel[0]&sel[1],sel[2]&sel[3]]
        sums=np.zeros((3,4,len(cuts),3))
        for leg in range(4):
            for kind,mask in enumerate([sel[leg],same,pair[leg//2]]):
                for ci in range(len(cuts)):
                    keep=mask[:,ci];sums[kind,leg,ci]=[keep.sum(),*vals[leg][keep,:2].sum(0)]
        for axis in [1,2]:
            with np.load(a.ROOT/'current_measured_photometry_20261107'/f'axis{axis}/case{case}.npz') as z:
                np.testing.assert_allclose(sums[0,(axis-1)*2:axis*2],z['sim'].sum(-2),rtol=1e-12,atol=1e-8)
        np.testing.assert_array_equal(sums[1,:,:,0],np.broadcast_to(sums[1,0,:,0],sums[1,:,:,0].shape))
        full.append(sums[0].reshape(2,2,len(cuts),3));common.append(sums[1].reshape(2,2,len(cuts),3));paircommon.append(sums[2].reshape(2,2,len(cuts),3))
        receipts.append({'case':case,'matching':rec,'common_ids_sha256':hashlib.sha256(ids[same[:,4]].tobytes()).hexdigest(),'original_sums_replayed':True})
        print('CASE_DONE',case,'central_common',int(sums[1,0,4,0]),'seconds',round(time.monotonic()-start,1),flush=True)
    arrays={'full':np.array(full),'common_four':np.array(common),'common_axis_pair':np.array(paircommon)}
    path=ROOT/'membership_sums.npz';np.savez(path,**arrays,cases=cfg['cases'],cuts=cuts)
    n=len(cfg['cases']);w=np.vstack([np.ones(n),np.random.default_rng(202611071940).multinomial(n,np.ones(n)/n,10000),1-np.eye(n)])
    v={name:a.values(s,w) for name,s in arrays.items()};remainder={k:v['full'][k]-v['common_four'][k] for k in v['full']}
    out={'definition':'Measured-only exact decomposition: full response = common-all-four selected-ID response + remainder from changing membership and normalization. Common-cohort response changes the population; this is not a causal error allocation.', 'cases':cfg['cases'],'source_sha256':a.sha(__file__),'data_sha256':a.sha(path),'receipts':receipts,'cuts':{}}
    for ci,cut in enumerate(cuts):
        row={name:{k:a.entry(q[:,ci],n) for k,q in d.items()} for name,d in {**v,'membership_and_normalization_remainder':remainder}.items()}
        denominator=arrays['full'][:,:,:,ci,0].sum(0);count=arrays['common_four'][:,0,0,ci,0].sum()
        row['common_four_count']=int(count);row['full_counts_by_axis_leg']=denominator.tolist();row['common_fraction_by_axis_leg']=(count/denominator).tolist()
        # Point and all resamples close the stated response identity.
        np.testing.assert_allclose(v['common_four']['matrix'][:,ci]+remainder['matrix'][:,ci],v['full']['matrix'][:,ci],rtol=1e-13,atol=1e-13)
        out['cuts'][cut]=row
    (H/'membership_report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('CENTRAL',json.dumps(out['cuts']['mag25_radius_gt0.60']),flush=True)
    print('DONE_MEASURED_MEMBERSHIP',flush=True)
if __name__=='__main__':main()
