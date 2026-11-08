"""Full coherent conditional-shape + emulator response at observed photometry."""
import importlib.util,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scripts.evaluate_smooth_classifier_joint as ev
from scripts import joint_response_axes as axes
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_measured_photometry_20261106')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e=module('joint_measured_helpers',HERE.parent/'alignment_factorized_constant_20261105/helpers.py')
cp=module('joint_measured_conditional',HERE.parent/'alignment_conditioned_photometry_20261105/evaluate.py')
portable=module('joint_measured_portable',HERE.parent/'alignment_target9_20261106/portable_draws.py')
BANKS=[list(range(64)),list(range(16))+list(range(32,48)),list(range(16,32))+list(range(48,64))]
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    manifest=json.loads((HERE/'manifest.json').read_text());index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    if index:assert (ROOT/'pilot_passed.json').exists()
    task=manifest['tasks'][index];case,axis=task['case'],task['axis']
    dest=ROOT/f'axis{axis}';dest.mkdir(parents=True,exist_ok=True);out=dest/f'case{case}.npz';assert not out.exists()
    e.CUTS=manifest['cuts'];e.CUT_NAMES=tuple(x[0] for x in e.CUTS);e.SIMS=Path(manifest['sim_roots'][str(axis)])
    if axis==2:
        for leg in ['0.02','-0.02']:assert (e.SIMS.parent/f'case{case}_{leg}_verified.json').exists()
    models={};first_condition=None
    for name,pin in manifest['arms'].items():
        assert e.sha256(pin['path'])==pin['sha256']
        model,condition,_,_=cp.fm.load(pin['path']);models[name]=model
        if first_condition is None:first_condition=condition
        else:
            for key,value in first_condition.to_state().items():np.testing.assert_array_equal(condition.to_state()[key],value)
    preset,paths,conditions=e.preset();emulator=DiskResponseMoment(paths['emulator'].parent,nthread=4,device='cuda')
    parent=Path(manifest['parent_root'])
    with np.load(parent/f'parent/case{case}.npz') as z:ids=z['input_index'];raw=z['context'].astype(float)
    truth,truth_path=e.truth(case,.02);np.testing.assert_array_equal(ids,truth['index_input'])
    assert np.all(truth[f'gamma{axis}_input']==.02) and np.all(truth[f'gamma{3-axis}_input']==0)
    frames=axes.shear_frames(e.full_context(raw),.02,axis)
    contexts=[torch.as_tensor(first_condition.transform_frame(f),dtype=torch.float32,device='cuda') for f in frames]
    values=[];usable=[];selection=[];receipts=[]
    for h in [.02,-.02]:
        v,u,receipt=ev.measured_leg(e.SIMS,case,h,ids)
        assert np.isfinite(v[u]).all() and (v[u,2:]>0).all()
        values.append(v);usable.append(u);selection.append(e.output_cuts(v)&u[:,None]);receipts.append(receipt)
    union=usable[0]|usable[1]
    pairs=e.pair_table(e.scene_frame(truth),np.flatnonzero(union),conditions)
    coefficients=np.zeros((2,len(ids)));started=time.monotonic()
    print('PAIRS_READY',case,axis,'usable',list(map(np.sum,usable)),'pairs',len(pairs),flush=True)
    for leg in range(2):
        bid=ids[usable[leg]];pp=pairs.loc[pairs['index_input_p'].isin(bid)]
        b,record=predict_scene_draw_responses(emulator,pp,bid,values[leg][usable[leg],None,:])
        coefficients[leg,usable[leg]]=b[:,0]
    del pairs,emulator
    bins=np.digitize(raw[:,4],manifest['intrinsic_size_edges_arcsec']);nc=len(e.CUTS)
    sim=np.zeros((2,nc,5,3));sums=np.zeros((len(models),3,2,nc,5,5))
    for leg in range(2):
        for cut in range(nc):
            for size in range(5):
                keep=selection[leg][:,cut]&(bins==size)
                sim[leg,cut,size]=[keep.sum(),*values[leg][keep,:2].sum(0)]
    checks={};support={name:[{'draws':0,'rounded_boundary':0,'selected_draws':0,'selected_rounded_boundary':0,'max_squared_radius':0.} for _ in range(2)] for name in models}
    center=e.CUT_NAMES.index('mag25_radius_gt0.60')
    print('READY',case,axis,'GPU',torch.cuda.get_device_name(),flush=True)
    with torch.no_grad():
        for block,lo in enumerate(range(0,len(ids),1024)):
            hi=min(len(ids),lo+1024);nb=hi-lo
            latent=portable.latent_bank(nb,64,4,7301+10000019*case+1000003*block)
            for leg in range(2):
                keep=usable[leg][lo:hi]
                if not keep.any():continue
                z=torch.from_numpy(latent[keep]).to('cuda').reshape(-1,4)
                ctx=contexts[leg][lo:hi][keep,None,:].expand(-1,64,-1).reshape(len(z),-1)
                photo=torch.as_tensor(values[leg][lo:hi][keep,2:],dtype=torch.float64,device='cuda')
                expanded_photo=photo[:,None,:].expand(-1,64,-1).reshape(len(z),2)
                weights=selection[leg][lo:hi][keep];sizes=bins[lo:hi][keep]
                velocity=torch.zeros((int(keep.sum()),1,2),dtype=torch.float64,device='cuda')
                velocity[:,:,axis-1]=torch.as_tensor(coefficients[leg,lo:hi][keep,None],device='cuda')*(.02 if leg==0 else -.02)
                for model_index,(name,model) in enumerate(models.items()):
                    if name not in checks:
                        joint,_=model.forward_physical(z[:256],ctx[:256]);valid=(joint[:,2:]>0).all(-1);assert bool(valid.any())
                        reconstructed=cp.conditional_physical(model,z[:256][valid],ctx[:256][valid],joint[valid,2:])
                        torch.testing.assert_close(reconstructed,joint[valid],rtol=2e-5,atol=2e-6)
                        changed=z[:256][valid].clone();changed[:,2:]+=3
                        torch.testing.assert_close(cp.conditional_physical(model,changed,ctx[:256][valid],joint[valid,2:]),reconstructed,rtol=0,atol=0)
                        checks[name]={'reconstruction_max_abs':float((reconstructed-joint[valid]).abs().max()),'unused_photo_latent_exact':True}
                    v=cp.conditional_physical(model,z,ctx,expanded_photo).reshape(-1,64,4)
                    torch.testing.assert_close(v[...,2:],photo[:,None,:].expand(-1,64,-1),rtol=0,atol=0)
                    if not bool(torch.isfinite(v).all()):
                        raise ValueError(f'nonfinite conditional output: {case=} {axis=} {block=} {leg=} {name=}')
                    radius2=v[...,:2].square().sum(-1);largest=float(radius2.max())
                    # Canonical radial_tanh returns representable boundary values
                    # without clipping when tanh saturates (flow_disk.py:31--46).
                    # We use forward expectations, not inverse-density evaluation.
                    assert largest<=1+16*torch.finfo(torch.float64).eps,(case,axis,block,leg,name,largest)
                    boundary=radius2>=1;selected=torch.as_tensor(weights[:,center],device='cuda')[:,None]
                    audit=support[name][leg];audit['draws']+=radius2.numel();audit['rounded_boundary']+=int(boundary.sum())
                    audit['selected_draws']+=int(selected.sum())*64;audit['selected_rounded_boundary']+=int((boundary&selected).sum())
                    audit['max_squared_radius']=max(audit['max_squared_radius'],largest)
                    np.testing.assert_array_equal(e.output_cuts(v[:,0]).cpu().numpy(),weights)
                    transported,ld=disk_response_transport(v[...,:2],velocity)
                    assert bool(torch.isfinite(transported).all() and torch.isfinite(ld).all())
                    shift=transported-v[...,:2]
                    for bank,draws in enumerate(BANKS):
                        moment=torch.cat([torch.ones((len(v),1),dtype=torch.float64,device='cuda'),
                            v[:,draws,:2].mean(1),shift[:,draws].mean(1)],-1).cpu().numpy()
                        for size in range(5):
                            take=sizes==size
                            sums[model_index,bank,leg,:,size]+=np.einsum('nk,nq->kq',weights[take].astype(float),moment[take])
            if block%100==0:print('PROGRESS',case,axis,block,round(time.monotonic()-started,1),flush=True)
    assert len(checks)==len(models) and np.isfinite(sums).all()
    np.testing.assert_array_equal(sums[...,0],np.broadcast_to(sim[...,0],sums[...,0].shape))
    np.testing.assert_allclose(sums[:,0],.5*(sums[:,1]+sums[:,2]),rtol=1e-12,atol=2e-7)
    with np.load(Path(manifest['generated_roots']['baseline'])/f'axis{axis}/case{case}.npz') as old:
        np.testing.assert_array_equal(old['cuts'],e.CUT_NAMES);np.testing.assert_allclose(sim,old['sim'],rtol=1e-12,atol=1e-8)
    np.savez(out,model=sums,sim=sim,case=case,axis=axis,cuts=e.CUT_NAMES,arms=list(models))
    receipt={'case':case,'axis':axis,'models':manifest['arms'],'emulator_sha256':preset['emulator']['sha256'],
        'measured':receipts,'usability':'actual, independently per leg','common_leg_intersection':False,
        'measured_photometry_exact':True,'selected_masses_exact':True,'simulation_replay_passed':True,
        'half_bank_identity_passed':True,'conditional_checks':checks,'representable_disk_boundary_audit':support,'source_sha256':e.sha256(__file__),
        'conditional_source_sha256':e.sha256(cp.__file__),'portable_source_sha256':e.sha256(portable.__file__),
        'parent_sha256':e.sha256(parent/f'parent/case{case}.npz'),'gpu':torch.cuda.get_device_name(),'seconds':time.monotonic()-started}
    out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('DONE',case,axis,receipt['seconds'],flush=True)
if __name__=='__main__':main()
