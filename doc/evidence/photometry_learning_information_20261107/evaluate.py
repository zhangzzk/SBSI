"""Persistent paired-endpoint constant-shear evaluation with frozen auxiliaries."""
import gc,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scripts.evaluate_smooth_classifier_joint as ev
from scripts import joint_response_axes as axes
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('moment_evaluation_common',HERE/'common.py');setup=importlib.util.module_from_spec(s);s.loader.exec_module(setup);c=setup.c
class Worker:
    def __init__(self):
        self.cfg=c.manifest();self.e=c.e;self.e.CUTS=self.cfg['cuts'];self.e.CUT_NAMES=tuple(x[0] for x in self.e.CUTS)
        self.models={};self.pins={};self.source=c.sha(__file__)
        training=json.loads((c.ROOT/'training_completed.json').read_text())
        for arm in self.cfg['arms']:
            pin=training['arms'][arm];path=c.ROOT/arm/'final.pt';assert pin['reload_exact'] and c.sha(path)==pin['sha256']
            model,condition,_,_=c.m.load(path);self.models[arm]=(model,condition);self.pins[arm]=pin['sha256']
        self.preset,self.paths,self.conditions=self.e.preset()
        self.emulator=DiskResponseMoment(self.paths['emulator'].parent,nthread=4,device='cuda')
        assert self.emulator.conditioned and self.emulator.response_coordinate=='disk_velocity'
        print('WORKER_MODELS_READY',torch.cuda.get_device_name(),flush=True)
    def evaluate(self,task):
        case,axis=task['case'],task['axis'];e=self.e;cfg=self.cfg;active=[];paths={}
        for arm in cfg['arms']:
            out=c.ROOT/'constant'/arm/f'axis{axis}'/f'case{case}.npz';out.parent.mkdir(parents=True,exist_ok=True);paths[arm]=out
            if out.with_suffix('.json').exists():
                d=json.loads(out.with_suffix('.json').read_text());assert d['checkpoint_sha256']==self.pins[arm] and d['source_sha256']==self.source and c.sha(out)==d['data_sha256']
            else:
                assert not out.exists();active.append(arm)
        if not active:print('SKIP_CASE',case,axis,flush=True);return
        e.SIMS=Path(cfg['sim_roots'][str(axis)]);center=e.CUT_NAMES.index('mag25_radius_gt0.60')
        if axis==2:
            for leg in ['0.02','-0.02']:assert (e.SIMS.parent/f'case{case}_{leg}_verified.json').exists()
        parent=Path(cfg['parent_root']);flags=parent if axis==1 else c.ROOT.parent/'alignment_axis2_constant_20261105'
        with np.load(parent/f'parent/case{case}.npz') as z,np.load(flags/f'c8/case{case}.npz') as f:
            ids=z['input_index'];context=z['context'].astype(float);np.testing.assert_array_equal(ids,f['input_index']);probability=np.stack([f['probability_plus'],f['probability_minus']])
        truth,_=e.truth(case,.02);np.testing.assert_array_equal(truth['index_input'],ids)
        assert np.all(truth[f'gamma{axis}_input']==.02) and np.all(truth[f'gamma{3-axis}_input']==0)
        pairs=e.pair_table(e.scene_frame(truth),np.arange(len(truth)),self.conditions);pair_ids=pairs['index_input_p'].to_numpy()
        score=c.c.score_pairs(self.emulator,pairs,ids)
        with np.load(c.ROOT.parent/f'photometry_fixed_cell_20261106/targets{case}.npz') as z:
            ix=pd.Index(ids).get_indexer(z['input_index']);assert (ix>=0).all();np.testing.assert_allclose(score[ix],z['reference_coefficient'].sum(-1),rtol=2e-10,atol=2e-7)
        frames=axes.shear_frames(e.full_context(context),.02,axis)
        for frame in frames:frame[c.c.FEATURE]=score
        contexts={arm:[torch.as_tensor(condition.transform_frame(frame),dtype=torch.float32,device='cuda') for frame in frames] for arm,(model,condition) in self.models.items()}
        for leg in [0,1]:
            for arm in cfg['arms']:torch.testing.assert_close(contexts['control'][leg],contexts[arm][leg],rtol=0,atol=0)
        measured=[];usable=[];selection=[];receipts=[]
        for shear in [.02,-.02]:
            v,u,r=ev.measured_leg(e.SIMS,case,shear,ids);measured.append(v);usable.append(u);selection.append(e.output_cuts(v)&u[:,None]);receipts.append(r)
        common=None;target_pass=None
        if axis==1:
            all_values=[measured[0]];all_u=[usable[0]];split=e.DATA/'sbsi/galsbi_resim/constant_split_20261105/sims'
            for h in [.02,-.02]:
                v,u,_=ev.measured_leg(split,case,h,ids);all_values.append(v);all_u.append(u)
            all_values.append(measured[1]);all_u.append(usable[1]);common=np.logical_and.reduce(all_u)
            target_pass=np.mean([e.output_cuts(v)[:,center] for v in all_values],axis=0)
        bins=np.digitize(context[:,4],cfg['intrinsic_size_edges_arcsec']);nc=len(e.CUTS)
        sums={arm:np.zeros((3,2,nc,5,5)) for arm in active};association={arm:np.zeros(5) for arm in active};sim=np.zeros((2,nc,5,3))
        for leg in range(2):
            for k in range(nc):
                for b in range(5):
                    keep=selection[leg][:,k]&(bins==b);sim[leg,k,b]=[keep.sum(),*measured[leg][keep,:2].sum(0)]
        started=time.monotonic();print('READY',case,axis,'arms',active,'pairs',len(pairs),flush=True)
        with torch.no_grad():
            for block,lo in enumerate(range(0,len(ids),1024)):
                hi=min(len(ids),lo+1024);p0=np.searchsorted(pair_ids,ids[lo]);p1=np.searchsorted(pair_ids,ids[hi-1],side='right')
                draws_by_arm={arm:c.portable.physical_context_draws(self.models[arm][0],[ctx[lo:hi] for ctx in contexts[arm]],64,7301+10000019*case+1000003*block) for arm in active}
                # The frozen emulator depends on photometry and fixed truth pairs,
                # so share exact tree cells across arms and shear legs.
                joint=torch.cat([v for arm in active for v in draws_by_arm[arm]],dim=1).cpu().numpy()
                pooled,_=predict_scene_draw_responses(self.emulator,pairs.iloc[p0:p1],ids[lo:hi],joint)
                if block==0:
                    pooling_max_abs=0.
                    for ai,arm in enumerate(active):
                        for leg,v in enumerate(draws_by_arm[arm]):
                            separate,_=predict_scene_draw_responses(self.emulator,pairs.iloc[p0:p1],ids[lo:hi],v.cpu().numpy())
                            part=pooled[:,(2*ai+leg)*64:(2*ai+leg+1)*64]
                            pooling_max_abs=max(pooling_max_abs,float(np.abs(part-separate).max()))
                            # Summing pair predictions in200k-row batches can
                            # change the final floating-point addition grouping.
                            np.testing.assert_allclose(part,separate,rtol=2e-14,atol=2e-14)
                    print('POOLED_EMULATOR_ROUNDOFF_CHECK',case,axis,pooling_max_abs,flush=True)
                for ai,arm in enumerate(active):
                    values=draws_by_arm[arm]
                    pass_probability=np.zeros(hi-lo)
                    for leg,v in enumerate(values):
                        assert bool(torch.isfinite(v).all())
                        response=pooled[:,(2*ai+leg)*64:(2*ai+leg+1)*64]
                        shear=torch.zeros(2,device='cuda',dtype=torch.float64);shear[axis-1]=.02 if leg==0 else -.02
                        transformed,ld=disk_response_transport(v[...,:2],torch.as_tensor(response,device='cuda')[...,None]*shear)
                        assert bool(torch.isfinite(transformed).all() and torch.isfinite(ld).all())
                        shift=transformed-v[...,:2];keep=e.output_cuts(v);mass,flow,emu=e.draw_moments(keep,v,shift)
                        own=torch.cat([mass[...,None],flow,emu],-1).cpu().numpy()
                        uncut=torch.cat([torch.ones_like(mass[:,:1]),v[...,:2].mean(1),shift.mean(1)],-1).cpu().numpy();actual=selection[leg][lo:hi,:,None]*uncut[:,None,:]
                        for mode,q,weight in [(0,own,probability[leg,lo:hi]),(1,own,usable[leg][lo:hi].astype(float)),(2,actual,np.ones(hi-lo))]:
                            for b in range(5):
                                take=bins[lo:hi]==b;sums[arm][mode,leg,:,b]+=np.einsum('n,nkq->kq',weight[take],q[take])
                        pass_probability+=.5*mass[:,center].cpu().numpy()
                    if common is not None:
                        take=common[lo:hi];y=target_pass[lo:hi][take];p=pass_probability[take];b=score[lo:hi][take]
                        association[arm]+=[take.sum(),y.sum(),(y*b).sum(),p.sum(),(p*b).sum()]
                if block%100==0:print('PROGRESS',case,axis,block,round(time.monotonic()-started,1),flush=True)
        with np.load(c.ROOT.parent/f'alignment_target9_20261106/axis{axis}/case{case}.npz') as z:
            np.testing.assert_array_equal(z['cuts'],e.CUT_NAMES);np.testing.assert_allclose(sim,z['sim'],rtol=1e-12,atol=1e-8)
        for arm in active:
            np.testing.assert_array_equal(sums[arm][2,...,0],sim[...,0]);assert np.isfinite(sums[arm]).all()
            out=paths[arm];temp=out.with_name(out.stem+'.tmp.npz');np.savez(temp,model=sums[arm],sim=sim,cuts=e.CUT_NAMES,case=case,axis=axis,association=association[arm]);temp.replace(out)
            c.write(out.with_suffix('.json'),{'arm':arm,'case':case,'axis':axis,'checkpoint_sha256':self.pins[arm],'source_sha256':self.source,'data_sha256':c.sha(out),
                'emulator_sha256':self.preset['emulator']['sha256'],'classifier_sha256':self.preset['classifier']['sha256'],'measured':receipts,
                'simulation_replay_passed':True,'reference_score_replay_passed':True,'true_selection_mass_exact':True,'pooled_emulator_first_block_max_abs':pooling_max_abs,'pooled_emulator_roundoff_check_passed':True,'seconds_for_both_arms':time.monotonic()-started})
        print('DONE_CASE',case,axis,time.monotonic()-started,flush=True);gc.collect()
def run(worker=None):
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4);cfg=c.manifest()
    if worker is not None:assert (c.ROOT/'pilot_passed.json').exists()
    tasks=[cfg['pilot_task']] if worker is None else cfg['worker_tasks'][worker]
    instance=Worker()
    for index in tasks:instance.evaluate(cfg['constant_tasks'][index])
    if worker is None:
        task=cfg['constant_tasks'][cfg['pilot_task']]
        receipts={arm:json.loads((c.ROOT/'constant'/arm/f"axis{task['axis']}"/f"case{task['case']}.json").read_text()) for arm in cfg['arms']}
        c.write(c.ROOT/'pilot_passed.json',receipts);print('PILOT_PASSED',flush=True)
    print('DONE_WORKER',worker,flush=True)
if __name__=='__main__':run(int(os.environ['SLURM_ARRAY_TASK_ID']))
