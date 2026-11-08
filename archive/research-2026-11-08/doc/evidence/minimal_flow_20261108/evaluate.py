"""Small-model pilots on both axes/nine cuts, with a live teacher replay gate."""
import json
import os
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
from scripts import joint_response_axes as axes
from sbsi.output_conditioned_response import DiskResponseMoment,predict_scene_draw_responses
from sbsi.disk_response_transport import disk_response_transport
H=Path(__file__).resolve().parent
c=types.ModuleType('minimal_evaluate_common');c.__file__=str(H/'common.py')
exec(compile((H/'common.py').read_bytes(),c.__file__,'exec'),c.__dict__)
m=c.load('minimal_evaluate_model',H/'model.py')
e=c.old.c.e

class Worker:
    def __init__(self):
        c.verify(); self.cfg=c.config();cfg=self.cfg
        self.models={};self.identities={}
        for arm in cfg['arms']:
            path=c.ROOT/arm['name']/'final.pt';r=json.loads(path.with_name('completed.json').read_text())
            assert c.sha(path)==r['sha256'] and r['reload_exact']
            assert r['provenance']['manifest']==c.sha(H/'manifest.json')
            model,condition,_,_=m.load(path);self.models[arm['name']]=(model,condition);self.identities[arm['name']]=r['sha256']
        teacher,condition,_,_=c.old.load(cfg['teacher']['path']);self.models['teacher']=(teacher,condition)
        self.identities['teacher']=cfg['teacher']['sha256']
        self.preset,self.paths,self.conditions=e.preset()
        assert self.preset['emulator']['sha256']==cfg['emulator_sha256'] and self.preset['classifier']['sha256']==cfg['classifier_sha256']
        self.emulator=DiskResponseMoment(self.paths['emulator'].parent,nthread=4,device='cuda')
        assert self.emulator.conditioned and self.emulator.response_coordinate=='disk_velocity'
        e.CUTS=cfg['cuts'];e.CUT_NAMES=tuple(x[0] for x in e.CUTS)
        p=Path(cfg['measurement_sums']);assert c.sha(p)==cfg['measurement_sums_sha256']
        with np.load(p) as z:self.sim=z['new'];self.reference=z['model'];self.cases=list(z['cases']);np.testing.assert_array_equal(z['cuts'],e.CUT_NAMES)
        print('EVALUATION_MODELS_READY',torch.cuda.get_device_name(),flush=True)

    def evaluate(self,index):
        cfg=self.cfg;case,axis=[cfg['tasks'][index][k] for k in ('case','axis')]
        dest=c.ROOT/'constant'/f'axis{axis}'/f'case{case}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.with_suffix('.json').exists():
            r=json.loads(dest.with_suffix('.json').read_text());assert c.sha(dest)==r['data_sha256'] and r['models']==self.identities and r['source_sha256']==c.sha(__file__)
            print('SKIP_EVALUATION',case,axis,flush=True);return
        assert not dest.exists()
        parent=Path(cfg['parent_root']);flagroot=parent if axis==1 else c.ROOT.parent/'alignment_axis2_constant_20261105'
        flag=flagroot/f'c8/case{case}.npz';par=parent/f'parent/case{case}.npz'
        with np.load(par) as z,np.load(flag) as f:
            ids=z['input_index'];context=z['context'].astype(float);np.testing.assert_array_equal(ids,f['input_index']);prob=np.stack([f['probability_plus'],f['probability_minus']])
        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        if axis==2:
            meta=json.loads(flag.with_suffix('.json').read_text());assert meta['axis']==2 and meta['parent_sha256']==c.sha(par) and meta['classifier_sha256']==cfg['classifier_sha256']
        e.SIMS=Path(cfg['sim_roots'][str(axis)])
        truth,_=e.truth(case,.02);np.testing.assert_array_equal(ids,truth.index_input)
        assert (truth[f'gamma{axis}_input']==.02).all() and (truth[f'gamma{3-axis}_input']==0).all()
        pairs=e.pair_table(e.scene_frame(truth),np.arange(len(truth)),self.conditions);pairids=pairs.index_input_p.to_numpy()
        score=c.old.c.score_pairs(self.emulator,pairs,ids)
        frames=axes.shear_frames(e.full_context(context),.02,axis)
        for frame in frames:frame[c.old.c.FEATURE]=score
        contexts={name:[torch.as_tensor(pre.transform_frame(f),dtype=torch.float32,device='cuda') for f in frames] for name,(_,pre) in self.models.items()}
        names=list(self.models);sums=np.zeros((len(names),2,9,5));started=time.monotonic();pool_error=0.
        print('CASE_READY',case,axis,'parents',len(ids),'models',names,flush=True)
        with torch.no_grad():
            for block,lo in enumerate(range(0,len(ids),1024)):
                hi=min(lo+1024,len(ids));p0=np.searchsorted(pairids,ids[lo]);p1=np.searchsorted(pairids,ids[hi-1],side='right');pp=pairs.iloc[p0:p1]
                values={name:c.portable.physical_context_draws(self.models[name][0],[x[lo:hi] for x in contexts[name]],64,7301+10000019*case+1000003*block) for name in names}
                joined=torch.cat([v for name in names for v in values[name]],dim=1).cpu().numpy()
                pooled,_=predict_scene_draw_responses(self.emulator,pp,ids[lo:hi],joined)
                if block==0:
                    for ai,name in enumerate(names):
                        for leg,v in enumerate(values[name]):
                            single,_=predict_scene_draw_responses(self.emulator,pp,ids[lo:hi],v.cpu().numpy())
                            part=pooled[:,(2*ai+leg)*64:(2*ai+leg+1)*64]
                            pool_error=max(pool_error,float(np.abs(single-part).max()));np.testing.assert_allclose(single,part,rtol=2e-14,atol=2e-14)
                    print('POOL_REPLAY_PASSED',case,axis,pool_error,flush=True)
                for ai,name in enumerate(names):
                    for leg,v in enumerate(values[name]):
                        assert torch.isfinite(v).all() and (v[:,:,:2].square().sum(-1)<=1+1e-12).all() and (v[:,:,2:]>0).all()
                        shear=torch.zeros(2,dtype=torch.float64,device='cuda');shear[axis-1]=.02 if leg==0 else -.02
                        response=pooled[:,(2*ai+leg)*64:(2*ai+leg+1)*64]
                        transformed,_=disk_response_transport(v[:,:,:2],torch.as_tensor(response,device='cuda')[:,:,None]*shear)
                        mass,flow,emu=e.draw_moments(e.output_cuts(v),v,transformed-v[:,:,:2])
                        q=torch.cat([mass[:,:,None],flow,emu],-1).cpu().numpy()
                        sums[ai,leg]+=np.einsum('n,nkq->kq',prob[leg,lo:hi],q)
                if block%100==0:print('EVAL_PROGRESS',case,axis,block,round(time.monotonic()-started,1),flush=True)
        ci=self.cases.index(case);sl=slice((axis-1)*2,axis*2);ref=self.reference[ci,sl]
        replay=float(np.abs(sums[-1,...,1:]/sums[-1,...,:1]-ref[...,1:]/ref[...,:1]).max())
        assert replay<1e-9,replay
        np.testing.assert_allclose(sums[-1,...,0],ref[...,0],rtol=1e-12,atol=2e-7)
        tmp=dest.with_name(dest.stem+'.tmp.npz');np.savez(tmp,model=sums,sim=self.sim[ci,sl],names=names,cuts=e.CUT_NAMES,case=case,axis=axis);tmp.replace(dest)
        c.write(dest.with_suffix('.json'),dict(case=case,axis=axis,models=self.identities,data_sha256=c.sha(dest),source_sha256=c.sha(__file__),
                    manifest_sha256=c.sha(H/'manifest.json'),parent_sha256=c.sha(par),classifier_rows_sha256=c.sha(flag),
                    measurement_sums_sha256=cfg['measurement_sums_sha256'],pool_max_abs=pool_error,teacher_replay_max_abs=replay,seconds=time.monotonic()-started))
        print('EVALUATION_CASE_DONE',case,axis,'teacher replay',replay,flush=True)

def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4);cfg=c.config();w=Worker()
    if sys.argv[1]=='pilot':
        w.evaluate(0)
        c.write(c.ROOT/'evaluation_pilot_passed.json',{'manifest_sha256':c.sha(H/'manifest.json')})
    else:
        gate=json.loads((c.ROOT/'evaluation_pilot_passed.json').read_text());assert gate['manifest_sha256']==c.sha(H/'manifest.json')
        for index in cfg['worker_tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]:w.evaluate(index)
if __name__=='__main__':main()
