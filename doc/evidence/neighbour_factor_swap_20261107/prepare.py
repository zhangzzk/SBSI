"""Assemble frozen products; validate factors and reuse original response probes."""
import copy,gc,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sbsi.flow_disk import radial_atanh
from sbsi.flow_physical_disk import positive_inverse
from sbsi.flow_guard_response import evaluate_guard_response
from scripts.train_fixed_g0_flow import nll_pass
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('swap_prepare_common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def canonical(v):return json.dumps(v,sort_keys=True,default=lambda x:x.tolist())
def tensor(path,dtype):return torch.as_tensor(np.load(path),dtype=dtype,device='cuda')
def conditional_shape_density(model,y,ctx):
    """Evaluate the factor directly, accounting for the joint float32 ld sum."""
    b=model.base;p=y[:,2:]*b.physical_scales+b.physical_means
    up=positive_inverse(p/b.physical_units)
    wp=((up-b.physical_coordinate_means)/b.physical_coordinate_scales).to(ctx.dtype)
    us,ldisk=radial_atanh(y[:,:2]*b.disk_shape_scales+b.disk_shape_means)
    ws=((us-b.disk_coordinate_means)/b.disk_coordinate_scales).to(ctx.dtype)
    zp,lp=b.base.photometry.inverse(wp,ctx)
    zs,ls=b.base.shape.inverse(ws,torch.cat([ctx,wp],-1))
    js=ldisk+(b.disk_shape_scales.log()-b.disk_coordinate_scales.log()).sum()
    jp=(b.physical_scales.log()-b.physical_units.log()-b.physical_coordinate_scales.log()-F.logsigmoid(up)).sum(-1)
    log2pi=np.log(2*np.pi)
    conditional=-.5*(zs.double().square()+log2pi).sum(-1)+ls.double()+js
    photometry=-.5*(zp.double().square()+log2pi).sum(-1)+lp.double()+jp
    roundoff=(lp+ls).double()-lp.double()-ls.double()
    # Joint code sums factor log determinants in float32 before casting.
    torch.testing.assert_close(model.log_prob(y,ctx),conditional+photometry+roundoff,rtol=0,atol=1e-10)
    return conditional,float(roundoff.abs().max())
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    cfg=c.manifest();dest=c.ROOT/'assembly_completed.json'
    provenance={'manifest_sha256':c.sha(HERE/'manifest.json'),'source_sha256':c.sha(__file__),'sources':cfg['sources']}
    if dest.exists():
        d=json.loads(dest.read_text());assert d['provenance']==provenance
        for arm,pin in d['arms'].items():assert c.sha(c.ROOT/arm/'final.pt')==pin['sha256']
        print('SKIP_ASSEMBLY',flush=True);return
    sources={};states=[]
    for name,pin in cfg['sources'].items():
        assert c.sha(pin['path'])==pin['sha256'];sources[name]=c.m.load(pin['path'])
        model,condition,target,artifact=sources[name]
        states.append(canonical([condition.to_state(),target.to_state(),artifact['config']]))
    assert len(set(states))==1
    prefixes=('base.base.photometry.','base.base.shape.')
    ref=sources['initial'][0].state_dict()
    for name,(model,_,_,_) in sources.items():
        sd=model.state_dict();assert sd.keys()==ref.keys()
        for key,value in ref.items():
            if not key.startswith(prefixes):torch.testing.assert_close(sd[key],value,rtol=0,atol=0)
    prep=Path(cfg['prepared_root']);old=Path(cfg['original_prepared_root'])
    vc=tensor(prep/'validation/nll_context.npy',torch.float32);vt=tensor(old/'validation/nll_target.npy',torch.float64)
    pops={}
    for split in ['train','validation']:
        pc=tensor(prep/split/'probe_context.npy',torch.float32);pd=tensor(old/split/'probe_design.npy',torch.float64)
        stats=json.loads((old/split/'statistics.json').read_text())
        pops[split]=c.g.Population(pc,torch.arange(len(pc),device='cuda').reshape(-1,2),pd,stats,list(range(9)),cfg['training']['relative_response_scale'])
    z=torch.from_numpy(c.portable.latent_bank(256,4,4,202611070102)).to('cuda').reshape(-1,4)
    ctx,y=vc[:len(z)],vt[:len(z)];receipts={}
    print('ASSEMBLY_READY',torch.cuda.get_device_name(),flush=True)
    for name,(pname,sname) in cfg['recipes'].items():
        start=time.monotonic();pmodel,condition,target,artifact=sources[pname];smodel=sources[sname][0]
        model=copy.deepcopy(pmodel);model.base.base.shape.load_state_dict(smodel.base.base.shape.state_dict());model.eval()
        out=c.ROOT/name;out.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():
            v,ld=model.forward_physical(z,ctx);pv,_=pmodel.forward_physical(z,ctx)
            torch.testing.assert_close(v[:,2:],pv[:,2:],rtol=0,atol=0)
            q,joint_roundoff=conditional_shape_density(model,y,ctx)
            sq,_=conditional_shape_density(smodel,y,ctx)
            torch.testing.assert_close(q,sq,rtol=0,atol=0)
            conditional_difference=float((q-sq).abs().max())
            if pname==sname:
                torch.testing.assert_close(v,pv,rtol=0,atol=0)
                torch.testing.assert_close(model.log_prob(y,ctx),pmodel.log_prob(y,ctx),rtol=0,atol=0)
        c.m.save(out/'final.pt',model,condition,target,artifact['config'],{'factor_recipe':[pname,sname],'provenance':provenance,'optimizer_steps':0})
        loaded,_,_,_=c.m.load(out/'final.pt')
        with torch.no_grad():
            vv,ll=loaded.forward_physical(z,ctx)
            torch.testing.assert_close(v,vv,rtol=0,atol=0);torch.testing.assert_close(ld,ll,rtol=0,atol=0)
            torch.testing.assert_close(model.log_prob(y,ctx),loaded.log_prob(y,ctx),rtol=0,atol=0)
        probe={'nll':nll_pass(model,vt,vc)}
        for split,pop in pops.items():
            probe[split]=evaluate_guard_response(model,pop,np.arange(len(pop.pairs)),draws=cfg['training']['probe_draws'],groups=cfg['training']['probe_groups'],batch_size=1024,seed=202611068201)
        if name in cfg['sources']:
            path=(c.previous.ROOT/'initial_probe.json') if name=='initial' else c.previous.ROOT/name/'probe.json'
            old_probe=json.loads(path.read_text());assert abs(probe['nll']['nll']-old_probe['nll']['nll'])<2e-5
            for split in pops:np.testing.assert_allclose(probe[split]['mean_response'],old_probe[split]['mean_response'],rtol=0,atol=cfg['reference_response_atol'])
        c.write(out/'probe.json',probe)
        receipts[name]={'sha256':c.sha(out/'final.pt'),'recipe':[pname,sname],'reload_exact':True,
            'photo_source_exact':True,'conditional_shape_log_density_max_abs':conditional_difference,
            'joint_float32_logdet_roundoff_max_abs':joint_roundoff,
            'identity_exact':pname==sname,'seconds':time.monotonic()-start}
        print('DONE_PRODUCT',name,'nll',probe['nll']['nll'],'train_center_residual',probe['train']['mean_residual'][4].tolist(),flush=True)
        del model,loaded;gc.collect();torch.cuda.empty_cache()
    result={'provenance':provenance,'standardizers_and_outer_transforms_exact':True,'optimizer_steps':0,'arms':receipts}
    c.write(dest,result);c.write(HERE/'assembly_completed.json',result);print('ASSEMBLY_PASSED',flush=True)
if __name__=='__main__':main()
