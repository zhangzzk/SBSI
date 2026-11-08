import os,runpy,time
from pathlib import Path
A=runpy.run_path(str(Path(__file__).with_name('shared.py')));globals().update({k:v for k,v in A.items() if not k.startswith('__')})
import torch
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
    matched=json.loads((ROOT/'matched.json').read_text())
    for p,expected in matched['hashes'].items():assert sha(p)==expected,p
    m=load('boundary_flow',H.parent/'neighbour_conditioned_flow_20261106/model.py')
    portable=load('boundary_portable',H.parent/'alignment_target9_20261106/portable_draws.py')
    frozen=C/F.name;training=json.loads((frozen/'training_completed.json').read_text());assert training['provenance']['initial']==cfg['initial_checkpoint']
    pins={'start':cfg['initial_checkpoint'],**{k:{'path':str(frozen/k/'final.pt'),'sha256':v['sha256']} for k,v in training['arms'].items()}}
    assert list(pins)==['start','control','weighted']
    models=[];conds=[]
    for name,pin in pins.items():
        assert sha(pin['path'])==pin['sha256'];model,condition,_,_=m.load(pin['path']);models.append(model);conds.append(condition)
    assert all(x.to_state()['mean']==conds[0].to_state()['mean'] and x.to_state()['scale']==conds[0].to_state()['scale'] for x in conds)
    with np.load(ROOT/'matched.npz') as f:
        u=f['u'];B=f['B'];direction=f['direction'];group=f['group'];baseline=f['measured'][1]
    ctx=np.load(prep/'validation/probe_context.npy',mmap_mode='r').reshape(len(u),2,17)
    sums=np.zeros((3,2,200,9,len(labels)))
    seed=202611069912+100000000;blocksize=1024;half=list(range(16))+list(range(32,48))
    asgpu=lambda x:torch.as_tensor(x,device='cuda')
    start=time.monotonic()
    print('READY',torch.cuda.get_device_name(),'models',list(pins),'pairs',len(u),flush=True)
    with torch.inference_mode():
        for ib,lo in enumerate(range(0,len(u),blocksize)):
            sl=slice(lo,min(lo+blocksize,len(u)));context=asgpu(np.array(ctx[sl]));direction_gpu=asgpu(direction[sl]);ug=asgpu(u[sl]);bg=asgpu(B[sl])
            for ai,model in enumerate(models):
                p,q=portable.physical_context_draws(model,[context[:,0],context[:,1]],64,seed+ib)
                assert all(bool(torch.isfinite(t).all()&(t[:,:,2:]>0).all()) for t in (p,q))
                val=features(p,q,direction_gpu,ug,bg,torch,lambda x:x.mean(1))
                vhalf=features(p[:,half],q[:,half],direction_gpu,ug,bg,torch,lambda x:x.mean(1))
                add(sums[ai,0],group[sl],val.cpu().numpy());add(sums[ai,1],group[sl],vhalf.cpu().numpy())
            if ib%32==0 or ib==255:print('PROGRESS',ib+1,'/256',round(time.monotonic()-start,1),flush=True)
    for v in sums.reshape(-1,200,9,len(labels)):
        np.testing.assert_allclose(v[:,:,:3],baseline[:,:,:3],rtol=1e-12,atol=3e-10)
    dest=ROOT/'models.npz';np.savez(dest,sums=sums)
    write(ROOT/'models.json',dict(job=os.environ['SLURM_JOB_ID'],pins=pins,seed=seed,draws=64,block=blocksize,
        gpu=torch.cuda.get_device_name(),seconds=time.monotonic()-start,hashes={str(p):sha(p) for p in [dest,H/'shared.py',Path(__file__),Path(portable.__file__),ROOT/'matched.json']}))
    print('MODEL REDUCTIONS COMPLETE',flush=True)
if __name__=='__main__':main()
