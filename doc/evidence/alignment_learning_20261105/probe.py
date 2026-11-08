"""Frozen checkpoints: train/validation alignment, strata, and angular density."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scripts.train_fixed_g0_flow import load_split
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model

ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
OUT=ROOT/'alignment_learning_20261105'
FAMILIES={'fs2':'v36_f8_20260924_v1','galsbi':'galsbi_f8c8_20261102'}
GROUPS=['all']+[f'n{i}' for i in range(4)]+[f'mag{i}' for i in range(4)]+[f'crowd{i}' for i in range(4)]+[f'angle{i}' for i in range(4)]+[f'ellip{i}' for i in range(3)]+[f'n{i}_mag{j}' for i in range(4) for j in range(4)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def group_indices(raw):
    n=np.digitize(raw[:,2],[1,2,4]);m=np.digitize(raw[:,3],[23,24,25])
    c=np.digitize(raw[:,6],[.01,.1,1])
    a=np.minimum(3,((np.arctan2(raw[:,1],raw[:,0])+np.pi)*2/np.pi).astype(int))
    e=np.digitize(np.linalg.norm(raw[:,:2],axis=1),[.2,.5])
    return np.stack([np.zeros(len(raw),int),1+n,5+m,9+c,13+a,17+e,20+4*n+m],1)

def add_profile(out,v,intr,groups):
    nd=v.shape[1]
    rad=torch.bucketize(v[...,2].contiguous(),torch.arange(2.,5.001,.1,device='cuda',dtype=torch.float64),right=True)
    dot=(v[...,:2]*intr[:,None,:]).sum(-1)
    cross=v[...,1]*intr[:,None,0]-v[...,0]*intr[:,None,1]
    power=intr.square().sum(-1)[:,None].expand_as(dot)
    cols=torch.stack([torch.ones_like(dot),power,dot,(dot<0).double(),torch.minimum(dot,torch.zeros_like(dot)),torch.maximum(dot,torch.zeros_like(dot)),cross],-1)/nd
    for j,flux in enumerate([100.,10**(-.4*(25.8-30))]):
        values=cols*(v[...,3]>flux)[...,None]
        for g in groups.T:
            cell=(g[:,None]*32+rad).reshape(-1)
            out[j].index_add_(0,cell,values.reshape(-1,7))

def main():
    a=argparse.ArgumentParser();a.add_argument('family',choices=FAMILIES);args=a.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    torch.set_num_threads(4)
    source=ROOT/FAMILIES[args.family];domain=source/'harmonized/domain'
    manifest=json.loads((domain/'manifest.json').read_text())
    cases={'train':manifest['split']['train_cases'][:40],'validation':manifest['split']['validation_cases']}
    paths={'nll':source/'base/flow/nll/selected.pt','adopted':source/'joint/joint_probability_refinement_full/selected.pt'}
    prior=json.loads((ROOT/'fs2_galsbi_transfer_20261105'/args.family/'protocol.json').read_text())
    assert sha(paths['adopted'])==prior['checkpoint_sha256']
    nll_receipt=json.loads((source/'base/flow/nll_completed.json').read_text())
    assert sha(paths['nll'])==nll_receipt['selected_sha256']
    models={k:load_measurement_model(p,device='cuda') for k,p in paths.items()}
    for b in models.values():b.model.eval()
    output=OUT/args.family;output.mkdir(parents=True,exist_ok=True)
    protocol={'family':args.family,'cases':cases,'groups':GROUPS,'branches':['simulation','nll_bank0','nll_bank1','adopted_bank0','adopted_bank1'],
              'columns':['count','intrinsic_power','dot','negative_dot_count','negative_dot','positive_dot','perpendicular_dot'],
              'models':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},'source_sha256':sha(__file__),
              'density_columns':['count','sum_logp','sum_logp_squared','sum_rotated_minus_logp','sum_rotation_difference_squared'],
              'seed':'202611051900 + 10000019*case + 1000003*bank + block','job':os.environ['SLURM_JOB_ID']}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic()
    for split,ids in cases.items():
        for case in ids:
            dest=output/f'{split}_case{case:03d}.npz'
            if dest.exists():raise FileExistsError(dest)
            d=load_split(domain,[case]);raw=d['context_raw'][d['pairs'][:,0]]
            keep=(raw[:,4]>=.15)&(raw[:,4]<.25)
            raw=raw[keep];target=d['pair_zero_target'][keep]
            groups=group_indices(raw);intr=torch.as_tensor(raw[:,:2],device='cuda',dtype=torch.float64)
            gi=torch.as_tensor(groups,device='cuda')
            profiles=torch.zeros((5,2,len(GROUPS)*32,7),device='cuda',dtype=torch.float64)
            density=np.zeros((2,2,32,5))
            rotated=raw.copy();rotated[:,0]=-raw[:,1];rotated[:,1]=raw[:,0]
            t_rot=target.copy();t_rot[:,0]=-target[:,1];t_rot[:,1]=target[:,0]
            contexts={k:b.context_tensor(pd.DataFrame(raw,columns=FLOW_FEATURES)) for k,b in models.items()}
            rotcontext={k:b.context_tensor(pd.DataFrame(rotated,columns=FLOW_FEATURES)) for k,b in models.items()}
            targets={k:torch.as_tensor(b.target_transform.transform_array(target),device='cuda') for k,b in models.items()}
            rotatedtargets={k:torch.as_tensor(b.target_transform.transform_array(t_rot),device='cuda') for k,b in models.items()}
            with torch.no_grad():
                for block,lo in enumerate(range(0,len(raw),2048)):
                    hi=min(lo+2048,len(raw));s=slice(lo,hi)
                    measured=torch.as_tensor(target[s,None,:],device='cuda')
                    add_profile(profiles[0],measured,intr[s],gi[s])
                    for j,(name,b) in enumerate(models.items()):
                        for bank in range(2):
                            seed=202611051900+10000019*case+1000003*bank+block
                            v=physical_context_draws(b.model,[contexts[name][s]],16,seed)[0]
                            assert bool(torch.isfinite(v).all())
                            add_profile(profiles[1+2*j+bank],v,intr[s],gi[s])
                        lp=b.model.log_prob(targets[name][s],contexts[name][s]).cpu().numpy()
                        lr=b.model.log_prob(rotatedtargets[name][s],rotcontext[name][s]).cpu().numpy()-lp
                        assert np.isfinite(lp).all() and np.isfinite(lr).all()
                        rb=np.digitize(target[s,2],np.arange(2,5.001,.1))
                        for m,flux in enumerate([100.,10**(-.4*(25.8-30))]):
                            w=(target[s,3]>flux)
                            for k,x in enumerate([np.ones(len(lp)),lp,lp*lp,lr,lr*lr]):
                                density[j,m,:,k]+=np.bincount(rb,weights=x*w,minlength=32)
            arr=profiles.cpu().numpy().reshape(5,2,len(GROUPS),32,7)
            for start,width in [(1,4),(5,4),(9,4),(13,4),(17,3),(20,16)]:
                np.testing.assert_allclose(arr[:,:,start:start+width].sum(2),arr[:,:,0],rtol=1e-10,atol=1e-8)
            np.testing.assert_allclose(arr[...,4]+arr[...,5],arr[...,2],rtol=1e-10,atol=1e-8)
            np.savez(dest,profile=arr,density=density)
            (dest.with_suffix('.json')).write_text(json.dumps({'case':case,'split':split,'small_pairs':len(raw),'all_pairs':len(keep),'matching':d['per_case_matching'],'seconds':time.monotonic()-started},indent=2)+'\n')
            print('DONE',args.family,split,case,len(raw),round(time.monotonic()-started,1),flush=True)
    (output/'completed.json').write_text(json.dumps({'seconds':time.monotonic()-started,'source_sha256':sha(__file__)})+'\n')

if __name__=='__main__':main()
