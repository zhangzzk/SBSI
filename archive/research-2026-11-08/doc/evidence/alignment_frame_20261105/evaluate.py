"""Frozen-family standalone comparison; run with the installed F8 interpreter."""
import argparse
import gc
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from scripts.train_fixed_g0_flow import load_split, nll_pass
from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model

D = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
ROOT = D / 'alignment_frame_20261105'
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('experimental_frame',HERE/'frame_model.py')
fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)
FAMILIES = {
    'fs2': ('v36_f8_20260924_v1', 'bd51dbbbeeb72763f6982a958cb8f3a0b2d3906d4f78935d9d959fa85a17797e'),
    'galsbi': ('galsbi_f8c8_20261102', 'abac069f66cc24fbe19a4127b9acfeb5b87c09414c662dc9380ffb115b9a4dbb'),
}
CUT_NAMES = ['uncut', 'mag25', 'mag25.8', 'radius3', 'radius3_mag25',
             'radius3_mag25.8', 'radius3.75_mag25.8']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(v):
    r, f = v[..., 2], v[..., 3]
    bright, faint = f > 100, f > 10**(-.4*(25.8-30))
    return torch.stack([torch.ones_like(r, dtype=torch.bool), bright, faint,
                        r > 3, (r > 3) & bright, (r > 3) & faint,
                        (r >= 3.75) & faint], -1).double()


def summarize(v):
    w = weights(v)
    # object, cut, [mass,e1,e2], averaging only the draw dimension
    return torch.cat([w[..., None], w[..., None]*v[..., None, :2]], -1).mean(1)


def alignment(profile, v, intrinsic, sizebin):
    n, draws = v.shape[:2]
    radius = v[..., 2]
    measured_bin = torch.bucketize(radius.contiguous(),
                                  torch.arange(2., 5.001, .1, dtype=torch.float64, device='cuda'), right=True)
    cell = (sizebin[:, None]*32 + measured_bin).reshape(-1)
    dot = (v[..., :2]*intrinsic[:, None, :]).sum(-1)
    power = (intrinsic**2).sum(-1)[:, None].expand(n, draws)
    logr = torch.where(radius > 0, torch.log(radius), torch.zeros_like(radius))
    cols = torch.stack([torch.ones_like(dot), power, dot, logr, logr**2, dot*logr, dot**2], -1)
    for j, limit in enumerate([100., 10**(-.4*(25.8-30))]):
        keep = ((v[..., 3] > limit) & (radius > 0)).double()
        profile[j].index_add_(0, cell, (cols*keep[..., None]/draws).reshape(-1, 7))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('family', choices=['cartesian','frame'])
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    root = D/'galsbi_f8c8_20261102'
    domain = root/'harmonized/domain'
    assert (ROOT/args.family/'completed.json').exists()
    checkpoint = ROOT/args.family/'selected.pt'
    expected=sha(checkpoint)
    manifest = json.loads((domain/'manifest.json').read_text())
    cases = manifest['split']['validation_cases']
    out = ROOT/(args.family+'_eval')
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(family=args.family, source_sha256=sha(__file__), checkpoint=str(checkpoint),
                   checkpoint_sha256=expected, domain_manifest_sha256=sha(domain/'manifest.json'),
                   cases=cases, cuts=CUT_NAMES, branches=['measured', 'model_bank0', 'model_bank1'],
                   size_edges=[.15,.25,.4,.7], crowd_edges=[.1,.3,1,3], draws_per_bank=16,
                   profile_radius_edges=np.arange(2,5.001,.1).tolist(),
                   profile_columns=['count','intrinsic_power','dot','ln_radius','ln_radius_sq','dot_ln_radius','dot_sq'],
                   job=os.environ['SLURM_JOB_ID'], runtime=torch.__version__)
    (out/'protocol.json').write_text(json.dumps(receipt, indent=2)+'\n')
    model,condition,target,artifact=fm.load(checkpoint)
    model.eval()
    started = time.monotonic()
    validation=load_split(domain,cases)
    ix=np.random.default_rng(202611052701).choice(len(validation['target_raw']),500_000,replace=False)
    vc=torch.as_tensor(condition.transform_frame(pd.DataFrame(validation['context_raw'][ix],columns=FLOW_FEATURES)),device='cuda')
    vt=torch.as_tensor(target.transform_array(validation['target_raw'][ix]),device='cuda')
    nll={'selected':nll_pass(model,vt,vc)}
    np.testing.assert_allclose(nll['selected']['nll'],artifact['metadata']['validation_nll'],rtol=0,atol=1e-6)
    if args.family=='cartesian':
        for name,relative in [('retained','joint/joint_probability_refinement_full/selected.pt'),('original_nll','base/flow/nll/selected.pt')]:
            original=load_measurement_model(root/relative,device='cuda')
            original_context=original.context_tensor(pd.DataFrame(validation['context_raw'][ix],columns=FLOW_FEATURES))
            original_target=torch.as_tensor(original.target_transform.transform_array(validation['target_raw'][ix]),device='cuda')
            np.testing.assert_array_equal(original_target.cpu().numpy(),vt.cpu().numpy())
            nll[name]=nll_pass(original.model,original_target,original_context)
            del original,original_context,original_target
    (out/'validation_nll.json').write_text(json.dumps(nll,indent=2)+'\n')
    del validation,vc,vt;gc.collect();torch.cuda.empty_cache()
    print('VALIDATION_NLL',args.family,nll,flush=True)
    for case in cases:
        dest = out/f'case{case:03d}.npz'
        if dest.exists():
            raise FileExistsError(dest)
        data = load_split(domain, [case])
        raw = data['context_raw'][data['pairs'][:,0]]
        size = np.digitize(raw[:,4],[.15,.25,.4,.7])
        cell = size*5 + np.digitize(raw[:,6],[.1,.3,1,3])
        ci = torch.as_tensor(cell, device='cuda')
        sizebin = torch.as_tensor(size, device='cuda')
        intrinsic = torch.as_tensor(raw[:,:2], dtype=torch.float64, device='cuda')
        context = torch.as_tensor(condition.transform_frame(pd.DataFrame(data['context_raw'], columns=FLOW_FEATURES)),device='cuda')
        pairs = torch.as_tensor(data['pairs'], device='cuda')
        gamma = torch.as_tensor(data['gamma'], dtype=torch.float64, device='cuda')
        measured = [torch.as_tensor(data[k], dtype=torch.float64, device='cuda')
                    for k in ['pair_zero_target','pair_shear_target']]
        base = torch.zeros((3,25,7,3), dtype=torch.float64, device='cuda')
        cross = torch.zeros((3,25,7,3,2), dtype=torch.float64, device='cuda')
        normal = torch.zeros((25,2,2), dtype=torch.float64, device='cuda')
        count = torch.bincount(ci, minlength=25).double()
        normal.index_add_(0, ci, gamma[:,:,None]*gamma[:,None,:])
        profile = torch.zeros((3,2,160,7), dtype=torch.float64, device='cuda')
        nonpositive_radius = np.zeros((3,2),dtype=np.int64)
        print('CASE_START', args.family, case, len(pairs), 'elapsed', time.monotonic()-started, flush=True)
        with torch.no_grad():
            for block, lo in enumerate(range(0,len(pairs),2048)):
                hi = min(len(pairs),lo+2048)
                index, g = ci[lo:hi], gamma[lo:hi]
                for branch in range(3):
                    if branch == 0:
                        values = [v[lo:hi,None,:] for v in measured]
                    else:
                        seed = 202611050700 + 10000019*case + 1000003*(branch-1) + block
                        values = physical_context_draws(model,
                                    [context[pairs[lo:hi,j]] for j in [0,1]],16,seed)
                    assert all(bool(torch.isfinite(v).all()) for v in values)
                    for leg, v in enumerate(values):
                        nonpositive_radius[branch,leg] += int((v[...,2] <= 0).sum())
                    m0, m1 = map(summarize,values)
                    base[branch].index_add_(0,index,m0)
                    cross[branch].index_add_(0,index,(m1-m0)[...,None]*g[:,None,None,:])
                    alignment(profile[branch],values[0],intrinsic[lo:hi],sizebin[lo:hi])
        arrays = {k:v.cpu().numpy() for k,v in [('base',base),('cross',cross),('normal',normal),
                                                ('count',count),('profile',profile)]}
        for v in arrays.values():
            assert np.isfinite(v).all()
        np.savez(dest,**arrays,case=case,nonpositive_radius=nonpositive_radius)
        (out/f'case{case:03d}.json').write_text(json.dumps({'case':case,'pair_count':len(pairs),
            'per_case':data['per_case_matching'],'elapsed':time.monotonic()-started},indent=2)+'\n')
        print('CASE_DONE',args.family,case,'elapsed',time.monotonic()-started,flush=True)
    (out/'completed.json').write_text(json.dumps({'cases':cases,'seconds':time.monotonic()-started,
                                                 'source_sha256':sha(__file__)},indent=2)+'\n')


if __name__ == '__main__':
    main()
