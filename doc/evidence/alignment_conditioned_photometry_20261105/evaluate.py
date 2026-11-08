"""Frozen-family standalone comparison; run with the installed F8 interpreter."""
import argparse
import gc
import importlib.util
from types import SimpleNamespace
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
SOURCE = D / 'alignment_factorized_20261105'
ROOT = D / 'alignment_conditioned_photometry_20261105'
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('experimental_frame',HERE.parent/'alignment_factorized_20261105/frame_model.py')
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


from sbsi.flow_physical_disk import positive_inverse
from sbsi.flow_disk import radial_tanh


def conditional_physical(model, z, context, photometry):
    """Draw only shape conditional on exact physical photometry; no joint density."""
    b=model.base
    assert photometry.dtype == torch.float64
    assert bool(torch.isfinite(photometry).all()) and bool((photometry>0).all())
    w=(positive_inverse(photometry/b.physical_units)-b.physical_coordinate_means)/b.physical_coordinate_scales
    shape,_=b.base.shape.forward(z[:,:2],torch.cat([context,w.to(context.dtype)],-1))
    e,_=radial_tanh(shape.double()*b.disk_coordinate_scales+b.disk_coordinate_means)
    if model.kind=='frame':e=model.rotate(e,context,inverse=True)
    return torch.cat([e,photometry],-1)


def conditional_draws(model,contexts,photometry,draws,seed):
    # Keep the original four-dimensional latent RNG stream, including unused
    # photometry coordinates, to preserve paired shape draws exactly.
    assert draws>=2 and draws%2==0
    n=len(contexts[0]);device=contexts[0].device;dtype=contexts[0].dtype
    g=torch.Generator(device=device).manual_seed(int(seed))
    half=torch.randn(n,draws//2,4,dtype=dtype,device=device,generator=g)
    z=torch.cat([half,-half],1).reshape(n*draws,4)
    return [conditional_physical(model,z,c[:,None,:].expand(n,draws,model.context_dim).reshape(n*draws,model.context_dim),
             p[:,None,:].expand(n,draws,2).reshape(n*draws,2)).reshape(n,draws,4)
            for c,p in zip(contexts,photometry)]


@torch.no_grad()
def verify_conditioning(model,context):
    g=torch.Generator(device=context.device).manual_seed(202611057301)
    z=torch.randn(len(context),4,device=context.device,dtype=context.dtype,generator=g)
    joint,_=model.forward_physical(z,context)
    reconstructed=conditional_physical(model,z,context,joint[:,2:])
    error=float((reconstructed-joint).abs().max())
    torch.testing.assert_close(reconstructed,joint,rtol=2e-5,atol=2e-6)
    changed=z.clone();changed[:,2:]+=3
    torch.testing.assert_close(conditional_physical(model,changed,context,joint[:,2:]),reconstructed,rtol=0,atol=0)
    sampled=conditional_draws(model,[context,context],[joint[:,2:],joint[:,2:]],16,7301)
    torch.testing.assert_close(sampled[0],sampled[1],rtol=0,atol=0)
    torch.testing.assert_close(sampled[0][...,2:],joint[:,None,2:].expand(-1,16,-1),rtol=0,atol=0)
    assert bool((sampled[0][...,:2].square().sum(-1)<1).all())
    return dict(joint_sample_reconstruction_max_error=error,unused_photometry_latent='exactly independent',
                conditional_leg_crn='exact',measured_photometry_preserved='exact',disk_support='passed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=['cartesian','frame'])
    args = p.parse_args()
    args.family='fs2' if args.arm=='fs2' else 'galsbi'
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    sub, expected = FAMILIES[args.family]
    root = D/sub
    domain = root/'harmonized/domain'
    assert (SOURCE/args.arm/'completed.json').exists()
    checkpoint = SOURCE/args.arm/'selected.pt'
    expected={'cartesian':'37c4afc13b045bbd118eea0d20572961eef5f6649d6b9329c1e5d09fe820afbb','frame':'7215268fc7c3543e0dd9a597ee82fb702ef1a44e5f6971c0aaa71de7bfcdfa31'}[args.arm]
    assert sha(checkpoint)==expected
    manifest = json.loads((domain/'manifest.json').read_text())
    cases = manifest['split']['validation_cases']
    out = ROOT/(args.arm+'_eval')
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(arm=args.arm, family=args.family, source_sha256=sha(__file__), checkpoint=str(checkpoint),
                   checkpoint_sha256=expected, domain_manifest_sha256=sha(domain/'manifest.json'),
                   cases=cases, cuts=CUT_NAMES, branches=['measured', 'conditional_bank0', 'conditional_bank1'], conditioning='each leg actual radius/flux; conditional shape, measured selection',
                   size_edges=[.15,.25,.4,.7], crowd_edges=[.1,.3,1,3], draws_per_bank=16,
                   profile_radius_edges=np.arange(2,5.001,.1).tolist(),
                   profile_columns=['count','intrinsic_power','dot','ln_radius','ln_radius_sq','dot_ln_radius','dot_sq'],
                   job=os.environ['SLURM_JOB_ID'], runtime=torch.__version__)
    (out/'protocol.json').write_text(json.dumps(receipt, indent=2)+'\n')
    model,condition,target,artifact=fm.load(checkpoint)
    bundle=SimpleNamespace(model=model,condition_preprocessor=condition,target_transform=target,context_tensor=lambda frame: torch.as_tensor(condition.transform_frame(frame),device='cuda'))
    bundle.model.eval()
    started = time.monotonic()
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
        context = bundle.context_tensor(pd.DataFrame(data['context_raw'], columns=FLOW_FEATURES))
        pairs = torch.as_tensor(data['pairs'], device='cuda')
        gamma = torch.as_tensor(data['gamma'], dtype=torch.float64, device='cuda')
        measured = [torch.as_tensor(data[k], dtype=torch.float64, device='cuda')
                    for k in ['pair_zero_target','pair_shear_target']]
        assert all(bool((v[:,2:] > 0).all()) for v in measured), 'Nonpositive measured photometry; stop without clipping or dropping'
        if case == cases[0]:
            verification=verify_conditioning(model,context[:32])
            (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
            print('VERIFIED',verification,flush=True)
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
                        values = conditional_draws(bundle.model,
                                    [context[pairs[lo:hi,j]] for j in [0,1]],
                                    [v[lo:hi,2:] for v in measured],16,seed)
                    assert all(bool(torch.isfinite(v).all()) for v in values)
                    if branch:
                        for v,meas in zip(values,measured):
                            torch.testing.assert_close(weights(v),weights(meas[lo:hi,None,:]).expand(-1,16,-1),rtol=0,atol=0)
                    for leg, v in enumerate(values):
                        nonpositive_radius[branch,leg] += int((v[...,2] <= 0).sum())
                    m0, m1 = map(summarize,values)
                    base[branch].index_add_(0,index,m0)
                    cross[branch].index_add_(0,index,(m1-m0)[...,None]*g[:,None,None,:])
                    alignment(profile[branch],values[0],intrinsic[lo:hi],sizebin[lo:hi])
        # Actual membership must be identical in every bank, in both response terms.
        for branch in [1,2]:
            torch.testing.assert_close(base[branch,...,0],base[0,...,0],rtol=0,atol=0)
            torch.testing.assert_close(cross[branch,...,0,:],cross[0,...,0,:],rtol=1e-12,atol=1e-12)
        arrays = {k:v.cpu().numpy() for k,v in [('base',base),('cross',cross),('normal',normal),
                                                ('count',count),('profile',profile)]}
        for v in arrays.values():
            assert np.isfinite(v).all()
        with np.load(SOURCE/(args.arm+'_eval')/f'case{case:03d}.npz') as old:
            for k in ['base','cross','profile']:np.testing.assert_allclose(arrays[k][0],old[k][0],rtol=1e-12,atol=1e-8)
            np.testing.assert_array_equal(arrays['count'],old['count'])
            np.testing.assert_allclose(arrays['normal'],old['normal'],rtol=1e-12,atol=1e-12)
        np.savez(dest,**arrays,case=case,nonpositive_radius=nonpositive_radius)
        (out/f'case{case:03d}.json').write_text(json.dumps({'case':case,'pair_count':len(pairs),
            'per_case':data['per_case_matching'],'elapsed':time.monotonic()-started},indent=2)+'\n')
        print('CASE_DONE',args.family,case,'elapsed',time.monotonic()-started,flush=True)
    (out/'completed.json').write_text(json.dumps({'cases':cases,'seconds':time.monotonic()-started,
                                                 'source_sha256':sha(__file__)},indent=2)+'\n')


if __name__ == '__main__':
    main()
