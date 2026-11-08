"""Compare latent banks and physical draws on two GPU families."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
if sys.argv[1]=='compare':
    metas=[json.loads((ROOT/f'device_{x}.json').read_text()) for x in ['a40','v100']]
    result={'devices':metas,'portable_source_sha256':hashlib.sha256((HERE.parent/'alignment_target9_20261106/portable_draws.py').read_bytes()).hexdigest(),'new_max_abs_difference':{},'legacy_hashes_equal':{}}
    with np.load(ROOT/'device_a40.npz') as a,np.load(ROOT/'device_v100.npz') as b:
        for k in a.files:
            close=np.isclose(a[k],b[k],rtol=2e-5,atol=2e-6)
            result.setdefault('original_tolerance_check',{})[k]={'passed':bool(close.all()),'mismatched_elements':int((~close).sum()),'total_elements':int(close.size)}
            result['new_max_abs_difference'][k]=float(np.max(np.abs(a[k]-b[k])))
            aa,bb=a[k],b[k]
            wa=(aa[...,2]>3)&(aa[...,3]>100);wb=(bb[...,2]>3)&(bb[...,3]>100)
            result.setdefault('roundoff_diagnostics',{})[k]={'max_abs_per_column':np.max(np.abs(aa-bb),axis=(0,1)).tolist(),'mean_shape_difference':(aa[...,:2]-bb[...,:2]).mean((0,1)).tolist(),'cut_mask_disagreements':int((wa!=wb).sum()),'draw_count':wa.size}
    for n in metas[0]['banks']:
        assert metas[0]['banks'][n]['portable_sha256']==metas[1]['banks'][n]['portable_sha256']
        result['legacy_hashes_equal'][n]=metas[0]['banks'][n]['legacy_sha256']==metas[1]['banks'][n]['legacy_sha256']
    result['physical_allclose_passed']=all(v['passed'] for v in result['original_tolerance_check'].values())
    result['purpose']='Descriptive hardware-difference audit; records original pointwise tolerance failures rather than relaxing them. Independent whole-catalogue replay controls four-state acceptance.'
    (ROOT/'v100_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
else:
    import torch
    from scripts import joint_response_axes as axes
    e=module('helper',HERE.parent/'alignment_factorized_constant_20261105/helpers.py')
    fm=module('factorized',HERE.parent/'alignment_factorized_20261105/frame_model.py')
    portable=module('portable',HERE.parent/'alignment_target9_20261106/portable_draws.py')
    m=json.loads((HERE/'manifest.json').read_text());model,condition,_,_=fm.load(m['arms']['nine_weak']['path'])
    with np.load(e.OUT/'parent/case1000.npz') as z:context=z['context']
    frames=axes.shear_frames(e.full_context(context[:1024]),.02,1)
    contexts=[torch.as_tensor(condition.transform_frame(f),dtype=torch.float32,device='cuda') for f in frames]
    info={'gpu':torch.cuda.get_device_name(),'tf32':torch.backends.cuda.matmul.allow_tf32,'precision':torch.get_float32_matmul_precision(),'banks':{}};out={}
    for n in [1024,len(context)%1024]:
        seed=7301+10000019*1000
        bank=portable.latent_bank(n,64,4,seed)
        old=torch.randn(n,32,4,dtype=torch.float32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed)).cpu().numpy()
        info['banks'][str(n)]={'portable_sha256':digest(bank),'legacy_sha256':digest(old)}
        with torch.no_grad():values=portable.physical_context_draws(model,[c[:n] for c in contexts],64,seed)
        for leg,v in enumerate(values):out[f'{n}_{leg}']=v.cpu().numpy()
    tag=sys.argv[1];np.savez(ROOT/f'device_{tag}.npz',**out);(ROOT/f'device_{tag}.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
