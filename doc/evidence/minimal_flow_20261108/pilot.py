"""Numerical prerequisites, including exact no-score assembly and reload."""
import os
import json
import types
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent
c=types.ModuleType('minimal_pilot_common');c.__file__=str(H/'common.py')
exec(compile((H/'common.py').read_bytes(),c.__file__,'exec'),c.__dict__)

def main():
    assert os.environ.get('SLURM_JOB_ID'); c.verify(); cfg=c.config(); torch.set_num_threads(4)
    t=c.load('minimal_pilot_train',H/'train.py');m=t.m
    assert c.sha(cfg['teacher']['path'])==cfg['teacher']['sha256']
    teacher,condition,target,art=c.old.load(cfg['teacher']['path']);teacher.requires_grad_(False)
    print('TEACHER',art['config'],'features',condition.base.feature_names,flush=True)
    source=np.load(Path(cfg['prepared_root'])/'train/context.npy',mmap_mode='r')
    x=torch.from_numpy(np.array(source[:32])).to('cuda');records={}
    for arm in cfg['arms']:
        student=m.initialize(teacher,arm);student.requires_grad_(True)
        check=t.verify(student,teacher,condition,target,arm,x)
        out=c.ROOT/arm['name'];out.mkdir(exist_ok=True)
        m.save(out/'initial.pt',student,condition,target,art['config'],arm,{'initial_only':True})
        reload,pre,_,_=m.load(out/'initial.pt')
        z=torch.from_numpy(c.portable.latent_bank(len(x),2,4,202611081702)).to('cuda')[:,0]
        with torch.no_grad():
            a=student.forward_physical(z,m.contexts(x,arm))[0]
            b=reload.forward_physical(z,m.contexts(x,arm))[0]
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        check.update(parameters=sum(p.numel() for p in student.parameters()),reload_exact=True,
                     context_dim=student.context_dim,teacher_parameters=sum(p.numel() for p in teacher.parameters()))
        records[arm['name']]=check
        print('NUMERICAL_ARM_PASSED',arm['name'],check,flush=True)
    c.write(c.ROOT/'numerical_passed.json',dict(manifest_sha256=c.sha(H/'manifest.json'),records=records))
    c.write(H/'numerical_passed.json',dict(manifest_sha256=c.sha(H/'manifest.json'),records=records))
    print('NUMERICAL_PILOT_PASSED',flush=True)
if __name__=='__main__': main()
