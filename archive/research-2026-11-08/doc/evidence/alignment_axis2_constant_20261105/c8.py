"""Frozen C8 predictions for g2 legs; original parent contexts are unchanged."""
import os,json,importlib.util
from pathlib import Path
import numpy as np
import torch
from sbsi.selection_model import load_selection_model
from scripts.joint_response_axes import shear_frames
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_axis2_constant_20261105')
spec=importlib.util.spec_from_file_location('constant_helpers',HERE.parent/'alignment_factorized_constant_20261105/helpers.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
assert os.environ.get('SLURM_JOB_ID')
torch.set_num_threads(4);m=json.loads((HERE/'manifest.json').read_text());_,paths,_=e.preset()
model=load_selection_model(paths['classifier'],device='cpu')
chunk=int(os.environ['SLURM_ARRAY_TASK_ID']);cases=m['cases'][chunk*4:(chunk+1)*4]
for case in cases:
 with np.load(Path(m['parent_root'])/f'parent/case{case}.npz') as z:ids=z['input_index'];context=z['context']
 if case==m['cases'][0]:
  replay=[]
  for f in shear_frames(e.full_context(context),.02,1):replay.append(model.predict_proba(f.drop(columns=['nbr_flux_near','nbr_flux_far','nbr_flux_max'])))
  with np.load(Path(m['parent_root'])/f'c8/case{case}.npz') as z:
   for val,key in zip(replay,['probability_plus','probability_minus']):np.testing.assert_allclose(val,z[key],atol=2e-7,rtol=1e-6)
  print('ORIGINAL_AXIS1_C8_REPLAY_PASSED',flush=True)
 out={}
 for name,f in zip(['plus','minus'],shear_frames(e.full_context(context),.02,2)):
  q=np.asarray(model.predict_proba(f.drop(columns=['nbr_flux_near','nbr_flux_far','nbr_flux_max'])),float)
  assert np.isfinite(q).all() and np.all((q>=0)&(q<=1));out['probability_'+name]=q
 dest=ROOT/f'c8/case{case}.npz';assert not dest.exists();np.savez(dest,input_index=ids,**out)
 dest.with_suffix('.json').write_text(json.dumps(dict(case=case,axis=2,classifier_sha256=e.sha256(paths['classifier']),parent_sha256=e.sha256(Path(m['parent_root'])/f'parent/case{case}.npz'),rows=len(ids)),indent=2)+'\n')
 print('C8_DONE',case,flush=True)
