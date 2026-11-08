"""Check crowding definitions and cached raw inputs without loading model code."""
import ast
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
BASE=Path('/project/ls-gruen/users/zekang.zhang')
RUNTIME=Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/sbsi/v39_inference.py')
TRAINING=BASE/'sbsi_flow_f8_20260924/scripts/build_close_crowding_lookup.py'

def function(path,name):
    node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name)
    namespace={'np':np,'cKDTree':cKDTree,'CLOSE_SCALES_ARCSEC':(.5,1.,2.)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    current=function(RUNTIME,'physical_close_crowding');training=function(TRAINING,'close_crowding')
    x=np.array([0.,.1,.4,1.,2.,4.,6.,8.,9.999,10.001,20.,25.])/3600
    scene=pd.DataFrame({'RA':180+x/np.cos(np.deg2rad(-.5)),'DEC':np.full(len(x),-.5),'r':np.linspace(20,28,len(x))})
    truth=scene.rename(columns={c:c+'_input' for c in scene})
    np.testing.assert_allclose(current(scene),training(truth),rtol=0,atol=2e-12)
    rows=np.array([9,1,5,0]);np.testing.assert_allclose(current(scene,rows),training(truth)[rows],rtol=0,atol=2e-12)
    results=[]
    for case in [1000,1040,1080,1120]:
        path=BASE/f'sbsi/galsbi_resim/constant_g002/sims/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather'
        columns=['index_input','RA_input','DEC_input','r_input','Re_input','axis_ratio_input','e1_input_rot0','e2_input_rot0','sersic_n_input']
        t=pd.read_feather(path,columns=columns).sort_values('index_input',kind='stable').reset_index(drop=True)
        with np.load(BASE/f'sbsi_caches/galsbi_dev_constant_20261103/parent/case{case}.npz') as z:
            ids=z['input_index'];context=z['context']
        np.testing.assert_array_equal(ids,t['index_input'])
        rows=np.sort(np.random.default_rng(202611069511+case).choice(len(ids),256,replace=False))
        data=pd.DataFrame({'RA':t['RA_input'],'DEC':t['DEC_input'],'r':t['r_input']})
        crowd=current(data,rows)
        expected=np.column_stack([t['e1_input_rot0'].to_numpy()[rows],t['e2_input_rot0'].to_numpy()[rows],
            t['sersic_n_input'].to_numpy()[rows],t['r_input'].to_numpy()[rows],
            t['Re_input'].to_numpy(float)[rows]*np.sqrt(t['axis_ratio_input'].to_numpy(float)[rows]),crowd])
        np.testing.assert_allclose(context[rows],expected,rtol=0,atol=2e-12)
        results.append({'case':case,'rendered_rows':len(ids),'sampled_rows':len(rows),
                        'max_absolute_context_difference':float(np.abs(context[rows]-expected).max())})
        print('CASE',results[-1],flush=True)
    out={'passed':True,'cases':results,'training_runtime_formula_fixture_passed':True,
         'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),RUNTIME,TRAINING]},
         'limitations':'Training/runtime crowding formula parity on synthetic geometry plus256 cached context rows per constant scene. Not an audit of every training-cache row or conditional density calibration.'}
    (HERE/'context_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print('DONE: coordinate, relative-flux, radius conversion and sampled cached contexts pass',flush=True)

if __name__=='__main__':main()
