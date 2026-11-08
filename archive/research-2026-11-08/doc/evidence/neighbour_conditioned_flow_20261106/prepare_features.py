"""Build truth-only reference-neighbour features without using measured labels."""
import argparse, gc, importlib.util, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.train_fixed_g0_flow import load_case
from sbsi.output_conditioned_response import DiskResponseMoment
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neighbour_common',HERE/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def one(case,constant=False):
    folder=c.ROOT/('constant_features' if constant else 'features');folder.mkdir(exist_ok=True)
    out=folder/f'case{case}.npz'
    if out.with_suffix('.json').exists():
        receipt=json.loads(out.with_suffix('.json').read_text())
        assert receipt['source_sha256']==c.sha(__file__) and receipt['data_sha256']==c.sha(out)
        print('VERIFIED_EXISTING',case,flush=True);return
    assert not out.exists()
    started=time.monotonic();dm=json.loads((c.DOMAIN/'manifest.json').read_text())
    if constant:
        truth,path=c.e.truth(case,.02)
        with np.load(c.e.OUT/f'parent/case{case}.npz') as z:ids=z['input_index']
    else:
        arrays=[load_case(c.DOMAIN/f'flow/{leg}/case{case:03d}.npz') for leg in ['g0','g05']]
        ids=np.union1d(arrays[0]['input_index'],arrays[1]['input_index'])
        path=Path(dm['sources']['sims'])/f'case{case}_0.0/real0/catalogues/input/gals_info_{c.e.TILE}.feather'
        truth=pd.read_feather(path).sort_values('index_input',kind='stable').reset_index(drop=True)
        other=pd.read_feather(Path(dm['sources']['sims'])/f'case{case}_0.05/real0/catalogues/input/gals_info_{c.e.TILE}.feather',
            columns=['index_input','RA_input','DEC_input','Re_input','r_input','sersic_n_input'])
        other=other.sort_values('index_input',kind='stable').reset_index(drop=True)
        for key in other.columns:np.testing.assert_array_equal(truth[key],other[key])
        del arrays,other
    loc=pd.Index(truth['index_input']).get_indexer(ids)
    if (loc<0).any():raise ValueError(f'{int((loc<0).sum())} unmatched training parents')
    preset,paths,conditions=c.e.preset()
    emulator=DiskResponseMoment(paths['emulator'].parent,nthread=8,device='cuda')
    pairs=c.e.pair_table(c.e.scene_frame(truth),loc,conditions)
    print('PAIRS_READY',case,'constant',constant,'parents',len(ids),'pairs',len(pairs),flush=True)
    score=c.score_pairs(emulator,pairs,ids);check=None
    if constant:
        with np.load(c.ROOT.parent/f'photometry_fixed_cell_20261106/targets{case}.npz') as z:
            ix=pd.Index(ids).get_indexer(z['input_index']);assert (ix>=0).all()
            previous=z['reference_coefficient'].sum(-1)
        np.testing.assert_allclose(score[ix],previous,rtol=2e-10,atol=2e-7)
        check=float(np.max(np.abs(score[ix]-previous)))
    np.savez(out,input_index=ids,score=score)
    c.write(out.with_suffix('.json'),{'case':case,'parents':len(ids),'pairs':len(pairs),'unmatched_parents':0,
        'constant':constant,'reference_radius_pixels':3.5,'reference_magnitude':24.5,'prediction_device':'cuda',
        'truth_path':str(path),'truth_sha256':c.sha(path),'emulator_sha256':preset['emulator']['sha256'],
        'source_sha256':c.sha(__file__),'common_sha256':c.sha(HERE/'common.py'),'data_sha256':c.sha(out),
        'previous_score_replay_max_abs':check,'seconds':time.monotonic()-started})
    print('DONE',case,'seconds',round(time.monotonic()-started,1),flush=True)
    del pairs,truth,emulator;gc.collect()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.pilot:
        one(136);one(1000,constant=True)
        c.write(c.ROOT/'feature_pilot_passed.json',{'training_case':136,'constant_case':1000,'source_sha256':c.sha(__file__)})
    else:
        assert (c.ROOT/'feature_pilot_passed.json').exists()
        task=int(os.environ['SLURM_ARRAY_TASK_ID'])
        for case in range(task*10,min(200,(task+1)*10)):one(case)
if __name__=='__main__':main()
