"""Explicit paths and frozen feature contract for the neighbour-input trial."""
import hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_conditioned_flow_20261106')
DOMAIN=ROOT.parent/'galsbi_f8c8_20261102/harmonized/domain'
OLD=ROOT.parent/'alignment_regularized_20261106/prepared'
BASE=ROOT.parent/'alignment_factorized_20261105/cartesian/selected.pt'
BASE_SHA='37c4afc13b045bbd118eea0d20572961eef5f6649d6b9329c1e5d09fe820afbb'
FEATURE='neighbour_reference_coefficient'
CASES=[1000,1040,1080,1120]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
e=module('neighbour_frozen_helpers',HERE.parent/'alignment_factorized_constant_20261105/helpers.py')
fm=module('neighbour_factorized',HERE.parent/'alignment_factorized_20261105/frame_model.py')
def sha(path):return e.sha256(path)
def write(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=convert,allow_nan=False)+'\n');tmp.replace(path)
def score_pairs(emulator,pairs,ids):
    """Frozen physical coefficient sum at3.5px/24.5mag; no measured quantities."""
    import pandas as pd
    ids=np.asarray(ids);loc=pd.Index(ids).get_indexer(pairs['index_input_p'])
    if (loc<0).any():raise ValueError('pair primary outside feature parent set')
    answer=np.zeros(len(ids))
    for lo in range(0,len(pairs),200000):
        pp=pairs.iloc[lo:lo+200000]
        values=emulator.predict_pairs(pp,np.full(len(pp),3.5),np.full(len(pp),10.**(.4*(30-24.5))))
        answer+=np.bincount(loc[lo:lo+len(pp)],weights=values,minlength=len(ids))
    assert np.isfinite(answer).all()
    return answer
