"""Reuse the pinned evaluated science pipeline, isolated under this trial's cache."""
import gc,importlib.util,os,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixedshape_setup',H/'common.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
assert os.environ.get('SLURM_JOB_ID')
s.verify()
stage=sys.argv[1]
if stage=='train':
    s.load('fixedshape_train',H/'train.py').main();gc.collect();torch.cuda.empty_cache()
    s.bound('own_evaluate').main();gc.collect();torch.cuda.empty_cache()
    s.bound('evaluate').run(None)
elif stage=='worker':
    s.bound('evaluate').run(int(os.environ['SLURM_ARRAY_TASK_ID']))
elif stage=='report':
    s.load('fixedshape_report',H/'report.py').main()
else:raise ValueError(stage)
