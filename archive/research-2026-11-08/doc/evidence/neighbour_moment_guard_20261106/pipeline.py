import gc,importlib.util,os,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
def module(name):
    s=importlib.util.spec_from_file_location('moment_pipeline_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
assert os.environ.get('SLURM_JOB_ID')
if sys.argv[1]=='train':
    module('train').main();gc.collect();torch.cuda.empty_cache()
    module('own_evaluate').main();gc.collect();torch.cuda.empty_cache()
    module('evaluate').run(None)
elif sys.argv[1]=='worker':module('evaluate').run(int(os.environ['SLURM_ARRAY_TASK_ID']))
elif sys.argv[1]=='report':module('report').main()
else:raise ValueError(sys.argv[1])
