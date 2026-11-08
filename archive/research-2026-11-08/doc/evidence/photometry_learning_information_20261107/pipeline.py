"""Pinned stages with persistent workers and numerical prerequisite gates."""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
import torch

H = Path(__file__).resolve().parent
pins = json.loads((H / 'source_pins.json').read_text())
for path, expected in pins.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
namespace = {'__file__': str(H / 'common.py'), '__name__': 'cutprob_bootstrap'}
exec(compile((H / 'common.py').read_bytes(), str(H / 'common.py'), 'exec'), namespace)
s = type('Setup', (), namespace)
assert os.environ.get('SLURM_JOB_ID')
s.verify()
print('PINNED_STAGE', sys.argv[1], 'pipeline_sha256', pins[str(H / 'pipeline.py')], flush=True)
stage = sys.argv[1]
if stage == 'train':
    s.load('cutprob_tests', H / 'verify.py').main()
    s.load('cutprob_reducer_tests', H / 'verify_reducer.py').main()
    s.load('factorial_tests', H / 'verify_factorial.py').main()
    s.load('cutprob_training', H / 'train.py').main()
    gc.collect(); torch.cuda.empty_cache()
    s.load('cutprob_own', H / 'own_evaluate.py').main()
    gc.collect(); torch.cuda.empty_cache()
    s.load('factorial_coherent_eval', H / 'evaluate.py').run(None)
elif stage == 'worker':
    s.load('factorial_coherent_eval', H / 'evaluate.py').run(int(os.environ['SLURM_ARRAY_TASK_ID']))
elif stage == 'report':
    s.load('cutprob_report', H / 'report.py').main()
elif stage == 'verify':
    s.load('cutprob_tests', H / 'verify.py').main()
    s.load('cutprob_reducer_tests', H / 'verify_reducer.py').main()
    s.load('factorial_tests', H / 'verify_factorial.py').main()
else:
    raise ValueError(stage)
