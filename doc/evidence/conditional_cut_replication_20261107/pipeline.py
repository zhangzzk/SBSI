"""Execute source-pinned replication stages."""
import hashlib
import json
import os
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
pins = json.loads((H / 'source_pins.json').read_text())
for path, expected in pins.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
ns = {'__file__': str(H / 'common.py'), '__name__': 'cutrep_bootstrap'}
exec(compile((H / 'common.py').read_bytes(), str(H / 'common.py'), 'exec'), ns)
assert os.environ.get('SLURM_JOB_ID')
ns['verify']()
stage = sys.argv[1]
print('PINNED_REPLICATION_STAGE', stage, flush=True)
if stage == 'pilot':
    ns['load']('cutrep_verification', H / 'verify.py').main()
    ns['load']('cutrep_evaluate', H / 'evaluate.py').run(None)
elif stage == 'worker':
    ns['load']('cutrep_evaluate', H / 'evaluate.py').run(int(os.environ['SLURM_ARRAY_TASK_ID']))
elif stage == 'report':
    ns['load']('cutrep_report', H / 'report.py').main()
else:
    raise ValueError(stage)
