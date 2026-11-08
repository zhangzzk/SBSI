"""Restore the continuing array after the diagnostic GPUs have exited."""
import json
import subprocess
from pathlib import Path
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106')
r=subprocess.run(['scontrol','update','JobId=17788286','ArrayTaskThrottle=6','Nice=0'],capture_output=True,text=True)
s=subprocess.run(['scontrol','show','job','17788286'],capture_output=True,text=True)
if 'JobState=PENDING' in s.stdout:
    assert 'ArrayTaskThrottle=6' in s.stdout and 'Nice=0' in s.stdout,s.stdout
# Completed subtasks can produce a nonzero update exit status even when the
# pending array is correctly updated. Preserve both the receipt and verification.
(ROOT/'release_receipt.json').write_text(json.dumps({'update_returncode':r.returncode,'update_stdout':r.stdout,'update_stderr':r.stderr,'verified_job':s.stdout},indent=2)+'\n')
print('RESTORED pending coherent array to throttle6 and Nice0',flush=True)
