"""Restore six scan workers after all training workers have exited successfully."""
import json
import os
import subprocess
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
root=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106')
p=subprocess.run(['squeue','-h','-r','-j','17786000','-t','PENDING','-o','%i'],capture_output=True,text=True,check=True)
pending=p.stdout.split();result={'pending':pending}
if pending:
    update=subprocess.run(['scontrol','update','JobId=17786000','ArrayTaskThrottle=6'],capture_output=True,text=True)
    result['update']={'returncode':update.returncode,'stdout':update.stdout,'stderr':update.stderr}
    verify=subprocess.run(['scontrol','show','job',pending[0]],capture_output=True,text=True,check=True)
    assert 'ArrayTaskThrottle=6' in verify.stdout,verify.stdout
    result['verified']=verify.stdout
(root/'scan_resource_release.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
