"""Expand this test after the earlier neighbour-flow array has drained."""
import json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
j=json.loads((HERE/'jobs.json').read_text())['jobs']
active=subprocess.run(['squeue','-h','-j',j['neighbour_array'],'-t','R,CG','-o','%i'],capture_output=True,text=True,check=True).stdout
assert not active.strip(),active
results={}
for name in ['four_remaining','full_remaining']:
    p=subprocess.run(['scontrol','update',f"JobId={j[name]}",'ArrayTaskThrottle=6'],capture_output=True,text=True)
    # scontrol can warn about finished tasks while updating pending tasks.
    results[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    state=subprocess.run(['scontrol','show','job',j[name]],capture_output=True,text=True)
    results[name]['state']=state.stdout
    if 'JobState=PENDING' in state.stdout:assert 'ArrayTaskThrottle=6' in state.stdout
(HERE/'release_receipt.json').write_text(json.dumps(results,indent=2)+'\n');print(results,flush=True)
