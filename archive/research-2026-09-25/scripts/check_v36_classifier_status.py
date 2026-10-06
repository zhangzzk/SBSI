"""Write a bounded read-only receipt for the classifier simplification jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_occam_20260925_v1')


def main():
    launch = json.loads((ROOT/'launch.json').read_text())
    jobs = ','.join(str(j) for j in launch['jobs'].values())
    result = subprocess.run(['sacct', '-j', jobs, '-X', '-o', 'JobID,State,Elapsed,ExitCode', '-P'],
                            capture_output=True, text=True, check=True)
    states = {p.name: json.loads(p.read_text()) for p in ROOT.glob('*_status.json')}
    history = {}
    for path in ROOT.glob('C*/history.json'):
        h = json.loads(path.read_text())
        history[path.parent.name] = {k: v for k, v in h.items() if k != 'history'} | {'last_epoch': h['history'][-1]}
    receipt = dict(time=datetime.now(timezone.utc).isoformat(), scheduler=result.stdout,
                   states=states, history=history, comparison_exists=(ROOT/'comparison.json').exists())
    path = ROOT / ('status_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
