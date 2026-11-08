#!/usr/bin/env python
"""Bounded read-only scheduler and experiment progress snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    launch = json.loads((a.root / 'launch.json').read_text())
    jobs = ','.join(str(v) for v in launch['jobs'].values())
    result = subprocess.run(['sacct', '-X', '-P', '-n', '-j', jobs,
        '--format=JobID,State,ExitCode,Elapsed'], check=True, text=True, capture_output=True)
    now = datetime.now(timezone.utc)
    status = {path.stem: json.loads(path.read_text()) for path in (a.root / 'status').glob('*.json')}
    report = dict(checked_at=now.isoformat(), scheduler=result.stdout, stages=status,
        development_comparison_complete=(a.root / 'development_comparison.json').exists(),
        production_accepted=False, note='Read-only local receipt; no conversational notification or model promotion.')
    for phase in ('nll', 'guard'):
        path = a.root / f'base/flow/{phase}/history.json'
        if path.exists():
            history = json.loads(path.read_text())
            if isinstance(history, list):
                report[f'{phase}_progress'] = dict(records=len(history), last=history[-1] if history else None)
    output = a.root / f'status_{now.strftime("%Y%m%dT%H%M%SZ")}.json'
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
