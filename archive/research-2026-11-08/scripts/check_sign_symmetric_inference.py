#!/usr/bin/env python
"""Run an integration preflight or save a bounded study status receipt."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('@STUDY_ROOT@')
PYTHON = '/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python'


def preflight():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler required')
    snapshot = ROOT / 'source_snapshot'
    os.chdir(snapshot)
    sys.path[:0] = [str(snapshot), str(snapshot / 'scripts')]
    from signsym_candidate import CANDIDATE, IDENTITY
    CANDIDATE.validate()
    import sbsi
    assert Path(sbsi.__file__).resolve().parent == snapshot / 'sbsi'
    subprocess.run([PYTHON, '-m', 'pytest', 'tests/test_flow_sign_symmetric.py', '-q'], check=True)
    ids = {name: [int(x) for x in (ROOT / f'{name}_ids.txt').read_text().split()[:2]]
           for name in ('bright', 'faint')}
    smoke = ROOT / 'preflight'
    smoke.mkdir(exist_ok=False)
    (smoke / 'source_snapshot').symlink_to(snapshot, target_is_directory=True)
    ladder = ['--betas', '1', '.7883', '.6167', '.4449', '.2849', '.1473', '.05014', '.005',
              '--narrow-step-probability', '.5']
    for label, observations, shear, temper in (
        ('bright', ids['bright'], [0, 0], ladder),
        ('faint', ids['faint'], [0, 0], []),
        ('nonzero', ids['bright'][:1], [.02, .004], ladder),
    ):
        command = ['bash', str(snapshot / 'jobs/job_galsbi_tempered.sh'), label,
                   '--reference', str(ROOT / 'reference'), '--shear', *map(str, shear),
                   '--seed', '2028500001', '--initial-bank-seed', '2033500001',
                   '--observation-indices', *map(str, observations),
                   '--steps', '64', '--warmup', '32', '--profile-stride', '16', *temper]
        subprocess.run(command, env=dict(os.environ, SBSI_TEMPER_ROOT=str(smoke)), check=True)
        result = json.loads((smoke / label / 'result.json').read_text())
        protocol = json.loads((smoke / label / 'protocol.json').read_text())
        assert protocol['model_identity'] == IDENTITY
        assert len(result['observations']) == len(observations)
    # Use the actual zero-centred nodes, draws and chunk size for a one-chunk integration check.
    from run_galsbi_bright import selection
    plan = json.loads((ROOT / 'normalization/original/it0/plan.json').read_text())
    plan['selection_chunks'] = 1
    (smoke / 'plan.json').write_text(json.dumps(plan))
    selection(smoke, plan, 0, reference=ROOT / 'reference')
    receipt = json.loads((smoke / 'selection0/result.json').read_text())
    assert receipt['model_identity'] == IDENTITY
    (ROOT / 'preflight_passed.json').write_text(json.dumps(dict(status='passed',
        model_identity=IDENTITY, numerical_settings='Short smoke only; production recipe unchanged'), indent=2))
    print('SYMMETRIC_INFERENCE_PREFLIGHT_PASSED', flush=True)


def status():
    jobs_path = ROOT / 'jobs.json'
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    report = dict(jobs=jobs, stages={}, normalization={})
    for stage in ('c1', 'c2'):
        folder = ROOT / stage
        labels = [line.split()[0] for line in (folder / 'units.txt').read_text().splitlines()]
        progress = {}
        for label in labels:
            path = folder / label / 'progress.json'
            if path.exists():
                progress[label] = json.loads(path.read_text())
        report['stages'][stage] = dict(planned=len(labels),
            done=sum((folder / 'claims' / label / 'done').exists() for label in labels),
            failed=[label for label in labels if (folder / 'claims' / label / 'failed').exists()],
            progress=progress)
        estimate = folder / 'estimate.json'
        if estimate.exists():
            report['stages'][stage]['estimate'] = json.loads(estimate.read_text())
    for side in ('original', 'twin'):
        for replica in range(4):
            p = ROOT / 'normalization' / side / 'it0' / f'selection{replica}'
            report['normalization'][f'{side}_{replica}'] = dict(
                completed=(p / 'result.json').exists(),
                progress=json.loads((p / 'progress.json').read_text()) if (p / 'progress.json').exists() else None)
    ids = [str(x) for k, x in jobs.items() if k != 'checks']
    if ids:
        report['sacct'] = subprocess.run(['sacct', '-X', '-j', ','.join(ids), '-n',
            '-o', 'JobID,State,ExitCode,Elapsed'], capture_output=True, text=True).stdout
    output = ROOT / 'logs' / f'status_{os.environ.get("SLURM_JOB_ID", "manual")}.json'
    output.write_text(json.dumps(report, indent=2))
    print(output, flush=True)


if __name__ == '__main__':
    preflight() if sys.argv[1:] == ['preflight'] else status()
