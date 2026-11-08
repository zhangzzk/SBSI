#!/usr/bin/env python
"""Train symmetric F8 through the base response guard, audit and record the candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def record_candidate(root, experiment):
    """Publish a research candidate only after an audit of the exact checkpoint."""
    flow = (root / 'base/flow/guard/selected.pt').resolve()
    audit_path = root / 'base_symmetry.json'
    audit = json.loads(audit_path.read_text())
    checkpoint_hash = sha256(flow)
    if (audit.get('status') != 'passed' or Path(audit.get('flow', '')).resolve() != flow
            or audit.get('flow_sha256') != checkpoint_hash):
        raise ValueError('base-guard checkpoint or symmetry audit mismatch')
    write(root / 'trained_candidate.json', dict(flow=str(flow), flow_sha256=checkpoint_hash,
        selection_stage='base_guard', later_refinement=False,
        symmetry_audit_sha256=sha256(audit_path),
        fixed_components=experiment['fixed_components'], production_accepted=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['base'], required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler allocation required')
    root = a.root.resolve()
    experiment = json.loads((root / 'experiment.json').read_text())
    if (experiment['training'].get('stages') != ['base']
            or experiment['training'].get('later_refinement') is not False):
        raise ValueError('requires a new frozen base-only study; preserve historical runs')
    for file, key in ((Path(__file__), 'runner_sha256'),
                      (root / 'audit_sign_symmetric_flow.py', 'audit_sha256')):
        if sha256(file) != experiment[key]:
            raise ValueError(f'frozen orchestration changed: {file}')
    checkout = Path(experiment['checkout'])
    for name, expected in experiment['checkout_hashes'].items():
        if sha256(checkout / name) != expected:
            raise ValueError(f'frozen source changed: {name}')
    import sbsi
    from sbsi.fixed_g0_domain import FLOW_FEATURES, PARENT_FEATURES
    if Path(sbsi.__file__).resolve().parent != checkout / 'sbsi':
        raise ValueError(f'wrong installed SBSI: {sbsi.__file__}')
    if list(FLOW_FEATURES) != experiment['raw_features'] or len(PARENT_FEATURES) != 8:
        raise ValueError('flow/classifier contracts differ')
    if 'nbr_flux_near' not in PARENT_FEATURES or 'nbr_flux_near' in FLOW_FEATURES:
        raise ValueError('classifier features were changed with flow')
    # Training must consume completed, hashed projections of the original data.
    for domain in (root / 'base/domain',):
        receipt = json.loads((domain / 'completed_projection.json').read_text())
        if sha256(domain / 'manifest.json') != receipt['manifest_sha256']:
            raise ValueError('domain projection changed')
    key = a.stage
    status = dict(stage=key, job_id=os.environ['SLURM_JOB_ID'],
                  started_at=datetime.now(timezone.utc).isoformat(), state='started', commands=[])
    status_path = root / 'status' / f'{key}.json'
    write(status_path, status)

    def run(module, *arguments):
        command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
        status['commands'].append(dict(command=command, state='running'))
        write(status_path, status)
        print('STAGE_COMMAND', key, module, ' '.join(map(str, arguments)), flush=True)
        subprocess.run(command, cwd='/tmp', check=True)
        status['commands'][-1]['state'] = 'completed'
        write(status_path, status)

    try:
        run('pytest', checkout / 'tests/test_guard_density_gate.py', '-q')
        # Exercise the real data loader, transform and physical density before
        # loading all 200 scenes; this checks semantic eight-input ordering.
        import numpy as np
        import torch
        from scripts.train_fixed_g0_flow import load_case, build_transforms_and_config
        from sbsi.measurement_model import build_flow
        import pandas as pd
        data = load_case(root / 'base/domain/flow/g0/case000.npz')
        subset = dict(context_raw=data['context'][:1024], target_raw=data['target'][:1024])
        condition, target, config = build_transforms_and_config(subset)
        assert config['context_dim'] == 16 and config['target_dim'] == 4
        model = build_flow(config)
        x = torch.as_tensor(condition.transform_frame(pd.DataFrame(subset['context_raw'], columns=FLOW_FEATURES)))
        y = torch.as_tensor(target.transform_array(subset['target_raw']), dtype=torch.float64)
        loss = -model.log_prob(y, x).mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite projected-data density')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite projected-data gradient')
        assert np.isfinite(subset['context_raw']).all()
        print('F8_PREFLIGHT_PASSED inputs=8 context=16 physical_outputs=4 density_and_gradient_finite=True', flush=True)
        del model, data, subset, x, y, loss
        run('scripts.train_full_domain_guard_flow', '--domain-root', root / 'base/domain',
            '--output-root', root / 'base/flow')
        flow = root / 'base/flow/guard/selected.pt'
        audit = root / 'base_symmetry.json'
        subprocess.run([sys.executable, '-u', str(root / 'audit_sign_symmetric_flow.py'),
            '--root', str(root), '--flow', str(flow), '--output', str(audit)],
            cwd='/tmp', check=True)
        record_candidate(root, experiment)
        status['state'] = 'completed'
    except BaseException as error:
        status['state'] = 'failed'
        status['error'] = repr(error)
        raise
    finally:
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(status_path, status)
    print('F8_STAGE_COMPLETE', key, flush=True)


if __name__ == '__main__':
    main()
