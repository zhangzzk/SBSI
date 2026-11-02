#!/usr/bin/env python
"""Run the frozen F8 training stages on the GalSBI re-simulation domain.

Copy of /project/ls-gruen/users/zekang.zhang/sbsi_caches/v39_f8_control_20261005/run_stage.py (itself a copy of
v36_f8_20260924_v1/run_v36_f8_stage.py) with only the data changed: base and harmonized domains are the same
directory, labels are recomputed (no equality with the FS2 label banks), and there is no ConstGold evaluation
stage. Runs under the frozen venv, which imports the frozen checkout named in experiment.json.
"""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['base', 'labels', 'refine'], required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler allocation required')
    root = a.root
    experiment = json.loads((root / 'experiment.json').read_text())
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
    # Training must consume the completed, hashed domain.
    for domain in (root / 'base/domain', root / 'harmonized/domain'):
        receipt = json.loads((domain / 'completed_projection.json').read_text())
        if sha256(domain / 'manifest.json') != receipt['manifest_sha256']:
            raise ValueError('domain changed after it was built')
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

    domain, banks = root / 'harmonized/domain', root / 'joint'
    banks.mkdir(exist_ok=True)
    try:
        if a.stage == 'base':
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
        elif a.stage == 'labels':
            for split in ('validation', 'train'):
                run('scripts.per_leg_guard_response', '--domain-root', domain, '--split', split,
                    '--output', banks / f'per_leg_guard_{split}_targets.json')
            run('scripts.audit_harmonized_usability_response', '--domain-root', domain,
                '--output', banks / 'validation_usability_response.json')
            manifest = json.loads((domain / 'manifest.json').read_text())
            files = {}
            for split in ('train', 'validation'):
                path = banks / f'per_leg_guard_{split}_targets.json'
                d = json.loads(path.read_text())
                if d['cases'] != manifest['split'][f'{split}_cases']:
                    raise ValueError('label split changed')
                files[path.name] = sha256(path)
            write(banks / 'per_leg_guard_target_banks.json', dict(files=files,
                domain_manifest_sha256=sha256(domain / 'manifest.json'),
                source_sha256=sha256(checkout / 'scripts/per_leg_guard_response.py'),
                status='labels computed on the GalSBI re-simulation domain', calibration_claim=False))
        else:
            run('scripts.audit_guard_control_variate', '--domain-root', domain,
                '--flow', root / 'base/flow/guard/selected.pt', '--output-root', root / 'harmonized/guard_reference')
            run('scripts.refine_guard_control_variate_flow', '--domain-root', domain,
                '--flow', root / 'base/flow/guard/selected.pt', '--reference-root', root / 'harmonized/guard_reference',
                '--output-root', root / 'harmonized/refinement')
            flow = root / 'harmonized/refinement/selected.pt'
            for smoke in (True, False):
                suffix, device = ('smoke', 'cpu') if smoke else ('full', 'cuda')
                options = ['--smoke'] if smoke else []
                run('scripts.audit_per_leg_guard_gradient', '--domain-root', domain, '--flow', flow,
                    '--target-bank', banks / 'per_leg_guard_train_targets.json',
                    '--output-root', banks / f'per_leg_cv_{suffix}', '--device', device, *options)
            for smoke in (True, False):
                suffix, device = ('smoke', 'cpu') if smoke else ('full', 'cuda')
                options = ['--smoke'] if smoke else []
                run('scripts.audit_joint_guard_probability_gradient', '--domain-root', domain, '--flow', flow,
                    '--reference-root', banks / f'per_leg_cv_{suffix}',
                    '--output-root', banks / f'joint_gradient_{suffix}', '--device', device, *options)
            for smoke in (True, False):
                suffix, device = ('smoke', 'cpu') if smoke else ('full', 'cuda')
                options = ['--smoke'] if smoke else []
                run('scripts.refine_joint_guard_probability_flow', '--domain-root', domain, '--flow', flow,
                    '--reference-root', banks / f'per_leg_cv_{suffix}', '--target-bank-root', banks,
                    '--probability-audit-root', banks / f'joint_gradient_{suffix}',
                    '--output-root', banks / f'joint_probability_refinement_{suffix}', '--device', device, *options)
            final = banks / 'joint_probability_refinement_full/selected.pt'
            write(root / 'trained_candidate.json', dict(flow=str(final), flow_sha256=sha256(final),
                production_accepted=False))
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
