#!/usr/bin/env python
"""Run frozen F8 stages in the experiment's installed, isolated environment."""
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
    p.add_argument('--stage', choices=['base', 'labels', 'refine', 'evaluate'], required=True)
    p.add_argument('--chunk', type=int, choices=range(8))
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
    if 'nbr_flux_near' not in PARENT_FEATURES or 'nbr_flux_near' in FLOW_FEATURES:
        raise ValueError('classifier features were changed with flow')
    # Training must consume completed, hashed projections of the original data.
    for domain in (root / 'base/domain', root / 'harmonized/domain'):
        receipt = json.loads((domain / 'completed_projection.json').read_text())
        if sha256(domain / 'manifest.json') != receipt['manifest_sha256']:
            raise ValueError('domain projection changed')
    key = a.stage if a.chunk is None else f'{a.stage}_{a.chunk}'
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

    cache = root.parent
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
            import numpy as np
            files = {}
            for split in ('train', 'validation'):
                path = banks / f'per_leg_guard_{split}_targets.json'
                d = json.loads(path.read_text())
                old = json.loads((cache / 'flow_joint_closecrowd_20260924_v1' / path.name).read_text())
                if d['cases'] != manifest['split'][f'{split}_cases']:
                    raise ValueError('label split changed')
                # Recompute labels, then verify features alone did not alter them.
                for name in ('target', 'baseline_mass', 'baseline_mean_shape', 'shape_normal'):
                    np.testing.assert_allclose(d['per_leg'][name], old['per_leg'][name], rtol=0, atol=2e-12)
                files[path.name] = sha256(path)
            write(banks / 'per_leg_guard_target_banks.json', dict(files=files,
                domain_manifest_sha256=sha256(domain / 'manifest.json'),
                source_sha256=sha256(checkout / 'scripts/per_leg_guard_response.py'),
                status='recomputed labels reproduce frozen full-feature targets', calibration_claim=False))
        elif a.stage == 'refine':
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
                fixed_components=experiment['fixed_components'], production_accepted=False))
        else:
            if a.chunk is None:
                raise ValueError('evaluation requires a chunk')
            candidate = json.loads((root / 'trained_candidate.json').read_text())
            if sha256(candidate['flow']) != candidate['flow_sha256']:
                raise ValueError('trained flow changed')
            geometry = cache / 'scene_classifier_geometry_20260918_v1'
            parent = cache / 'constgold_truth_parent_closecrowd_20260924_v1/parent'
            population = cache / 'constgold_truth_parent_unbounded_20260922_v1'
            classifier = cache / 'scene_classifier_unbounded_20260922_v1/smooth_classifier_full/selected.pt'
            emulator = cache.parent / 'blendemu_runs/disk_response_moment_20260918_v1/model'
            if sha256(classifier) != experiment['fixed_components']['classifier_sha256']:
                raise ValueError('fixed classifier changed')
            if sha256(emulator / 'model.json') != experiment['fixed_components']['disk_emulator_sha256']:
                raise ValueError('fixed emulator changed')
            start = 40 + 10 * a.chunk
            result = root / f'evaluation/chunk{a.chunk}'
            result.mkdir(parents=True, exist_ok=False)
            arguments = ['--flow', candidate['flow'], '--parent-root', parent,
                '--constgold-root', cache.parent / 'lsst_sims_fs2_25876_constant',
                '--pair-root', population / 'pairs', '--geometry-root', population / 'geometry',
                '--classifier-control', geometry / 'full_control/selected.pt',
                '--classifier-geometry', geometry / 'full_geometry/selected.pt',
                '--classifier-transported', geometry / 'transported_classifier_full/selected.pt',
                '--classifier-smooth', classifier, '--response-candidate', emulator,
                '--response-composition', 'mobius', '--response-device', 'cuda', '--response-threads', '4',
                '--cohorts', 'truth_parent', '--draws', '64', '--batch-size', '1024', '--seed', '7301',
                '--h', '.02', '--axis', '1', '--device', 'cuda', '--predictive-diagnostics',
                '--output', result / 'results.json', '--dump-rows', result / 'rows',
                '--rblend', population / f'rblend/cases{start}_{start+9}.feather']
            for seed in (20260913, 20260914, 20260915):
                arguments.extend(['--classifier', cache / f'fixed_g0_m258_r060_v1/classifier/seed{seed}/selected.pt'])
            arguments.extend(['--cases', *range(start, start + 10)])
            run('scripts.evaluate_smooth_classifier_joint', *arguments)
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
