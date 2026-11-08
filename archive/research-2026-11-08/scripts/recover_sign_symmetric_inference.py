"""Resume the million-galaxy study after reviewed full-unit cache-off validation.

Create an overlay: preserved C1 outputs plus the two verified repaired units;
new C2 chains use the validated sampler. Original runs are never modified.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PYTHON = '/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def load(path):
    spec = importlib.util.spec_from_file_location('original_extension', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_repaired(original, candidate, receipt, model):
    """Reject changed scientific settings, incomplete output, or failed replay."""
    permitted = {'implementation_sha256', 'invariant_work', 'nuisance_likelihood_cache'}
    changed = {k for k in original.keys() | candidate.keys() if original.get(k) != candidate.get(k)}
    if changed != permitted or candidate.get('nuisance_likelihood_cache') is not False:
        raise ValueError('unexpected repaired protocol differences')
    if candidate['model_identity'] != model or candidate['steps'] != 20480:
        raise ValueError('wrong model or incomplete chain plan')
    audit = receipt['current_state_audit']
    if (receipt['status'] != 'completed_full_unit'
            or receipt['observations'] != len(candidate['observations'])
            or audit['likelihood_calls_skipped'] != 0 or audit['checked_states'] <= 0
            or audit['maximum_summary_change'] != 0 or audit['maximum_log_kernel_change'] != 0):
        raise ValueError('full-unit validation failed')
    return sorted(changed)


def setup(parent, root):
    if root.exists():
        raise ValueError('fresh recovery directory required')
    original = load(parent/'study.py')
    original.verify(parent)
    validation = parent/'cacheoff_validation_v1'
    for name, sha in json.loads((validation/'setup_manifest.json').read_text()).items():
        if digest(validation/name) != sha:
            raise ValueError('validation setup changed: '+name)
    if json.loads((validation/'preflight_passed.json').read_text())['status'] != 'passed':
        raise ValueError('paired GPU preflight did not pass')
    subprocess.run(['sha256sum', '--check', 'SHA256SUMS', '--quiet'],
                   cwd=validation/'source_snapshot', check=True)
    rule = json.loads((parent/'rule.json').read_text())
    repaired = {}
    vrule = json.loads((validation/'rule.json').read_text())
    for run in vrule['runs']:
        label = run['label']
        old = parent/'c1'/label
        new = validation/label
        if digest(old/'protocol.json') != run['original_protocol_sha256']:
            raise ValueError('original failed protocol changed')
        receipt = json.loads((validation/(label+'_validation.json')).read_text())
        changed = validate_repaired(json.loads((old/'protocol.json').read_text()),
            json.loads((new/'protocol.json').read_text()), receipt, rule['model_identity'])
        if digest(new/'result.json') != receipt['result_sha256']:
            raise ValueError('validated result changed')
        result = json.loads((new/'result.json').read_text())
        if (result['protocol_sha256'] != digest(new/'protocol.json')
                or result['comoving_identity_replay_max_abs'] > 1e-6
                or [x['reference_observation'] for x in result['observations']] != run['indices']
                or json.loads((new/'progress.json').read_text())['step'] != 20480):
            raise ValueError('incomplete or mismatched validated unit')
        repaired[label] = dict(source=str(new), changed_protocol_keys=changed,
            hashes={name:digest(new/name) for name in ['protocol.json', 'result.json', 'local_profiles.npz']},
            validation_receipt_sha256=digest(validation/(label+'_validation.json')),
            prior_validation_policy='No automatic promotion; reviewed and promoted by this recovery amendment')
    root.mkdir()
    for stage in ['c1', 'c2']:
        (root/stage/'claims').mkdir(parents=True)
        shutil.copy2(parent/stage/'units.txt', root/stage/'units.txt')
        (root/stage/'source_snapshot').symlink_to(validation/'source_snapshot')
    (root/'logs').mkdir()
    for name in ['reference', 'prepared.json', 'detect_resources.py', 'reused_references.json']:
        (root/name).symlink_to(parent/name)
    for path in parent.glob('*_ids.txt'):
        (root/path.name).symlink_to(path)
    (root/'source_snapshot').symlink_to(validation/'source_snapshot')
    reused = json.loads((parent/'reused_units.json').read_text())
    count = 0
    for line in (root/'c1/units.txt').read_text().splitlines():
        label, filename, first, last = line.split()
        count += int(last)-int(first)+1
        if label in repaired:
            (root/'c1'/label).symlink_to(repaired[label]['source'])
            claim = root/'c1/claims'/label
            claim.mkdir()
            save(claim/'done', dict(recovery='validated full likelihood', source=repaired[label]['source']))
            reused[label] = repaired[label]['hashes']
        else:
            if not (parent/'c1/claims'/label/'done').exists() or (parent/'c1/claims'/label/'failed').exists():
                raise ValueError('unresolved original unit: '+label)
            (root/'c1'/label).symlink_to(parent/'c1'/label)
            (root/'c1/claims'/label).symlink_to(parent/'c1/claims'/label)
    if count != 999285 or len(repaired) != 2:
        raise ValueError('unexpected recovery coverage')
    # The estimator expects reference overrides for every reused unit.
    refs = json.loads((parent/'reused_references.json').read_text())
    for label in repaired:
        refs[label] = str((parent/'reference').resolve())
    (root/'reused_references.json').unlink()
    save(root/'reused_references.json', refs)
    save(root/'reused_units.json', reused)
    rule['recovery'] = dict(original_root=str(parent), validation=str(validation), repaired=repaired,
        c1_policy='1043 unchanged cached units plus two verified full-likelihood units; no omissions',
        c2_policy='all 311 units use validated full-likelihood sampler, original seeds and groups',
        maximum_concurrent_gpus=4,
        limitations='Numerical replay validation is not posterior convergence or independent scientific acceptance')
    save(root/'rule.json', rule)
    shutil.copy2(__file__, root/'study.py')
    (root/'verify.py').write_text('from pathlib import Path\nimport study\nstudy.verify(Path(__file__).resolve().parent)\n')
    for name in ['worker.sh', 'pipeline.sh', 'estimator_base.py']:
        text = (parent/name).read_text()
        before = f"H = Path('{parent}')" if name.endswith('.py') else f'H={parent}\n'
        after = f"H = Path('{root}')" if name.endswith('.py') else f'H={root}\n'
        if text.count(before) != 1:
            raise ValueError('expected one root binding: '+name)
        (root/name).write_text(text.replace(before, after))
    shutil.copy2(parent/'monitor.py', root/'monitor.py')
    (root/'monitor.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec '+PYTHON+' '+str(root/'monitor.py')+' '+str(root)+'\n')
    files = ['rule.json', 'study.py', 'verify.py', 'worker.sh', 'pipeline.sh', 'estimator_base.py',
             'reused_units.json', 'reused_references.json', 'c1/units.txt', 'c2/units.txt', 'monitor.py', 'monitor.sh']
    save(root/'setup_manifest.json', {name:digest(root/name) for name in files})
    save(root/'promotion.json', dict(status='reviewed_recovery_ready', galaxies=count, repaired=repaired,
        original_artifacts_untouched=True, setup_manifest_sha256=digest(root/'setup_manifest.json')))
    verify(root)
    job = subprocess.check_output(['sbatch', '--parsable', '--job-name=signsym-recovery-c1',
        '--output='+str(root/'logs/c1_estimate_%j.out'), str(root/'pipeline.sh'), 'estimate-c1'], text=True).strip()
    save(root/'jobs.json', dict(prepare=os.environ['SLURM_JOB_ID'], estimate_c1=job))
    monitor = subprocess.check_output(['sbatch', '--parsable', '--begin=now+2hours', '--partition=cluster',
        '--cpus-per-task=1', '--mem=2G', '--time=00:10:00', '--job-name=signsym-recovery-check',
        '--output='+str(root/'logs/monitor_%j.out'), str(root/'monitor.sh')], text=True).strip()
    save(root/'monitor_setup.json', dict(first_check=monitor, on_disk_only=True))
    print('RECOVERY_READY', root, 'C1_JOB', job, flush=True)


def verify(root):
    rule = json.loads((root/'rule.json').read_text())
    parent = Path(rule['recovery']['original_root'])
    load(parent/'study.py').verify(parent)
    for name, sha in json.loads((root/'setup_manifest.json').read_text()).items():
        if digest(root/name) != sha:
            raise ValueError('recovery setup changed: '+name)
    subprocess.run(['sha256sum', '--check', 'SHA256SUMS', '--quiet'], cwd=root/'source_snapshot', check=True)
    for label, item in rule['recovery']['repaired'].items():
        for name, sha in item['hashes'].items():
            if digest(root/'c1'/label/name) != sha:
                raise ValueError('repaired output changed: '+label+'/'+name)
    print('RECOVERY_INPUTS_VERIFIED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'c1', 'c2', 'check'])
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler required')
    if args.action == 'setup':
        setup(args.parent.resolve(), args.root.resolve())
        return
    root = Path(__file__).resolve().parent
    verify(root)
    rule = json.loads((root/'rule.json').read_text())
    original = load(Path(rule['recovery']['original_root'])/'study.py')
    if args.action == 'check':
        original.status(root)
    else:
        print('RECOVERY_REDUCTION_STARTED', args.action, flush=True)
        original.estimate(root, args.action)
        print('RECOVERY_REDUCTION_COMPLETE', args.action, flush=True)


if __name__ == '__main__':
    main()
