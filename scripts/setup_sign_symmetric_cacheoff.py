"""Validate full nuisance-move likelihoods in an isolated frozen sampler copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

PYTHON = '/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python'

REPLACEMENT = '''        if block in invariant_visits:
            # Check the retained state against itself, not an analytically
            # invariant proposal whose reconstructed float32 inputs can differ.
            if invariant_visits[block] % 1024 == 0:
                valid = np.isfinite(logprior)
                audit_like, audit_summary = evaluate_scene(scene, valid, with_b=True)
                invariant_audit['checked_states'] += int(valid.sum())
                if valid.any():
                    invariant_audit['maximum_summary_change'] = max(invariant_audit['maximum_summary_change'],
                        float(np.nanmax(np.abs(audit_summary[valid]-summaries[valid]))))
                    invariant_audit['maximum_log_kernel_change'] = max(invariant_audit['maximum_log_kernel_change'],
                        float(np.max(np.abs(audit_like[valid]-loglike[valid]))))
                if (invariant_audit['maximum_summary_change'] > protocol['replay_tolerances']['summary_absolute']
                        or invariant_audit['maximum_log_kernel_change'] > protocol['replay_tolerances']['log_kernel_absolute']):
                    save(args.output/'current_state_replay_failure.json', invariant_audit)
                    raise ValueError('retained state disagrees with full likelihood replay')
            invariant_visits[block] += 1
        # Every proposal receives its own implemented numerical likelihood.
        # No RNG calls, support changes, or tolerance changes are introduced.
        newlike, newsummary = evaluate_scene(newscene, np.isfinite(newprior))
'''


def transform(source):
    first = '        if block in invariant_visits:\n'
    last = '        accepted = metropolis_accept('
    assert source.count(first) == source.count(last) == 1
    start, stop = source.index(first), source.index(last)
    source = source[:start]+REPLACEMENT+source[stop:]
    old = """        invariant_work=('z/SED moves reuse the likelihood, environment moves the target prior; '
                        'every 1024th such move is replayed in full'),"""
    new = """        invariant_work=('z/SED likelihood reuse disabled; each proposal evaluated in full; '
                        'environment prior reuse retained; every 1024th nuisance visit replays the current state'),
        nuisance_likelihood_cache=False,"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace('invariant_audit = dict(likelihood_calls_skipped=0,',
        "invariant_audit = dict(mode='current_state_replay_without_nuisance_cache', likelihood_calls_skipped=0,")
    source = source.replace('checked_proposals=0, maximum_summary_change=0.',
                            'checked_states=0, maximum_summary_change=0.')
    return source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def setup(parent, root):
    root.mkdir(exist_ok=False)
    (root/'logs').mkdir()
    snapshot = root/'source_snapshot'
    shutil.copytree((parent/'source_snapshot').resolve(), snapshot, symlinks=True)
    subprocess.run(['sha256sum', '--check', 'SHA256SUMS', '--quiet'], cwd=snapshot, check=True)
    script = snapshot/'scripts/check_galsbi_local_sampler.py'
    original = digest(script)
    script.write_text(transform(script.read_text()))
    hashes = snapshot/'SHA256SUMS'
    hashes.write_text('\n'.join(digest(script)+'  '+line.split(maxsplit=1)[1]
        if line.split(maxsplit=1)[1].lstrip('./') == 'scripts/check_galsbi_local_sampler.py' else line
        for line in hashes.read_text().splitlines())+'\n')
    subprocess.run(['sha256sum', '--check', 'SHA256SUMS', '--quiet'], cwd=snapshot, check=True)
    runs = []
    for j, line in enumerate((parent/'c1/units.txt').read_text().splitlines()):
        label, filename, first, last = line.split()
        if not (parent/'c1/claims'/label/'failed').exists():
            continue
        protocol = json.loads((parent/'c1'/label/'protocol.json').read_text())
        ids = list(map(int, (parent/filename).read_text().splitlines()[int(first):int(last)+1]))
        assert ids == protocol['observations'] and protocol['seed'] == 2028500001+j
        runs.append(dict(label=label, indices=ids, seed=protocol['seed'],
            bank_seed=protocol['initial_bank_seed'], original_protocol_sha256=digest(parent/'c1'/label/'protocol.json')))
    assert len(runs) == 2
    save(root/'rule.json', dict(parent=str(parent), runs=runs,
        model_identity=json.loads((parent/'rule.json').read_text())['model_identity'],
        original_sampler_sha256=original, sampler_sha256=digest(script),
        model_prior_selection='unchanged; no new normalization required for evaluating the same full likelihood',
        policy='Isolated validation only; no automatic promotion. No nuisance cache, no tolerance relaxation. '
               'Same original unit seeds, observation batches, steps, warmup, tempering and proposal law.',
        validation='paired 64-step cached/full evaluations with exact initial/final states and profiles within 1e-6; '
                   'then full original failed units with current-state and co-moving replay gates'))
    shutil.copy2(__file__, root/'study.py')
    shutil.copy2(parent/'detect_resources.py', root/'detect_resources.py')
    (root/'job.sh').write_text('''#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --constraint=x86-64-v3
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
''' + f'H={root}\nPY={PYTHON}\n' + '''"$PY" "$H/detect_resources.py" -o "$H/logs/resources_${SLURM_JOB_ID}.json"
exec "$PY" -u "$H/study.py" "$@"
''')
    save(root/'setup_manifest.json', {name:digest(root/name) for name in
         ['rule.json', 'study.py', 'job.sh', 'detect_resources.py']})
    print(root)


def command(root, output, label, run, *, indices=None, shear=(0, 0), short=False):
    rule = json.loads((root/'rule.json').read_text())
    args = ['bash', str(root/'source_snapshot/jobs/job_galsbi_tempered.sh'), label,
            '--reference', str(Path(rule['parent'])/'reference'), '--shear', *map(str, shear),
            '--seed', str(run['seed']), '--initial-bank-seed', str(run['bank_seed']),
            '--observation-indices', *map(str, run['indices'] if indices is None else indices)]
    if run['label'].startswith('bright'):
        args += ['--betas', '1', '.7883', '.6167', '.4449', '.2849', '.1473', '.05014', '.005',
                 '--narrow-step-probability', '.5']
    if short:
        args += ['--steps', '64', '--warmup', '32', '--profile-stride', '16']
    subprocess.run(args, env=dict(os.environ, SBSI_TEMPER_ROOT=str(output)), check=True)


def preflight(root):
    import numpy as np
    rule = json.loads((root/'rule.json').read_text())
    outputs = []
    for name, source in [('baseline', Path(rule['parent'])/'source_snapshot'), ('candidate', root/'source_snapshot')]:
        out = root/('preflight_'+name)
        out.mkdir(exist_ok=False)
        (out/'source_snapshot').symlink_to(source)
        outputs.append(out)
    comparisons = []
    for run, shear, name in [(rule['runs'][0], (0, 0), 'bright'),
                             (rule['runs'][1], (0, 0), 'faint'),
                             (rule['runs'][0], (.01902, .00077), 'nonzero')]:
        for out in outputs:
            command(root, out, name, run, indices=run['indices'][:2], shear=shear, short=True)
        for filename in ['initial_state.npz', 'final_scenes.npz']:
            with np.load(outputs[0]/name/filename) as a, np.load(outputs[1]/name/filename) as b:
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        with np.load(outputs[0]/name/'local_profiles.npz') as a, np.load(outputs[1]/name/'local_profiles.npz') as b:
            maximum = max(float(np.max(np.abs(a[key]-b[key]))) for key in a.files)
            assert maximum <= 1e-6
        audit = json.loads((outputs[1]/name/'invariant_replay.json').read_text())
        assert audit['likelihood_calls_skipped'] == 0 and audit['checked_states'] > 0
        comparisons.append(dict(label=name, exact_initial_and_final_states=True, maximum_profile_difference=maximum,
                                current_state_audit=audit))
    save(root/'preflight_passed.json', dict(status='passed', comparisons=comparisons,
        limitation='Short integration and deterministic path replay; not posterior convergence'))
    print('CACHEOFF_PREFLIGHT_PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'preflight', 'unit'])
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.action == 'setup':
        setup(args.parent.resolve(), args.root.resolve())
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler required')
    root = Path(__file__).resolve().parent
    for name, sha in json.loads((root/'setup_manifest.json').read_text()).items():
        assert digest(root/name) == sha, name
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.action == 'preflight':
        preflight(root)
    else:
        assert json.loads((root/'preflight_passed.json').read_text())['status'] == 'passed'
        rule = json.loads((root/'rule.json').read_text())
        run = rule['runs'][args.index]
        command(root, root, run['label'], run)
        result = json.loads((root/run['label']/'result.json').read_text())
        assert len(result['observations']) == len(run['indices'])
        audit = json.loads((root/run['label']/'invariant_replay.json').read_text())
        assert audit['likelihood_calls_skipped'] == 0 and audit['checked_states'] > 0
        assert result['comoving_identity_replay_max_abs'] <= 1e-6
        save(root/(run['label']+'_validation.json'), dict(status='completed_full_unit',
            observations=len(run['indices']), current_state_audit=audit,
            production_promoted=False, result_sha256=digest(root/run['label']/'result.json')))


if __name__ == '__main__':
    main()
