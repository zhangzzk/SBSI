"""Preserve a fixed 10M-scene prefix and reuse the frozen 100k inference recipe.

This is an explicitly smaller research evaluation, not completion of the 52M run.
The prefix is fixed before viewing estimates; only its selection budget changes.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

CHUNKS = 611
SCENES = 8192
PYTHON = '/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_selection(root):
    import numpy as np
    rule = json.loads((root/'rule.json').read_text())
    pieces = []
    for replica in range(2):
        receipt = json.loads((root/f'normalization/selection{replica}.json').read_text())
        path = root/f'normalization/selection{replica}.npz'
        expected = dict(model_identity=rule['model_identity'], chunks=CHUNKS,
                        scenes_per_chunk=SCENES, flow_draws=16, replica=replica,
                        seed_base=2027207001+10000*replica,
                        source_plan_sha256=rule['source_plan_sha256'])
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('preliminary selection receipt does not match fixed budget/identity')
        if receipt['data_sha256'] != digest(path):
            raise ValueError('preliminary selection checksum mismatch')
        with np.load(path) as data:
            values = data['values']
            if not np.array_equal(data['grid'], rule['shear_nodes']):
                raise ValueError('preliminary selection shear nodes differ')
            if values.shape != (CHUNKS, 5, 1, 2):
                raise ValueError('preliminary selection prefix incomplete')
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError('invalid selection probabilities')
            pieces.append(values[:, :, 0, 1])
    result = np.concatenate(pieces)
    if np.any(result.mean(0) <= 0):
        raise ValueError('zero preliminary selection probability')
    return result


def capture(root):
    """Wait for two fixed prefixes; preserve validated copies before stopping workers."""
    import numpy as np
    rule = json.loads((root/'rule.json').read_text())
    parent = Path(rule['parent'])
    if digest(parent/'normalization/original/it0/plan.json') != rule['source_plan_sha256']:
        raise ValueError('source selection plan changed')
    deadline = time.monotonic() + 6*3600
    pending = set(range(2))
    while pending:
        for replica in sorted(pending):
            source = parent/f'normalization/original/it0/selection{replica}'
            progress = json.loads((source/'progress.json').read_text())
            print('PREFIX_PROGRESS', replica, progress['chunks'], CHUNKS, flush=True)
            if progress['chunks'] < CHUNKS:
                continue
            try:
                raw = (source/'chunks.npz').read_bytes()
                with np.load(io.BytesIO(raw)) as data:
                    values, grid = data['values'].copy(), data['grid'].copy()
            except (OSError, EOFError, ValueError, zipfile.BadZipFile):
                # Source writer replaces the archive every chunk; retry a concurrent write.
                continue
            if values.shape[0] < CHUNKS:
                continue
            if (values.shape[1:] != (5, 1, 2) or not np.isfinite(values).all()
                    or np.any((values < 0) | (values > 1))
                    or not np.array_equal(grid, rule['shear_nodes'])):
                raise ValueError('invalid live selection source')
            path = root/f'normalization/selection{replica}.npz'
            np.savez_compressed(path, values=values[:CHUNKS], grid=grid)
            receipt = dict(model_identity=rule['model_identity'], replica=replica,
                chunks=CHUNKS, scenes_per_chunk=SCENES, flow_draws=16,
                seed_base=2027207001+10000*replica,
                source_plan_sha256=rule['source_plan_sha256'], source=str(source),
                source_captured_chunks=len(values), source_archive_sha256=hashlib.sha256(raw).hexdigest(),
                data_sha256=digest(path), status='fixed_prefix_preliminary')
            save(root/f'normalization/selection{replica}.json', receipt)
            subprocess.run(['scancel', f"{rule['selection_job']}_{replica}"], check=True)
            pending.remove(replica)
            print('PREFIX_CAPTURED', replica, CHUNKS, flush=True)
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError('selection prefix not ready within six hours')
            time.sleep(60)
    load_selection(root)
    save(root/'normalization/complete.json', dict(scenes=2*CHUNKS*SCENES,
         chunks=2*CHUNKS, status='preliminary_budget_complete'))


def estimate(root, stage):
    rule = json.loads((root/'rule.json').read_text())
    path = Path(rule['parent'])/'estimate.py'
    if digest(path) != rule['parent_estimator_sha256']:
        raise ValueError('frozen parent estimator changed')
    spec = importlib.util.spec_from_file_location('frozen_estimator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.H = root
    module.selection = lambda: load_selection(root)
    module.__doc__ = ('Preliminary 10,010,624-scene normalization: first 611 chunks from original '
        'replicas 0 and 1, 8192 scenes/chunk, 16 flow draws, unchanged five shear nodes. '
        'Same frozen C1/C2 estimator and sampler; normalization uncertainty recomputed from 1222 chunks. '
        'This supersedes the parent estimator documentation about using all eight selection replicas.\n')
    sys.argv = [str(path), stage]
    module.main()
    result = root/stage/'estimate.json'
    out = json.loads(result.read_text())
    out['preliminary_normalization'] = dict(scenes=2*CHUNKS*SCENES,
        rule_sha256=digest(root/'rule.json'), full_budget_scenes=8*800*SCENES)
    save(result, out)


def status(root):
    rule = json.loads((root/'rule.json').read_text())
    parent = Path(rule['parent'])
    report = dict(preliminary=True, normalization_scenes=2*CHUNKS*SCENES, stages={})
    for stage in ('c1', 'c2'):
        labels = [line.split()[0] for line in (root/stage/'units.txt').read_text().splitlines()]
        report['stages'][stage] = dict(planned=len(labels),
            done=sum((root/stage/'claims'/label/'done').exists() for label in labels),
            failed=[label for label in labels if (root/stage/'claims'/label/'failed').exists()])
        p = root/stage/'estimate.json'
        if p.exists():
            report['stages'][stage]['estimate'] = json.loads(p.read_text())
    report['normalization'] = {str(i): dict(
        captured=(root/f'normalization/selection{i}.json').exists(),
        source_progress=json.loads((parent/f'normalization/original/it0/selection{i}/progress.json').read_text()))
        for i in range(2)}
    save(root/f'logs/status_{os.environ.get("SLURM_JOB_ID", "manual")}.json', report)
    print(json.dumps(report), flush=True)


def setup(parent):
    root = parent/'preliminary_10m'
    root.mkdir(exist_ok=False)
    for name in ('logs', 'normalization', 'c1', 'c2'):
        (root/name).mkdir()
    for name in ('source_snapshot', 'reference', 'bright_ids.txt', 'faint_ids.txt', 'detect_resources.py'):
        (root/name).symlink_to(parent/name)
    for name in ('units.txt', 'claims'):
        (root/'c1'/name).symlink_to(parent/'c1'/name)
    for line in (parent/'c1/units.txt').read_text().splitlines():
        label = line.split()[0]
        (root/'c1'/label).symlink_to(parent/'c1'/label)
    if (parent/'c1/left_out.txt').exists():
        raise ValueError('review omitted observations before preparing preliminary study')
    shutil.copy2(parent/'c2/units.txt', root/'c2/units.txt')
    # C2 units reference the original root's magnitude ID files.
    for line in (root/'c2/units.txt').read_text().splitlines():
        name = line.split()[1]
        if not (root/name).exists():
            (root/name).symlink_to(parent/name)
    plan_path = parent/'normalization/original/it0/plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['selection_chunk_size'] == SCENES and plan['selection_flow_draws'] == 16
    assert plan['selection_seed_base'] == 2027207001
    jobs = json.loads((parent/'jobs.json').read_text())
    rule = dict(parent=str(parent), model_identity=plan['model_identity'],
        normalization_scenes=2*CHUNKS*SCENES, chunks_per_replica=CHUNKS,
        replicas=[0, 1], seed_family='original', source_plan_sha256=digest(plan_path),
        shear_nodes=plan['shear_nodes'], selection_job=jobs['selection'],
        parent_estimator_sha256=digest(parent/'estimate.py'),
        selection='Fixed first 611 chunks of each original replica 0 and 1; not selected by values',
        authorization='Owner requested approximately 10 million normalization scenes for faster evaluation',
        limits='100k observed galaxies unchanged; preliminary result; no independent acceptance')
    save(root/'rule.json', rule)
    shutil.copy2(__file__, root/'study.py')
    worker = (parent/'worker.sh').read_text()
    old = f'H={parent}\n'
    assert worker.count(old) == 1
    (root/'worker.sh').write_text(worker.replace(old, f'H={root}\n'))
    verify = f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({str(root)!r})
subprocess.run([{PYTHON!r}, {str(parent/'verify.py')!r}],check=True)
for name,sha in json.loads((root/'manifest.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
        raise ValueError('preliminary input changed: '+name)
print('PRELIMINARY_INPUTS_VERIFIED',flush=True)
'''
    (root/'verify.py').write_text(verify)
    template = Path(__file__).resolve().parents[1]/'jobs/job_sign_symmetric_preliminary.sh'
    (root/'pipeline.sh').write_text(template.read_text().replace('@STUDY_ROOT@', str(root)))
    save(root/'manifest.json', {name: digest(root/name) for name in
        ('rule.json', 'study.py', 'worker.sh', 'pipeline.sh', 'verify.py', 'c2/units.txt')})
    print(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'capture', 'c1', 'c2', 'check'])
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    if args.action == 'setup':
        setup(args.parent.resolve())
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('use scheduler for capture and inference')
    root = Path(__file__).resolve().parent
    if args.action == 'capture':
        capture(root)
    elif args.action == 'check':
        status(root)
    else:
        estimate(root, args.action)


if __name__ == '__main__':
    main()
