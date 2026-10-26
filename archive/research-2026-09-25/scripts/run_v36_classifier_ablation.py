"""Train paired classifier ablations or reweight immutable F8 draws under Slurm."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts.train_full_parent_classifier import sha256, write_json


def check_protocol(root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    p = json.loads((root / 'protocol.json').read_text())
    for path, expected in p['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen input changed: ' + path)
    return p


def train(root, p, index):
    seed, arm = p['seeds'][index // 2], p['arms'][index % 2]
    command = [sys.executable, '-u', '-m', 'scripts.train_smooth_classifier',
               '--prepared-root', p['prepared'], '--output-root', str(root / f'{arm}_{seed}'),
               '--arm', 'smooth', '--feature-arm', arm, '--seed', str(seed),
               '--device', 'cuda', '--frozen-protocol', str(root / 'protocol.json')]
    # A bounded real-data round trip exercises each input subset and checkpoint
    # export before loading the full training population.
    smoke = list(command)
    smoke[smoke.index('--prepared-root') + 1] = str(root / 'smoke_prepared')
    smoke[smoke.index('--output-root') + 1] = str(root / f'smoke_{arm}_{seed}')
    subprocess.run([*smoke, '--smoke'], check=True, cwd=p['checkout'])
    subprocess.run(command, check=True, cwd=p['checkout'])


def frame_inputs(p, case, z):
    import pandas as pd
    from sbsi.fixed_g0_domain import PARENT_FEATURES
    from scripts.joint_response_axes import shear_frames
    from scripts.scene_classifier_joint import classifier_frames
    from scripts.transported_classifier_joint import state_geometry_frames
    from scripts.smooth_scene_classifier_features import FEATURES, augment_state

    cache = Path(p['f8_root']).parent
    domain = cache / 'constgold_truth_parent_unbounded_20260922_v1'
    path = domain / f'pairs/case{case:03d}.feather'
    meta = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != meta['sha256']:
        raise ValueError('pair hash changed')
    pairs = pd.read_feather(path)
    ids, context = z['input_index'], z['context'].astype(np.float64)
    if pairs.duplicated(['index_input_p', 'index_input_s']).any() or not np.isin(pairs['index_input_p'], ids).all():
        raise ValueError('pair identity mismatch')
    frames = {d: f[list(PARENT_FEATURES)] for d, f in zip(('plus', 'minus'), shear_frames(context, .02, 1))}
    static, report = classifier_frames(frames, ids, context[:, :8], case, domain / 'geometry')
    moved, transport = state_geometry_frames(static, ids, context[:, :8], pairs, meta['conditions'])
    result = {d: pd.DataFrame(augment_state(ids, context[:, :8], f.to_numpy(), pairs,
                transport['psf_half_light_radius']), columns=FEATURES) for d, f in moved.items()}
    return result, dict(pair_sha256=meta['sha256'], geometry=report)


def metrics(prob, usable):
    usable = np.asarray(usable, dtype=float)
    q = np.clip(prob, 1e-12, 1-1e-12)
    return dict(bce=float(np.mean(-usable*np.log(q)-(1-usable)*np.log1p(-q))),
                brier=float(np.mean((prob-usable)**2)),
                calibration=float(np.mean(prob-usable)))


def evaluate(root, p, chunk):
    import torch
    from sbsi.selection_model import load_selection_model
    torch.set_num_threads(4)
    paths = {'retained': Path(p['old_classifier'])}
    for seed in p['seeds']:
        for arm in p['arms']:
            name = f'{arm}_{seed}'
            path = root / name / 'selected.pt'
            done = json.loads((path.parent / 'completed.json').read_text())
            if sha256(path) != done['selected_sha256']:
                raise ValueError('trained classifier changed')
            paths[name] = path
    models = {name: load_selection_model(path, device='cpu') for name, path in paths.items()}
    f8 = Path(p['f8_root'])
    baseline = json.loads((f8 / 'development_comparison.json').read_text())
    source_rows = {r['case']: r for r in baseline['row_replay']}
    records = []
    for case in range(40+chunk*10, 50+chunk*10):
        started = time.monotonic()
        path = f8 / f'evaluation/chunk{chunk}/rows/case{case:03d}.npz'
        if sha256(path) != source_rows[case]['sha256']:
            raise ValueError('saved F8 draws changed')
        with np.load(path) as z:
            frames, provenance = frame_inputs(p, case, z)
            if not z['cohort_truth_parent'].all():
                raise ValueError('all target-role parents must remain')
            bins = np.digitize(z['context'][:, 3], [24, 25.5, 27.5])
            sums = {name: [] for name in ['measured', *models]}
            scores = {name: [] for name in models}
            for leg in ('plus', 'minus'):
                values, usable = z[f'measured_{leg}'], z[f'usable_{leg}']
                mass = z[f'mass_{leg}']
                shape = z[f'flow_{leg}'] + z[f'blend_{leg}']
                cuts = np.stack([np.ones(len(values), bool),
                    (values[:, 2] > 3) & (values[:, 3] > 10**(.4*(30-25.8))),
                    (values[:, 2] >= 3.75) & (values[:, 3] > 10**(.4*(30-25.8)))], axis=1)
                keep = usable[:, None] & cuts
                rows = np.concatenate([keep[..., None], np.where(keep[..., None], values[:, None, :2], 0)], axis=2)
                sums['measured'].append(np.stack([rows[bins == b].sum(0) for b in range(4)], axis=1).tolist())
                for name, model in models.items():
                    prob = np.asarray(model.predict_proba(frames[leg]), float)
                    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
                        raise ValueError('invalid probabilities')
                    if name == 'retained':
                        np.testing.assert_allclose(prob, z[f'probability_modeled_smooth_{leg}'], atol=1e-6, rtol=2e-6)
                        # Keep the exact retained evaluation reference after checking
                        # independent reconstruction on this worker.
                        prob = z[f'probability_modeled_smooth_{leg}']
                    scores[name].append(metrics(prob, usable))
                    rows = np.concatenate([(prob[:, None]*mass)[..., None], prob[:, None, None]*shape], axis=2)
                    sums[name].append(np.stack([rows[bins == b].sum(0) for b in range(4)], axis=1).tolist())
            for name in ('measured', 'retained'):
                arm = 'measured' if name == 'measured' else 'modeled_smooth'
                np.testing.assert_allclose(np.asarray(sums[name])[:, 1], source_rows[case]['sums'][arm], atol=2e-6, rtol=2e-10)
            records.append(dict(case=case, sums=sums, scores=scores, provenance=provenance,
                                saved_rows_sha256=source_rows[case]['sha256']))
        print(f'CLASSIFIER_REWEIGHT case={case} seconds={time.monotonic()-started:.1f}', flush=True)
    write_json(root / f'evaluation_{chunk}.json', dict(records=records,
        classifier_hashes={k: sha256(v) for k, v in paths.items()}, protocol_sha256=sha256(root/'protocol.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'evaluate'), required=True)
    parser.add_argument('--index', type=int, required=True)
    a = parser.parse_args()
    p = check_protocol(a.root)
    state = a.root / f'{a.stage}_{a.index}_status.json'
    write_json(state, dict(state='running', job=os.environ['SLURM_JOB_ID']))
    try:
        (train if a.stage == 'train' else evaluate)(a.root, p, a.index)
    except Exception as error:
        write_json(state, dict(state='failed', error=repr(error)))
        raise
    write_json(state, dict(state='completed', job=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    main()
