"""Operational full-scene prediction with smoke smooth weights and fixed old arms."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import smooth_classifier_joint as smooth
from scripts.evaluate_constgold_truth_parent import measured_leg
from scripts.joint_response_axes import shear_frames
from scripts.train_full_parent_classifier import sha256, write_json


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    torch.set_num_threads(4)
    root = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1')
    path = root/'smooth_joint_smoke.json'
    if path.exists():
        raise FileExistsError(path)
    protocol = json.loads((root/'classifier_selected_density_protocol.json').read_text())
    for source, expected in protocol['files'].items():
        if sha256(Path(source)) != expected:
            raise ValueError('frozen reference source changed')
    spec = next(e for e in protocol['endpoints'] if e['name'] == 'fresh_g1')
    case = spec['cases'][0]
    parent = Path(spec['parent'])
    manifest = json.loads((parent/'manifest.json').read_text())
    data_path = parent/f'case{case:03d}.npz'
    if sha256(data_path) != manifest['case_reports'][str(case)]['sha256']:
        raise ValueError('parent hash mismatch')
    with np.load(data_path) as data:
        ids, context = data['input_index'], data['context'].astype(float)
    arms, provenance = smooth.load_arms(protocol['classifiers'], protocol['control'], protocol['geometry'],
        protocol['transported'], root/'smooth_classifier_smoke/selected.pt', 'cpu', operational=True)
    base = dict(zip(('plus', 'minus'), shear_frames(context, spec['h'], spec['axis'])))
    frames, _ = smooth.classifier_frames(base, ids, context, case, Path(spec['geometry_root']))
    pairs_path = Path(spec['pair_root'])/f'case{case:03d}.feather'
    metadata = json.loads(pairs_path.with_suffix('.json').read_text())
    if sha256(pairs_path) != metadata['sha256']:
        raise ValueError('pair hash mismatch')
    probabilities, coordinate = smooth.predict_arms(arms, frames, ids, context, pd.read_feather(pairs_path), metadata['conditions'])
    old = json.loads(Path(protocol['coherent_reference']).read_text())
    record = next(e for e in old['results'] if e['specification']['name'] == spec['name'])['records'][0]
    replay = {}
    for leg, sign in (('plus', 1), ('minus', -1)):
        _, usable, report = measured_leg(Path(spec['catalogue']), case, sign*spec['h'], ids)
        if report != record['measurement'][leg]:
            raise ValueError('measurement receipt mismatch')
        for a in smooth.ARMS:
            p = probabilities[a][leg]
            if p.shape != usable.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('invalid probability output')
            if a != 'modeled_smooth':
                # Brier replay avoids log rounding at saturated probabilities.
                difference = (np.square(p-usable).sum()-record['arms'][a]['legs'][leg]['brier'])/len(ids)
                if abs(difference) > 2e-7:
                    raise ValueError('old prediction replay mismatch')
                replay[a+'/'+leg] = difference
    write_json(path, {'case': case, 'parents': len(ids), 'old_brier_replay_per_row': replay,
                     'classifier_provenance': provenance, 'coordinate': coordinate,
                     'calibration_claim': False, 'operational_smoke_only': True,
                     'source_sha256': sha256(Path(__file__))})
    print('SMOOTH_JOINT_SMOKE_COMPLETE', path, flush=True)


if __name__ == '__main__':
    main()
