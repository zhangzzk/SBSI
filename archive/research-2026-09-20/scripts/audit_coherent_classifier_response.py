"""Read-only coherent-scene usability audit on fixed joint-validation parents.

Intrinsic-shape-weighted event derivatives diagnose classifier transfer; they
are not measured-shape response or multiplicative shear bias.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.selection_model import load_selection_model_ensemble
from scripts.audit_classifier_measurement_transfer import calibration, predict, summarize
from scripts.evaluate_constgold_truth_parent import measured_leg, raw_root
from scripts.joint_response_axes import shear_frames
from scripts.train_full_parent_classifier import sha256, write_json

BIN_NAMES = ['all', 'r_lt24', 'r_24_to25', 'r_25_to25.8', 'r_25.8_to26']


def sufficient(context, plus, minus, h):
    """Full-parent antithetic moments, retaining detections entering or leaving."""
    context = np.asarray(context, float)
    n = len(context)
    if context.shape != (n, 8) or not np.isfinite(context).all() or h <= 0:
        raise ValueError('finite full-parent contexts and positive step required')
    if (context[:, 3] >= 26).any():
        raise ValueError('strict true-r<26 required')
    entries = []
    for leg in (plus, minus):
        p, u, d = [np.asarray(leg[k]) for k in ('p', 'u', 'd')]
        if any(a.shape != (n,) for a in (p, u, d)) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('aligned probabilities and events required')
        if not np.isin(u, [0, 1]).all() or not np.isin(d, [0, 1]).all() or np.any(u > d):
            raise ValueError('binary U implies D required')
        entries.append(np.stack((p, u, d), 1).astype(float))
    magnitude = context[:, 3]
    masks = np.stack((np.ones(n, bool), magnitude < 24, (magnitude >= 24) & (magnitude < 25),
                      (magnitude >= 25) & (magnitude < 25.8), magnitude >= 25.8), 1)
    basis = np.column_stack((np.ones(n), context[:, :2]))
    return {'n': masks.sum(0), 'response_sum': np.einsum('nb,nk,no->bko', masks, (entries[0]-entries[1])/(2*h), basis),
            'event_sum': np.einsum('nb,nk->bk', masks, (entries[0]+entries[1])/2)}


def response_summary(rows):
    n = len(rows)
    rng = np.random.default_rng(20260914)
    weights = np.concatenate((np.ones((1, n)), rng.multinomial(n, np.full(n, 1/n), size=10000))) if n > 1 else np.ones((1, n))
    total = {key: np.einsum('bc,c...->b...', weights, np.asarray([r[key] for r in rows])) for key in rows[0]}
    if np.any(total['n'] == 0):
        raise ValueError('all prespecified magnitude bins must contain parents')
    response = total['response_sum']/total['n'][:, :, None, None]
    values = {'response': response, 'prediction_minus_U': response[:, :, 0]-response[:, :, 1],
              'U_minus_D': response[:, :, 1]-response[:, :, 2],
              'mean_event_probability': total['event_sum']/total['n'][:, :, None]}
    return {key: {'estimate': v[0], 'case_bootstrap_se': v[1:].std(0, ddof=1) if n > 1 else None,
                  'case_bootstrap_ci95': np.moveaxis(np.quantile(v[1:], [.025, .975], axis=0), 0, -1) if n > 1 else None}
            for key, v in values.items()}


def audit_endpoint(spec, ensemble, smoke):
    parent = Path(spec['parent'])
    manifest_path = parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    joint_path = Path(spec['joint_result'])
    joint = json.loads(joint_path.read_text())
    if (joint['cases'] != spec['cases'] or joint['h'] != spec['h'] or joint.get('shear_axis', 1) != spec['axis']
            or joint['inputs']['parent_manifest_sha256'] != sha256(manifest_path)):
        raise ValueError('joint validation population or extraction mismatch')
    cases = spec['cases'][:1] if smoke else spec['cases']
    records = []
    for case in cases:
        path = parent/f'case{case:03d}.npz'
        if sha256(path) != manifest['case_reports'][str(case)]['sha256']:
            raise ValueError('prepared parent changed')
        with np.load(path) as stored:
            ids, context = stored['input_index'], stored['context'].astype(float)
        if len(np.unique(ids)) != len(ids):
            raise ValueError('duplicate parent identities')
        legs, record = {}, {'case': case, 'parent_sha256': sha256(path), 'legs': {}}
        frames = shear_frames(context, spec['h'], spec['axis'])
        for name, sign, frame in zip(('plus', 'minus'), (1, -1), frames):
            root = raw_root(Path(spec['catalogue']), case, sign*spec['h'])
            paths = [root/'CrossMatch/tile180.0_-0.5_rot0_matched.feather',
                     root/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather']
            hashes = {str(p): sha256(p) for p in paths}
            _, usable, report = measured_leg(Path(spec['catalogue']), case, sign*spec['h'], ids)
            if report != joint['case_reports'][str(case)]['legs'][name]:
                raise ValueError('measured labels disagree with frozen joint evaluation')
            cross = pd.read_feather(paths[0], columns=['id_input'])
            detected = np.isin(ids, cross['id_input'])
            logs = predict(ensemble, frame.to_numpy())
            legs[name] = {'p': np.exp(logs[:, 0]), 'u': usable, 'd': detected}
            record['legs'][name] = {'calibration': calibration(logs, usable, detected), 'measurement': report, 'sha256': hashes}
            for p, expected in hashes.items():
                if sha256(Path(p)) != expected:
                    raise ValueError('catalogue changed during audit')
        record['sufficient'] = sufficient(context, legs['plus'], legs['minus'], spec['h'])
        records.append(record)
        print(f"COHERENT_CLASSIFIER_CASE endpoint={spec['name']} case={case} n={len(ids)}", flush=True)
    return {'specification': spec, 'cases': cases, 'records': records,
            'parent_manifest_sha256': sha256(manifest_path), 'joint_result_sha256': sha256(joint_path),
            'response': response_summary([r['sufficient'] for r in records]),
            'calibration': {leg: summarize([r['legs'][leg]['calibration'] for r in records], 'calibration') for leg in ('plus', 'minus')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('catalogue audit requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen audit input changed: '+path)
    torch.set_num_threads(4)
    ensemble = load_selection_model_ensemble(protocol['classifiers'], device='cpu')
    for spec in protocol['endpoints']:
        joint = json.loads(Path(spec['joint_result']).read_text())
        if joint['inputs']['classifiers'] != [{'path': p, 'sha256': sha256(Path(p))} for p in protocol['classifiers']]:
            raise ValueError('classifier differs from joint evaluation')
    results = [audit_endpoint(spec, ensemble, args.smoke) for spec in protocol['endpoints']]
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('source/model changed during audit: '+path)
    write_json(args.output, {'format_version': 1, 'smoke': args.smoke, 'optimizer_steps': 0,
        'protocol_sha256': sha256(args.protocol), 'results': results, 'magnitude_bins': BIN_NAMES,
        'basis': ['one', 'intrinsic_e1', 'intrinsic_e2'], 'components': ['predicted_U', 'observed_U', 'raw_crossmatch_D'],
        'normalization': 'full truth-parent count separately in each truth-magnitude bin; no measured selection',
        'extraction': 'coherent antithetic central difference; intrinsic shape fixed between legs',
        'bootstrap_seed': 20260914,
        'limitations': ['event/truth-shape moments are not measured shear response or m',
                        'no flow/emulator predictions; cannot establish selected joint-density calibration',
                        'main80 overlaps classifier train/tune cases; independent40 and fresh4 are classifier-unseen',
                        'different h and populations are reported separately; no pooling or empirical correction']})
    print('COHERENT_CLASSIFIER_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
