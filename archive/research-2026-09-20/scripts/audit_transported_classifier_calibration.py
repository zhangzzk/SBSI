"""Coordinate-aware fixed coherent-scene probability and usability-response audit.

No flow/emulator predictions or checkpoint selection are performed.
"""
import argparse
import json
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_classifier_measurement_transfer import calibration, predict, summarize
from scripts.audit_coherent_classifier_response import sufficient, response_summary, BIN_NAMES
from scripts.evaluate_constgold_truth_parent import measured_leg, raw_root
from scripts.joint_response_axes import shear_frames
from scripts.scene_classifier_features import GEOMETRY_FEATURES
from scripts.transported_classifier_joint import load_arms, classifier_frames, state_geometry_frames, ARMS
from scripts.train_full_parent_classifier import sha256, write_json

PAIRS = tuple(combinations(ARMS, 2))


def geometry_error(geometry, plus_probability, minus_probability, plus_u, minus_u):
    geometry = np.asarray(geometry, float)
    residual = .5*(np.asarray(plus_probability)-np.asarray(plus_u, float)
                    +np.asarray(minus_probability)-np.asarray(minus_u, float))
    if geometry.shape != (len(residual), len(GEOMETRY_FEATURES)) or not np.isfinite(geometry).all():
        raise ValueError('aligned finite five-column geometry required')
    return {'n': np.asarray(len(residual)), 'residual_sum': residual.sum(),
            'geometry_residual_sum': residual@geometry}


def case_summary(n, sums, weights):
    values = np.einsum('bc,c...->b...', weights, np.asarray(sums))
    denominator = weights@np.asarray(n)
    values /= denominator.reshape((-1,)+(1,)*(values.ndim-1))
    return {'estimate': values[0], 'case_bootstrap_se': values[1:].std(0, ddof=1) if len(n) > 1 else None,
            'case_bootstrap_ci95': np.moveaxis(np.quantile(values[1:], [.025, .975], axis=0), 0, -1) if len(n) > 1 else None}


def endpoint_summary(records):
    n = len(records)
    rng = np.random.default_rng(20260914)
    weights = np.concatenate((np.ones((1, n)), rng.multinomial(n, np.full(n, 1/n), size=10000))) if n > 1 else np.ones((1, n))
    report = {}
    for arm in ARMS:
        rows = [r['arms'][arm] for r in records]
        report[arm] = {'calibration': {leg: summarize([r['legs'][leg] for r in rows], 'calibration') for leg in ('plus', 'minus')},
                      'response': response_summary([r['sufficient'] for r in rows]),
                      'geometry_residual': case_summary([r['geometry']['n'] for r in rows], [r['geometry']['geometry_residual_sum'] for r in rows], weights)}
    comparisons = {}
    for left, right in PAIRS:
        key = right+'_minus_'+left
        comparisons[key] = {}
        counts = [sum(r['arms'][left]['legs'][leg]['n'] for leg in ('plus', 'minus')) for r in records]
        for metric in ('bce', 'brier', 'gap'):
            delta = [sum(r['arms'][right]['legs'][leg][metric]-r['arms'][left]['legs'][leg][metric] for leg in ('plus', 'minus')) for r in records]
            comparisons[key][metric] = case_summary(counts, delta, weights)
    return {'arms': report, 'paired_probability_changes': comparisons}


def audit_endpoint(spec, arms, old_result, smoke):
    parent, geometry_root = Path(spec['parent']), Path(spec['geometry_root'])
    manifest_path = parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    geometry_manifest = json.loads((geometry_root/'manifest.json').read_text())
    joint = json.loads(Path(spec['joint_result']).read_text())
    if (joint['cases'] != spec['cases'] or joint['h'] != spec['h'] or joint.get('shear_axis', 1) != spec['axis']
            or joint['inputs']['parent_manifest_sha256'] != sha256(manifest_path)):
        raise ValueError('joint validation population mismatch')
    old_records = {r['case']: r for r in old_result['records']}
    cases = spec['cases'][:1] if smoke else spec['cases']
    records = []
    for case in cases:
        path = parent/f'case{case:03d}.npz'
        if sha256(path) != manifest['case_reports'][str(case)]['sha256']:
            raise ValueError('prepared parent changed')
        with np.load(path) as stored:
            ids, context = stored['input_index'], stored['context'].astype(float)
        base_frames = dict(zip(('plus', 'minus'), shear_frames(context, spec['h'], spec['axis'])))
        frames, geometry_provenance = classifier_frames(base_frames, ids, context, case, geometry_root)
        geometry = frames['plus'][list(GEOMETRY_FEATURES)].to_numpy()
        pair_path = Path(spec['pair_root'])/f'case{case:03d}.feather'
        metadata = json.loads(pair_path.with_suffix('.json').read_text())
        if sha256(pair_path) != geometry_manifest['case_records'][str(case)]['pair_sha256'] or metadata['sha256'] != sha256(pair_path):
            raise ValueError('frozen geometry pairs changed')
        pairs = pd.read_feather(pair_path)
        transported, coordinate_report = state_geometry_frames(frames, ids, context, pairs, metadata['conditions'])
        record = {'case': case, 'parent_sha256': sha256(path), 'geometry_provenance': geometry_provenance,
                  'measurement': {}, 'coordinate_report': coordinate_report, 'arms': {arm: {'legs': {}} for arm in ARMS}}
        events, probabilities = {}, {arm: {} for arm in ARMS}
        for leg, sign in (('plus', 1), ('minus', -1)):
            catalog = raw_root(Path(spec['catalogue']), case, sign*spec['h'])
            cross_path = catalog/'CrossMatch/tile180.0_-0.5_rot0_matched.feather'
            hashes = old_records[case]['legs'][leg]['sha256']
            for p, expected in hashes.items():
                if sha256(Path(p)) != expected:
                    raise ValueError('measurement catalogue differs from prior audit: '+p)
            _, u, measured_report = measured_leg(Path(spec['catalogue']), case, sign*spec['h'], ids)
            if measured_report != joint['case_reports'][str(case)]['legs'][leg]:
                raise ValueError('measurement report changed')
            d = np.isin(ids, pd.read_feather(cross_path, columns=['id_input'])['id_input'])
            events[leg] = {'u': u, 'd': d}
            record['measurement'][leg] = measured_report
            for arm, bundle in arms.items():
                selected_frames = (transported if arm in ARMS[3:] else frames if arm == 'modeled_geometry' else base_frames)
                raw = selected_frames[leg].to_numpy()
                logs = predict(bundle, raw)
                probabilities[arm][leg] = np.exp(logs[:, 0])
                stats = calibration(logs, u, d)
                record['arms'][arm]['legs'][leg] = stats
                if arm == 'modeled_usable':
                    old = old_records[case]['legs'][leg]['calibration']
                    for field in ('n', 'usable', 'detected', 'detected_unusable'):
                        if stats[field] != old[field]:
                            raise ValueError('historical classifier label replay failed')
                    differences = {field: float((stats[field]-old[field])/len(ids)) for field in ('bce', 'brier', 'gap')}
                    if max(map(abs, differences.values())) > 2e-7:
                        raise ValueError('historical classifier probability replay failed')
                    record.setdefault('reference_replay_per_row_difference', {})[leg] = differences
            for p, expected in hashes.items():
                if sha256(Path(p)) != expected:
                    raise ValueError('measurement changed during audit')
        for arm in ARMS:
            plus = events['plus'] | {'p': probabilities[arm]['plus']}
            minus = events['minus'] | {'p': probabilities[arm]['minus']}
            record['arms'][arm]['sufficient'] = sufficient(context, plus, minus, spec['h'])
            record['arms'][arm]['geometry'] = geometry_error(geometry, plus['p'], minus['p'], plus['u'], minus['u'])
        records.append(record)
        print(f"SCENE_CLASSIFIER_CALIBRATION_CASE endpoint={spec['name']} case={case} parents={len(ids)}", flush=True)
    return {'specification': spec, 'cases': cases, 'records': records, **endpoint_summary(records)}


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
    arms, provenance = load_arms(protocol['classifiers'], protocol['control'], protocol['geometry'], protocol['transported'], 'cpu')
    old = json.loads(Path(protocol['old_coherent_audit']).read_text())
    old_endpoints = {r['specification']['name']: r for r in old['results']}
    results = [audit_endpoint(spec, arms, old_endpoints[spec['name']], args.smoke) for spec in protocol['endpoints']]
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('audit source/model changed: '+path)
    write_json(args.output, {'format_version': 1, 'smoke': args.smoke, 'optimizer_steps': 0,
        'results': results, 'classifier_provenance': provenance, 'protocol_sha256': sha256(args.protocol),
        'magnitude_bins': BIN_NAMES, 'response_basis': ['one', 'intrinsic_e1', 'intrinsic_e2'],
        'response_components': ['predicted_U', 'observed_U', 'raw_crossmatch_D'],
        'geometry_coordinates': GEOMETRY_FEATURES, 'bootstrap_seed': 20260914,
        'geometry_moment': 'full-parent mean of geometry times even-leg (predicted U minus observed U)',
        'limitations': ['event and geometry residual moments are not measured shear response or m',
            'main80 overlaps training/tuning; independent40 and fresh4 are unseen to all classifier arms',
            'probability scores do not establish selected joint-density or emulator calibration',
            'all declared arms/endpoints retained; no checkpoint reselection or empirical correction',
            'modeled_coordinate is counterfactual static-trained; modeled_transported uses consistent coordinates']})
    print('SCENE_CLASSIFIER_CALIBRATION_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
