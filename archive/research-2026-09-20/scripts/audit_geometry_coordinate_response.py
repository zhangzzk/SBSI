"""Counterfactual major-axis feature transport on fixed classifier checkpoints.

The existing classifier was trained with static geometry. This diagnostic is
not a validated change of model and never replaces its frozen predictions.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_classifier_measurement_transfer import calibration, predict, summarize
from scripts.audit_coherent_classifier_response import sufficient, response_summary, BIN_NAMES
from scripts.evaluate_constgold_truth_parent import measured_leg, raw_root
from scripts.joint_response_axes import shear_frames
from scripts.scene_classifier_features import geometry_features, GEOMETRY_FEATURES, PAIR_COLUMNS
from scripts.scene_classifier_joint import classifier_frames, load_arms
from scripts.scene_geometry_coordinates import transport_major_distance
from scripts.train_full_parent_classifier import sha256, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('catalogue diagnostic requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen source or artifact changed: '+path)
    torch.set_num_threads(4)
    arms, provenance = load_arms(protocol['classifiers'], protocol['control'], protocol['geometry'], 'cpu')
    bundle = arms['modeled_geometry']
    previous = {r['specification']['name']: r for r in json.loads(Path(protocol['previous']).read_text())['results']}
    source_audit = {r['specification']['name']: {row['case']: row for row in r['records']}
                    for r in json.loads(Path(protocol['old_coherent_audit']).read_text())['results']}
    results = []
    for spec in protocol['endpoints']:
        old = previous[spec['name']]
        old_rows = {r['case']: r for r in old['records']}
        manifest = json.loads((Path(spec['geometry_root'])/'manifest.json').read_text())
        rows = []
        for case in spec['cases'][:1] if args.smoke else spec['cases']:
            old_row = old_rows[case]
            parent = Path(spec['parent'])/f'case{case:03d}.npz'
            if sha256(parent) != old_row['parent_sha256']:
                raise ValueError('parent changed')
            with np.load(parent) as data:
                ids, context = data['input_index'], data['context'].astype(float)
            frames = dict(zip(('plus', 'minus'), shear_frames(context, spec['h'], spec['axis'])))
            static, geometry_provenance = classifier_frames(frames, ids, context, case, spec['geometry_root'])
            pair_path = Path(spec['pair_root'])/f'case{case:03d}.feather'
            pair_meta = json.loads(pair_path.with_suffix('.json').read_text())
            if (sha256(pair_path) != manifest['case_records'][str(case)]['pair_sha256']
                    or pair_meta['sha256'] != sha256(pair_path)
                    or pair_meta['conditions'] != protocol['conditions']):
                raise ValueError('geometry pair provenance or conditions mismatch')
            pairs = pd.read_feather(pair_path, columns=[*PAIR_COLUMNS, 'Re_input_p_scaled'])
            baseline, _ = geometry_features(ids, pairs)
            np.testing.assert_array_equal(baseline, static['plus'][list(GEOMETRY_FEATURES)].to_numpy())
            zero_pairs = transport_major_distance(ids, context, context[:, :2], pairs, protocol['psf_half_light_radius'])
            zero, _ = geometry_features(ids, zero_pairs)
            np.testing.assert_array_equal(zero, baseline)
            events = {mode: {} for mode in ('static', 'transported')}
            record = {'case': case, 'geometry_provenance': geometry_provenance, 'legs': {}}
            for leg, sign in (('plus', 1), ('minus', -1)):
                root = raw_root(Path(spec['catalogue']), case, sign*spec['h'])
                hashes = source_audit[spec['name']][case]['legs'][leg]['sha256']
                for path, expected in hashes.items():
                    if sha256(Path(path)) != expected:
                        raise ValueError('measurement catalogue changed: '+path)
                _, u, report = measured_leg(Path(spec['catalogue']), case, sign*spec['h'], ids)
                if report != old_row['measurement'][leg]:
                    raise ValueError('measurement report changed')
                d = np.isin(ids, pd.read_feather(root/'CrossMatch/tile180.0_-0.5_rot0_matched.feather', columns=['id_input'])['id_input'])
                moved = transport_major_distance(ids, context, frames[leg].iloc[:, :2].to_numpy(), pairs,
                                                protocol['psf_half_light_radius'])
                geometry, _ = geometry_features(ids, moved)
                transported = frames[leg].assign(**{name: geometry[:, i] for i, name in enumerate(GEOMETRY_FEATURES)})
                record['legs'][leg] = {'changed_geometry_parents': int(np.any(geometry != baseline, axis=1).sum()), 'calibration': {}}
                for mode, frame in (('static', static[leg]), ('transported', transported)):
                    logs = predict(bundle, frame.to_numpy())
                    p = np.exp(logs[:, 0])
                    events[mode][leg] = {'p': p, 'u': u, 'd': d}
                    record['legs'][leg]['calibration'][mode] = calibration(logs, u, d)
                for path, expected in hashes.items():
                    if sha256(Path(path)) != expected:
                        raise ValueError('measurement catalogue changed during audit: '+path)
            record['sufficient'] = {mode: sufficient(context, e['plus'], e['minus'], spec['h']) for mode, e in events.items()}
            prior = old_row['arms']['modeled_geometry']['sufficient']
            record['static_replay_prediction_per_parent_max_difference'] = {}
            for key, value in prior.items():
                actual, expected = record['sufficient']['static'][key], np.asarray(value)
                if key == 'n':
                    np.testing.assert_array_equal(actual, expected)
                    continue
                np.testing.assert_array_equal(actual[:, 1:], expected[:, 1:])
                n = np.asarray(prior['n'])
                divisor = n.reshape((-1,)+(1,)*(actual[:, 0].ndim-1))
                difference = float(np.max(np.abs((actual[:, 0]-expected[:, 0])/divisor)))
                record['static_replay_prediction_per_parent_max_difference'][key] = difference
                if difference > protocol['prediction_replay_per_parent_tolerance']:
                    raise ValueError('static probability moment replay failed')
            rows.append(record)
            print('GEOMETRY_COORDINATE_CASE', spec['name'], case, flush=True)
        results.append({'specification': spec, 'records': rows,
            'response': {mode: response_summary([r['sufficient'][mode] for r in rows]) for mode in events},
            'calibration': {mode: {leg: summarize([r['legs'][leg]['calibration'][mode] for r in rows], 'calibration')
                                   for leg in ('plus', 'minus')} for mode in events}})
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen source or artifact changed during audit: '+path)
    write_json(args.output, {'status': 'completed', 'smoke': args.smoke, 'results': results,
        'classifier_provenance': provenance, 'protocol_sha256': sha256(args.protocol),
        'magnitude_bins': BIN_NAMES, 'optimizer_steps': 0, 'calibration_claim': False,
        'limitations': ['counterfactual feature transport on a classifier trained with static geometry',
            'not a validated replacement model; training and evaluation would need a shared convention',
            'all declared endpoints and magnitude bins retained; no joint bias selection',
            'event moments have no measured cut and are not shear m; no flow or emulator evaluation']})
    print('GEOMETRY_COORDINATE_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
