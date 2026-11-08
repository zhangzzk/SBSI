"""Fixed smooth-crowding associations of shear-even classifier usability errors.

All five frozen arms are evaluated. Linear partial associations control for
all thirteen static classifier inputs; they do not prove conditional independence
or establish shear calibration. No classifier or flow is trained here.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from scripts.audit_classifier_measurement_transfer import predict
from scripts.evaluate_constgold_truth_parent import measured_leg
from scripts.joint_response_axes import shear_frames
from scripts.scene_classifier_features import geometry_features
from scripts.transported_classifier_joint import ARMS, classifier_frames, load_arms, state_geometry_frames
from scripts.train_full_parent_classifier import sha256, write_json

SCALES = (.5, 1., 2., 4.)


def smooth_crowding(ids, pairs):
    # Reuse all finite-value, identity, duplicate and unmatched-row checks.
    geometry_features(ids, pairs)
    index = pd.Index(ids).get_indexer(pairs['index_input_p'])
    distance = pairs['distance_scaled'].to_numpy(float)
    ratio = 10.**(-.4*(pairs['r_input_s_scaled'].to_numpy(float)-pairs['r_input_p_scaled'].to_numpy(float)))
    features = np.column_stack([np.log1p(np.bincount(index,
        weights=ratio*np.exp(-.5*(distance/s)**2), minlength=len(ids))) for s in SCALES])
    if not np.isfinite(features).all():
        raise ValueError('finite smooth crowding required')
    return features


def partial_associations(xx, xg, gg, xr, gr, rr, n):
    xx, xg, gg, xr, gr, rr, n = [np.asarray(v, float) for v in (xx, xg, gg, xr, gr, rr, n)]
    if not all(np.isfinite(v).all() for v in (xx, xg, gg, xr, gr, rr, n)) or np.any(n <= 0):
        raise ValueError('finite sufficient statistics and positive counts required')
    inverse = np.linalg.pinv(xx, rcond=1e-10, hermitian=True)
    projected = np.einsum('...i,...ij->...j', xr, inverse)
    cross = gr-np.einsum('...i,...ij->...j', projected, xg)
    residual_r = rr-np.einsum('...i,...i->...', projected, xr)
    residual_g = np.diagonal(gg-np.einsum('...ij,...ik,...kl->...jl', xg, inverse, xg), axis1=-2, axis2=-1)
    if np.any(residual_r <= 0) or np.any(residual_g <= 0):
        raise ValueError('positive residual variances required for partial correlation')
    return {'partial_covariance': cross/n[..., None],
            'partial_correlation': cross/np.sqrt(residual_r[..., None]*residual_g)}


def summarize(records):
    n = len(records)
    if n < 2:
        raise ValueError('at least two independent cases required')
    rng = np.random.default_rng(20260919)
    weights = np.concatenate((np.ones((1, n)), rng.multinomial(n, np.full(n, 1/n), size=2000)))
    def total(field):
        return np.einsum('bc,c...->b...', weights, np.asarray([r[field] for r in records]))
    common = {k: total(k) for k in ('xx', 'xg', 'gg', 'n')}
    report = {}
    for arm in ARMS:
        arm_stats = {k: np.einsum('bc,c...->b...', weights, np.asarray([r['arms'][arm][k] for r in records]))
                     for k in ('xr', 'gr', 'rr', 'residual_sum')}
        result = partial_associations(**common, **{k: arm_stats[k] for k in ('xr', 'gr', 'rr')})
        result['mean_probability_error'] = arm_stats['residual_sum']/common['n']
        report[arm] = {k: {'estimate': v[0], 'case_bootstrap_se': v[1:].std(axis=0, ddof=1),
                          'case_bootstrap_ci95': np.moveaxis(np.quantile(v[1:], [.025, .975], axis=0), 0, -1)}
                       for k, v in result.items()}
    return report


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm and a new output required')
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen input changed: '+path)
    torch.set_num_threads(4)
    arms, provenance = load_arms(protocol['classifiers'], protocol['control'], protocol['geometry'], protocol['transported'], 'cpu')
    prior = json.loads(Path(protocol['coherent_reference']).read_text())
    references = {e['specification']['name']: {r['case']: r for r in e['records']} for e in prior['results']}
    measured_sources = json.loads(Path(protocol['old_coherent_audit']).read_text())
    measurement_hashes = {e['specification']['name']: {r['case']: r for r in e['records']}
                          for e in measured_sources['results']}
    results = []
    for spec in protocol['endpoints']:
        records = []
        parent, geometry_root = Path(spec['parent']), Path(spec['geometry_root'])
        manifest = json.loads((parent/'manifest.json').read_text())
        geometry_manifest = json.loads((geometry_root/'manifest.json').read_text())
        for case in spec['cases']:
            path = parent/f'case{case:03d}.npz'
            if sha256(path) != manifest['case_reports'][str(case)]['sha256']:
                raise ValueError('parent hash mismatch')
            with np.load(path) as data:
                ids, context = data['input_index'], data['context'].astype(float)
            base = dict(zip(('plus', 'minus'), shear_frames(context, spec['h'], spec['axis'])))
            base['zero'] = pd.DataFrame(context, columns=FLOW_FEATURES)
            frames, geometry_receipt = classifier_frames(base, ids, context, case, geometry_root)
            pair_path = Path(spec['pair_root'])/f'case{case:03d}.feather'
            metadata = json.loads(pair_path.with_suffix('.json').read_text())
            pair_hash = sha256(pair_path)
            if pair_hash != metadata['sha256'] or pair_hash != geometry_manifest['case_records'][str(case)]['pair_sha256']:
                raise ValueError('pair hash mismatch')
            pairs = pd.read_feather(pair_path)
            moved, coordinate_receipt = state_geometry_frames(
                {k: frames[k] for k in ('plus', 'minus')}, ids, context, pairs, metadata['conditions'])
            g = smooth_crowding(ids, pairs)
            standardized = arms['modeled_geometry'].preprocessor.transform_frame(frames['zero']).astype(float)
            # All thirteen standardized values; constant missing-indicator columns
            # (if present) do not change the projection under the fixed pseudoinverse.
            design = np.column_stack((np.ones(len(ids)), standardized))
            residuals = {a: np.zeros(len(ids)) for a in ARMS}
            previous = references[spec['name']][case]
            replay = {}
            for leg, sign in (('plus', 1), ('minus', -1)):
                for p, expected in measurement_hashes[spec['name']][case]['legs'][leg]['sha256'].items():
                    if sha256(Path(p)) != expected:
                        raise ValueError('measurement hash mismatch')
                _, usable, measured_report = measured_leg(Path(spec['catalogue']), case, sign*spec['h'], ids)
                if measured_report != previous['measurement'][leg]:
                    raise ValueError('measurement receipt mismatch')
                for arm, bundle in arms.items():
                    selected = moved if arm in ARMS[3:] else frames if arm == 'modeled_geometry' else base
                    logs = predict(bundle, selected[leg].to_numpy())
                    probability = np.exp(logs[:, 0])
                    residuals[arm] += .5*(probability-usable)
                    old = previous['arms'][arm]['legs'][leg]
                    bce = -np.where(usable, logs[:, 0], logs[:, 1]).sum()
                    error = (bce-old['bce'])/len(ids)
                    if old['n'] != len(ids) or old['usable'] != int(usable.sum()) or abs(error) > 2e-7:
                        raise ValueError('frozen probability/label replay mismatch')
                    replay[arm+'/'+leg] = float(error)
            record = {'case': case, 'n': len(ids), 'xx': design.T@design, 'xg': design.T@g, 'gg': g.T@g,
                      'geometry_receipt': geometry_receipt, 'coordinate_receipt': coordinate_receipt,
                      'parent_sha256': sha256(path), 'pair_sha256': pair_hash, 'probability_replay_per_row': replay,
                      'arms': {a: {'xr': design.T@v, 'gr': g.T@v, 'rr': v@v, 'residual_sum': v.sum()}
                               for a, v in residuals.items()}}
            records.append(record)
            print('CONTINUOUS_CROWDING_CASE', spec['name'], case, len(ids), flush=True)
        results.append({'specification': spec, 'records': records, 'summary': summarize(records)})
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen input changed during audit: '+path)
    write_json(args.output, {'results': results, 'classifier_provenance': provenance,
        'kernel_scales': SCALES, 'protocol_sha256': sha256(args.protocol), 'optimizer_steps': 0,
        'calibration_claim': False, 'bootstrap_seed': 20260919, 'bootstrap_replicates': 2000,
        'definition': 'shear-even (predicted U minus observed U), associated with baseline log1p flux-ratio Gaussian radial sums',
        'controls': 'intercept and thirteen standardized static geometry classifier inputs',
        'limitations': ['linear partial associations do not prove nonlinear conditional independence',
                       'baseline geometry diagnostic; no new shear-response feature contract or model is introduced',
                       'no measured-output cut; U includes detection and usable measurement',
                       'case errors omit training uncertainty; four fresh cases are imprecise',
                       'all five arms, four kernel scales and declared endpoints retained; no feature/model selection']})
    print('CONTINUOUS_CROWDING_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
