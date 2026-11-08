"""Read-only classifier calibration across corrected measurement labels.

The truth-shape-weighted moments are conditional-U diagnostics, NOT shear m.
Cases are grouped by the classifier's original training/tuning membership.
"""
import argparse
import math
import os
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.selection_model import load_selection_model_ensemble
from scripts.audit_harmonized_usability_response import load_aligned
from scripts.train_full_parent_classifier import sha256, write_json


def ensemble_logs(logits):
    """Log arithmetic-mean Bernoulli probabilities, without clipping tails."""
    logits = logits.double()
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('finite member-by-row logits required')
    return (torch.logsumexp(F.logsigmoid(logits), 0)-math.log(len(logits)),
            torch.logsumexp(F.logsigmoid(-logits), 0)-math.log(len(logits)))


@torch.no_grad()
def predict(ensemble, context):
    result = []
    for start in range(0, len(context), 65536):
        raw = torch.as_tensor(context[start:start+65536], dtype=torch.float32)
        logits = torch.stack([m.logits_from_raw_tensor(raw).flatten() for m in ensemble.members])
        lp, lq = ensemble_logs(logits)
        native = torch.sigmoid(logits).mean(0).double()
        if torch.max(torch.abs(native-lp.exp())) > 2e-7:
            raise ValueError('stable probability disagrees with native ensemble')
        result.append(torch.stack((lp, lq), 1).numpy())
    return np.concatenate(result)


def match_ids(new_ids, old_ids):
    """Sorted identity join; return both indices and explicit excluded counts."""
    if len(np.unique(new_ids)) != len(new_ids) or len(np.unique(old_ids)) != len(old_ids):
        raise ValueError('unique identities required')
    _, ni, oi = np.intersect1d(new_ids, old_ids, return_indices=True)
    return ni, oi, {'matched': len(ni), 'new_without_old': len(new_ids)-len(ni),
                    'old_outside_new': len(old_ids)-len(oi)}


def calibration(logs, usable, detected):
    p = np.exp(logs[:, 0])
    u, d = np.asarray(usable, bool), np.asarray(detected, bool)
    if np.any(u & ~d):
        raise ValueError('usable must imply detection with shape row')
    bins = np.minimum((10*p).astype(int), 9)
    return {'n': np.asarray(len(p)), 'bce': -np.where(u, logs[:, 0], logs[:, 1]).sum(),
            'brier': np.square(p-u).sum(), 'gap': (p-u).sum(), 'usable': u.sum(),
            'detected': d.sum(), 'detected_unusable': (d & ~u).sum(),
            'reliability': np.stack([np.bincount(bins, weights=w, minlength=10)
                                     for w in (np.ones(len(p)), p, u)], 1)}


def response_moments(e0, gamma, p0, pg, u0, ug, d0, dg):
    # Components are predicted U, observed U, matched detection; U-D isolates
    # measurement failures among matched detection+shape rows.
    delta = np.stack((pg-p0, ug.astype(float)-u0, dg.astype(float)-d0), 1)
    return {'normal': gamma.T@gamma,
            'number': np.einsum('nk,ni->ki', delta, gamma),
            'truth_shape': np.einsum('nk,no,ni->koi', delta, e0, gamma)}


def summarize(rows, kind):
    weights = np.ones((1, len(rows)))
    if len(rows) > 1:
        rng = np.random.default_rng(20260914)
        weights = np.concatenate((weights, rng.multinomial(len(rows), np.full(len(rows), 1/len(rows)), size=10000)))
    totals = {k: np.einsum('bc,c...->b...', weights, np.asarray([r[k] for r in rows])) for k in rows[0]}
    if kind == 'calibration':
        values = {k: v/totals['n'] for k, v in totals.items() if k not in ('n', 'reliability')}
        bins = totals['reliability'][0]
        values['reliability_gap'] = np.divide(totals['reliability'][:, :, 1]-totals['reliability'][:, :, 2],
                                              totals['reliability'][:, :, 0],
                                              out=np.full_like(totals['reliability'][:, :, 0], np.nan),
                                              where=totals['reliability'][:, :, 0] > 0)
    else:
        inverse = np.linalg.inv(totals['normal'])
        values = {'number': totals['number']@inverse, 'truth_shape': totals['truth_shape']@inverse[:, None]}
        values.update({k+'_prediction_minus_usable': v[:, 0]-v[:, 1] for k, v in list(values.items())})
    report = {k: {'estimate': v[0], 'case_bootstrap_se': v[1:].std(0, ddof=1) if len(rows) > 1 else None,
                  'case_bootstrap_ci95': np.moveaxis(np.quantile(v[1:], [.025, .975], axis=0), 0, -1) if len(rows) > 1 else None}
              for k, v in values.items()}
    if kind == 'calibration':
        report['reliability_totals_count_probability_usable'] = bins
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-root', type=Path, required=True)
    parser.add_argument('--old-prepared', type=Path, required=True)
    parser.add_argument('--classifier', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('catalogue audit requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    paths = [Path(__file__), Path('sbsi/selection_model.py'),
             Path('scripts/audit_harmonized_usability_response.py'),
             args.domain_root/'manifest.json', args.old_prepared/'manifest.json', *args.classifier]
    pins = {str(p.resolve()): sha256(p) for p in paths}
    manifest = json.loads((args.domain_root/'manifest.json').read_text())
    old_manifest = json.loads((args.old_prepared/'manifest.json').read_text())
    if manifest['flow']['features'] != list(FLOW_FEATURES) or old_manifest['features'] != list(FLOW_FEATURES):
        raise ValueError('feature contract changed')
    ensemble = load_selection_model_ensemble(args.classifier, device='cpu')
    train, tune = set(range(40, 100)), set(range(100, 120))
    for member in ensemble.members:
        meta = member.metadata
        if (set(meta['training_cases']) != train or set(meta['tune_cases']) != tune
                or meta['prepared_manifest_sha256'] != sha256(args.old_prepared/'manifest.json')
                or member.preprocessor.feature_names != list(FLOW_FEATURES)):
            raise ValueError('classifier provenance mismatch')
    cases = [0, 40, 100] if args.smoke else list(range(200))
    records = []
    for case in cases:
        legs, provenance = load_aligned(args.domain_root, manifest, case)
        record = {'case': case, 'group': 'train' if case in train else 'tune' if case in tune else 'unseen',
                  'provenance': provenance, 'legs': {}}
        for name in ('g0', 'g05'):
            leg = legs[name]
            with np.load(args.domain_root/f'usability/{name}/case{case:03d}.npz') as stored:
                leg['context'], leg['detected'] = stored['context'], stored['detected']
            if not np.isfinite(leg['context']).all() or not np.all(leg['context'][:, 3] < 26.):
                raise ValueError('finite strict true-r<26 population required')
            leg['logs'] = predict(ensemble, leg['context'])
            leg['p'] = np.exp(leg['logs'][:, 0])
            record['legs'][name] = calibration(leg['logs'], leg['usable'], leg['detected'])
            record['legs'][name]['cross_without_shape'] = np.asarray(manifest['case_records'][str(case)]['legs'][name]['cross_without_shape'])
        z, g = legs['g0'], legs['g05']
        record['moments'] = response_moments(z['context'][:, :2], g['gamma'], z['p'], g['p'],
                                              z['usable'], g['usable'], z['detected'], g['detected'])
        if case in train | tune:
            path = args.old_prepared/f'cases/case{case:03d}.npz'
            if sha256(path) != old_manifest['case_records'][str(case)]['prepared_sha256']:
                raise ValueError('old prepared labels changed')
            with np.load(path) as old:
                ni, oi, counts = match_ids(z['input_index'], old['ids'])
                record['old_overlap'] = {'alignment': counts, 'sha256': sha256(path), 'legs': {}}
                for name, suffix in (('g0', '0'), ('g05', '1')):
                    leg = legs[name]
                    old_logs = predict(ensemble, old['x'+suffix][oi])
                    old_u, old_d = old['u'+suffix][oi].astype(bool), old['d'+suffix][oi].astype(bool)
                    diff = leg['context'][ni]-old['x'+suffix][oi]
                    def transitions(a, b):
                        return np.bincount(2*a.astype(int)+b.astype(int), minlength=4).reshape(2, 2)
                    record['old_overlap']['legs'][name] = {
                        'old_calibration': calibration(old_logs, old_u, old_d),
                        'new_calibration': calibration(leg['logs'][ni], leg['usable'][ni], leg['detected'][ni]),
                        'U_transitions_old_rows_new_columns': transitions(old_u, leg['usable'][ni]),
                        'D_transitions_old_rows_new_columns': transitions(old_d, leg['detected'][ni]),
                        'feature_mean_absolute_change': np.abs(diff).mean(0),
                        'feature_max_absolute_change': np.abs(diff).max(0),
                        'probability_mean_change': (leg['p'][ni]-np.exp(old_logs[:, 0])).mean(),
                        'probability_max_absolute_change': np.abs(leg['p'][ni]-np.exp(old_logs[:, 0])).max()}
        records.append(record)
        print(f"CLASSIFIER_TRANSFER_CASE case={case} group={record['group']} n={len(z['p'])} gap0={record['legs']['g0']['gap']/len(z['p']):.8f}", flush=True)
    groups = {}
    for name in ('train', 'tune', 'unseen'):
        rows = [r for r in records if r['group'] == name]
        groups[name] = {'cases': [r['case'] for r in rows],
                        'calibration': {leg: summarize([r['legs'][leg] for r in rows], 'calibration') for leg in ('g0', 'g05')},
                        'response_moments': summarize([r['moments'] for r in rows], 'response')}
    for path, expected in pins.items():
        if sha256(Path(path)) != expected:
            raise ValueError('audit source/model changed: '+path)
    write_json(args.output, {'format_version': 1, 'smoke': args.smoke, 'optimizer_steps': 0,
        'pins': pins, 'groups': groups, 'records': records, 'features': list(FLOW_FEATURES),
        'moment_component_order': ['predicted_U', 'observed_U', 'matched_detection_with_shape_row'],
        'bootstrap_seed': 20260914, 'reliability_bin_edges': np.linspace(0, 1, 11),
        'limitations': ['random-direction forward h0.05; not coherent or antithetic joint shear calibration',
            'truth-shape-weighted usability moments are diagnostics, not measured shear response or m',
            'D uses crossmatch joined to a shape row; cross_without_shape is reported separately',
            'corrected-label calibration uses the entire new parent; old comparisons use identity overlap',
            'classifier-unseen scenes are not necessarily unseen to the flow or emulator',
            'smoke has one case per split and no case uncertainty; no empirical correction or model update']})
    print('CLASSIFIER_TRANSFER_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
