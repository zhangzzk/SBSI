"""Relative selected parent/output densities for five classifiers and one fixed flow.

The common transported measurement density cancels from log density ratios.
For arm a, the selected law is p_a(U|i) q(y|i) S(y) / M_a,
where M_a=sum_i p_a(U|i) P_q(S|i). This is not a per-parent conditional
score or a marginalized output-only score. No model is fitted here.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_classifier_measurement_transfer import predict
from scripts.evaluate_constgold_truth_parent import CUTS, measured_leg, output_cuts
from scripts.joint_response_axes import shear_frames
from scripts.normalize_selected_density_comparison import score_change
from scripts.summarize_classifier_joint_stability import read_verified
from scripts.train_full_parent_classifier import sha256, write_json
from scripts.transported_classifier_joint import ARMS, classifier_frames, load_arms, state_geometry_frames


def selected_statistics(log_usable, values, usable):
    logs = np.asarray(log_usable, float)
    if logs.shape != usable.shape or not np.isfinite(logs).all() or np.any(logs > 1e-7):
        raise ValueError('finite aligned log probabilities required')
    selected = output_cuts(values) & usable[:, None]
    return {cut: {'count': int(selected[:, k].sum()),
                  'negative_log_usable_sum': float(-logs[selected[:, k]].sum())}
            for k, cut in enumerate(CUTS)}


def check_pins(protocol):
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen input changed: '+path)


@torch.no_grad()
def collect(protocol):
    check_pins(protocol)
    torch.set_num_threads(4)
    arms, provenance = load_arms(protocol['classifiers'], protocol['control'],
                                protocol['geometry'], protocol['transported'], 'cpu')
    prior = json.loads(Path(protocol['coherent_reference']).read_text())
    refs = {e['specification']['name']: {r['case']: r for r in e['records']} for e in prior['results']}
    original = json.loads(Path(protocol['old_coherent_audit']).read_text())
    sources = {e['specification']['name']: {r['case']: r for r in e['records']} for e in original['results']}
    results = []
    for spec in protocol['endpoints']:
        parent, geometry = Path(spec['parent']), Path(spec['geometry_root'])
        manifest = json.loads((parent/'manifest.json').read_text())
        geom_manifest = json.loads((geometry/'manifest.json').read_text())
        records = []
        for case in spec['cases']:
            path = parent/f'case{case:03d}.npz'
            if sha256(path) != manifest['case_reports'][str(case)]['sha256']:
                raise ValueError('parent hash mismatch')
            with np.load(path) as data:
                ids, context = data['input_index'], data['context'].astype(float)
            base = dict(zip(('plus', 'minus'), shear_frames(context, spec['h'], spec['axis'])))
            frames, receipt = classifier_frames(base, ids, context, case, geometry)
            pair_path = Path(spec['pair_root'])/f'case{case:03d}.feather'
            metadata = json.loads(pair_path.with_suffix('.json').read_text())
            pair_hash = sha256(pair_path)
            if pair_hash != metadata['sha256'] or pair_hash != geom_manifest['case_records'][str(case)]['pair_sha256']:
                raise ValueError('pair hash mismatch')
            moved, coordinate = state_geometry_frames(frames, ids, context, pd.read_feather(pair_path), metadata['conditions'])
            record = {'case': case, 'geometry': receipt, 'coordinate': coordinate, 'legs': {}}
            for leg, sign in (('plus', 1), ('minus', -1)):
                for p, expected in sources[spec['name']][case]['legs'][leg]['sha256'].items():
                    if sha256(Path(p)) != expected:
                        raise ValueError('measurement hash mismatch')
                values, usable, measured = measured_leg(Path(spec['catalogue']), case, sign*spec['h'], ids)
                previous = refs[spec['name']][case]
                if measured != previous['measurement'][leg]:
                    raise ValueError('measurement receipt mismatch')
                outputs = {}
                for arm, bundle in arms.items():
                    frame = moved if arm in ARMS[3:] else frames if arm == 'modeled_geometry' else base
                    logs = predict(bundle, frame[leg].to_numpy())
                    old = previous['arms'][arm]['legs'][leg]
                    bce = float(-np.where(usable, logs[:, 0], logs[:, 1]).sum())
                    error = (bce-old['bce'])/len(ids)
                    if old['n'] != len(ids) or old['usable'] != int(usable.sum()) or abs(error) > 2e-7:
                        raise ValueError('probability replay mismatch')
                    outputs[arm] = {'cuts': selected_statistics(logs[:, 0], values, usable),
                                    'bce': bce, 'bce_replay_per_row': error}
                record['legs'][leg] = {'measurement': measured, 'arms': outputs}
            records.append(record)
            print('CLASSIFIER_SELECTED_CASE', spec['name'], case, flush=True)
        results.append({'specification': spec, 'records': records,
                        'parent_manifest_sha256': sha256(parent/'manifest.json')})
    check_pins(protocol)
    return {'results': results, 'classifier_provenance': provenance, 'optimizer_steps': 0}


def compare_endpoint(observed, joint, provenance):
    spec = observed['specification']
    if (joint['cases'] != spec['cases'] or joint['h'] != spec['h']
            or joint['shear_axis'] != spec['axis'] or joint['cuts'] != list(CUTS)
            or joint['inputs']['parent_manifest_sha256'] != observed['parent_manifest_sha256']
            or joint['inputs']['classifier_arms'] != provenance):
        raise ValueError('classifier density/mass contract mismatch')
    n = len(spec['cases'])
    if n < 2 or [r['case'] for r in observed['records']] != spec['cases']:
        raise ValueError('complete ordered case records required')
    weights = np.concatenate((np.ones((1, n)),
        np.random.default_rng(20260919).multinomial(n, np.full(n, 1/n), size=2000)))
    report = {}
    for leg in ('plus', 'minus'):
        for cut in CUTS:
            reference = ARMS[0]
            stats = {a: [r['legs'][leg]['arms'][a]['cuts'][cut] for r in observed['records']] for a in ARMS}
            count = np.array([r['count'] for r in stats[reference]])
            mass = {}
            for a in ARMS:
                raw = joint['per_case_sufficient']['truth_parent/'+cut+'/'+a]
                np.testing.assert_array_equal(count, np.array([r['count'] for r in stats[a]]))
                np.testing.assert_array_equal(count, np.asarray(raw['measured'][leg]['denominator'])[:, 0])
                mass[a] = np.asarray(raw['model'][leg]['denominator'])[:, 0]
            for record in observed['records']:
                if record['legs'][leg]['measurement'] != joint['case_reports'][str(record['case'])]['legs'][leg]:
                    raise ValueError('measurement mass provenance mismatch')
            for a in ARMS:
                delta = np.array([x['negative_log_usable_sum']-y['negative_log_usable_sum']
                                  for x, y in zip(stats[a], stats[reference])])/count
                changes = score_change(count, delta, mass[reference], mass[a], weights)
                report[leg+'/'+cut+'/'+a] = {
                    'scores': {k: {'estimate': float(v[0]), 'case_bootstrap_se': float(v[1:].std(ddof=1)),
                                   'case_bootstrap_ci95': np.quantile(v[1:], [.025, .975])} for k, v in changes.items()},
                    'observed_count': int(count.sum()),
                    'per_case': {'count': count, 'raw_delta_nll': delta,
                                 'reference_mass': mass[reference], 'candidate_mass': mass[a]}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observed', type=Path)
    parser.add_argument('--mode', choices=('collect', 'early', 'full'), required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm and new output required')
    protocol = json.loads(args.protocol.read_text())
    check_pins(protocol)
    if args.mode == 'collect':
        out = collect(protocol)
    else:
        digest = sha256(args.observed)
        observed = json.loads(args.observed.read_text())
        if observed['protocol_sha256'] != sha256(args.protocol):
            raise ValueError('observed statistics protocol mismatch')
        root = Path(protocol['joint_root'])
        reports = {}
        for endpoint in observed['results']:
            name = endpoint['specification']['name']
            if args.mode == 'early' and not name.startswith('fresh_'):
                continue
            label = {'main80': 'main', 'independent40': 'independent'}.get(name, name)
            suffix = '_cuda_early' if args.mode == 'early' else '_cuda'
            path = root/f'transported_joint_{label}{suffix}.json'
            receipt_path = root/f'transported_joint_{label}_comparison{suffix}.json'
            joint, receipt = read_verified(path, receipt_path)
            expected_protocol = protocol['early_joint_protocol'] if args.mode == 'early' else protocol['full_joint_protocol']
            if receipt['protocol_sha256'] != sha256(Path(expected_protocol)) or receipt['endpoint'] != label:
                raise ValueError('joint protocol/endpoint receipt mismatch')
            reports[name] = {'comparison': compare_endpoint(endpoint, joint, observed['classifier_provenance']),
                             'joint_sha256': sha256(path), 'receipt_sha256': sha256(receipt_path)}
            print('CLASSIFIER_SELECTED_COMPARISON', name, flush=True)
        if sha256(args.observed) != digest:
            raise ValueError('observed statistics changed')
        out = {'results': reports, 'observed_sha256': digest,
               'reference_arm': ARMS[0], 'bootstrap_seed': 20260919, 'bootstrap_replicates': 2000}
    check_pins(protocol)
    out.update(protocol_sha256=sha256(args.protocol), calibration_claim=False,
               definition='pooled finite parent identity and physical output conditional on predicted U and measured selection',
               limitations=['same flow/emulator in every arm; only classifier density ratios are measured',
                            'relative selected score does not establish absolute density adequacy or shear calibration',
                            '64-draw selection masses; uncertainty excludes finite-draw and training variation',
                            'all declared arms/cuts/legs retained; development data, no checkpoint selection'])
    write_json(args.output, out)
    print('CLASSIFIER_SELECTED_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
