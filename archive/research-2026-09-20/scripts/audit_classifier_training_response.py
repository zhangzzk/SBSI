"""Audit fixed classifiers on the response information in their training pairs.

This read-only audit cannot establish generalization or select a checkpoint.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.audit_classifier_measurement_transfer import predict, response_moments, summarize
from scripts.scene_classifier_joint import load_arms
from scripts.train_full_parent_classifier import sha256, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full prepared-data audit requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen input changed: '+path)
    torch.set_num_threads(4)
    root, domain = Path(protocol['prepared_root']), Path(protocol['domain_root'])
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['cases'] != list(range(40, 120)) or not manifest['with_corrected_labels']:
        raise ValueError('fixed train60 and tune20 corrected-label cases required')
    arms, provenance = load_arms(protocol['classifiers'], protocol['control'], protocol['geometry'], 'cpu')
    previous = {r['case']: r for r in json.loads(Path(protocol['transfer_result']).read_text())['records']}
    rows = []
    for case in manifest['cases']:
        path = root/f'cases/case{case:03d}.npz'
        metadata = manifest['case_records'][str(case)]
        if sha256(path) != metadata['prepared_sha256']:
            raise ValueError('prepared labels changed')
        with np.load(path) as stored:
            data = {key: stored[key] for key in ('ids', 'x0', 'x1', 'u0', 'u1', 'd0', 'd1')}
        gamma = None
        for index, leg in enumerate(('g0', 'g05')):
            raw_path = domain/f'usability/{leg}/case{case:03d}.npz'
            if sha256(raw_path) != metadata['label_provenance'][leg]['usability_sha256']:
                raise ValueError('source prepared leg changed')
            with np.load(raw_path) as raw:
                np.testing.assert_array_equal(raw['input_index'], data['ids'])
                np.testing.assert_array_equal(raw['usable'], data[f'u{index}'])
                np.testing.assert_array_equal(raw['detected'], data[f'd{index}'])
                np.testing.assert_array_equal(raw['context'].astype(np.float32), data[f'x{index}'][:, :8])
                if index == 0:
                    np.testing.assert_array_equal(raw['gamma'], 0.)
                else:
                    gamma = raw['gamma']
                    np.testing.assert_allclose(np.linalg.norm(gamma, axis=1), .05, rtol=0, atol=1e-12)
        record = {'case': case, 'group': 'train' if case < 100 else 'tune',
                  'prepared_sha256': sha256(path), 'moments': {}}
        for arm, bundle in arms.items():
            dimension = 13 if arm == 'modeled_geometry' else 8
            p0, p1 = [np.exp(predict(bundle, data[f'x{i}'][:, :dimension])[:, 0]) for i in (0, 1)]
            moments = response_moments(data['x0'][:, :2], gamma, p0, p1,
                                       data['u0'], data['u1'], data['d0'], data['d1'])
            record['moments'][arm] = moments
            if arm == 'modeled_usable':
                # Normalized responses magnify small float32 rounding; retain
                # raw sufficient-statistic replay with a declared tolerance.
                differences = {k: float(np.max(np.abs(np.asarray(v)-np.asarray(previous[case]['moments'][k]))))
                               for k, v in moments.items()}
                for k, v in moments.items():
                    np.testing.assert_allclose(v, previous[case]['moments'][k], rtol=1e-6, atol=1e-6)
                record['reference_replay_max_differences'] = differences
        rows.append(record)
        print('TRAINING_CLASSIFIER_RESPONSE_CASE', case, flush=True)
    groups = {group: {arm: summarize([r['moments'][arm] for r in rows if r['group'] == group], 'response')
                      for arm in arms} for group in ('train', 'tune')}
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('input changed during audit: '+path)
    write_json(args.output, {'status': 'completed', 'groups': groups, 'records': rows,
        'classifier_provenance': provenance, 'protocol_sha256': sha256(args.protocol),
        'moment_component_order': ['predicted_U', 'observed_U', 'matched_detection_with_shape_row'],
        'bootstrap_seed': 20260914, 'optimizer_steps': 0, 'calibration_claim': False,
        'limitations': ['training60 and tuning20 reused from checkpoint training, not independent acceptance',
            'random-direction forward h0.05, not coherent antithetic m',
            'intrinsic-shape moments include all truth parents and no measured-output cut',
            'no checkpoint reselection, empirical response correction or model update']})
    print('TRAINING_CLASSIFIER_RESPONSE_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
