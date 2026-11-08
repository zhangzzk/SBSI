"""Retain scene influence and frozen latent-repeat evidence for both components."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from plots.constgold_property_bias import make_case_weights
from scripts.compare_truth_parent_response import require_latent_repeat
from scripts.summarize_classifier_joint_stability import read_verified
from scripts.summarize_smooth_response_vectors import response_vectors
from scripts.train_full_parent_classifier import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm and new output required')
    endpoints, sources = {}, {}
    for endpoint in ('main', 'independent', 'fresh_g1', 'fresh_g2'):
        path = args.root/f'smooth_joint_{endpoint}.json'
        receipt_path = args.root/f'smooth_joint_{endpoint}_comparison.json'
        baseline, receipt = read_verified(path, receipt_path)
        if receipt['maximum_reference_sufficient_difference'] != 0:
            raise ValueError('exact reference replay required')
        sources.update({str(p): sha256(p) for p in (path, receipt_path)})
        n = len(baseline['cases'])
        weights = np.concatenate((np.ones((1, n)), np.eye(n), np.ones((n, n))-np.eye(n),
                                  make_case_weights(10000, n, 20260914)))
        replica = None
        if endpoint.startswith('fresh'):
            repeat_path = args.root/f'smooth_joint_{endpoint}_seed7302.json'
            comparison_path = args.root/f'smooth_joint_{endpoint}_latent_comparison.json'
            comparison = json.loads(comparison_path.read_text())
            if (comparison['baseline_sha256'] != sha256(path)
                    or comparison['candidate_sha256'] != sha256(repeat_path)
                    or comparison['comparison_kind'] != 'independent_latent_repeat'):
                raise ValueError('latent repeat receipt mismatch')
            replica = json.loads(repeat_path.read_text())
            require_latent_repeat(baseline, replica)
            sources.update({str(p): sha256(p) for p in (repeat_path, comparison_path)})
        records = {}
        for key, sufficient in baseline['per_case_sufficient'].items():
            original = response_vectors(sufficient, weights, baseline['h'], baseline['shear_axis'])
            records[key] = {field: {'point': v[0], 'per_scene': v[1:1+n],
                                    'leave_one_out': v[1+n:1+2*n],
                                    'leave_one_out_min_max': np.stack((v[1+n:1+2*n].min(0), v[1+n:1+2*n].max(0))).T}
                            for field, v in original.items()}
            if replica is not None:
                other = replica['per_case_sufficient'][key]
                for leg in ('plus', 'minus'):
                    for field in ('numerator', 'denominator'):
                        np.testing.assert_array_equal(sufficient['measured'][leg][field], other['measured'][leg][field])
                repeated = response_vectors(other, weights, baseline['h'], baseline['shear_axis'])
                records[key]['latent_repeat'] = {}
                for field in original:
                    delta = repeated[field]-original[field]
                    boot = delta[1+2*n:]
                    records[key]['latent_repeat'][field] = {'repeat_point': repeated[field][0],
                        'delta': delta[0], 'paired_case_se': boot.std(0, ddof=1),
                        'paired_case_ci95': np.quantile(boot, [.025, .975], axis=0).T}
        endpoints[endpoint] = {'cases': baseline['cases'], 'axis': baseline['shear_axis'],
                               'h': baseline['h'], 'summary': records}
        print('CROSS_RESPONSE_AUDIT_COMPLETE', endpoint, flush=True)
    write_json(args.output, {'endpoints': endpoints, 'sources': sources,
        'script_sha256': sha256(Path(__file__)), 'vector_helper_sha256': sha256(Path('scripts/summarize_smooth_response_vectors.py')),
        'calibration_claim': False,
        'limitations': ['all arms/cuts/scenes retained; no exclusions or favorable seed selection',
                       'single-scene estimates and omission ranges are influence diagnostics, not confidence intervals',
                       'second-stream comparison covers the two completed four-scene endpoints only',
                       'two streams do not bound finite-draw bias; scene bootstrap omits training and simulation-family uncertainty']})


if __name__ == '__main__':
    main()
