"""Audit independent g1 scenes without opening any measured response values."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_constgold_truth_pairing import INVARIANT_COLUMNS, canonical_truth, frame_sha256, truth_path
from scripts.train_fixed_g0_flow import sha256, write_json


def generated_scene(path):
    # Positions plus identities detect copied scene realizations independently
    # of the applied shear and catalogue column order.
    frame = pd.read_feather(path, columns=['index', 'RA', 'DEC'])
    frame = frame.sort_values('index', kind='stable').reset_index(drop=True)
    if (not np.array_equal(frame['index'], np.arange(len(frame)))
            or not np.isfinite(frame.to_numpy()).all()):
        raise ValueError(f'invalid generated scene identities/positions: {path}')
    return frame_sha256(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--independent-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full-scene truth audit requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    training = {}
    for case in range(200):
        path = args.training_root/f'gals{case}_0.2.feather'
        training[str(case)] = generated_scene(path)
        if case % 50 == 0:
            print(f'INDEPENDENT_TRUTH_TRAINING_FINGERPRINT case={case}', flush=True)
    if len(set(training.values())) != len(training):
        raise ValueError('duplicate generated scenes in the training inventory')
    reports, seen = {}, set(training.values())
    for case in range(200, 240):
        generated = [args.independent_root/f'gals{case}_{h}.feather' for h in (.05, -.05)]
        fingerprints = [generated_scene(p) for p in generated]
        if fingerprints[0] != fingerprints[1] or fingerprints[0] in seen:
            raise ValueError('independent scene was copied or antithetic scene identity differs')
        seen.add(fingerprints[0])
        paths = [truth_path(args.independent_root, case, h) for h in (.05, -.05)]
        plus, minus = [canonical_truth(p) for p in paths]
        pd.testing.assert_frame_equal(plus[list(INVARIANT_COLUMNS)], minus[list(INVARIANT_COLUMNS)], check_exact=True)
        for h, frame in ((.05, plus), (-.05, minus)):
            np.testing.assert_array_equal(frame['gamma1_input'], h)
            np.testing.assert_array_equal(frame['gamma2_input'], 0.)
        source = pd.read_feather(generated[0], columns=['index', 'r'])
        parent = source.loc[(source['index'] >= len(source)//2) & (source['r'] < 26), 'index'].to_numpy(np.int64)
        present = np.isin(parent, plus['index_input'])
        reports[str(case)] = {
            'generated_scene_fingerprint': fingerprints[0], 'generated_rows': len(source),
            'rendered_rows': len(plus), 'generated_role_boundary': len(source)//2,
            'true_mag26_target_parent': len(parent), 'unrendered_parent_dropped': int((~present).sum()),
            'unrendered_parent_ids': parent[~present].tolist(),
            'invariant_truth_sha256': frame_sha256(plus[list(INVARIANT_COLUMNS)]),
            'source_sha256': {str(p): sha256(p) for p in [*generated, *paths]},
        }
        print(f'INDEPENDENT_TRUTH_CASE case={case} parent={len(parent)} dropped={(~present).sum()}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        'training_cases': list(range(200)), 'independent_cases': list(range(200, 240)),
        'training_scene_fingerprints': training, 'case_reports': reports,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'no_measured_response_values_opened': True,
        'limitations': ['scene identity independence is not measurement-version compatibility',
                        'existing g1 +/-0.05 only, not fresh multi-axis acceptance',
                        'finite source population may reuse intrinsic galaxy templates across distinct scenes'],
    })
    print(f'INDEPENDENT_TRUTH_AUDIT_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
