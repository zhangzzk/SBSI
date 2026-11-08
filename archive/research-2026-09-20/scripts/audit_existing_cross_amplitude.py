"""Measured response at both existing amplitudes; no model fitting or new images."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from sbsi.shear_map import shear_jacobian_at_zero
from scripts.evaluate_smooth_classifier_joint import CUTS, measured_leg, output_cuts
from scripts.evaluate_constgold_fixed_g0_response import raw_root
from scripts.summarize_classifier_joint_stability import read_verified
from scripts.train_full_parent_classifier import sha256, write_json


def finite_response(plus, minus, plus_mask, minus_mask, h):
    if not plus_mask.any() or not minus_mask.any():
        raise ValueError('empty diagnostic sample')
    return (plus[plus_mask, :2].mean(0)-minus[minus_mask, :2].mean(0))/(2*h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm and new output required')
    parent_root = args.prepared_root/'parent'
    manifest_path = parent_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = {str(manifest_path): sha256(manifest_path)}
    results = {}
    for axis in (1, 2):
        path = args.result_root/f'smooth_joint_fresh_g{axis}.json'
        receipt_path = args.result_root/f'smooth_joint_fresh_g{axis}_comparison.json'
        result, receipt = read_verified(path, receipt_path)
        if (receipt['maximum_reference_sufficient_difference'] != 0
                or result['inputs']['parent_manifest_sha256'] != sha256(manifest_path)):
            raise ValueError('verified parent and exact replay required')
        results[axis] = result
        sources.update({str(p): sha256(p) for p in (path, receipt_path)})
    prior_moments = {}
    for case in results[1]['cases']:
        path = parent_root/f'case{case}.npz'
        expected = manifest['case_reports'][str(case)]['sha256']
        if sha256(path) != expected:
            raise ValueError('parent changed')
        sources[str(path)] = expected
        with np.load(path, allow_pickle=False) as stored:
            context = stored['context'].astype(float)
            prior_moments[str(case)] = {'parent_rows': len(context),
                'intrinsic_shape_mean': context[:, :2].mean(0),
                'bare_intrinsic_mean_jacobian': shear_jacobian_at_zero(context[:, 0], context[:, 1]).mean(0)}
            if case == 20010:
                ids = stored['input_index'].copy()
                intrinsic = context.copy()
    case = 20010  # Only existing full scene with both amplitudes on both axes.
    record = manifest['case_reports'][str(case)]
    legs = {}
    for axis in (1, 2):
        for h in (.02, .04):
            root = args.prepared_root/'catalogues'/f'g{axis}_h{h}'
            for sign in (1, -1):
                directory = raw_root(root, case, sign*h)
                for suffix in ('Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather',
                               'CrossMatch/tile180.0_-0.5_rot0_matched.feather'):
                    path = (directory/suffix).resolve()
                    expected = record['source_sha256'][str(path)]
                    if sha256(path) != expected:
                        raise ValueError('measurement source changed')
                    sources[str(path)] = expected
                values, usable, report = measured_leg(root, case, sign*h, ids)
                legs[(axis, h, sign)] = (values, usable, output_cuts(values), report)
    all_usable = np.logical_and.reduce([v[1] for v in legs.values()])
    common_selected = [np.logical_and.reduce([v[1] & v[2][:, k] for v in legs.values()])
                       for k in range(len(CUTS))]
    summaries = {}
    for axis in (1, 2):
        for h in (.02, .04):
            plus, minus = legs[(axis, h, 1)], legs[(axis, h, -1)]
            matched = plus[1] & minus[1]
            for k, cut in enumerate(CUTS):
                masks = {
                    'per_leg': (plus[1] & plus[2][:, k], minus[1] & minus[2][:, k]),
                    'matched_usable': (matched & plus[2][:, k], matched & minus[2][:, k]),
                    'all_eight_usable': (all_usable & plus[2][:, k], all_usable & minus[2][:, k]),
                    'all_eight_selected': (common_selected[k], common_selected[k])}
                entry = {}
                for name, (pm, mm) in masks.items():
                    response = finite_response(plus[0], minus[0], pm, mm, h)
                    entry[name] = {'measured_response': response,
                                   'plus_count': int(pm.sum()), 'minus_count': int(mm.sum())}
                    if h == .02 and name in ('per_leg', 'matched_usable'):
                        branch = 'actual_usable_flags' if name == 'per_leg' else 'matched_usable'
                        stats = results[axis]['per_case_sufficient'][f'truth_parent/{cut}/{branch}']['measured']
                        ci = results[axis]['cases'].index(case)
                        for leg, values, mask in (('plus', plus[0], pm), ('minus', minus[0], mm)):
                            np.testing.assert_array_equal(stats[leg]['denominator'][ci], [int(mask.sum())])
                            np.testing.assert_allclose(stats[leg]['numerator'][ci], values[mask, :2].sum(0)[None, :], atol=1e-11, rtol=1e-12)
                summaries[f'g{axis}/h{h}/{cut}'] = entry
    write_json(args.output, {'case': case, 'sources': sources, 'summary': summaries,
        'parent_moments': prior_moments,
        'common_selected_bare_intrinsic_jacobian': {cut: shear_jacobian_at_zero(
            intrinsic[mask, 0], intrinsic[mask, 1]).mean(0) for cut, mask in zip(CUTS, common_selected)},
        'leg_reports': {str(key): v[3] for key, v in legs.items()},
        'script_sha256': sha256(Path(__file__)), 'measured_helper_sha256': sha256(Path('scripts/evaluate_smooth_classifier_joint.py')),
        'calibration_claim': False,
        'limitations': ['one scene; no scene uncertainty or amplitude chosen by its result',
                       'all-eight masks condition on future measurements and are mechanism diagnostics only',
                       'bare intrinsic Jacobian is not measured responsivity and supplies no correction',
                       'measurement-only diagnostic; no h0.04 model calibration or training claim']})
    print('EXISTING_CROSS_AMPLITUDE_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
