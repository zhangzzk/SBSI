"""Describe object and fixed-grid contributions without excluding measurements."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_smooth_classifier_joint import CUTS, measured_leg, output_cuts
from scripts.evaluate_constgold_fixed_g0_response import raw_root
from scripts.summarize_classifier_joint_stability import read_verified
from scripts.train_full_parent_classifier import sha256, write_json


def contributions(values, ids, blocks):
    values = np.asarray(values, float)
    count = np.bincount(blocks, minlength=64)
    sums = np.bincount(blocks, weights=values, minlength=64)
    if np.any(count == 0) or not np.isfinite(values).all():
        raise ValueError('finite values and 64 occupied spatial blocks required')
    mean = values.mean()
    omitted = (values.sum()-sums)/(len(values)-count)
    order = np.argsort(np.abs(values), kind='stable')[::-1]
    return {'mean': mean, 'objects': len(values), 'block_counts': count,
        'block_means': sums/count, 'leave_one_block_out': omitted,
        'block_jackknife_se': np.sqrt(63/64*np.square(omitted-omitted.mean()).sum()),
        'leave_one_block_out_min_max': [omitted.min(), omitted.max()],
        'object_quantile_levels': [0, .001, .01, .1, .5, .9, .99, .999, 1],
        'object_quantiles': np.quantile(values, [0, .001, .01, .1, .5, .9, .99, .999, 1]),
        'positive_contribution': values[values > 0].sum()/len(values),
        'negative_contribution': values[values < 0].sum()/len(values),
        'absolute_ranked_tail_contributions': {str(f): {
            'count': max(1, int(np.ceil(f*len(values)))),
            'contribution_to_full_mean': values[order[:max(1, int(np.ceil(f*len(values))))]].sum()/len(values)}
            for f in (.0001, .001, .01)},
        'largest_absolute_objects': [{'id': int(ids[i]), 'response': values[i], 'block': int(blocks[i])}
                                     for i in order[:20]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm and new output required')
    manifest_path = args.prepared_root/'parent/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = {str(manifest_path): sha256(manifest_path)}
    results = {}
    for axis in (1, 2):
        path = args.result_root/f'smooth_joint_fresh_g{axis}.json'
        receipt = args.result_root/f'smooth_joint_fresh_g{axis}_comparison.json'
        results[axis], checked = read_verified(path, receipt)
        if (checked['maximum_reference_sufficient_difference'] != 0
                or results[axis]['inputs']['parent_manifest_sha256'] != sha256(manifest_path)):
            raise ValueError('verified original results required')
        sources.update({str(p): sha256(p) for p in (path, receipt)})
    cases = {}
    for case in results[1]['cases']:
        record = manifest['case_reports'][str(case)]
        path = args.prepared_root/'parent'/f'case{case}.npz'
        if sha256(path) != record['sha256']:
            raise ValueError('parent changed')
        sources[str(path)] = record['sha256']
        with np.load(path, allow_pickle=False) as data:
            ids = data['input_index']
        truth_paths = [Path(p) for p in record['source_sha256']
                       if '/g0/' in p and p.endswith('/input/gals_info_tile180.0_-0.5.feather')]
        if len(truth_paths) != 1:
            raise ValueError('unique g0 truth required')
        path = truth_paths[0]
        if sha256(path) != record['source_sha256'][str(path)]:
            raise ValueError('position source changed')
        sources[str(path)] = record['source_sha256'][str(path)]
        truth = pd.read_feather(path, columns=['index_input', 'RA_input', 'DEC_input'])
        coordinates = truth.set_index('index_input').loc[ids, ['RA_input', 'DEC_input']].to_numpy(float)
        bounds = np.stack((truth[['RA_input', 'DEC_input']].min().to_numpy(),
                           truth[['RA_input', 'DEC_input']].max().to_numpy()))
        grid = np.minimum(7, np.floor(8*(coordinates-bounds[0])/(bounds[1]-bounds[0])).astype(int))
        if not np.isfinite(coordinates).all() or (grid < 0).any():
            raise ValueError('invalid scene coordinates')
        blocks = grid[:, 0]+8*grid[:, 1]
        legs = {}
        for axis in (1, 2):
            for h in ((.02, .04) if case == 20010 else (.02,)):
                root = args.prepared_root/'catalogues'/f'g{axis}_h{h}'
                for sign in (1, -1):
                    for suffix in ('Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather',
                                   'CrossMatch/tile180.0_-0.5_rot0_matched.feather'):
                        path = (raw_root(root, case, sign*h)/suffix).resolve()
                        expected = record['source_sha256'][str(path)]
                        if sha256(path) != expected:
                            raise ValueError('measurement changed')
                        sources[str(path)] = expected
                    value, usable, _ = measured_leg(root, case, sign*h, ids)
                    legs[(axis, h, sign)] = value, usable, output_cuts(value)
                    if h == .02:
                        ci = results[axis]['cases'].index(case)
                        direction = 'plus' if sign == 1 else 'minus'
                        for k, cut in enumerate(CUTS):
                            mask = usable & legs[(axis, h, sign)][2][:, k]
                            expected = results[axis]['per_case_sufficient'][f'truth_parent/{cut}/actual_usable_flags']['measured'][direction]
                            np.testing.assert_array_equal(expected['denominator'][ci], [mask.sum()])
                            np.testing.assert_allclose(expected['numerator'][ci], value[mask, :2].sum(0)[None], atol=1e-11)
        summaries = {}
        for scope in (('h02_four_legs', 'both_amplitudes_eight_legs') if case == 20010 else ('h02_four_legs',)):
            retained = {key: v for key, v in legs.items() if scope != 'h02_four_legs' or key[1] == .02}
            for k, cut in enumerate(CUTS):
                mask = np.logical_and.reduce([v[1] & v[2][:, k] for v in retained.values()])
                for axis in (1, 2):
                    cross = 2-axis
                    diffs = {}
                    for h in sorted({key[1] for key in retained}):
                        diffs[h] = (retained[(axis, h, 1)][0][mask, cross]-retained[(axis, h, -1)][0][mask, cross])/(2*h)
                        summaries[f'{scope}/{cut}/g{axis}/h{h}'] = contributions(diffs[h], ids[mask], blocks[mask])
                    if len(diffs) == 2:
                        summaries[f'{scope}/{cut}/g{axis}/h04_minus_h02'] = contributions(diffs[.04]-diffs[.02], ids[mask], blocks[mask])
        cases[str(case)] = {'grid_bounds_RA_DEC': bounds, 'summary': summaries}
        print('CROSS_CONTRIBUTIONS_COMPLETE', case, flush=True)
    write_json(args.output, {'cases': cases, 'sources': sources, 'script_sha256': sha256(Path(__file__)),
        'calibration_claim': False, 'grid': '8x8 equal-width cells over full rendered g0 truth bounds',
        'limitations': ['fixed masks condition on all declared legs; diagnostic only, not acceptance population',
                       'block jackknife is within-scene descriptive uncertainty and may miss dependence beyond cells',
                       'ranked tails are contributions to the full retained mean; no rows excluded or winsorized',
                       'all original cuts/axes/scenes reported; single-scene amplitude comparison only']})


if __name__ == '__main__':
    main()
