"""Inspect retained shape jumps, detection centroids and fit flags; no refitting."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_smooth_classifier_joint import CUTS, measured_leg, output_cuts
from scripts.evaluate_constgold_fixed_g0_response import raw_root
from scripts.train_full_parent_classifier import sha256, write_json


def describe(mask, response, p, m, flags):
    maximum_shape = np.maximum(np.linalg.norm(p[:, :2], axis=1), np.linalg.norm(m[:, :2], axis=1))
    centroid_shift = np.linalg.norm(p[:, 4:6]-m[:, 4:6], axis=1)
    quantiles = [.0, .1, .5, .9, 1.]
    return {'count': int(mask.sum()), 'contribution_to_full_mean': response[mask].sum()/len(response),
        'max_endpoint_shape_quantiles': np.quantile(maximum_shape[mask], quantiles),
        'centroid_shift_pixel_quantiles': np.quantile(centroid_shift[mask], quantiles),
        'mean_leg_magnitude_quantiles': np.quantile((p[mask, 2]+m[mask, 2])/2, quantiles),
        'mean_leg_radius_pixel_quantiles': np.quantile((p[mask, 3]+m[mask, 3])/2, quantiles),
        'opposing_shape_vectors_fraction': np.mean(np.sum(p[mask, :2]*m[mask, :2], axis=1) < 0),
        'endpoint_shape_above_0p9_fraction': np.mean(maximum_shape[mask] > .9),
        'centroid_shift_above_one_pixel_fraction': np.mean(centroid_shift[mask] > 1),
        'any_nonzero_fit_flags': int(np.any(flags[mask] != 0, axis=1).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Slurm and new output required')
    audit = json.loads(args.audit.read_text())
    manifest_path = args.prepared_root/'parent/manifest.json'
    if sha256(manifest_path) != audit['sources'][str(manifest_path)]:
        raise ValueError('parent manifest differs from contribution audit')
    manifest = json.loads(manifest_path.read_text())
    sources = {str(args.audit): sha256(args.audit), str(manifest_path): sha256(manifest_path)}
    cases = {}
    for case_name in audit['cases']:
        case = int(case_name)
        parent_path = args.prepared_root/'parent'/f'case{case}.npz'
        if sha256(parent_path) != audit['sources'][str(parent_path)]:
            raise ValueError('parent changed')
        sources[str(parent_path)] = sha256(parent_path)
        with np.load(parent_path, allow_pickle=False) as data:
            ids = data['input_index']
        receipt_path = args.image_root/f'case{case}/measurement_manifest.json'
        if sha256(receipt_path) != manifest['case_reports'][case_name]['measurement_manifest_sha256']:
            raise ValueError('measurement receipt changed')
        receipt = json.loads(receipt_path.read_text())
        status_path = receipt_path.parent/'fit_status.npz'
        sources.update({str(p): sha256(p) for p in (receipt_path, status_path)})
        with np.load(status_path, allow_pickle=False) as status:
            if len(np.unique(status['ids'])) != len(status['ids']):
                raise ValueError('ambiguous fit identity')
            index = pd.Index(status['ids']).get_indexer(ids)
            if (index < 0).any():
                raise ValueError('missing fit identities')
            shapes, fit_flags = status['shapes'][index], status['flags'][index]
            for i, leg in enumerate(receipt['legs']):
                if int((status['flags'][:, i] > 0).sum()) != leg['nonzero_fit_flags']:
                    raise ValueError('fit-status flags disagree with completion receipt')
        legs = {}
        for axis in (1, 2):
            root = args.prepared_root/'catalogues'/f'g{axis}_h0.02'
            for sign in (1, -1):
                catalogue = raw_root(root, case, sign*.02)
                cross_path = (catalogue/'CrossMatch/tile180.0_-0.5_rot0_matched.feather').resolve()
                shape_path = (catalogue/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather').resolve()
                for path in (cross_path, shape_path):
                    if sha256(path) != audit['sources'][str(path)]:
                        raise ValueError('source differs from contribution audit')
                    sources[str(path)] = sha256(path)
                cross = pd.read_feather(cross_path, columns=['id_input', 'id_detec'])
                shape = pd.read_feather(shape_path, columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS', 'X_IMAGE', 'Y_IMAGE'])
                aligned = cross.merge(shape, left_on='id_detec', right_on='NUMBER', validate='one_to_one').set_index('id_input').reindex(ids)
                values = aligned[['NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS', 'X_IMAGE', 'Y_IMAGE']].to_numpy(float)
                slot = next(i for i, v in enumerate(receipt['legs']) if v['group'] == f'g{axis}_h0.02' and v['h'] == sign*.02)
                np.testing.assert_allclose(values[:, :2], shapes[:, slot], rtol=0, atol=0, equal_nan=True)
                measured, usable, _ = measured_leg(root, case, sign*.02, ids)
                legs[(axis, sign)] = values, usable, output_cuts(measured), fit_flags[:, slot]
        summaries = {}
        for k, cut in enumerate(CUTS):
            mask = np.logical_and.reduce([v[1] & v[2][:, k] for v in legs.values()])
            for axis in (1, 2):
                p, m = legs[(axis, 1)][0][mask], legs[(axis, -1)][0][mask]
                flags = np.stack((legs[(axis, 1)][3][mask], legs[(axis, -1)][3][mask]), axis=1)
                selected_ids = ids[mask]
                response = (p[:, 2-axis]-m[:, 2-axis])/.04
                expected = audit['cases'][case_name]['summary'][f'h02_four_legs/{cut}/g{axis}/h0.02']
                np.testing.assert_allclose(response.mean(), expected['mean'], atol=1e-12)
                order = np.argsort(np.abs(response), kind='stable')[::-1]
                np.testing.assert_array_equal(selected_ids[order[:20]], [v['id'] for v in expected['largest_absolute_objects']])
                tail = np.zeros(len(response), bool)
                tail[order[:int(np.ceil(.01*len(response)))]] = True
                summaries[f'{cut}/g{axis}'] = {
                    'all': describe(np.ones(len(response), bool), response, p, m, flags),
                    'largest_one_percent': describe(tail, response, p, m, flags),
                    'remaining_ninety_nine_percent': describe(~tail, response, p, m, flags),
                    'largest_twenty': [{'id': int(selected_ids[i]), 'cross_response': response[i],
                        'plus_shape_mag_radius_xy': p[i], 'minus_shape_mag_radius_xy': m[i],
                        'fit_flags': flags[i]} for i in order[:20]]}
        cases[case_name] = summaries
        print('CROSS_TAIL_OBJECTS_COMPLETE', case, flush=True)
    write_json(args.output, {'cases': cases, 'sources': sources, 'script_sha256': sha256(Path(__file__)),
        'quantile_levels': [0, .1, .5, .9, 1], 'calibration_claim': False,
        'limitations': ['all measurements retained; ranked groups describe contributions, not exclusion rules',
                       'four-leg common-selected samples are conditional mechanism diagnostics',
                       'large fitted-shape changes and zero fit flags do not prove optimizer failure',
                       'no stamps inspected, images changed, or objects remeasured']})


if __name__ == '__main__':
    main()
