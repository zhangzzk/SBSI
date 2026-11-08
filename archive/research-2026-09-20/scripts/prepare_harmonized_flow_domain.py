"""Prepare a full-parent flow domain from harmonized measurement catalogues.

Construct contexts from raw truth and the existing truth-only crowd features.
Determine validity independently in each new measurement leg, preserving
newly successful fits outside the old valid-row cache. Keep the original
full-data train/validation case split.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS, FlowRows
from scripts.prepare_full_domain_flow import save_rows
from scripts.prepare_full_parent_classifier import base_context
from scripts.train_fixed_g0_flow import sha256, write_json


def independent_measurement_rows(case, ids, context, gamma, values):
    """Retain each leg's valid physical targets without any old-row anchor."""
    ids, context, gamma, values = map(np.asarray, (ids, context, gamma, values))
    n = len(ids)
    if (ids.shape != (n,) or len(np.unique(ids)) != n or context.shape != (n, 8)
            or gamma.shape != (n, 2) or values.shape != (n, 4)):
        raise ValueError('unique aligned parent, contexts, shear and measurements required')
    if not np.isfinite(context).all() or not np.isfinite(gamma).all():
        raise ValueError('nonfinite truth conditions')
    usable = np.isfinite(values).all(1) & (np.square(values[:, :2]).sum(1) < 1)
    usable &= (values[:, 2:] > 0).all(1)
    return FlowRows(np.full(int(usable.sum()), case, dtype=np.int32), ids[usable].astype(np.int64),
                    context[usable].astype(np.float32), values[usable].astype(np.float64),
                    gamma[usable].astype(np.float64), usable)


def generated_secondary_parent(generated, rendered_ids, measured_ids):
    """Use generator identities for role/cut membership, then join rendered rows.

Rendering may omit objects, so half the rendered row count is not the role
boundary. Magnitude membership must likewise use the same generated truth
that selected the measurement job, rather than a serialized rendered copy.
"""
    source_ids = generated['index'].to_numpy(np.int64)
    if not np.array_equal(np.sort(source_ids), np.arange(len(generated))):
        raise ValueError('generated truth must have unique contiguous identities')
    mask = (source_ids >= len(generated)//2) & (generated['r'].to_numpy(float) < 26.)
    parent = np.sort(source_ids[mask])
    np.testing.assert_array_equal(parent, measured_ids)
    rendered = pd.Index(rendered_ids)
    if not rendered.is_unique:
        raise ValueError('duplicate rendered truth identities')
    position = rendered.get_indexer(parent)
    present = position >= 0
    return parent[present], position[present], {
        'generated_rows': len(generated), 'generated_secondary_boundary': len(generated)//2,
        'generated_truth_parent_rows': len(parent), 'rendered_truth_parent_rows': int(present.sum()),
        'unrendered_truth_parent_dropped': int((~present).sum()),
        'unrendered_truth_parent_ids': parent[~present].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--crowd-catalogue', type=Path, required=True)
    parser.add_argument('--baseline-domain', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full-parent domain preparation requires Slurm')
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if len(set(args.cases)) != len(args.cases):
        raise ValueError('duplicate cases')
    baseline = json.loads((args.baseline_domain/'manifest.json').read_text())
    if baseline['flow']['features'] != list(FLOW_FEATURES) or baseline['flow']['targets'] != list(FLOW_TARGETS):
        raise ValueError('baseline feature/target contract differs')
    split = {key: [c for c in baseline['split'][key] if c in args.cases]
             for key in ('train_cases', 'validation_cases')}
    if (not all(split.values()) or sorted(split['train_cases']+split['validation_cases']) != sorted(args.cases)):
        raise ValueError('requested cases must retain nonempty disjoint original train/validation groups')
    args.output_root.mkdir(parents=True)
    partial, records = {}, {}
    for case in sorted(args.cases):
        receipt_path = args.measurement_root/f'case{case:03d}_manifest.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt['case'] != case or receipt['role'] != 'secondary' or receipt['centering'] != 'subpixel'
                or receipt['amplitudes'] != [0., .05]):
            raise ValueError('unexpected measurement role, centering or amplitudes')
        with np.load(args.measurement_root/f'case{case:03d}_fit_status.npz') as fitted:
            measured_ids = fitted['ids']
        truths = [pd.read_feather(args.raw_root/f'case{case}_{h}/real0/catalogues/input/gals_info_tile180.0_-0.5.feather')
                  for h in (0., .05)]
        invariant = [c for c in truths[0] if c not in ('gamma1_input', 'gamma2_input')]
        pd.testing.assert_frame_equal(truths[0][invariant], truths[1][invariant], check_exact=True)
        if truths[0]['index_input'].duplicated().any():
            raise ValueError('duplicate truth identities')
        generated_path = args.raw_root/f'gals{case}_0.2.feather'
        if sha256(generated_path) != receipt['truth_sha256']:
            raise ValueError('generated truth differs from measurement parent provenance')
        generated = pd.read_feather(generated_path, columns=['index', 'r'])
        ids, index, parent_report = generated_secondary_parent(generated, truths[0]['index_input'], measured_ids)
        gammas = [truth.iloc[index][['gamma1_input', 'gamma2_input']].to_numpy(float) for truth in truths]
        np.testing.assert_array_equal(gammas[0], 0.)
        np.testing.assert_allclose(np.linalg.norm(gammas[1], axis=1), .05, atol=1e-12)
        partial[case] = {'ids': ids, 'base': [base_context(truth, index) for truth in truths],
                         'gamma': gammas, 'crowd': [], 'receipt': receipt}
        records[str(case)] = {'receipt_sha256': sha256(receipt_path),
                              'measurement_script_sha256': receipt['script_sha256'],
                              'shape_module_sha256': receipt['shape_module_sha256'],
                              'seed_formula': receipt['seed_formula'], 'parent_rows': len(ids),
                              'parent_alignment': parent_report, 'generated_truth_sha256': receipt['truth_sha256'],
                              'truth_sha256': [sha256(args.raw_root/f'case{case}_{h}/real0/catalogues/input/gals_info_tile180.0_-0.5.feather')
                                               for h in (0., .05)]}
    for field in ('measurement_script_sha256', 'shape_module_sha256', 'seed_formula'):
        if len({row[field] for row in records.values()}) != 1:
            raise ValueError(f'inconsistent measurement provenance: {field}')
    columns = ['case', 'input_index', 'nbr_flux_near', 'nbr_flux_far', 'nbr_flux_max']
    with ipc.open_file(args.crowd_catalogue) as reader:
        for bi in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(bi)]).select(columns).to_pandas()
            for case, group in frame.loc[frame['case'].isin(args.cases)].groupby('case', sort=False):
                keep = group['input_index'].isin(partial[int(case)]['ids'])
                partial[int(case)]['crowd'].append(group.loc[keep].drop(columns='case'))
            if bi == 0 or (bi+1) % 200 == 0:
                print(f'HARMONIZED_CROWD batches={bi+1}/{reader.num_record_batches}', flush=True)
    stores = {'g0': {}, 'g05': {}}
    counts = {leg: {'truth_parent_rows': 0, 'usable_rows': 0, 'invalid_rows_after_truth_cut': 0} for leg in stores}
    for case in sorted(args.cases):
        data = partial.pop(case)
        ids = data['ids']
        crowd = pd.concat(data['crowd'], ignore_index=True).set_index('input_index')
        if crowd.index.duplicated().any():
            raise ValueError('duplicate crowd identities')
        crowd = crowd.reindex(ids)[columns[2:]].to_numpy(np.float32)
        if not np.isfinite(crowd).all():
            raise ValueError('missing/nonfinite full-parent crowd features')
        leg_reports = {}
        for hi, leg in enumerate(('g0', 'g05')):
            receipt = data['receipt']['case_reports'][hi]
            shape_path = Path(receipt['output'])
            if sha256(shape_path) != receipt['output_sha256'] or sha256(receipt['source_shapes']) != receipt['source_shapes_sha256']:
                raise ValueError('changed measurement source')
            cross_path = args.raw_root/f'case{case}_{receipt["h"]}/real0/catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather'
            if sha256(cross_path) != receipt['cross_sha256']:
                raise ValueError('changed crossmatch')
            cross = pd.read_feather(cross_path, columns=['id_input', 'id_detec'])
            measured = pd.read_feather(shape_path, columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO'])
            if (cross['id_input'].duplicated().any() or cross['id_detec'].duplicated().any()
                    or measured['NUMBER'].duplicated().any()):
                raise ValueError('nonunique detection/shape identities')
            joined = cross.merge(measured, left_on='id_detec', right_on='NUMBER', validate='one_to_one').set_index('id_input')
            aligned = joined.reindex(ids)
            values = aligned[['NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO']].to_numpy(float)
            with np.errstate(over='ignore', invalid='ignore'):
                values[:, 3] = 10**(-.4*(values[:, 3]-30))
            context = np.column_stack((data['base'][hi], crowd))
            rows = independent_measurement_rows(case, ids, context, data['gamma'][hi], values)
            stores[leg][case] = [rows]
            old_path = args.baseline_domain/f'flow/{leg}/case{case:03d}.npz'
            if sha256(old_path) != baseline['flow'][f'{leg}_cases'][str(case)]['sha256']:
                raise ValueError('changed baseline domain case')
            with np.load(old_path) as old:
                shared, ni, oi = np.intersect1d(rows.input_index, old['input_index'], return_indices=True)
                np.testing.assert_allclose(rows.context[ni], old['context'][oi], atol=2e-6, rtol=2e-6)
                np.testing.assert_array_equal(rows.target[ni, 2:], old['target'][oi, 2:])
                old_count = len(old['input_index'])
            parent_out = args.output_root/'usability'/leg
            parent_out.mkdir(parents=True, exist_ok=True)
            np.savez(parent_out/f'case{case:03d}.npz', input_index=ids, context=context,
                     gamma=data['gamma'][hi], usable=rows.usable_mask,
                     detected=np.isin(ids, joined.index))
            leg_reports[leg] = {'old_rows': old_count, 'new_rows': len(rows.input_index), 'shared_rows': len(shared),
                                'old_only_dropped': old_count-len(shared), 'new_only_retained': len(rows.input_index)-len(shared),
                                'truth_without_detection_dropped': int((~np.isin(ids, joined.index)).sum()),
                                'cross_without_shape': len(cross)-len(joined),
                                'baseline_domain_sha256': sha256(old_path)}
            counts[leg]['truth_parent_rows'] += len(ids)
            counts[leg]['usable_rows'] += len(rows.input_index)
            counts[leg]['invalid_rows_after_truth_cut'] += len(ids)-len(rows.input_index)
        records[str(case)]['legs'] = leg_reports
        print(f'HARMONIZED_CASE case={case} counts={leg_reports}', flush=True)
    outputs = {leg: save_rows(args.output_root, leg, store) for leg, store in stores.items()}
    write_json(args.output_root/'manifest.json', {
        'format_version': 1,
        'population': {'name': 'full_valid_measurement_domain_true_mag_lt26_subpixel_common_rng_v1',
                       'anchor_leg': None, 'predicate': None, 'measured_selection_cut': None,
                       'truth_analysis_cut': {'column': 'r_input_p', 'operator': '<', 'value': 26.},
                       'independent_per_leg_validity': True, 'sheared_leg_recut': None,
                       'inherited_parent_support_only': False},
        'flow': {'features': list(FLOW_FEATURES), 'targets': list(FLOW_TARGETS),
                 'g0_cases': outputs['g0'], 'g05_cases': outputs['g05'],
                 'per_leg_validity': 'detected; finite truth/targets; open shape disk; positive radius and flux'},
        'split': baseline['split'] | split, 'counts': counts, 'case_records': records,
        'measurement': {'centering': 'subpixel', 'stamp_size': 48, 'pixel_scale': .2,
                        'seed_formula': next(iter(records.values()))['seed_formula']},
        'sources': {'measurement_root': str(args.measurement_root), 'raw_root': str(args.raw_root),
                    'crowd_sha256': sha256(args.crowd_catalogue),
                    'baseline_manifest_sha256': sha256(args.baseline_domain/'manifest.json'),
                    'script_sha256': sha256(__file__)},
        'limitations': ['reused development scenes; no calibration acceptance',
                        'unrendered generated-parent identities are dropped and explicitly reported before context construction',
                        'usability records cover the declared true-r<26 parent only']})
    print(f'HARMONIZED_DOMAIN_COMPLETE {args.output_root}', flush=True)


if __name__ == '__main__':
    main()
