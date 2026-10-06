"""Validate common-seed pilot measurements and compare paired label variance."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_response_label_amplitude import measured_shapes
from scripts.train_fixed_g0_flow import sha256, write_json


def paired_marginal_summary(original, candidate, selected):
    """Compare per-leg marginals without conditioning on common fit success.

The paired influence error describes object sampling in these fixed scenes;
it is not independent-scene or model-training uncertainty. ``selected`` also
includes the shared finite positive photometry requirement.
    """
    original, candidate = np.asarray(original), np.asarray(candidate)
    selected = np.asarray(selected, dtype=bool)
    n = len(original)
    if original.shape != (n, 2) or candidate.shape != (n, 2) or selected.shape != (n,) or n < 2:
        raise ValueError('aligned full-parent shapes and selection required')
    means, influences, valid = [], [], []
    for values in (original, candidate):
        good = selected & np.isfinite(values).all(1) & (np.square(values).sum(1) < 1)
        if not good.any():
            raise ValueError('empty selected usable marginal')
        e = np.where(good[:, None], values, 0.)
        features = np.column_stack((e, e[:, 0]**2, e[:, 1]**2, e[:, 0]*e[:, 1]))
        mean = features.sum(0)/good.sum()
        influences.append(good[:, None]*(features-mean)/good.sum())
        means.append(mean)
        valid.append(good)
    difference = influences[1]-influences[0]
    mass_difference = valid[1].astype(float)-valid[0]
    return {'full_parent_rows': n, 'original_selected_usable': int(valid[0].sum()),
            'candidate_selected_usable': int(valid[1].sum()),
            'original_only': int((valid[0] & ~valid[1]).sum()),
            'candidate_only': int((valid[1] & ~valid[0]).sum()),
            'selected_mass_difference': float(mass_difference.mean()),
            'selected_mass_paired_object_se': float(mass_difference.std(ddof=1)/np.sqrt(n)),
            'moment_names': ['e1', 'e2', 'e1_squared', 'e2_squared', 'e1_e2'],
            'original_moments': means[0], 'candidate_moments': means[1],
            'moment_difference': means[1]-means[0],
            'moment_difference_paired_object_se': np.sqrt(n/(n-1)*np.square(difference).sum(0))}


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('pilot catalogue audit requires Slurm')
    base = Path('/project/ls-gruen/users/zekang.zhang')
    source = base/'lsst_sims_fs2_25876'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, default=base/'blendemu_runs/common_optimizer_primary_pilot_20260917_v1')
    parser.add_argument('--cases', type=int, nargs='+', default=[0, 1, 40, 41])
    parser.add_argument('--reference-pilot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        raise ValueError('duplicate pilot cases')
    pilot = args.pilot_root
    prepared = base/'blendemu_runs/output_conditioned_response_20260917_v1/prepared'
    cache = base/'sbsi_caches/output_conditioned_response_20260917_v1'
    manifest = json.loads((prepared/'manifest.json').read_text())
    reports = []
    for case in args.cases:
        record = json.loads((pilot/f'case{case:03d}_manifest.json').read_text())
        for leg in record['case_reports']:
            if sha256(leg['output']) != leg['output_sha256'] or sha256(leg['source_shapes']) != leg['source_shapes_sha256']:
                raise ValueError('changed original/pilot measurement catalogue')
        replay = None
        if case in (0, 1) and record.get('centering', 'geometric') == 'geometric':
            with np.load(cache/f'measurement_seed_case{case}.npz') as probe, np.load(pilot/f'case{case:03d}_fit_status.npz') as full:
                index = pd.Index(full['ids']).get_indexer(probe['ids'])
                if (index < 0).any():
                    raise ValueError('pilot misses probe identities')
                values, expected = full['shapes'][index], probe['measured'][:, :, 0]
                np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
                replay = {'rows': len(index), 'maximum_difference': float(np.max(np.abs(values-expected)))}
        legs = {arm: {h: measured_shapes(root, case, h)[0] for h in (0., .02, .2)}
                for arm, root in (('original', source), ('common_optimizer', pilot))}
        reference_legs, centering_comparison = None, {}
        if args.reference_pilot is not None:
            reference_record = json.loads((args.reference_pilot/f'case{case:03d}_manifest.json').read_text())
            if reference_record['seed_formula'] != record['seed_formula'] or reference_record['truth_sha256'] != record['truth_sha256']:
                raise ValueError('centering comparison requires identical truth and optimizer streams')
            for old, new in zip(reference_record['case_reports'], record['case_reports']):
                for key in ('h', 'source_shapes_sha256', 'cross_sha256', 'image', 'image_bytes', 'image_mtime_ns', 'psf_sha256'):
                    if old[key] != new[key]:
                        raise ValueError(f'centering comparison source mismatch: {key}')
                if sha256(old['output']) != old['output_sha256']:
                    raise ValueError('reference pilot measurement changed')
            reference_legs = {h: measured_shapes(args.reference_pilot, case, h)[0] for h in (0., .02, .2)}
        marginals = {}
        with np.load(pilot/f'case{case:03d}_fit_status.npz') as fitted:
            parent_ids = fitted['ids']
        for leg in record['case_reports']:
            h = leg['h']
            columns = ['NUMBER', 'FLUX_RADIUS', 'MAG_AUTO']
            old_phot = pd.read_feather(leg['source_shapes'], columns=columns)
            new_phot = pd.read_feather(leg['output'], columns=columns)
            pd.testing.assert_frame_equal(old_phot, new_phot, check_exact=True)
            cross = pd.read_feather(source/f'case{case}_{h}/real0/catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather')
            aligned = cross.merge(old_phot, left_on='id_detec', right_on='NUMBER', validate='one_to_one').set_index('id_input').reindex(parent_ids)
            radius, magnitude = aligned['FLUX_RADIUS'].to_numpy(), aligned['MAG_AUTO'].to_numpy()
            usable_phot = np.isfinite(radius) & (radius > 0) & np.isfinite(magnitude)
            masks = (usable_phot, usable_phot & (radius > 3) & (magnitude < 25.8),
                     usable_phot & (radius >= 3.75) & (magnitude < 25.8))
            old_shapes = legs['original'][h].reindex(parent_ids).to_numpy(float)
            new_shapes = legs['common_optimizer'][h].reindex(parent_ids).to_numpy(float)
            marginals[str(h)] = [paired_marginal_summary(old_shapes, new_shapes, mask) for mask in masks]
            if reference_legs is not None:
                reference_shapes = reference_legs[h].reindex(parent_ids).to_numpy(float)
                centering_comparison[str(h)] = [paired_marginal_summary(reference_shapes, new_shapes, mask) for mask in masks]
        sums = np.zeros((3, 2, 2, 2))  # cut, arm, amplitude, scalar first/second moment
        pair_counts = np.zeros(3, dtype=np.int64)
        ids = [[] for _ in range(3)]
        dropped = total = 0
        for part in manifest['parts']:
            if part['case'] != case:
                continue
            path = prepared/part['path']
            if sha256(path) != part['sha256']:
                raise ValueError('prepared pair changed')
            with np.load(path) as data:
                primary, x = data['primary'], data['x']
                shapes = {arm: {h: leg.reindex(primary).to_numpy(float) for h, leg in values.items()}
                          for arm, values in legs.items()}
                common = np.logical_and.reduce([np.isfinite(v).all(1) for values in shapes.values() for v in values.values()])
                total += len(primary)
                dropped += int((~common).sum())
                angle = 2*np.deg2rad(data['shear_angle'].astype(float))
                u = np.column_stack((np.cos(angle), np.sin(angle)))
                cuts = (common, common & (x[:, 7] > np.log(3)) & (x[:, 8] < 25.8),
                        common & (x[:, 7] >= np.log(3.75)) & (x[:, 8] < 25.8))
                for ki, keep in enumerate(cuts):
                    pair_counts[ki] += keep.sum()
                    ids[ki].append(primary[keep])
                    for ai, arm in enumerate(('original', 'common_optimizer')):
                        for hi, h in enumerate((.02, .2)):
                            label = np.sum((shapes[arm][h][keep]-shapes[arm][0.][keep])*u[keep], axis=1)/h
                            sums[ki, ai, hi] += [label.sum(), np.square(label).sum()]
        primary_counts = np.array([len(np.unique(np.concatenate(group))) for group in ids])
        means = sums[..., 0]/pair_counts[:, None, None]
        variance = sums[..., 1]/pair_counts[:, None, None]-np.square(means)
        response = sums[..., 0]/primary_counts[:, None, None]
        reports.append({'case': case, 'probe_replay': replay, 'per_leg_marginals': marginals,
                        'reference_pilot_marginal_comparison': centering_comparison,
                        'centering': record.get('centering', 'geometric'), 'pair_counts': pair_counts,
                        'represented_primaries': primary_counts, 'input_pairs': total, 'invalid_or_unmatched_dropped': dropped,
                        'scalar_response': response, 'pair_label_variance': variance,
                        'common_over_original_variance': variance[:, 1]/variance[:, 0],
                        'paired_common_point02_minus_point2_response': response[:, 1, 0]-response[:, 1, 1],
                        'sums': sums, 'pilot_manifest_sha256': sha256(pilot/f'case{case:03d}_manifest.json')})
        print(f'COMMON_OPTIMIZER_AUDIT case={case} replay={replay} '
              f'variance_ratio={variance[:, 1]/variance[:, 0]}', flush=True)
    output = args.output or cache/'common_optimizer_pilot.json'
    if output.exists():
        raise FileExistsError(output)
    write_json(output, {'cases': reports, 'cuts': ['uncut', 'radius_gt3_mag_lt25.8', 'radius_ge3.75_mag_lt25.8'],
                        'arms': ['original', 'common_optimizer'], 'amplitudes': [.02, .2],
                        'script_sha256': sha256(__file__),
                        'limitations': ['four reused pilot scenes; not calibration acceptance',
                                        'common usable pairs across original and new measurements',
                                        'per-leg marginal checks use separate usability; paired object errors condition on the four fixed scenes',
                                        'pair label variance includes physical response heterogeneity',
                                        'one optimizer realization per pilot object']})
    print(f'COMMON_OPTIMIZER_AUDIT_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
