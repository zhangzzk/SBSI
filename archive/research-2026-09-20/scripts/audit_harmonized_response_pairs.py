"""Compare fresh small/large-amplitude labels using exact physical pair IDs.

Report each amplitude's own usable population and their common pairs. Paired
uncertainty groups all neighbours of one measured primary together. No old
response-catalogue acceptance mask enters the comparison.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_fixed_g0_flow import sha256, write_json


def primary_response_difference(primary, small, large):
    primary, small, large = np.asarray(primary), np.asarray(small), np.asarray(large)
    if primary.ndim != 1 or small.shape != primary.shape or large.shape != primary.shape:
        raise ValueError('aligned pair identities and scalar labels required')
    if not len(primary) or not np.isfinite(small).all() or not np.isfinite(large).all():
        raise ValueError('nonempty finite labels required')
    unique, index = np.unique(primary, return_inverse=True)
    a = np.bincount(index, weights=small, minlength=len(unique))
    b = np.bincount(index, weights=large, minlength=len(unique))
    difference = a-b
    return {'pairs': len(primary), 'represented_primaries': len(unique),
            'response_small': float(a.mean()), 'response_large': float(b.mean()),
            'paired_small_minus_large': float(difference.mean()),
            'paired_primary_se': float(difference.std(ddof=1)/np.sqrt(len(unique))) if len(unique) > 1 else None,
            'pair_label_variance_small': float(small.var()), 'pair_label_variance_large': float(large.var())}


def baseline_photometry(measurement_manifest):
    m = json.loads(measurement_manifest.read_text())
    leg = next(v for v in m['case_reports'] if v['h'] == 0.)
    path = Path(leg['output'])
    if sha256(path) != leg['output_sha256']:
        raise ValueError('changed zero-shear measurements')
    shape = pd.read_feather(path, columns=['NUMBER', 'FLUX_RADIUS', 'MAG_AUTO'])
    cross_path = path.parents[1]/'CrossMatch/tile180.0_-0.5_rot0_matched.feather'
    if sha256(cross_path) != leg['cross_sha256']:
        raise ValueError('changed baseline identities')
    cross = pd.read_feather(cross_path, columns=['id_input', 'id_detec'])
    return cross.merge(shape, left_on='id_detec', right_on='NUMBER', validate='one_to_one').set_index('id_input')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('response-pair scan requires Slurm')
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.cases)) != len(args.cases):
        raise ValueError('duplicate cases')
    reports = []
    for case in sorted(args.cases):
        parts, metadata = [], []
        for h in (.02, .2):
            path = args.prepared_root/f'case{case:03d}_h{h}.npz'
            m = json.loads(path.with_suffix('.json').read_text())
            if (m['format'] != 'harmonized_response_case_v1' or m['case'] != case or m['amplitude'] != h
                    or sha256(path) != m['sha256']
                    or sha256(m['measurement_manifest']) != m['measurement_manifest_sha256']):
                raise ValueError('changed or incompatible fresh label part')
            with np.load(path) as stored:
                parts.append({key: stored[key] for key in stored.files})
            metadata.append(m)
        for field in ('features', 'rescale', 'pair_definition', 'population', 'measurement_manifest_sha256', 'baseline_source'):
            if metadata[0][field] != metadata[1][field]:
                raise ValueError(f'forward amplitudes differ in {field}')
        keys = [pd.MultiIndex.from_arrays([p['primary'], p['secondary']]) for p in parts]
        if any(not key.is_unique for key in keys):
            raise ValueError('duplicate physical pair')
        other = keys[1].get_indexer(keys[0])
        a, b = np.flatnonzero(other >= 0), other[other >= 0]
        np.testing.assert_array_equal(parts[0]['x'][a], parts[1]['x'][b])
        angles = [np.deg2rad(2*p['shear_angle'][i].astype(float)) for p, i in zip(parts, (a, b))]
        np.testing.assert_allclose(np.stack((np.cos(angles[0]), np.sin(angles[0]))),
                                   np.stack((np.cos(angles[1]), np.sin(angles[1]))), rtol=0, atol=2e-6)
        photometry = baseline_photometry(Path(metadata[0]['measurement_manifest']))
        cuts = ('uncut', 'mag25.8_radius_gt0.60', 'mag25.8_radius_ge0.75')

        def cut_masks(primary):
            values = photometry.reindex(primary)
            r, mag = values['FLUX_RADIUS'].to_numpy(float), values['MAG_AUTO'].to_numpy(float)
            if not np.isfinite(r).all() or not np.isfinite(mag).all() or np.any(r <= 0):
                raise ValueError('missing or invalid baseline photometry')
            return (np.ones(len(r), dtype=bool), (r > 3.) & (mag < 25.8), (r >= 3.75) & (mag < 25.8))

        own = []
        for p in parts:
            entries = {}
            for cut, mask in zip(cuts, cut_masks(p['primary'])):
                n = len(np.unique(p['primary'][mask]))
                entries[cut] = {'pairs': int(mask.sum()), 'represented_primaries': n,
                                'summed_projection_per_represented_primary': float(p['y'][mask].astype(float).sum()/n)}
            own.append(entries)
        common = {cut: primary_response_difference(parts[0]['primary'][a][mask],
                                                   parts[0]['y'][a][mask].astype(float),
                                                   parts[1]['y'][b][mask].astype(float))
                  for cut, mask in zip(cuts, cut_masks(parts[0]['primary'][a]))}
        report = {'case': case, 'independent_amplitude_populations': own, 'common_pairs': len(a),
                  'small_only_pairs': len(keys[0])-len(a), 'large_only_pairs': len(keys[1])-len(a),
                  'common_population': common, 'part_metadata': metadata}
        reports.append(report)
        print(f'HARMONIZED_AMPLITUDE case={case} common={common}', flush=True)
    write_json(args.output, {'cases': reports, 'amplitudes': [.02, .2], 'script_sha256': sha256(__file__),
                            'cut_coordinates': 'original float64 baseline measured radius in pixels and MAG_AUTO',
                            'uncertainty_unit': 'all pair projections from each primary are grouped before standard error',
                            'limitations': ['pilot scenes, not model fitting or calibration acceptance',
                                            'forward random-neighbour shear projections; common usable pair cohort',
                                            'primary errors condition on a scene and may omit spatial correlations',
                                            'response denominator is represented primaries, not the full truth parent']})


if __name__ == '__main__':
    main()
