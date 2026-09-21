"""Prepare state-major-axis geometry consistently for both training shear legs.

Creates a new versioned dataset; original features, labels and models remain
unchanged. The zero leg must exactly reproduce the original geometry.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.scene_classifier_features import geometry_features, GEOMETRY_FEATURES, PAIR_COLUMNS
from scripts.scene_geometry_coordinates import transport_major_distance
from scripts.train_full_parent_classifier import sha256, write_json

FORMAT = 'scene_classifier_state_major_geometry_v2'


def transport_prepared(data, pairs, psf_radius):
    result = {key: np.array(value, copy=True) for key, value in data.items()}
    context, ids = data['base_context'], data['ids']
    np.testing.assert_array_equal(data['x0'][:, :8], context)
    np.testing.assert_array_equal(data['x1'][:, 2:8], context[:, 2:8])
    changes = {}
    for leg in (0, 1):
        moved = transport_major_distance(ids, context, data[f'x{leg}'][:, :2], pairs, psf_radius)
        geometry, _ = geometry_features(ids, moved)
        if leg == 0:
            np.testing.assert_array_equal(geometry, data['geometry'])
            np.testing.assert_array_equal(geometry, data['x0'][:, 8:])
        result[f'geometry_g{leg}'] = geometry
        result[f'x{leg}'][:, 8:] = geometry
        changes[f'g{leg}'] = int(np.any(geometry != data[f'x{leg}'][:, 8:], axis=1).sum())
    return result, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full prepared geometry transformation requires Slurm')
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    protocol = json.loads(args.protocol.read_text())
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen preparation input changed: '+path)
    root = Path(protocol['prepared_root'])
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest['format_version'] != 'scene_classifier_geometry_v1'
            or manifest['cases'] != list(range(40, 120)) or not manifest['with_corrected_labels']):
        raise ValueError('original corrected-label 80-case preparation required')
    cases = [40, 100] if args.smoke else manifest['cases']
    (args.output_root/'cases').mkdir(parents=True)
    records = {}
    for case in cases:
        source = root/f'cases/case{case:03d}.npz'
        metadata = manifest['case_records'][str(case)]
        pair_path = Path(protocol['pair_root'])/f'case{case:03d}.feather'
        if sha256(source) != metadata['prepared_sha256'] or sha256(pair_path) != metadata['pair_sha256']:
            raise ValueError('original preparation or geometry pairs changed')
        with np.load(source) as stored:
            data = {key: stored[key] for key in stored.files}
        pairs = pd.read_feather(pair_path, columns=[*PAIR_COLUMNS, 'Re_input_p_scaled'])
        arrays, changes = transport_prepared(data, pairs, protocol['psf_half_light_radius'])
        path = args.output_root/f'cases/case{case:03d}.npz'
        np.savez(path, **arrays)
        records[str(case)] = metadata | {'original_prepared_sha256': metadata['prepared_sha256'],
                                         'prepared_sha256': sha256(path), 'changed_geometry_rows': changes}
        print('TRANSPORTED_GEOMETRY_PREPARED', case, changes, flush=True)
    for path, expected in protocol['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen preparation input changed during run: '+path)
    write_json(args.output_root/'manifest.json', manifest | {'format_version': FORMAT, 'cases': cases,
        'case_records': records, 'coordinate_convention': 'state_major_axis',
        'geometry_definition': 'flux-ratio annuli at distance/sqrt(state-major-Re**2+PSF-HLR**2)',
        'state_major_radius': 'circularized_Re/sqrt((1-|state_e|)/(1+|state_e|))',
        'geometry_features': GEOMETRY_FEATURES, 'protocol_sha256': sha256(args.protocol),
        'original_manifest_sha256': sha256(root/'manifest.json'),
        'limitations': ['new data contract requires state-dependent geometry at evaluation',
            'geometry stores zero-leg features; geometry_g0 and geometry_g1 store the corresponding training legs',
            'same inherited neighbour support; positions and fluxes held fixed under shape shear',
            'no model training, response fitting or acceptance occurs during preparation']})
    print('TRANSPORTED_GEOMETRY_COMPLETE', args.output_root, flush=True)


if __name__ == '__main__':
    main()
