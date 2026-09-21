"""Append fixed continuous truth-only crowding features to corrected training legs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.scene_classifier_features import PAIR_COLUMNS
from scripts.smooth_scene_classifier_features import FEATURES, FORMAT, SCALES, augment_prepared
from scripts.train_full_parent_classifier import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.output_root.exists():
        raise RuntimeError('Slurm and new output directory required')
    protocol = json.loads(args.protocol.read_text())
    def verify():
        for path, expected in protocol['files'].items():
            if sha256(Path(path)) != expected:
                raise ValueError('frozen smooth input changed: '+path)
    verify()
    root = Path(protocol['prepared_root'])
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest['format_version'] != 'scene_classifier_state_major_geometry_v2'
            or manifest['cases'] != list(range(40, 120)) or not manifest['with_corrected_labels']):
        raise ValueError('corrected state-major training data required')
    cases = [40, 100] if args.smoke else manifest['cases']
    (args.output_root/'cases').mkdir(parents=True)
    records = {}
    for case in cases:
        source = root/f'cases/case{case:03d}.npz'
        metadata = manifest['case_records'][str(case)]
        pairs_path = Path(protocol['pair_root'])/f'case{case:03d}.feather'
        if sha256(source) != metadata['prepared_sha256'] or sha256(pairs_path) != metadata['pair_sha256']:
            raise ValueError('prepared/pair hash mismatch')
        with np.load(source) as stored:
            data = {k: stored[k] for k in stored.files}
        pairs = pd.read_feather(pairs_path, columns=[*PAIR_COLUMNS, 'Re_input_p_scaled'])
        arrays = augment_prepared(data, pairs, protocol['psf_half_light_radius'])
        output = args.output_root/f'cases/case{case:03d}.npz'
        np.savez(output, **arrays)
        records[str(case)] = metadata | {'source_prepared_sha256': metadata['prepared_sha256'],
                                         'prepared_sha256': sha256(output)}
        print('SMOOTH_CLASSIFIER_PREPARED', case, len(arrays['ids']), flush=True)
    verify()
    write_json(args.output_root/'manifest.json', manifest | {
        'format_version': FORMAT, 'features': FEATURES, 'cases': cases, 'case_records': records,
        'kernel_scales': SCALES, 'protocol_sha256': sha256(args.protocol),
        'source_manifest_sha256': sha256(root/'manifest.json'),
        'limitations': ['truth-only features; no response targets or measured output features',
                        'inherited pair support and fixed neighbour positions/fluxes',
                        'smooth features use state major-axis distance in training and evaluation',
                        'development experiment; no shear-calibration or density acceptance claim']})
    print('SMOOTH_CLASSIFIER_PREPARATION_COMPLETE', args.output_root, flush=True)


if __name__ == '__main__':
    main()
