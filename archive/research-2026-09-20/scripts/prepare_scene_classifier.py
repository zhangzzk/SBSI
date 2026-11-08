"""Prepare corrected full-parent labels plus frozen truth-only geometry features."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.fixed_g0_domain import FLOW_FEATURES
from scripts.audit_harmonized_usability_response import load_aligned
from scripts.scene_classifier_features import GEOMETRY_FEATURES, PAIR_COLUMNS, geometry_features
from scripts.train_full_parent_classifier import sha256, write_json


def prepare(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full catalogue preparation requires Slurm')
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if len(set(args.cases)) != len(args.cases) or not args.cases:
        raise ValueError('distinct cases required')
    anchor_manifest = json.loads((args.anchor_root/'manifest.json').read_text())
    domain = None if args.domain_root is None else json.loads((args.domain_root/'manifest.json').read_text())
    if domain is not None and domain['flow']['features'] != list(FLOW_FEATURES):
        raise ValueError('corrected-domain feature contract mismatch')
    source_paths = [Path(__file__), Path('scripts/scene_classifier_features.py'),
                    args.anchor_root/'manifest.json', Path('/home/z/Zekang.Zhang/blendemu/blendemu/data_utils.py')]
    if domain is not None:
        source_paths += [args.domain_root/'manifest.json', Path('scripts/audit_harmonized_usability_response.py')]
    pins = {str(p.resolve()): sha256(p) for p in source_paths}
    (args.output_root/'cases').mkdir(parents=True)
    records = {}
    for case in args.cases:
        anchor_path = args.anchor_root/f'case{case:03d}.npz'
        pair_path = args.pair_root/f'case{case:03d}.feather'
        metadata_path = pair_path.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        anchor_hash, pair_hash = sha256(anchor_path), sha256(pair_path)
        if (metadata['case'] != case or metadata['sha256'] != pair_hash
                or metadata['anchor_sha256'] != anchor_hash
                or anchor_manifest['case_reports'][str(case)]['sha256'] != anchor_hash):
            raise ValueError('pair/anchor provenance mismatch')
        if metadata['conditions'] != {'pixel_size': .2, 'zero_point': 30., 'psf_fwhm': .73,
                                       'moffat_beta': 2.224, 'pixel_rms': .312}:
            raise ValueError('unexpected observation-condition scaling')
        with np.load(anchor_path) as stored:
            ids, context = stored['input_index'], stored['context'].astype(np.float32)
        if (context.shape != (len(ids), 8) or not np.isfinite(context).all()
                or np.any(context[:, 3] >= 26.)):
            raise ValueError('strict true-r<26 finite eight-input anchor required')
        pairs = pd.read_feather(pair_path, columns=list(PAIR_COLUMNS))
        geometry, counts = geometry_features(ids, pairs)
        del pairs
        arrays = {'ids': ids, 'base_context': context, 'geometry': geometry}
        report = {'anchor_sha256': anchor_hash, 'pair_sha256': pair_hash,
                  'pair_metadata_sha256': sha256(metadata_path), 'counts': counts,
                  'feature_min': geometry.min(0), 'feature_max': geometry.max(0)}
        if domain is not None:
            legs, provenance = load_aligned(args.domain_root, domain, case)
            label_ids = legs['g0']['input_index']
            position = pd.Index(ids).get_indexer(label_ids)
            missing = int((position < 0).sum())
            excluded = len(ids)-int((position >= 0).sum())
            if missing or excluded:
                raise ValueError(f'case{case}: label parents absent from anchor={missing}; anchor without labels={excluded}')
            arrays = {'ids': label_ids, 'base_context': context[position], 'geometry': geometry[position]}
            for index, leg in enumerate(('g0', 'g05')):
                path = args.domain_root/f'usability/{leg}/case{case:03d}.npz'
                with np.load(path) as stored:
                    np.testing.assert_array_equal(stored['input_index'], label_ids)
                    raw, detected = stored['context'].astype(np.float32), stored['detected']
                # The two experiments must have identical baseline truth/scene
                # conditions; only folded primary shape changes in the g05 leg.
                np.testing.assert_allclose(raw[:, 2:], arrays['base_context'][:, 2:], atol=2e-6, rtol=2e-6)
                if index == 0:
                    np.testing.assert_allclose(raw, arrays['base_context'], atol=2e-6, rtol=2e-6)
                usable = legs[leg]['usable']
                if np.any(usable & ~detected):
                    raise ValueError('usable label without matched detection')
                arrays[f'x{index}'] = np.column_stack((raw, arrays['geometry']))
                arrays[f'u{index}'], arrays[f'd{index}'] = usable, detected
            report.update(label_provenance=provenance, unmatched_label_parents=missing, anchor_without_labels=excluded)
        output = args.output_root/f'cases/case{case:03d}.npz'
        np.savez(output, **arrays)
        report['prepared_sha256'] = sha256(output)
        records[str(case)] = report
        print(f'SCENE_CLASSIFIER_PREPARED case={case} counts={counts}', flush=True)
    for path, expected in pins.items():
        if sha256(Path(path)) != expected:
            raise ValueError('preparation source changed: '+path)
    write_json(args.output_root/'manifest.json', {'format_version': 'scene_classifier_geometry_v1',
        'cases': args.cases, 'features': [*FLOW_FEATURES, *GEOMETRY_FEATURES], 'case_records': records,
        'with_corrected_labels': domain is not None, 'pins': pins,
        'annulus_edges': [0., .5, 1., 2., 4., 'infinity'],
        'distance_unit': 'sqrt(Re_primary**2 + PSF_half_light_radius**2), inherited from truth-pair rescale',
        'geometry_definition': 'log1p sum of neighbour/primary truth flux ratio in each disjoint distance annulus',
        'population': 'generated target-role parent; strict true r<26; no measured selection or usability intersection',
        'limitations': ['geometry uses the existing verified neighbour-pair support, not every source on the sky',
                        'no learned response column, measured size/flux, or U-dependent feature is read',
                        'static scalar geometry is shared by shape-shear legs; no neighbour orientation feature',
                        'preparation is not classifier fitting or downstream joint calibration']})
    print('SCENE_CLASSIFIER_PREPARATION_COMPLETE', args.output_root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor-root', type=Path, required=True)
    parser.add_argument('--pair-root', type=Path, required=True)
    parser.add_argument('--domain-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
