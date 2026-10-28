"""Append a uniform, disjoint draw to the frozen ConstGold image sample."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_constgold_inference_input import read_case, file_hash


def extension_indices(total, previous_count, previous_seed, additional_count, seed):
    """Replay the original choice, then draw uniformly from its complement."""
    previous = np.random.default_rng(previous_seed).choice(total, previous_count, replace=False)
    remaining = np.ones(total, dtype=bool)
    remaining[previous] = False
    additional = np.random.default_rng(seed).choice(np.flatnonzero(remaining), additional_count,
                                                   replace=False)
    return previous, additional


def freeze(source, destination, additional_count, seed):
    manifest = json.loads((source/'image_mock_manifest.json').read_text())
    for name, sha in manifest['output_sha256'].items():
        if file_hash(source/name) != sha:
            raise ValueError('original sample changed: '+name)
    if manifest['primary_mag'] is not None or manifest['primary_re'] is not None:
        raise ValueError('truth-property cuts prohibited')
    if manifest['n_objects'] != 500000 or manifest['n_eligible'] != 18157963:
        raise ValueError('unexpected parent sample population')
    frames, reports = [], []
    for old_report in manifest['per_case']:
        case = old_report['case']
        catalogue_root = Path(old_report['sources']['truth']['path']).parents[4]
        frame, report = read_case(catalogue_root, case, pixel_size=.2, zero_point=30.,
            primary_mag=None, primary_re=None, measured_radius_min=.6, measured_mag_max=25.8,
            strict_radius=True, physical_disk_usability=True)
        if report != old_report:
            raise ValueError(f'case {case}: source hashes or matching/selection accounting changed')
        frames.append(frame)
        reports.append(report)
        print(f'REPLAY_CASE {case} selected={len(frame)}', flush=True)
    eligible = pd.concat(frames, ignore_index=True)
    del frames
    if len(eligible) != manifest['n_eligible']:
        raise ValueError('eligible population changed')
    previous, additional = extension_indices(len(eligible), manifest['n_objects'],
        manifest['sample_seed'], additional_count, seed)
    old_measurements = pd.read_parquet(source/'measurements.parquet')
    old_truth = pd.read_parquet(source/'truth.parquet')
    identities = ['source_case', 'source_input_index', 'source_detection_id']
    replay = eligible.iloc[previous].reset_index(drop=True)
    pd.testing.assert_frame_equal(replay[old_measurements.columns], old_measurements, check_exact=True)
    pd.testing.assert_frame_equal(replay[identities], old_truth[identities], check_exact=True)
    chosen = eligible.iloc[additional].reset_index(drop=True)
    extra_truth = chosen[identities].copy()
    for column in old_truth.columns.difference(identities):
        if old_truth[column].nunique() != 1:
            raise ValueError('nonconstant provenance field: '+column)
        extra_truth[column] = old_truth[column].iloc[0]
    measurements = pd.concat([old_measurements, chosen[old_measurements.columns]], ignore_index=True)
    truth = pd.concat([old_truth, extra_truth[old_truth.columns]], ignore_index=True)
    for keys in (identities[:2], [identities[0], identities[2]]):
        if truth.duplicated(keys).any():
            raise ValueError('duplicate identities in expanded sample')
    destination.mkdir(exist_ok=False)
    measurements.to_parquet(destination/'measurements.parquet', index=False)
    truth.to_parquet(destination/'truth.parquet', index=False)
    expanded = dict(manifest, n_objects=len(truth), per_case=reports,
        sample_seed=None, extension_seed=seed,
        extension_sampling='Original 500000 rows retained exactly; uniform draw from their complement',
        previous_sample=str(source), previous_seed=manifest['sample_seed'],
        previous_manifest_sha256=file_hash(source/'image_mock_manifest.json'),
        previous_rows=len(old_truth), additional_rows=additional_count,
        original_sample_replay_exact=True, duplicate_identities=0, old_new_overlap=0,
        implementation_sha256=file_hash(__file__),
        adapter_sha256=file_hash(Path(__file__).with_name('prepare_constgold_inference_input.py')),
        output_sha256={name: file_hash(destination/name) for name in
                      ('measurements.parquet', 'truth.parquet')})
    (destination/'image_mock_manifest.json').write_text(json.dumps(expanded, indent=2)+'\n')
    print(f'EXPANDED_SAMPLE_FROZEN n={len(truth)} additional={additional_count}', flush=True)
