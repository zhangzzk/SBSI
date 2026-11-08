#!/usr/bin/env python
"""Project frozen flow domains without changing a row, target or usability label."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SOURCE_FEATURES = ['e1_input_p', 'e2_input_p', 'sersic_n_input_p', 'r_input_p',
                   'circularized_Re_input_p', 'nbr_flux_near', 'nbr_flux_far', 'nbr_flux_max',
                   'nbr_close_050', 'nbr_close_100', 'nbr_close_200']
FEATURES = SOURCE_FEATURES[:5] + SOURCE_FEATURES[8:]
INDICES = [SOURCE_FEATURES.index(name) for name in FEATURES]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def project_arrays(arrays):
    x = arrays['context']
    if x.ndim != 2 or x.shape[1] != len(SOURCE_FEATURES) or not np.isfinite(x).all():
        raise ValueError('expected finite frozen eleven-input context')
    return {k: (v[:, INDICES] if k == 'context' else v) for k, v in arrays.items()}


def project_file(task):
    source, output, expected = task
    digest = sha256(source)
    if digest != expected:
        raise ValueError(f'changed source: {source}')
    with np.load(source, allow_pickle=False) as stored:
        original = {k: stored[k] for k in stored.files}
    projected = project_arrays(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez(stream, **projected)
    with np.load(output, allow_pickle=False) as stored:
        if set(stored.files) != set(original):
            raise ValueError('array keys changed')
        for key, value in original.items():
            expected_array = value[:, INDICES] if key == 'context' else value
            actual = stored[key]
            if actual.dtype != expected_array.dtype:
                raise ValueError('array dtype changed')
            np.testing.assert_array_equal(actual, expected_array)
    return dict(source=str(source), output=str(output), source_sha256=digest,
                sha256=sha256(output), rows=len(original['context']),
                changed_arrays=['context'], removed_rows=0, altered_labels=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-banks', type=Path)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or a.output.exists():
        raise ValueError('scheduler allocation and new output required')
    source_manifest = a.source / 'manifest.json'
    manifest = json.loads(source_manifest.read_text())
    if manifest['flow']['features'] != SOURCE_FEATURES:
        raise ValueError('source feature contract changed')
    cases = sorted(manifest['split']['train_cases'] + manifest['split']['validation_cases'])
    if cases != list(range(200)):
        raise ValueError('expected original 200-case split')
    parent_hashes, bank_hashes = {}, {}
    if (a.source / 'usability').exists():
        if a.target_banks is None:
            raise ValueError('frozen full-parent hashes required')
        for split in ('train', 'validation'):
            path = a.target_banks / f'per_leg_guard_{split}_targets.json'
            bank = json.loads(path.read_text())
            if bank['domain_manifest_sha256'] != sha256(source_manifest):
                raise ValueError('target bank domain mismatch')
            bank_hashes[str(path)] = sha256(path)
            for record in bank['provenance']:
                for leg in ('g0', 'g05'):
                    parent_hashes[record['case'], leg] = record['sources'][leg]['usability_sha256']
    tasks = []
    for case in cases:
        for leg in ('g0', 'g05'):
            relative = Path('flow') / leg / f'case{case:03d}.npz'
            tasks.append((a.source / relative, a.output / relative,
                          manifest['flow'][f'{leg}_cases'][str(case)]['sha256']))
            if parent_hashes:
                relative = Path('usability') / leg / f'case{case:03d}.npz'
                tasks.append((a.source / relative, a.output / relative, parent_hashes[case, leg]))
    a.output.mkdir(parents=True)
    records = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for index, record in enumerate(pool.map(project_file, tasks), 1):
            records.append(record)
            if index == 1 or index % 40 == 0:
                print(f'PROJECTION_PROGRESS {index}/{len(tasks)} labels_unchanged=True', flush=True)
    projected_manifest = copy.deepcopy(manifest)
    projected_manifest['flow']['features'] = FEATURES
    for record in records:
        path = Path(record['output']).relative_to(a.output)
        if path.parts[0] == 'flow':
            leg, case = path.parts[1], str(int(path.stem[4:]))
            projected_manifest['flow'][f'{leg}_cases'][case]['sha256'] = record['sha256']
    projected_manifest['feature_projection'] = dict(source=str(a.source),
        source_manifest_sha256=sha256(source_manifest), source_features=SOURCE_FEATURES,
        features=FEATURES, indices=INDICES, source_target_bank_hashes=bank_hashes,
        unchanged=['row_order', 'identities', 'targets', 'gamma', 'usability', 'split', 'population'],
        script_sha256=sha256(__file__), records=records)
    (a.output / 'manifest.json').write_text(json.dumps(projected_manifest, indent=2) + '\n')
    (a.output / 'completed_projection.json').write_text(json.dumps(dict(files=len(records),
        manifest_sha256=sha256(a.output / 'manifest.json'), altered_labels=0, removed_rows=0), indent=2) + '\n')
    print(f'PROJECTION_COMPLETE files={len(records)} altered_labels=0 removed_rows=0', flush=True)


if __name__ == '__main__':
    main()
