"""Join existing physical crowding onto unchanged half-shear classifier labels."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_full_parent_classifier import sha256, write_json


def project(arrays, ids, context):
    if len(np.unique(ids)) != len(ids) or len(np.unique(arrays['ids'])) != len(arrays['ids']):
        raise ValueError('unique identities required')
    if len(ids) != len(arrays['ids']):
        raise ValueError('parent counts differ')
    index = pd.Index(ids).get_indexer(arrays['ids'])
    if np.any(index < 0):
        raise ValueError('unmatched parent; no row dropping allowed')
    if context.shape != (len(ids), 11):
        raise ValueError('full close-crowding parent expected')
    np.testing.assert_allclose(arrays['base_context'], context[index, :8], rtol=2e-6, atol=2e-6)
    extra = context[index, 8:11].astype(np.float32)
    if not np.isfinite(extra).all():
        raise ValueError('finite crowding required')
    out = dict(arrays)
    for leg in (0, 1):
        x = arrays[f'x{leg}']
        if x.shape != (len(ids), 17):
            raise ValueError('original seventeen-input classifier rows required')
        np.testing.assert_allclose(x[:, 2:5], arrays['base_context'][:, 2:5], rtol=2e-6, atol=2e-6)
        out[f'x{leg}'] = np.column_stack([x[:, :5], extra]).astype(np.float32)
        np.testing.assert_array_equal(out[f'x{leg}'][:, :5], x[:, :5])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    p = json.loads((a.root/'protocol.json').read_text())
    for path, expected in p['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen input changed: '+path)
    source, parent, target = (Path(p[k]) for k in ('source_prepared', 'close_parent', 'prepared'))
    manifest = json.loads((source/'manifest.json').read_text())
    parent_manifest = json.loads((parent/'manifest.json').read_text())
    if parent_manifest['close_crowding']['source_parent_manifest_sha256'] not in manifest['pins'].values():
        raise ValueError('half-shear and close parent lack shared parent provenance')
    (target/'cases').mkdir(parents=True)
    records = {}
    for case in manifest['cases']:
        path = source/f'cases/case{case:03d}.npz'
        pp = parent/f'case{case:03d}.npz'
        if sha256(path) != manifest['case_records'][str(case)]['prepared_sha256'] or sha256(pp) != parent_manifest['case_reports'][str(case)]['sha256']:
            raise ValueError('source data hash mismatch')
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(pp) as z:
            out = project(arrays, z['input_index'], z['context'])
        dest = target/'cases'/path.name
        np.savez(dest, **out)
        with np.load(dest) as z:
            for k in arrays:
                np.testing.assert_array_equal(z[k], out[k] if k in ('x0', 'x1') else arrays[k])
        records[str(case)] = dict(manifest['case_records'][str(case)],
            prepared_sha256=sha256(dest), source_prepared_sha256=sha256(path), close_parent_sha256=sha256(pp),
            unmatched_rows=0, rows_removed=0, labels_changed=0)
        print(f'C8_PROJECT case={case} rows={len(out["ids"])}', flush=True)
    changed = dict(manifest, features=p['raw_features'], format_version='scene_classifier_physical_f8_v1',
        coordinate_convention='physical_arcsec', case_records=records, geometry_features=[],
        kernel_scales=[.5, 1., 2.], distance_unit='arcsec',
        geometry_definition='exact F8 full-rendered-neighbour log1p relative flux Gaussian summaries,10arcsec support',
        source_manifest_sha256=sha256(source/'manifest.json'))
    for k in ('annulus_edges', 'state_major_radius'):
        changed.pop(k, None)
    changed['limitations'] = ['Reused development scenes.', 'Scene positions and truth fluxes fixed under these shear perturbations.']
    write_json(target/'manifest.json', changed)
    smoke = a.root/'smoke_prepared'
    (smoke/'cases').mkdir(parents=True)
    for case in (40, 100):
        name = f'case{case:03d}.npz'
        (smoke/'cases'/name).symlink_to(target/'cases'/name)
    write_json(smoke/'manifest.json', dict(changed, cases=[40, 100]))
    write_json(a.root/'prepared_receipt.json', dict(manifest_sha256=sha256(target/'manifest.json'),
        cases=manifest['cases'], labels_changed=0, rows_removed=0, unmatched_rows=0))


if __name__ == '__main__':
    main()
