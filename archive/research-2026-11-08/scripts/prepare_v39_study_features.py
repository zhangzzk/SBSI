"""Rebuild F8 physical crowding for the existing expanded finite prior."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

BASE = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior')
RUN = BASE/'inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1'
EXT = BASE/'inference_bright_extension_20260925_v1'
ROOT = BASE/'inference_v39_mock20k_distance_20260925_v1'
sys.path.insert(0, str(ROOT/'runtime'))
from sbsi.disk_inference_store import file_hash, write_json  # noqa: E402
from sbsi.v39_inference import FEATURES, physical_close_crowding  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f'source hash differs: {path}')


def main(index):
    start = time.monotonic()
    protocol = read(ROOT/'protocol.json')
    checked(__file__, protocol['feature_script_sha256'])
    checked(ROOT/'runtime/sbsi/v39_inference.py', protocol['implementation_sha256']['sbsi/v39_inference.py'])
    subset = read(RUN/'prior_subset20m_bright20/manifest.json')
    extension = read(EXT/'prior/manifest.json')
    checked(RUN/'prior_subset20m_bright20/manifest.json', protocol['subset_sha256'])
    checked(EXT/'prior/manifest.json', protocol['extension_sha256'])
    out = ROOT/f'features/shard_{index:02d}'
    out.mkdir(parents=True, exist_ok=False)
    if index < 20:
        record = subset['shards'][index]
        source = Path(record['root'])
        full = Path(record['source_manifest']).parent
        full_record = read(full/'manifest.json')
        checked(full/'manifest.json', record['source_manifest_sha256'])
        checked(full/'galaxies.parquet', full_record['output_sha256']['galaxies.parquet'])
        for name in ('flow_zero.parquet', 'source_atom_ids.npy', 'prior_weight.npy'):
            checked(source/name, record['output_sha256'][name])
        scene = pd.read_parquet(full/'galaxies.parquet', columns=['RA', 'DEC', 'r', 'case'])
        ids = np.load(source/'source_atom_ids.npy')-record['source_row_offset']
        assert len(ids) == record['n_rows'] and len(np.unique(ids)) == len(ids)
        zero = pd.read_parquet(source/'flow_zero.parquet').iloc[:, :5].copy()
        np.testing.assert_allclose(zero.r_input_p, scene.r.to_numpy()[ids], rtol=0, atol=1e-12)
        wanted = np.arange(len(scene)) if index == 0 else ids
        close = np.empty((len(wanted), 3))
        cases = scene.case.to_numpy()
        for case in np.unique(cases):
            members = np.flatnonzero(cases == case)
            positions = np.flatnonzero(cases[wanted] == case)
            local = np.searchsorted(members, wanted[positions])
            np.testing.assert_array_equal(members[local], wanted[positions])
            close[positions] = physical_close_crowding(scene.iloc[members], local, workers=4)
            print(f'CROWDING shard={index} case={case} targets={len(local)} seconds={time.monotonic()-start:.1f}', flush=True)
        if index == 0:
            np.save(out/'donor_close_float64.npy', close)
            np.save(out/'donor_magnitude.npy', scene.r.to_numpy())
            close = close[ids]
        weights = np.load(source/'prior_weight.npy').copy()
        weights[zero.r_input_p.to_numpy() < 20] *= extension['old_bright_weight_multiplier']
    else:
        record = extension['shards'][index-20]
        source = Path(record['root'])
        for name in ('flow_zero.parquet', 'context_donor.npy', 'prior_weight.npy'):
            checked(source/name, record['output_sha256'][name])
        donor_root = ROOT/'features/shard_00'
        donor_receipt = read(donor_root/'manifest.json')
        for name in ('donor_close_float64.npy', 'donor_magnitude.npy'):
            checked(donor_root/name, donor_receipt['output_sha256'][name])
        zero = pd.read_parquet(source/'flow_zero.parquet').iloc[:, :5].copy()
        donors = np.load(source/'context_donor.npy')
        donor_close = np.load(donor_root/'donor_close_float64.npy', mmap_mode='r')[donors]
        donor_mag = np.load(donor_root/'donor_magnitude.npy', mmap_mode='r')[donors]
        # Same complete physical environment, new target flux in F_j/F_i.
        close = np.log1p(np.expm1(donor_close)*10.**(-.4*(donor_mag-zero.r_input_p.to_numpy()))[:, None])
        weights = np.load(source/'prior_weight.npy')
    for column, name in enumerate(FEATURES[5:]):
        zero[name] = close[:, column].astype(np.float32)
    assert tuple(zero.columns) == FEATURES and np.isfinite(zero).all().all()
    zero.to_parquet(out/'flow_zero.parquet')
    np.save(out/'prior_weight.npy', weights)
    for name in ('pair_indptr.npy', 'pair_features.npy'):
        checked(source/name, record['output_sha256'][name])
    write_json(out/'manifest.json', dict(status='complete', shard=index, n_rows=len(zero),
        source_record=record, feature_names=list(FEATURES), conditions=subset['conditions'],
        prior_mass=float(weights.sum()), truth_cuts=None,
        source_script_sha256=file_hash(__file__), protocol_sha256=file_hash(ROOT/'protocol.json'),
        response_root=str(source), response_sha256={name: record['output_sha256'][name]
            for name in ('pair_indptr.npy', 'pair_features.npy')},
        output_sha256={p.name: file_hash(p) for p in out.iterdir() if p.is_file()},
        elapsed_seconds=time.monotonic()-start))
    print(f'FEATURES_COMPLETE shard={index} rows={len(zero)} seconds={time.monotonic()-start:.1f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True, choices=range(28))
    main(parser.parse_args().shard)
