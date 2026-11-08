#!/usr/bin/env python
"""Audit frozen population support and reserve numerical-validation rows.

This does not validate the likelihood, alter cuts, or authorize inference.
The held-out rows are new to proposal development, not independent image data.
Run through the scheduler: source truth tables and all prior shards are read.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validation_rows(magnitude, excluded, seed=20260922):
    """32 uniform + 8 per fixed magnitude stratum, without replacement."""
    magnitude = np.asarray(magnitude, float)
    if magnitude.ndim != 1 or not np.isfinite(magnitude).all():
        raise ValueError('finite magnitude vector required')
    available = ~np.isin(np.arange(len(magnitude)), excluded)
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.flatnonzero(available), 32, replace=False)
    available[selected] = False
    groups = [('uniform', selected)]
    for lo, hi in [(-np.inf, 20), (20, 22), (22, 24), (24, 25.8)]:
        pool = np.flatnonzero(available & (magnitude >= lo) & (magnitude < hi))
        rows = rng.choice(pool, 8, replace=False)
        groups.append((f'{lo}:{hi}', rows))
        available[rows] = False
    return {name: rows.tolist() for name, rows in groups}


def match_truth(frozen, source):
    """Reject ambiguous/missing matches instead of silently changing sample."""
    keys = ['source_case', 'source_input_index']
    if frozen.duplicated(keys).any() or source.index_input.duplicated().any():
        raise ValueError('duplicate frozen/source identities')
    joined = frozen.merge(source[['index_input', 'r_input', 'Re_input']],
                          left_on='source_input_index', right_on='index_input',
                          how='left', validate='one_to_one', indicator=True)
    missing = int((joined['_merge'] != 'both').sum())
    if missing:
        raise ValueError(f'{missing} unmatched frozen truth rows; audit rejected')
    if not np.isfinite(joined[['r_input', 'Re_input']].to_numpy()).all():
        raise ValueError('nonfinite matched truth properties')
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite population audit')
    root = args.run
    input_manifest = json.loads((root/'input/image_mock_manifest.json').read_text())
    for name, expected in input_manifest['output_sha256'].items():
        if file_hash(root/'input'/name) != expected:
            raise ValueError(f'frozen input hash differs: {name}')
    truth = pd.read_parquet(root/'input/truth.parquet')
    observed = pd.read_parquet(root/'input/measurements.parquet')
    if len(truth) != len(observed) or len(truth) != input_manifest['n_objects']:
        raise ValueError('frozen row count mismatch')
    truth = truth.assign(frozen_row=np.arange(len(truth)))
    by_case, outside = [], np.zeros(len(truth), bool)
    for receipt in input_manifest['per_case']:
        path = Path(receipt['sources']['truth']['path'])
        if file_hash(path) != receipt['sources']['truth']['sha256']:
            raise ValueError(f'source truth hash differs: {path}')
        source = pd.read_feather(path, columns=['index_input', 'r_input', 'Re_input'])
        selected = truth.loc[truth.source_case == receipt['case']]
        joined = match_truth(selected, source)
        mask = joined.r_input.to_numpy() >= 26
        outside[joined.frozen_row.to_numpy()] = mask
        by_case.append(dict(case=receipt['case'], n=len(joined), true_r_ge26=int(mask.sum())))
    if sum(row['n'] for row in by_case) != len(truth):
        raise ValueError('source case coverage differs')
    print(f'OBSERVATIONS n={len(truth)} true_r_ge26={outside.sum()}', flush=True)

    subset_path = root/'prior_subset24m/manifest.json'
    subset = json.loads(subset_path.read_text())
    prepared = json.loads((root/'disk_assembled_v1/manifest.json').read_text())
    if prepared['identity']['subset_sha256'] != file_hash(subset_path):
        raise ValueError('prior/preparation identity differs')
    probability_path = root/'disk_assembled_v1/probability.npy'
    if file_hash(probability_path) != prepared['output_sha256']['probability.npy']:
        raise ValueError('cached usability probabilities changed')
    probability = np.load(probability_path, mmap_mode='r')
    if probability.shape != (len(prepared['points']), subset['n_rows']):
        raise ValueError('probability shape differs')
    total_pu, outside_pu = np.zeros(len(probability)), np.zeros(len(probability))
    prior_cases, cursor, prior_outside, prior_rows = set(), 0, 0, []
    for shard in subset['shards']:
        path = Path(shard['root'])/'galaxies.parquet'
        if file_hash(path) != shard['output_sha256']['galaxies.parquet']:
            raise ValueError(f'prior source hash differs: {path}')
        source_manifest = Path(shard['source_manifest'])
        if file_hash(source_manifest) != shard['source_manifest_sha256']:
            raise ValueError('source feature provenance differs')
        prior_cases.update(json.loads(source_manifest.read_text())['cases'])
        frame = pd.read_parquet(path, columns=['r'])
        if len(frame) != shard['n_rows'] or not np.isfinite(frame.r).all():
            raise ValueError('invalid prior magnitude/coverage')
        mask = frame.r.to_numpy() >= 26
        values = probability[:, cursor:cursor+len(frame)]
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('invalid usability probability')
        total_pu += values.sum(1, dtype=np.float64)
        outside_pu += values[:, mask].sum(1, dtype=np.float64)
        prior_outside += int(mask.sum())
        prior_rows.append(dict(shard=shard['source_shard'], n=len(frame), true_r_ge26=int(mask.sum())))
        cursor += len(frame)
        print(f'PRIOR_AUDIT rows={cursor}/{subset["n_rows"]}', flush=True)
    if cursor != subset['n_rows'] or prior_cases.intersection(truth.source_case.unique()):
        raise ValueError('prior coverage or observation/prior case overlap')

    audit_path = root/'curvature_audit_v1.json'
    audit = json.loads(audit_path.read_text())
    # Exclude all 50 inspected worst rows, not only the 32 currently benchmarked.
    excluded = sorted(set(range(32)) | {row['row'] for row in audit['worst']})
    magnitude = 30 - 2.5*np.log10(observed.measured_flux_from_mag_auto.to_numpy())
    panel = validation_rows(magnitude, excluded)
    # Uniform full-population sample: do not censor development/tail observations.
    cohort = np.sort(np.random.default_rng(20260923).choice(len(truth), 100000, replace=False))
    args.output.mkdir(parents=True)
    np.save(args.output/'reserved_100k_rows.npy', cohort)
    report = dict(status='audit_only_not_launch_approval',
        source_sha256={str(p): file_hash(p) for p in [Path(__file__), root/'input/image_mock_manifest.json',
            subset_path, root/'disk_assembled_v1/manifest.json', audit_path]},
        observations=dict(n=len(truth), true_r_ge26=int(outside.sum()),
            fraction_true_r_ge26=float(outside.mean()), by_case=by_case, unmatched=0),
        prior=dict(n=cursor, true_r_ge26=prior_outside, fraction_true_r_ge26=prior_outside/cursor,
            cases=sorted(prior_cases), case_overlap=[], by_shard=prior_rows,
            usability_mass_fraction_true_r_ge26=(outside_pu/total_pu).tolist(), points=prepared['points'],
            warning='Usability-weighted mass is NOT measured-selected mass or posterior contribution.'),
        reserved_numerical_validation=dict(seed=20260922, excluded_development_rows=excluded,
            groups=panel, status='not_evaluated',
            warning='Proposal-development holdout only; original 500k moments already exist. Not independent model validation.'),
        reserved_production=dict(n=100000, seed=20260923, rows_sha256=file_hash(args.output/'reserved_100k_rows.npy'),
            n_true_r_ge26=int(outside[cohort].sum()), truth_cut=False,
            status='reserved_only_do_not_launch', workers=2, rows_per_worker=50000),
        limitations=['No prior-resolution or generator-equivalence test.',
            'No out-of-training-parent likelihood validation.', 'No new inference or selected-mass integration.'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'POPULATION_AUDIT_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
