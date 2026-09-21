#!/usr/bin/env python
"""Decompose saved disk-inference curvature without changing the estimator."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.combine_inference_partitions import _sha256
from sbsi.selection_normalization import load_population_normalization, log_mass_derivatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite audit')
    root = args.run
    frames = []
    reference = None
    cursor = 0
    receipts = []
    for part in sorted((root / 'production_lru_v1').glob('part_*')):
        payload = json.loads((part / 'result.json').read_text())
        if reference is None:
            reference = payload
        for name in ('initial_center', 'injected_shear', 'model_sha256',
                     'model_cache_sha256', 'scene_sha256', 'proposal_cache_sha256',
                     'mock_input_sha256', 'implementation_sha256', 'pipeline_config',
                     'pipeline_resolved_config_sha256', 'likelihood_config_sha256'):
            if payload[name] != reference[name]:
                raise ValueError(f'partition identity mismatch: {name}')
        partition = payload['observation_partition']
        path = part / payload['one_step_moments']['path']
        if partition['start'] != cursor or _sha256(path) != payload['one_step_moments']['sha256']:
            raise ValueError('partition coverage/hash mismatch')
        with np.load(path) as data:
            frame = {k: data[k] for k in data.files}
        np.testing.assert_array_equal(frame['object_rows'], np.arange(cursor, partition['stop']))
        cursor = partition['stop']
        frames.append(frame)
        receipts.append(dict(path=str(path), sha256=payload['one_step_moments']['sha256']))
    if cursor != reference['observation_partition']['n_total']:
        raise ValueError('incomplete input coverage')
    for name, expected in reference['mock_input_sha256'].items():
        if _sha256(root / 'input' / name) != expected:
            raise ValueError(f'input hash mismatch: {name}')
    observed = pd.read_parquet(root / 'input/measurements.parquet')
    truth = pd.read_parquet(root / 'input/truth.parquet')
    score = np.concatenate([p['score'] for p in frames])
    info = np.concatenate([p['information'] for p in frames])
    if len(observed) != cursor or len(truth) != cursor or not np.isfinite(score).all() or not np.isfinite(info).all():
        raise ValueError('input alignment or nonfinite moments')
    normalization = load_population_normalization(root / 'disk_assembled_v1/normalization.json')
    center = reference['initial_center']['center']
    log_mass, gradient, hessian = log_mass_derivatives(normalization, center)
    eig, vectors = np.linalg.eigh(info.sum(0))
    direction = vectors[:, 0]
    projection = np.einsum('i,nij,j->n', direction, info, direction)
    order = np.argsort(projection)

    def group(mask):
        n = int(np.sum(mask))
        matrix = info[mask].sum(0)
        return dict(n=n, score_sum=score[mask].sum(0).tolist(),
                    information_sum=matrix.tolist(), eigenvalues=np.linalg.eigvalsh(matrix).tolist(),
                    projected_information=float(projection[mask].sum()))

    radius = observed.measured_flux_radius.to_numpy()
    flux = observed.measured_flux_from_mag_auto.to_numpy()
    shape = np.linalg.norm(observed[['measured_ngmix_g1', 'measured_ngmix_g2']].to_numpy(), axis=1)
    magnitude = 30 - 2.5*np.log10(flux)
    bins = {}
    for name, values, edges in [('shape', shape, [0, .5, .8, .9, .95, .99, 1]),
                                ('radius_pixels', radius, [3, 4, 5, 7.5, 10, 20, np.inf]),
                                ('magnitude', magnitude, [-np.inf, 20, 22, 24, 25, 25.8])]:
        bins[name] = {f'{lo}:{hi}': group((values >= lo) & (values < hi)) for lo, hi in zip(edges[:-1], edges[1:])}
    tail = {}
    for n in [1, 10, 100, 1000, 5000]:
        keep = np.ones(cursor, dtype=bool)
        keep[order[:n]] = False
        tail[str(n)] = group(keep)
    worst = []
    for row in order[:50]:
        worst.append(dict(row=int(row), projected_information=float(projection[row]),
                          score=score[row].tolist(), information=info[row].tolist(),
                          observed=observed.iloc[row].to_dict(), truth=truth.iloc[row].to_dict()))
    weights = {}
    for key in ['weight_ess', 'weight_max_fraction', 'weight_relative_error', 'weight_pareto_k']:
        # Diagnostics are draw-level x observation, not a single flat vector.
        values = np.concatenate([p[key][-1] for p in frames])
        if values.shape != (cursor,):
            raise ValueError(f'weight diagnostic shape: {key} {values.shape}')
        finite = np.isfinite(values)
        weights[key] = dict(nonfinite=int((~finite).sum()),
                            quantiles=np.quantile(values[finite], [0, .01, .5, .9, .99, 1]).tolist())
    report = dict(status='diagnostic_only', n=cursor, center=center, source=receipts,
                  score_sum=score.sum(0).tolist(), information_sum=info.sum(0).tolist(),
                  information_eigenvalues=eig.tolist(), negative_direction=direction.tolist(),
                  normalization=dict(log_mass=log_mass, gradient=gradient.tolist(), hessian=hessian.tolist()),
                  numerator_information_eigenvalues=np.linalg.eigvalsh(info.sum(0)-cursor*hessian).tolist(),
                  projected_quantiles=np.quantile(projection, [0, .0001, .001, .01, .5, .99, 1]).tolist(),
                  bins=bins, diagnostic_exclusions_not_estimates=tail, worst=worst, weights=weights,
                  cases={str(case): group(truth.source_case.to_numpy() == case) for case in sorted(truth.source_case.unique())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=lambda x: x.item())+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source', 'cases', 'worst')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
