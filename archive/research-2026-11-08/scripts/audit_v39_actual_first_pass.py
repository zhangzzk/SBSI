"""Read-only curvature audit of the stopped actual-measurement first pass."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.selection_normalization import ExactPopulationNormalization, log_mass_derivatives


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return dict(finite_count=len(finite), total_count=len(values),
                p10_p50_p90=np.quantile(finite, [.1, .5, .9]).tolist() if len(finite) else None)


def audit(root, output):
    folder = root/'pass_00'
    center = np.asarray(json.loads((folder/'center.json').read_text())['center'])
    manifest = json.loads((root/'input/manifest.json').read_text())
    hashes = {}
    for name, expected in manifest['output_sha256'].items():
        path = root/'input'/name
        assert digest(path) == expected
        hashes[str(path)] = expected
    measured = pd.read_parquet(root/'input/measurements.parquet')
    truth = pd.read_parquet(root/'input/truth.parquet')
    parts = []
    for worker in range(4):
        path = folder/f'worker_{worker}.npz'
        receipt = json.loads((folder/f'worker_{worker}.json').read_text())
        assert receipt['status'] == 'complete' and receipt['center'] == center.tolist()
        assert receipt['normalization_sha256'] == digest(folder/'normalization.json')
        assert receipt['protocol_sha256'] == digest(root/'protocol.json')
        assert receipt['output_sha256'] == digest(path)
        hashes[str(path)] = receipt['output_sha256']
        with np.load(path) as data:
            parts.append({key: data[key] for key in data.files})
        np.testing.assert_array_equal(parts[-1]['object_rows'], np.arange(worker*5000, (worker+1)*5000))
    score = np.concatenate([p['ladder_score'] for p in parts], axis=1)
    info = np.concatenate([p['ladder_information'] for p in parts], axis=1)
    assert np.isfinite(score).all() and np.isfinite(info).all()
    np.testing.assert_array_equal(np.concatenate([p['score'] for p in parts]), score[-1])
    np.testing.assert_array_equal(np.concatenate([p['information'] for p in parts]), info[-1])
    _, gradient, hessian = log_mass_derivatives(
        ExactPopulationNormalization.from_payload(json.loads((folder/'normalization.json').read_text())), center)
    total = info[-1].sum(0)
    eigenvalues, vectors = np.linalg.eigh(total)
    direction = vectors[:, 0]
    projected = np.einsum('i,nij,j->n', direction, info[-1], direction)
    mag = measured.measured_mag_auto.to_numpy()
    diagnostics = {key: np.concatenate([p[key][-1] for p in parts]) for key in
                   ['weight_ess', 'weight_max_fraction', 'weight_relative_error', 'weight_pareto_k']}
    masks = {'mag_lt20': mag < 20, 'mag20_22': (mag >= 20) & (mag < 22),
             'mag22_24': (mag >= 22) & (mag < 24), 'mag_ge24': mag >= 24}
    groups = {}
    for name, mask in masks.items():
        groups[name] = dict(count=int(mask.sum()), score_sum=score[-1, mask].sum(0).tolist(),
            information_sum=info[-1, mask].sum(0).tolist(),
            curvature_in_negative_direction=float(projected[mask].sum()),
            negative_curvature_count=int((projected[mask] < 0).sum()),
            weight_diagnostics={k: quantiles(v[mask]) for k, v in diagnostics.items()})
    worst = np.argsort(projected)[:20]
    rows = []
    for row in worst:
        rows.append(dict(row=int(row), magnitude=float(mag[row]),
            case=int(truth.source_case.iloc[row]), input_index=int(truth.source_input_index.iloc[row]),
            measured_shape=measured.loc[row, ['measured_ngmix_g1', 'measured_ngmix_g2']].tolist(),
            negative_direction_curvature=float(projected[row]),
            diagnostics={key: float(value[row]) if np.isfinite(value[row]) else None for key, value in diagnostics.items()}))
    ladder = json.loads((root/'protocol.json').read_text())['draw_ladder']
    rungs = [dict(draws=draws, score_sum=s.sum(0).tolist(), information_sum=i.sum(0).tolist(),
                  eigenvalues=np.linalg.eigvalsh(i.sum(0)).tolist()) for draws, s, i in zip(ladder, score, info)]
    difference = info[-1]-info[-2]
    changed = np.einsum('i,nij,j->n', direction, difference, direction)
    result = dict(status='audit_complete_inference_stopped', n_observations=len(measured),
        center=center.tolist(), all_saved_moments_finite=True, exact_full_ladder_maximum_match=True,
        input_sha256=hashes, script_sha256=digest(__file__), rungs=rungs,
        eigenvalues=eigenvalues.tolist(), negative_direction=direction.tolist(),
        normalization_gradient=gradient.tolist(), normalization_hessian=hessian.tolist(),
        total_normalization_information=(len(measured)*hessian).tolist(),
        numerator_only_eigenvalues=np.linalg.eigvalsh(total-len(measured)*hessian).tolist(),
        magnitude_groups=groups, most_negative_rows=rows,
        paired_8k_to16k=dict(score_sum_change=(score[-1]-score[-2]).sum(0).tolist(),
            information_sum_change=difference.sum(0).tolist(),
            negative_direction_curvature_change=float(changed.sum()),
            row_conditional_curvature_change_se=float(np.std(changed, ddof=1)*np.sqrt(len(changed)))),
        diagnostic_only_omission=[dict(n_omitted=n, eigenvalues=np.linalg.eigvalsh(total-info[-1, worst[:n]].sum(0)).tolist())
                                  for n in [1, 5, 10, 20]],
        limitations=['No observations removed from inference; omission numbers only diagnose influence.',
                     'An indefinite local observed information matrix away from a maximum need not imply a software bug.',
                     'One finite-draw stencil cannot identify the global optimum or validate the model.',
                     'No replacement solver, truth-centering, or new shear estimate was applied.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['input_sha256', 'most_negative_rows']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
