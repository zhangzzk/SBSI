#!/usr/bin/env python
"""Summarize likelihood diagnosis without producing a shear estimate."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.probe_disk_curvature import stencil_moments
from sbsi.disk_inference_store import file_hash, write_json


def mixture_decomposition(log_terms, h):
    """I_mix = E[I_atom] - Cov(score_atom), approximated by finite differences.

The identity is exact for derivatives. The reported residual measures the
finite-step inconsistency, not an implementation correction to subtract.
"""
    terms = np.asarray(log_terms, dtype=float)
    if terms.ndim != 2 or terms.shape[0] != 9 or not np.isfinite(terms).all():
        raise ValueError('finite nine-node atom log terms required')
    total = np.logaddexp.reduce(terms, axis=1)
    posterior = np.exp(terms[0]-total[0])
    scores, infos = [], []
    for atom in range(terms.shape[1]):
        score, info = stencil_moments(terms[:, atom], h)
        scores.append(score)
        infos.append(info)
    scores, infos = np.array(scores), np.array(infos)
    mean_score = posterior@scores
    centered = scores-mean_score
    covariance = np.einsum('n,ni,nj->ij', posterior, centered, centered)
    mean_info = np.einsum('n,nij->ij', posterior, infos)
    mix_score, mix_info = stencil_moments(total, h)
    return dict(posterior_effective_atoms=float(1/(posterior@posterior)),
        mean_component_information=mean_info.tolist(), component_score_covariance=covariance.tolist(),
        difference_identity=(mean_info-covariance).tolist(), mixture_information=mix_info.tolist(),
        finite_step_information_residual=(mix_info-(mean_info-covariance)).tolist(),
        finite_step_score_residual=(mix_score-mean_score).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root/'summary.json'
    if output.exists():
        raise ValueError('refusing to overwrite summary')
    rows, provenance = [], {}
    for worker, expected_rows in enumerate(((142230, 0), (3563, 1))):
        root = args.root/f'worker_{worker}'
        complete = json.loads((root/'complete.json').read_text())
        if complete['rows'] != list(expected_rows):
            raise ValueError('worker row identity mismatch')
        provenance[str(root/'complete.json')] = file_hash(root/'complete.json')
        for row in expected_rows:
            ep, sp = root/f'row_{row}_exact.json', root/f'row_{row}_selected.json'
            exact_report, selected_report = json.loads(ep.read_text()), json.loads(sp.read_text())
            if exact_report['row'] != row or selected_report['row'] != row or exact_report['protocol'] != selected_report['protocol']:
                raise ValueError('row/protocol mismatch')
            for path in (ep, sp):
                provenance[str(path)] = file_hash(path)
            exact, selected = exact_report['exact'], selected_report['selected']
            decompositions = []
            for item in selected['stencils']:
                h = item['h']
                path = root/f'row_{row}_h{h:.7f}_atom_terms.npz'
                data = np.load(path)
                np.testing.assert_array_equal(data['atoms'], selected['atoms'])
                provenance[str(path)] = file_hash(path)
                decompositions.append(dict(h=h, **mixture_decomposition(data['log_terms'], h)))
            rows.append(dict(row=row, full_posterior_effective_atoms=exact['posterior_effective_atoms'],
                full_top1_posterior=[v[0] for v in exact['top_posterior_weights']],
                full_selected=exact['selected'], full_numerator=exact['numerator'],
                banks=exact['banks'], selected=selected, decompositions=decompositions))
    tail_root = args.root.parent/'tail_exact_16617850'
    tail_files = sorted(tail_root.glob('worker_*/row_*_exact.json'))
    tail_summary = None
    if len(tail_files) == 32:
        records = [json.loads(p.read_text()) for p in tail_files]
        if len({r['row'] for r in records}) != 32:
            raise ValueError('duplicate existing tail rows')
        for path in tail_files:
            provenance[str(path)] = file_hash(path)
        masses = [r['exact']['top_posterior_weights'][1][0] for r in records]
        center_mass = [sum(r['exact']['top_posterior_weights'][0]) for r in records]
        tail_summary = dict(n=32, plus_g1_top1_mass_quantiles=np.quantile(masses, [0, .5, 1]).tolist(),
            distinct_plus_g1_dominant_atoms=len({r['exact']['top_atoms'][1][0] for r in records}),
            center_top32_mass_quantiles=np.quantile(center_mass, [0, .5, 1]).tolist(),
            limitation='Previously selected worst32, not representative of the population.')
    write_json(output, dict(status='diagnostic_only', rows=rows, provenance=provenance, previous_tail=tail_summary,
        limitations='Four development observations; no population inference. Split-bank comparisons are numerator-only. Fine stencils/ablations are on a truncated fixed atom union, not the full prior. No model correction or production acceptance.'))
    for row in rows:
        print(json.dumps(dict(row=row['row'], effective_atoms=row['full_posterior_effective_atoms'][0],
            full_information=row['full_selected']['information'],
            minimum_union_capture=min(row['selected']['capture_at_original_nodes']),
            smallest_step=row['selected']['stencils'][-1],
            smallest_step_decomposition=row['decompositions'][-1]), indent=2), flush=True)


if __name__ == '__main__':
    main()
