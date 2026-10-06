#!/usr/bin/env python
"""Audit dominant-atom retrieval and test larger exact strata, without cuts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.probe_disk_curvature import load_runtime
from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_null import run_adaptive_section5
from sbsi.disk_inference_store import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--exact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite proposal probe')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    likelihood, proposal, mock, prepared = load_runtime(args.run)
    exact = json.loads(args.exact.read_text())
    if exact['preparation_sha256'] != file_hash(args.run / 'disk_assembled_v1/manifest.json'):
        raise ValueError('exact reference preparation differs')
    row = exact['row']
    observed = mock.measurements.iloc[[row]].reset_index(drop=True)
    atoms = np.unique(np.array(exact['top_atoms']).ravel())
    proxy = proposal._tilted_proxy('cuda')
    values = proposal._observed_values(observed)
    scores = proxy.score(values)[0]
    positions = torch.searchsorted(proxy.active, torch.as_tensor(atoms, device='cuda'))
    np.testing.assert_array_equal(proxy.active[positions].cpu().numpy(), atoms)
    q = proxy.mixture(values, delta=.1)[0]
    original = proposal.candidates(observed, n_candidates=1024,
        prefilter_candidates=131072, torch_device='cuda').indices[0]
    atom_reports = []
    for atom, position in zip(atoms, positions):
        atom_reports.append(dict(atom=int(atom), in_original_candidates=bool(atom in original),
            proxy_rank=int((scores > scores[position]).sum().item())+1,
            proposal_probability=float(q[position]),
            expected_hits_16k=float(q[position])*16384,
            context=likelihood.cache.zero_flow.iloc[int(atom)].to_dict()))
    write_json(args.output / 'retrieval.json', dict(exact_sha256=file_hash(args.exact), row=row, atoms=atom_reports))
    print('RETRIEVAL_COMPLETE', json.dumps(atom_reports[:5]), flush=True)
    del q, scores
    audit = json.loads((args.run / 'curvature_audit_v1.json').read_text())
    rows = np.unique([w['row'] for w in audit['worst'][:32]] + list(range(32)))
    window = MockCatalogue(mock.measurements.iloc[rows].reset_index(drop=True), mock.truth.iloc[rows].reset_index(drop=True))
    report = dict(status='diagnostic_only', exact_sha256=file_hash(args.exact), object_rows=rows.tolist(),
                  preparation_sha256=exact['preparation_sha256'], script_sha256=file_hash(__file__), arms=[])
    for k in (1024, 16384, 131072):
        moments = run_adaptive_section5(likelihood, window, proposal,
            center=prepared['identity']['center'], h=prepared['identity']['h'],
            draw_ladder=(16384,), n_candidates=k, epsilon=.1, proposal_seed=8701,
            min_ess=32, max_weight_fraction=.5, candidate_backend='torch',
            candidate_source='whole_catalogue_gaussian_proxy', estimator_mode='tilted_stratified',
            tilt_delta=.1, retain_full_ladder=True, full_information=True, object_ids=rows,
            object_chunk=4, atom_chunk=4096,
            progress=lambda done, total, elapsed: print(f'PROPOSAL K={k} rows={done}/{total} seconds={elapsed:.1f}', flush=True))
        path = args.output / f'whole_catalogue_k{k}.npz'
        np.savez_compressed(path, object_rows=rows, score=moments.score, information=moments.information,
                            weight_ess=moments.weight_ess, weight_max_fraction=moments.weight_max_fraction)
        worst_position = np.flatnonzero(rows == row)[0]
        report['arms'].append(dict(k=k, path=path.name, sha256=file_hash(path), seconds=moments.elapsed_seconds,
            worst_row_score=moments.score[worst_position].tolist(),
            worst_row_information=moments.information[worst_position].tolist(),
            exact_score_difference=(moments.score[worst_position]-exact['score']).tolist(),
            exact_information_difference=(moments.information[worst_position]-exact['information']).tolist()))
        write_json(args.output / f'progress_k{k}.json', report)
    report['seconds'] = time.monotonic()-started
    write_json(args.output / 'report.json', report)
    print('PROPOSAL_PROBE_COMPLETE', args.output, report['seconds'], flush=True)


if __name__ == '__main__':
    main()
