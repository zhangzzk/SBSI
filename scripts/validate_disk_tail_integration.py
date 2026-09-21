#!/usr/bin/env python
"""Persistent-worker exact benchmarks of the 32 most negative observations."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.probe_disk_auxiliary import StudentUnionProposal
from scripts.probe_disk_curvature import exact_observation, load_runtime
from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_null import run_adaptive_section5
from sbsi.disk_inference_store import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite exact panel')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    likelihood, base, mock, prepared = load_runtime(args.run)
    proposal = StudentUnionProposal(base, degrees_of_freedom=1.)
    audit_path = args.run / 'curvature_audit_v1.json'
    audit = json.loads(audit_path.read_text())
    entries = audit['worst'][:32][args.worker::2]
    identity = dict(status='diagnostic_only', worker=args.worker,
        audit_sha256=file_hash(audit_path), preparation_sha256=file_hash(args.run / 'disk_assembled_v1/manifest.json'),
        source_sha256={str(p): file_hash(p) for p in [Path(__file__), Path('scripts/probe_disk_curvature.py'), Path('scripts/probe_disk_auxiliary.py')]},
        protocol='audit_worst32; two persistent workers; exact24m; Cauchy-union K65536,262144; M16384 seed8701',
        object_rows=[entry['row'] for entry in entries])
    write_json(args.output / 'protocol.json', identity)
    for position, entry in enumerate(entries):
        row = entry['row']
        observed = mock.measurements.iloc[[row]].reset_index(drop=True)
        truth = mock.truth.iloc[[row]].reset_index(drop=True)
        # Reuse the two already completed exact benchmarks after hash/identity checks.
        old = {142230: 'curvature_probe_16617682_exact/report.json', 3563: 'exact_check_16617754/report.json'}.get(row)
        if old:
            reference = json.loads((args.run / old).read_text())
            if reference['preparation_sha256'] != identity['preparation_sha256'] or reference['row'] != row:
                raise ValueError('existing exact reference identity differs')
            exact = reference
            source = dict(path=old, sha256=file_hash(args.run / old))
        else:
            exact = exact_observation(likelihood, observed, prepared['points'][1:], prepared['identity']['h'],
                progress=lambda done, total: print(f'PANEL_EXACT row={row} atoms={done}/{total} seconds={time.monotonic()-started:.1f}', flush=True))
            source = dict(method='new_all_atom_sum')
        # Persist the expensive exact result before any candidate experiment.
        exact_path = args.output / f'row_{row}_exact.json'
        write_json(exact_path, dict(row=row, source=source, exact=exact, identity=identity))
        arms = []
        for k in (65536, 262144):
            moments = run_adaptive_section5(likelihood, MockCatalogue(observed, truth), proposal,
                center=prepared['identity']['center'], h=prepared['identity']['h'],
                draw_ladder=(16384,), n_candidates=k, proposal_prefilter_candidates=None,
                epsilon=.1, proposal_seed=8701, min_ess=32, max_weight_fraction=.5,
                candidate_backend='torch', estimator_mode='tilted_stratified', tilt_delta=.1,
                retain_full_ladder=True, full_information=True, object_ids=np.array([row]),
                object_chunk=1, atom_chunk=4096)
            arms.append(dict(k=k, score=moments.score[0].tolist(), information=moments.information[0].tolist(),
                score_error=(moments.score[0]-exact['score']).tolist(),
                information_error=(moments.information[0]-exact['information']).tolist()))
        write_json(args.output / f'row_{row}_comparison.json', dict(row=row, exact_sha256=file_hash(exact_path),
            exact=exact, production=entry, arms=arms))
        print(f'PANEL_COMPLETE objects={position+1}/{len(entries)} row={row} seconds={time.monotonic()-started:.1f}', flush=True)
    write_json(args.output / 'complete.json', dict(**identity, elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
