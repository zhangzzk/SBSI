#!/usr/bin/env python
"""Exact-sum and paired-sampler probes of the frozen V3.6 likelihood stencil.

No population cut, model, normalizer, expansion point, or production receipt
is changed. Exact mode scans every prior atom for one diagnostic observation.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import torch

from scripts import run_disk_inference as driver
from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.catalogue_likelihood import CatalogueSelection, OutputCut
from sbsi.catalogue_null import run_adaptive_section5
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.crowding import FLOW_FEATURES
from sbsi.disk_inference_store import FrozenDiskCache, ShardedDiskResponse, file_hash, write_json
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE, load_disk_response
from sbsi.selection_normalization import load_population_normalization


def load_runtime(root):
    prepared_root = root / 'disk_assembled_v1'
    subset = driver.load_subset_manifest(root / 'prior_subset24m/manifest.json')
    current = driver.identity(SimpleNamespace(input=root / 'input', subset=root / 'prior_subset24m/manifest.json'), subset)
    prepared = json.loads((prepared_root / 'manifest.json').read_text())
    driver.check_prepared_identity(prepared['identity'], current)
    if prepared['status'] != 'complete' or prepared['pilot_only']:
        raise ValueError('complete production preparation required')
    for name, expected in prepared['output_sha256'].items():
        if file_hash(prepared_root / name) != expected:
            raise ValueError(f'prepared hash mismatch: {name}')
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device='cuda')
    model = load_disk_response(V36_LIKE, device='cuda', nthread=4)
    responses = []
    for receipt in prepared['receipts']:
        source = receipt['subset_receipt']
        path = Path(source['root'])
        for name in ('pair_indptr.npy', 'pair_features.npy'):
            if file_hash(path / name) != source['output_sha256'][name]:
                raise ValueError(f'response hash mismatch: {path/name}')
        ptr = np.load(path / 'pair_indptr.npy', mmap_mode='r')[:receipt['n_rows']+1]
        features = np.load(path / 'pair_features.npy', mmap_mode='r')[:ptr[-1]]
        responses.append(CatalogueDiskResponse(ptr, features, model))
    response = ShardedDiskResponse(responses)
    zero = pd.DataFrame(np.load(prepared_root / 'zero.npy', mmap_mode='r'), columns=FLOW_FEATURES)
    probability = np.load(prepared_root / 'probability.npy', mmap_mode='r')
    cache = FrozenDiskCache(zero, dict(zip(map(tuple, prepared['points']), probability)), subset['conditions'])
    config = json.loads(driver.CONFIG.read_text())
    cut = OutputCut.from_specs(flow.target_transform.target_names, specs=config['measured_selection']['bounds'])
    likelihood = DiskCatalogueLikelihood(flow, cache, disk_response=response, selection=CatalogueSelection(cut))
    likelihood.population_normalization = load_population_normalization(prepared_root / 'normalization.json')
    if likelihood.population_normalization.identity != prepared['identity']:
        raise ValueError('normalizer identity mismatch')
    coords = ProposalCoordinateTable.load(prepared_root / 'proposal')
    proposal = DefensiveLocalProposal(coords, cache.prior.weights,
        local_base_weights=cache.get(0., 0.).detection_probability, score_dtype=torch.float64)
    flow.compile_log_prob(mode=None, dynamic=True)
    return likelihood, proposal, MockCatalogue.load(root / 'input'), prepared


def stencil_moments(log_values, h):
    """Nine nodes ordered centre,+x,-x,+y,-y,++,+-,-+,--."""
    z, xp, xm, yp, ym, pp, pm, mp, mm = np.asarray(log_values, dtype=float)
    score = np.array([(xp-xm)/(2*h), (yp-ym)/(2*h)])
    cross = -(pp-pm-mp+mm)/(4*h*h)
    information = np.array([[-(xp-2*z+xm)/h**2, cross], [cross, -(yp-2*z+ym)/h**2]])
    return score, information


def exact_observation(likelihood, observed, points, h, *, progress=None, atom_block=65536):
    """Sum every prior atom at all nine nodes for one observation."""
    if len(points) != 9 or atom_block < 1:
        raise ValueError('nine stencil nodes and positive atom block required')
    log_sums = np.full(9, -np.inf)
    top_values = [np.empty(0) for _ in points]
    top_atoms = [np.empty(0, dtype=np.int64) for _ in points]
    n = likelihood.disk_response.n_atoms
    for start in range(0, n, atom_block):
        atoms = np.arange(start, min(start+atom_block, n))
        for node, point in enumerate(points):
            terms = likelihood.log_importance_weights_tensor(observed, *point,
                atom_indices=atoms, proposal_probability=np.ones(len(atoms)), object_chunk=16)
            if terms.shape != (1, len(atoms)):
                raise ValueError('exact probe requires one observation and aligned atom terms')
            log_sums[node] = np.logaddexp(log_sums[node], torch.logsumexp(terms, 1).item())
            values, indices = torch.topk(terms[0], min(32, len(atoms)))
            merged_values = np.r_[top_values[node], values.cpu().numpy()]
            merged_atoms = np.r_[top_atoms[node], atoms[indices.cpu().numpy()]]
            keep = np.argsort(merged_values)[-32:][::-1]
            top_values[node], top_atoms[node] = merged_values[keep], merged_atoms[keep]
        if progress and (start == 0 or start // atom_block % 16 == 15 or start+atom_block >= n):
            progress(min(start+atom_block, n), n)
    normalizers = np.array([likelihood.population_normalization.log_mass(*p) for p in points])
    score, information = stencil_moments(log_sums-normalizers, h)
    if not np.isfinite(log_sums).all():
        raise ValueError('nonfinite exact log numerator')
    return dict(n_atoms=n, log_numerator=log_sums.tolist(),
                log_likelihood=(log_sums-normalizers).tolist(), score=score.tolist(),
                information=information.tolist(), top_atoms=[v.tolist() for v in top_atoms],
                top_posterior_weights=[np.exp(v-log_sums[i]).tolist() for i, v in enumerate(top_values)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('exact', 'sample'), required=True)
    parser.add_argument('--row', type=int, help='exact mode: another row from the audited worst-object list')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite probe')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    likelihood, proposal, mock, prepared = load_runtime(args.run)
    audit = json.loads((args.run / 'curvature_audit_v1.json').read_text())
    center, h = prepared['identity']['center'], prepared['identity']['h']
    print(f'PROBE_LOADED mode={args.mode} seconds={time.monotonic()-started:.1f}', flush=True)
    report = dict(status='diagnostic_only', mode=args.mode, center=center, h=h,
                  preparation_sha256=file_hash(args.run / 'disk_assembled_v1/manifest.json'),
                  audit_sha256=file_hash(args.run / 'curvature_audit_v1.json'),
                  probe_sha256=file_hash(__file__), implementation=driver.implementation())
    if args.mode == 'exact':
        row = audit['worst'][0]['row'] if args.row is None else args.row
        references = [entry for entry in audit['worst'] if entry['row'] == row]
        if len(references) != 1:
            raise ValueError('exact diagnostic row must occur once in the hash-checked audit')
        observed = mock.measurements.iloc[[row]].reset_index(drop=True)
        points = prepared['points'][1:]
        report.update(row=row, production=references[0], **exact_observation(
            likelihood, observed, points, h, progress=lambda done, total: print(
                f'EXACT atoms={done}/{total} seconds={time.monotonic()-started:.1f}', flush=True)))
    else:
        rows = np.unique([w['row'] for w in audit['worst'][:32]] + list(range(32)))
        window = MockCatalogue(mock.measurements.iloc[rows].reset_index(drop=True), mock.truth.iloc[rows].reset_index(drop=True))
        report['object_rows'] = rows.tolist()
        report['arms'] = []
        for seed in (8701, 8702):
            moments = run_adaptive_section5(likelihood, window, proposal, center=center, h=h,
                draw_ladder=(16384, 32768, 65536), n_candidates=1024, proposal_prefilter_candidates=131072,
                epsilon=.1, proposal_seed=seed, min_ess=32, max_weight_fraction=.5,
                candidate_backend='torch', estimator_mode='tilted_stratified', tilt_delta=.1,
                retain_full_ladder=True, full_information=True, object_ids=rows,
                object_chunk=4, atom_chunk=4096,
                progress=lambda done, total, elapsed: print(f'SAMPLE seed={seed} rows={done}/{total} seconds={elapsed:.1f}', flush=True))
            path = args.output / f'seed_{seed}.npz'
            np.savez_compressed(path, object_rows=rows, **{k: getattr(moments, k) for k in
                ('score', 'information', 'ladder_score', 'ladder_information', 'weight_ess', 'weight_max_fraction')})
            report['arms'].append(dict(seed=seed, path=path.name, sha256=file_hash(path), seconds=moments.elapsed_seconds))
    report['seconds'] = time.monotonic()-started
    write_json(args.output / 'report.json', report)
    print('PROBE_COMPLETE', args.output, report['seconds'], flush=True)


if __name__ == '__main__':
    main()
