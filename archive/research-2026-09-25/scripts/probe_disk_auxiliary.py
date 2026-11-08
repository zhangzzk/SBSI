#!/usr/bin/env python
"""Test an auxiliary-output exact stratum for poorly retrieved bright atoms.

Union the original shape-aware shortlist with radius/flux-only proxy ranks.
The complement still uses the original defensive proposal and exact weights.
This is an integration diagnostic, not a change to the likelihood or cuts.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.probe_disk_curvature import load_runtime
from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_null import run_adaptive_section5
from sbsi.catalogue_sampling import ProposalCandidates
from sbsi.disk_inference_store import file_hash, write_json


def union_candidates(original, auxiliary, count):
    """Preserve every original candidate and fill from a ranked unique list."""
    original, auxiliary = np.asarray(original), np.asarray(auxiliary)
    if (original.ndim != 2 or auxiliary.ndim != 2 or len(original) != len(auxiliary)
            or count < original.shape[1]):
        raise ValueError('aligned candidate matrices and sufficient union budget required')
    result = np.empty((len(original), count), dtype=np.int64)
    for i, (base, extra) in enumerate(zip(original, auxiliary)):
        if len(np.unique(base)) != len(base) or len(np.unique(extra)) != len(extra):
            raise ValueError('candidate lists must be unique')
        combined = np.r_[base, extra[~np.isin(extra, base)]]
        if len(combined) < count:
            raise ValueError('insufficient distinct candidates')
        result[i] = combined[:count]
    return result


class AuxiliaryUnionProposal:
    """Delegate sampling unchanged; augment only the exactly summed stratum."""
    def __init__(self, base, device='cuda'):
        self.base = base
        active = base.active_indices
        values = base.coordinates.values[active, 2:]
        dispersion = base.coordinates.dispersion[active, 2:]
        ivar = 1 / dispersion**2
        mass = base.local_base_weights[active]
        with np.errstate(divide='ignore'):
            constant = np.log(mass) - np.log(dispersion).sum(1) - .5*(values**2*ivar).sum(1)
        self.constant = torch.as_tensor(constant, device=device, dtype=torch.float64)
        self.a1 = torch.as_tensor(ivar, device=device, dtype=torch.float64)
        self.a2 = torch.as_tensor(values*ivar, device=device, dtype=torch.float64)
        self.active = torch.as_tensor(active, device=device)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def auxiliary_score(self, observed):
        x = torch.as_tensor(self.base._observed_values(observed)[:, 2:], device=self.a1.device, dtype=torch.float64)
        return self.constant - .5*(x.square() @ self.a1.T - 2*x @ self.a2.T)

    def candidates(self, observed, *, n_candidates, prefilter_candidates, torch_device):
        if n_candidates <= 1024:
            raise ValueError('union budget must exceed original K1024')
        if prefilter_candidates not in (None, 131072):
            raise ValueError('original candidate prefilter is fixed at 131072')
        original = self.base.candidates(observed, n_candidates=1024,
            prefilter_candidates=131072, torch_device=torch_device)
        score = self.auxiliary_score(observed)
        top = torch.topk(score, n_candidates, dim=1, sorted=True).indices
        ranked = self.active[top].cpu().numpy()
        indices = union_candidates(original.indices, ranked, n_candidates)
        # Candidate distances are diagnostics only in tilted-stratified mode;
        # report the actual auxiliary proxy gaps for the retained union.
        positions = torch.searchsorted(self.active, torch.as_tensor(indices, device=score.device))
        selected = torch.gather(score, 1, positions)
        distances = (score.max(1, keepdim=True).values-selected).cpu().numpy()
        return ProposalCandidates(indices, distances, distances.max(1))


class StudentUnionProposal(AuxiliaryUnionProposal):
    """Heavy-tailed four-output retrieval; the target density is unchanged."""
    def __init__(self, base, device='cuda', degrees_of_freedom=3.):
        if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
            raise ValueError('positive finite Student degrees of freedom required')
        self.base = base
        self.nu = float(degrees_of_freedom)
        active = base.active_indices
        self.location = torch.as_tensor(base.coordinates.values[active], device=device, dtype=torch.float64)
        self.dispersion = torch.as_tensor(base.coordinates.dispersion[active], device=device, dtype=torch.float64)
        self.active = torch.as_tensor(active, device=device)
        with np.errstate(divide='ignore'):
            constant = np.log(base.local_base_weights[active]) - np.log(base.coordinates.dispersion[active]).sum(1)
        self.constant = torch.as_tensor(constant, device=device, dtype=torch.float64)

    def auxiliary_score(self, observed):
        x = torch.as_tensor(self.base._observed_values(observed), device=self.location.device, dtype=torch.float64)
        score = self.constant.expand(len(x), -1).clone()
        # Bound memory to a few observation x atom matrices, not B x N x D.
        for axis in range(x.shape[1]):
            residual = (x[:, axis, None]-self.location[None, :, axis])/self.dispersion[None, :, axis]
            score -= .5*(self.nu+1)*torch.log1p(residual.square()/self.nu)
        return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--exact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proxy', choices=('gaussian_aux', 'student4'), default='gaussian_aux')
    parser.add_argument('--degrees-of-freedom', type=float, default=3.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite auxiliary probe')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    likelihood, base, mock, prepared = load_runtime(args.run)
    proposal = (AuxiliaryUnionProposal(base) if args.proxy == 'gaussian_aux' else
                StudentUnionProposal(base, degrees_of_freedom=args.degrees_of_freedom))
    exact = json.loads(args.exact.read_text())
    if exact['preparation_sha256'] != file_hash(args.run / 'disk_assembled_v1/manifest.json'):
        raise ValueError('exact reference preparation differs')
    row = exact['row']
    observed = mock.measurements.iloc[[row]].reset_index(drop=True)
    atoms = np.unique(np.array(exact['top_atoms']).ravel())
    score = proposal.auxiliary_score(observed)[0]
    positions = torch.searchsorted(proposal.active, torch.as_tensor(atoms, device='cuda'))
    ranks = {str(atom): int((score > score[pos]).sum().item())+1 for atom, pos in zip(atoms, positions)}
    del score
    print('AUXILIARY_RANKS', json.dumps(ranks), flush=True)
    audit = json.loads((args.run / 'curvature_audit_v1.json').read_text())
    rows = np.unique([w['row'] for w in audit['worst'][:32]] + list(range(32)))
    window = MockCatalogue(mock.measurements.iloc[rows].reset_index(drop=True), mock.truth.iloc[rows].reset_index(drop=True))
    report = dict(status='diagnostic_only', proxy=args.proxy, degrees_of_freedom=args.degrees_of_freedom,
                  exact_sha256=file_hash(args.exact), object_rows=rows.tolist(),
                  preparation_sha256=exact['preparation_sha256'], script_sha256=file_hash(__file__),
                  auxiliary_ranks=ranks, arms=[])
    for k in (4096, 16384, 65536):
        moments = run_adaptive_section5(likelihood, window, proposal,
            center=prepared['identity']['center'], h=prepared['identity']['h'],
            draw_ladder=(16384,), n_candidates=k, proposal_prefilter_candidates=131072,
            epsilon=.1, proposal_seed=8701, min_ess=32, max_weight_fraction=.5,
            candidate_backend='torch', estimator_mode='tilted_stratified', tilt_delta=.1,
            retain_full_ladder=True, full_information=True, object_ids=rows,
            object_chunk=4, atom_chunk=4096,
            progress=lambda done, total, elapsed: print(f'AUXILIARY K={k} rows={done}/{total} seconds={elapsed:.1f}', flush=True))
        path = args.output / f'auxiliary_union_k{k}.npz'
        np.savez_compressed(path, object_rows=rows, score=moments.score, information=moments.information,
                            weight_ess=moments.weight_ess, weight_max_fraction=moments.weight_max_fraction)
        index = np.flatnonzero(rows == row)[0]
        report['arms'].append(dict(k=k, path=path.name, sha256=file_hash(path), seconds=moments.elapsed_seconds,
            worst_row_score=moments.score[index].tolist(), worst_row_information=moments.information[index].tolist(),
            exact_score_difference=(moments.score[index]-exact['score']).tolist(),
            exact_information_difference=(moments.information[index]-exact['information']).tolist()))
        write_json(args.output / f'progress_k{k}.json', report)
    report['seconds'] = time.monotonic()-started
    write_json(args.output / 'report.json', report)
    print('AUXILIARY_PROBE_COMPLETE', args.output, report['seconds'], flush=True)


if __name__ == '__main__':
    main()
