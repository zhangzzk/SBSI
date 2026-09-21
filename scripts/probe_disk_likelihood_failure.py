#!/usr/bin/env python
"""Likelihood diagnosis, not a production estimator or model acceptance test.

Four predeclared development rows: two known failures and original rows 0/1.
All-atom sums measure posterior concentration and interleaved prior sensitivity.
The union of node-wise top32 atoms then localizes derivative mechanisms. That
union is deliberately truncated; its finer stencils are NOT exact24m results.
No new population normalizer is inferred from the frozen nine-node cache.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from scripts.probe_disk_curvature import load_runtime, stencil_moments
from sbsi.catalogue_likelihood import predict_detection_probability
from sbsi.crowding import FEATURES, classifier_features
from sbsi.disk_inference_store import file_hash, stencil, write_json
from sbsi.disk_response_transport import transported_physical_log_prob
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE, load_detection_classifier
from sbsi.output_conditioned_response import TRUTH_FEATURES
from sbsi.shear_map import apply_shear_to_ellipticity


def summarize_nodes(values, h):
    score, information = stencil_moments(values, h)
    return dict(log_values=np.asarray(values).tolist(), score=score.tolist(),
                information=information.tolist(), eigenvalues=np.linalg.eigvalsh(information).tolist())


def exact_partition_probe(likelihood, observed, points, h, progress, block=65536):
    """Four disjoint index-mod4 banks; retain original global masses in sums."""
    n = likelihood.disk_response.n_atoms
    sums = np.full((9, 4), -np.inf)
    squares = np.full(9, -np.inf)
    tops = [np.empty(0) for _ in points]
    ids = [np.empty(0, dtype=np.int64) for _ in points]
    for lo in range(0, n, block):
        atoms = np.arange(lo, min(lo+block, n))
        for node, point in enumerate(points):
            terms = likelihood.log_importance_weights_tensor(observed, *point,
                atom_indices=atoms, proposal_probability=np.ones(len(atoms)))[0]
            for group in range(4):
                mask = torch.as_tensor(atoms % 4 == group, device=terms.device)
                sums[node, group] = np.logaddexp(sums[node, group], torch.logsumexp(terms[mask], 0).item())
            squares[node] = np.logaddexp(squares[node], torch.logsumexp(2*terms, 0).item())
            value, position = torch.topk(terms, min(32, len(atoms)))
            merged = np.r_[tops[node], value.cpu().numpy()]
            merged_ids = np.r_[ids[node], atoms[position.cpu().numpy()]]
            order = np.argsort(merged)[-32:][::-1]
            tops[node], ids[node] = merged[order], merged_ids[order]
        if lo == 0 or lo // block % 16 == 15 or lo+block >= n:
            progress(min(lo+block, n), n)
    full = np.logaddexp.reduce(sums, axis=1)
    if not np.isfinite(full).all():
        raise ValueError('nonfinite full numerator')
    normalizer = np.array([likelihood.population_normalization.log_mass(*p) for p in points])
    banks = {}
    for groups in ((0,), (1,), (2,), (3,), (0, 2), (1, 3)):
        count = sum((n+3-g)//4 for g in groups)
        values = np.logaddexp.reduce(sums[:, groups], axis=1)+np.log(n/count)
        banks[str(groups)] = dict(n_atoms=count, **summarize_nodes(values, h),
            relative_log_shape_to_full=((values-values[0])-(full-full[0])).tolist())
    return dict(n_atoms=n, numerator=summarize_nodes(full, h),
        selected=summarize_nodes(full-normalizer, h), log_normalizer=normalizer.tolist(),
        posterior_effective_atoms=np.exp(2*full-squares).tolist(),
        top_atoms=[v.tolist() for v in ids],
        top_posterior_weights=[np.exp(v-full[i]).tolist() for i, v in enumerate(tops)],
        banks=banks, bank_limitation='Numerator sensitivity only: subset-specific selection normalizers NOT computed; index partitions are not fresh independently generated priors.')


class SelectedClassifier:
    """Recompute genuine state features for selected atoms, preserving CSR IDs."""
    def __init__(self, likelihood, prepared, atoms):
        self.detector = load_detection_classifier(V36_LIKE, device='cuda')
        conditions = likelihood.cache.conditions
        beta, fwhm = conditions['moffat_beta'], conditions['psf_fwhm']
        self.psf = fwhm/2*np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))
        self.groups = []
        self.n = len(atoms)
        offsets = likelihood.disk_response.offsets
        for shard, receipt in enumerate(prepared['receipts']):
            rows = np.flatnonzero((atoms >= offsets[shard]) & (atoms < offsets[shard+1]))
            if not len(rows):
                continue
            source = receipt['subset_receipt']
            root = Path(source['root'])
            for name in ('source_atom_ids.npy', 'pair_secondary.npy'):
                if file_hash(root/name) != source['output_sha256'][name]:
                    raise ValueError(f'identity hash mismatch: {root/name}')
            local = atoms[rows]-offsets[shard]
            source_ids = np.load(root/'source_atom_ids.npy', mmap_mode='r')[local]
            response = likelihood.disk_response.responses[shard]
            counts = response.indptr[local+1]-response.indptr[local]
            pair_rows = np.concatenate([np.arange(response.indptr[i], response.indptr[i+1]) for i in local])
            pairs = pd.DataFrame(response.features[pair_rows], columns=TRUTH_FEATURES)
            pairs['index_input_p'] = np.repeat(source_ids, counts)
            pairs['index_input_s'] = np.load(root/'pair_secondary.npy', mmap_mode='r')[pair_rows]
            zero = likelihood.cache.zero_flow.iloc[atoms[rows]].to_numpy(float)
            self.groups.append((rows, source_ids, zero, pairs))

    def evaluate(self, point):
        probability = np.empty(self.n)
        geometry = np.empty((self.n, 5))
        for rows, ids, zero, pairs in self.groups:
            shape = np.column_stack(apply_shear_to_ellipticity(zero[:, 0], zero[:, 1], *point))
            features = classifier_features(ids, zero, shape, pairs, self.psf)
            probability[rows] = predict_detection_probability(self.detector, pd.DataFrame(features, columns=FEATURES))
            geometry[rows] = features[:, 8:13]
        if not np.isfinite(probability).all() or np.any(probability <= 0):
            raise ValueError('selected atom classifier probabilities must be positive')
        return probability, geometry


def selected_probe(likelihood, observed, prepared, exact, *, output, row):
    atoms = np.unique(np.asarray(exact['top_atoms']).ravel())
    center, h = prepared['identity']['center'], prepared['identity']['h']
    points = prepared['points'][1:]
    actual = []
    for point in points:
        terms = likelihood.log_importance_weights_tensor(observed, *point,
            atom_indices=atoms, proposal_probability=np.ones(len(atoms)))
        actual.append(torch.logsumexp(terms[0], 0).item())
    capture = np.exp(np.array(actual)-np.array(exact['numerator']['log_values']))
    classifier = SelectedClassifier(likelihood, prepared, atoms)
    p0, geom0 = classifier.evaluate(center)
    cached_p = likelihood.cache.get(*center).detection_probability[atoms]
    np.testing.assert_allclose(p0, cached_p, rtol=2e-5, atol=1e-7)
    raw = observed[list(likelihood.target_names)].to_numpy(float)[0]
    b, _ = likelihood.disk_response.predict(atoms, np.full(len(atoms), raw[2]), np.full(len(atoms), raw[3]))
    device = likelihood.flow_model.device
    indices = torch.as_tensor(atoms, device=device)
    zero_context = likelihood._tensor_view(0., 0.).context.index_select(0, indices)
    physical = torch.as_tensor(np.broadcast_to(raw, (len(atoms), 4)).copy(), device=device)
    coefficients = torch.as_tensor(b, device=device)
    base64 = load_measurement_model(V36_LIKE.flow_checkpoints[0], device='cuda')
    base64.model.double()
    means = torch.as_tensor(base64.target_transform.means, device=device, dtype=torch.float64)
    scales = torch.as_tensor(base64.target_transform.scales, device=device, dtype=torch.float64)

    def density(point, *, precision='eager32', freeze_context=False, freeze_transport=False):
        cp = center if freeze_context else point
        context = likelihood._shear_standardized_context(
            zero_context.double() if precision == 'eager64' else zero_context, indices, *cp)
        tp = center if freeze_transport else point
        velocity = coefficients[:, None]*physical.new_tensor(tp)
        model = base64.model if precision == 'eager64' else likelihood.flow_model.model
        fn = model.log_prob if precision == 'compiled32' else lambda y, c: type(model).log_prob(model, y, c)
        return transported_physical_log_prob(physical, velocity, lambda y: fn((y-means)/scales, context))

    reports = []
    with torch.no_grad():
        for step in (h, h/2, h/4, h/8):
            values = {key: [] for key in ('compiled32', 'eager32', 'eager64', 'freeze_probability', 'freeze_context', 'freeze_transport')}
            jumps = []
            per_atom = []
            for point in stencil(center, step)[1:]:
                probability, geometry = classifier.evaluate(point)
                jumps.append(int(np.any(geometry != geom0, axis=1).sum()))
                lp = torch.as_tensor(np.log(probability)-np.log(exact['n_atoms']), device=device)
                lp0 = torch.as_tensor(np.log(p0)-np.log(exact['n_atoms']), device=device)
                d64 = density(point, precision='eager64')
                per_atom.append((d64+lp).cpu().numpy())
                terms = dict(compiled32=density(point, precision='compiled32')+lp,
                    eager32=density(point)+lp, eager64=d64+lp,
                    freeze_probability=d64+lp0,
                    freeze_context=density(point, precision='eager64', freeze_context=True)+lp,
                    freeze_transport=density(point, precision='eager64', freeze_transport=True)+lp)
                for key, term in terms.items():
                    values[key].append(torch.logsumexp(term, 0).item())
            per_atom = np.array(per_atom)
            np.savez_compressed(output/f'row_{row}_h{step:.7f}_atom_terms.npz',
                atoms=atoms, log_terms=per_atom, coefficients=b)
            reports.append(dict(h=step, geometry_changed_atoms=jumps,
                arms={key: summarize_nodes(v, step) for key, v in values.items()}))
    return dict(atoms=atoms.tolist(), capture_at_original_nodes=capture.tolist(),
        original_compiled_numerator=summarize_nodes(actual, h),
        classifier_center_max_abs_replay_error=float(np.max(np.abs(p0-cached_p))),
        coefficient_quantiles=np.quantile(b, [0, .25, .5, .75, 1]).tolist(),
        stencils=reports,
        limitation='All arms are unnormalized truncated numerators on the fixed top32 union. Finer-node capture is unknown. Frozen-component arms are causal code diagnostics, NOT replacement models. Eager64 promotes stored FP32 weights, not model retraining; classifier remains FP32.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite diagnostics')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    rows = ((142230, 0), (3563, 1))[args.worker]
    protocol = dict(status='diagnostic_only', rows=rows,
        selection='two previously inspected worst objects plus original rows0/1, not random or held-out',
        hypotheses=['finite atom concentration/resolution', 'classifier state discontinuities', 'flow/transport derivatives', 'FP32 finite-difference sensitivity'],
        script_sha256=file_hash(__file__), production_changes=False)
    write_json(args.output/'protocol.json', protocol)
    started = time.monotonic()
    likelihood, _, mock, prepared = load_runtime(args.run)
    protocol['preparation_sha256'] = file_hash(args.run/'disk_assembled_v1/manifest.json')
    write_json(args.output/'protocol.json', protocol)
    print(f'LIKELIHOOD_LOADED worker={args.worker} seconds={time.monotonic()-started:.1f}', flush=True)
    for row in rows:
        observed = mock.measurements.iloc[[row]].reset_index(drop=True)
        exact = exact_partition_probe(likelihood, observed, prepared['points'][1:], prepared['identity']['h'],
            lambda done, total: print(f'LIKELIHOOD_EXACT row={row} atoms={done}/{total} seconds={time.monotonic()-started:.1f}', flush=True))
        write_json(args.output/f'row_{row}_exact.json', dict(row=row, protocol=protocol, exact=exact))
        selected = selected_probe(likelihood, observed, prepared, exact, output=args.output, row=row)
        write_json(args.output/f'row_{row}_selected.json', dict(row=row, protocol=protocol, selected=selected))
        print(f'LIKELIHOOD_ROW_COMPLETE row={row} effective_atoms={exact["posterior_effective_atoms"][0]:.3f} capture={selected["capture_at_original_nodes"][0]:.6f} seconds={time.monotonic()-started:.1f}', flush=True)
    write_json(args.output/'complete.json', dict(**protocol, seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
