#!/usr/bin/env python
"""Autodiff reference on diagnostic atom unions, with classifier held fixed.

The first likelihood probe promotes neural weights to FP64 but deliberately
keeps the production context-shear helper (which uses FP32 raw shapes).
This independent reference also performs input shear/preprocessing in FP64.
It is a truncated numerator derivative, NOT a full selected likelihood.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import run_disk_inference as driver
from scripts.probe_disk_curvature import stencil_moments
from sbsi.disk_inference_store import file_hash, stencil, write_json
from sbsi.disk_response_transport import transported_physical_log_prob
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE


def shear_context64(raw, g, preprocessor):
    if raw.dtype != torch.float64 or g.dtype != torch.float64:
        raise ValueError('FP64 reference inputs required')
    first = preprocessor.feature_names.index('e1_input_p')
    second = preprocessor.feature_names.index('e2_input_p')
    e1, e2 = raw[:, first], raw[:, second]
    dr, di = 1+g[0]*e1+g[1]*e2, g[0]*e2-g[1]*e1
    nr, ni = e1+g[0], e2+g[1]
    values = raw.clone()
    values[:, first] = (nr*dr+ni*di)/(dr.square()+di.square())
    values[:, second] = (ni*dr-nr*di)/(dr.square()+di.square())
    return preprocessor.transform_tensor(values)


def derivatives(fn, center):
    point = center.detach().clone().requires_grad_(True)
    value = fn(point)
    gradient, = torch.autograd.grad(value, point, create_graph=True)
    hessian = torch.stack([torch.autograd.grad(gradient[i], point, retain_graph=True)[0] for i in range(2)])
    return dict(log_value=value.item(), score=gradient.detach().cpu().tolist(),
        information=(-hessian).detach().cpu().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite reference')
    torch.set_num_threads(4)
    prepared_root = args.run/'disk_assembled_v1'
    prepared = json.loads((prepared_root/'manifest.json').read_text())
    subset = driver.load_subset_manifest(args.run/'prior_subset24m/manifest.json')
    from types import SimpleNamespace
    current = driver.identity(SimpleNamespace(input=args.run/'input', subset=args.run/'prior_subset24m/manifest.json'), subset)
    driver.check_prepared_identity(prepared['identity'], current)
    for name in ('zero.npy', 'probability.npy'):
        if file_hash(prepared_root/name) != prepared['output_sha256'][name]:
            raise ValueError('prepared array hash mismatch')
    from sbsi.catalogue_closure import MockCatalogue
    mock = MockCatalogue.load(args.run/'input')
    zero = np.load(prepared_root/'zero.npy', mmap_mode='r')
    probability = np.load(prepared_root/'probability.npy', mmap_mode='r')
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device='cuda')
    flow.model.double()
    device = flow.device
    center = torch.tensor(prepared['identity']['center'], dtype=torch.float64, device=device)
    means = torch.tensor(flow.target_transform.means, dtype=torch.float64, device=device)
    scales = torch.tensor(flow.target_transform.scales, dtype=torch.float64, device=device)
    rows, provenance = [], {}
    for worker, row_ids in enumerate(((142230, 0), (3563, 1))):
        for row in row_ids:
            root = args.diagnostics/f'worker_{worker}'
            source = root/f'row_{row}_selected.json'
            report = json.loads(source.read_text())
            if report['protocol']['preparation_sha256'] != file_hash(prepared_root/'manifest.json'):
                raise ValueError('diagnostic preparation identity mismatch')
            path = root/f'row_{row}_h0.0010000_atom_terms.npz'
            data = np.load(path)
            atoms = data['atoms']
            np.testing.assert_array_equal(atoms, report['selected']['atoms'])
            provenance[str(source)], provenance[str(path)] = file_hash(source), file_hash(path)
            raw = torch.tensor(zero[atoms], dtype=torch.float64, device=device)
            observed = mock.measurements.iloc[row][list(flow.target_transform.target_names)].to_numpy(float)
            physical = torch.tensor(np.broadcast_to(observed, (len(atoms), 4)).copy(), device=device)
            b = torch.tensor(data['coefficients'], device=device)
            lp = torch.tensor(np.log(probability[1, atoms])-np.log(len(zero)), device=device)

            def atom_terms(g, arm='full'):
                cp = center if arm == 'freeze_context' else g
                tp = center if arm == 'freeze_transport' else g
                context = shear_context64(raw, cp, flow.condition_preprocessor)
                values = transported_physical_log_prob(physical, b[:, None]*tp,
                    lambda base: flow.model.log_prob((base-means)/scales, context))
                return values+lp

            def objective(g, arm='full'):
                return torch.logsumexp(atom_terms(g, arm), 0)

            arms = {}
            for arm in ('full', 'freeze_context', 'freeze_transport'):
                def fn(g):
                    return objective(g, arm)
                reference = derivatives(fn, center)
                finite = []
                with torch.no_grad():
                    for h in (.002, .001, .0005, .00025, .000125, .0000625, .00003125):
                        values = [fn(torch.tensor(p, dtype=torch.float64, device=device)).item()
                                  for p in stencil(center.cpu().tolist(), h)[1:]]
                        score, info = stencil_moments(values, h)
                        finite.append(dict(h=h, score=score.tolist(), information=info.tolist(),
                            max_information_error=float(np.max(np.abs(info-np.array(reference['information']))))))
                arms[arm] = dict(autodiff=reference, finite_differences=finite)
            dominant = np.unique(np.concatenate([np.argsort(data['log_terms'][i])[-3:] for i in (0, 1, 6)]))
            components = []
            for index in dominant:
                components.append(dict(atom=int(atoms[index]), context=dict(zip(
                    flow.condition_preprocessor.feature_names, raw[index].cpu().tolist())),
                    coefficient=float(b[index]),
                    autodiff=derivatives(lambda g: atom_terms(g)[index], center)))
            result = dict(row=row, measured_magnitude=float(30-2.5*np.log10(observed[3])),
                measured_radius_arcsec=float(.2*observed[2]), arms=arms, dominant_components=components)
            rows.append(result)
            print('DERIVATIVE_REFERENCE', json.dumps(result), flush=True)
    write_json(args.output, dict(status='diagnostic_only', rows=rows, provenance=provenance,
        script_sha256=file_hash(__file__),
        limitations='Fixed truncated atom union; classifier frozen at center; no selection normalizer. FP64 evaluation of stored FP32 neural weights with FP64 input shear. Does not validate the full likelihood or the population.'))


if __name__ == '__main__':
    main()
