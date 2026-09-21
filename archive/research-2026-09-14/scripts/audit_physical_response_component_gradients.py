#!/usr/bin/env python3
"""Audit shape/radius parameter-gradient balance for the marginal 6D loss."""

from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch

from sbsi.flow_grid_accumulation import response_plan
from sbsi.flow_physical_population import PhysicalResponsePopulation
from sbsi.flow_physical_response import (marginal_physical_response_statistics,
                                         physical_outputs)
from sbsi.measurement_model import TargetStandardizer, build_flow


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def flattened_gradient(model):
    return np.concatenate([
        (np.zeros(parameter.numel()) if parameter.grad is None
         else parameter.grad.detach().reshape(-1).cpu().numpy().astype(np.float64))
        for parameter in model.parameters()
    ])


def vector_statistics(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    mean = vectors.mean(axis=0)
    noise_squared = np.var(vectors, axis=0, ddof=1).sum()
    signal_squared = np.square(mean).sum()
    return {
        'mean_norm': float(np.sqrt(signal_squared)),
        'noise_rms': float(np.sqrt(noise_squared)),
        'noise_to_mean': float(np.sqrt(noise_squared/signal_squared)),
        'finite_rep_corrected_signal_squared': float(signal_squared-noise_squared/len(vectors)),
    }


def block_gradients(basis_mean, radius_weight):
    """Return output-path shape/radius gradients for sqrt(W) P sqrt(W)."""
    root = math.sqrt(radius_weight)
    shape = basis_mean[0] + root*basis_mean[1]
    radius = root*basis_mean[2] + radius_weight*basis_mean[3]
    return shape, radius


def balance_weight(basis_mean):
    """Solve ||g_shape(w)|| = ||g_radius(w)|| on a broad positive bracket."""
    def log_ratio(log_weight):
        shape, radius = block_gradients(basis_mean, math.exp(log_weight))
        shape_norm, radius_norm = np.linalg.norm(shape), np.linalg.norm(radius)
        if shape_norm <= 0 or radius_norm <= 0:
            raise ValueError('zero block gradient prevents a balance solution')
        return math.log(shape_norm/radius_norm)

    grid = np.linspace(math.log(1e-8), math.log(1e4), 241)
    values = np.asarray([log_ratio(point) for point in grid])
    crossings = np.flatnonzero(values[:-1]*values[1:] <= 0)
    if len(crossings) != 1:
        raise ValueError(f'expected one shape/radius balance crossing, found {len(crossings)}')
    lower, upper = grid[crossings[0]], grid[crossings[0]+1]
    for _ in range(80):
        middle = .5*(lower+upper)
        if log_ratio(lower)*log_ratio(middle) <= 0:
            upper = middle
        else:
            lower = middle
    return math.exp(.5*(lower+upper))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs-per-cell', type=int, default=359)
    parser.add_argument('--draws', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--nll-rows', type=int, default=131072)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1900000000001)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if (min(args.pairs_per_cell, args.draws, args.nll_rows,
            args.chunk_size) < 1 or args.repeats < 3 or args.seed < 0):
        raise ValueError('positive audit sizes and a nonnegative seed required')

    input_hashes = {str(path): sha256(path) for path in
                    (args.checkpoint, args.prepared, args.targets, args.raw)}
    if args.preflight:
        print(json.dumps({'input_hashes': input_hashes}, indent=2))
        print('PHYSICAL_COMPONENT_GRADIENT_PREFLIGHT_COMPLETE')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    transform = TargetStandardizer.from_state(checkpoint['target_transform'])
    expected_names = ['measured_ngmix_g1', 'measured_ngmix_g2',
                      'measured_flux_radius', 'measured_flux_from_mag_auto']
    if transform.target_names != expected_names:
        raise ValueError('checkpoint does not use the four physical output contract')
    model = build_flow(checkpoint['model_config']).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    data = torch.load(args.prepared, map_location='cpu', weights_only=False)
    context = data['context'].cuda()
    full_statistics = dict(np.load(args.targets))
    statistics = marginal_physical_response_statistics(full_statistics)
    population = PhysicalResponsePopulation(context, data['pairs'], data['bin_ids'], statistics)
    raw = np.load(args.raw, mmap_mode='r')
    if len(raw) != len(context):
        raise ValueError('raw NLL rows and prepared contexts are not aligned')

    names = ('shape_shape', 'shape_radius', 'radius_shape', 'radius_radius')
    all_basis, all_nll, records = [], [], []
    components = 6
    for repeat in range(args.repeats):
        started = time.monotonic()
        seed = args.seed + repeat*1000000000
        cells, local, groups = response_plan(
            population, population.n_cells, args.pairs_per_cell, seed)
        ids = torch.as_tensor(cells, device=context.device)
        divisor = torch.bincount(local, minlength=len(cells)).to(torch.float64)

        def chunks(stream):
            for block, start in enumerate(range(0, len(local), args.chunk_size)):
                stop = min(start+args.chunk_size, len(local))
                yield (groups[stream][start:stop], local[start:stop],
                       seed+100000+2*block+stream)

        means = []
        with torch.no_grad():
            for stream in range(2):
                total = torch.zeros((len(cells), components), device=context.device,
                                    dtype=torch.float64)
                for rows, bins, latent_seed in chunks(stream):
                    values = population.contributions(model, rows, args.draws, seed=latent_seed)
                    total.index_add_(0, bins, values)
                means.append(total/divisor[:, None])
            residuals = [means[0]-population.target[ids], means[1]-population.target[ids]]
            precision = population.base_precision[ids]
            coefficient_basis = [[], []]
            for stream, residual in enumerate((residuals[1], residuals[0])):
                factor = .5/(components*len(cells))
                blocks = [
                    torch.cat((torch.einsum('bij,bj->bi', precision[:, :4, :4], residual[:, :4]),
                               torch.zeros_like(residual[:, 4:])), dim=1),
                    torch.cat((torch.einsum('bij,bj->bi', precision[:, :4, 4:], residual[:, 4:]),
                               torch.zeros_like(residual[:, 4:])), dim=1),
                    torch.cat((torch.zeros_like(residual[:, :4]),
                               torch.einsum('bij,bj->bi', precision[:, 4:, :4], residual[:, :4])), dim=1),
                    torch.cat((torch.zeros_like(residual[:, :4]),
                               torch.einsum('bij,bj->bi', precision[:, 4:, 4:], residual[:, 4:])), dim=1),
                ]
                coefficient_basis[stream] = [factor*block for block in blocks]

        gradients = []
        for basis_index in range(4):
            model.zero_grad(set_to_none=True)
            for stream in range(2):
                for rows, bins, latent_seed in chunks(stream):
                    values = population.contributions(model, rows, args.draws, seed=latent_seed)
                    surrogate = (values*coefficient_basis[stream][basis_index][bins]
                                 /divisor[bins, None]).sum()
                    if not bool(torch.isfinite(surrogate)):
                        raise ValueError('nonfinite component-gradient replay')
                    surrogate.backward()
            gradient = flattened_gradient(model)
            if not np.isfinite(gradient).all():
                raise ValueError('nonfinite response component gradient')
            gradients.append(gradient)
        all_basis.append(gradients)

        nll_ids = np.random.default_rng(seed).choice(len(raw), args.nll_rows, replace=False)
        values = torch.as_tensor(transform.transform_array(physical_outputs(raw[nll_ids])),
                                 device=context.device)
        nll_indices = torch.as_tensor(nll_ids, device=context.device)
        model.zero_grad(set_to_none=True)
        nll = 0.
        for start in range(0, len(nll_ids), 4096):
            stop = min(start+4096, len(nll_ids))
            loss = -model.log_prob(values[start:stop], context[nll_indices[start:stop]]).sum()/len(nll_ids)
            loss.backward()
            nll += float(loss.detach())
        nll_gradient = flattened_gradient(model)
        if not np.isfinite(nll_gradient).all():
            raise ValueError('nonfinite NLL gradient')
        all_nll.append(nll_gradient)
        records.append({'repeat': repeat, 'seed': seed, 'nll': nll,
                        'basis_gradient_norms': {name: float(np.linalg.norm(value))
                                                 for name, value in zip(names, gradients)},
                        'nll_gradient_norm': float(np.linalg.norm(nll_gradient)),
                        'seconds': time.monotonic()-started})
        print('REPEAT_COMPLETE', repeat, records[-1], flush=True)

    all_basis = np.asarray(all_basis)
    all_nll = np.asarray(all_nll)
    basis_mean = all_basis.mean(axis=0)
    selected_weight = balance_weight(basis_mean)
    leave_one_out = [balance_weight(np.delete(all_basis, repeat, axis=0).mean(axis=0))
                     for repeat in range(args.repeats)]
    comparisons = {}
    for label, weight in [('proposed', .2), ('selected', selected_weight)]:
        shape, radius = block_gradients(basis_mean, weight)
        total = shape+radius
        comparisons[label] = {
            'radius_relative_weight': weight,
            'shape_gradient_norm': float(np.linalg.norm(shape)),
            'radius_gradient_norm': float(np.linalg.norm(radius)),
            'shape_to_radius_norm_ratio': float(np.linalg.norm(shape)/np.linalg.norm(radius)),
            'total_response_gradient_norm': float(np.linalg.norm(total)),
            'cosine_shape_radius': float(np.dot(shape, radius)/np.linalg.norm(shape)/np.linalg.norm(radius)),
            'shape_lambda_if_shape_is_0.1': .1,
            'radius_lambda_if_shape_is_0.1': .1*weight,
        }
    summary = {
        'definition': ('equal Euclidean norms of the shape-output and radius-output parameter-gradient '
                       'paths under sqrt(W) inverse(V_shape_radius) sqrt(W)'),
        'response_components': ['e1_g1', 'e1_g2', 'e2_g1', 'e2_g2',
                                'radius_parallel', 'radius_cross'],
        'flux_in_response_objective': False,
        'marginal_precision_identity_max_abs': float(np.max(np.abs(
            np.einsum('bij,bjk->bik', statistics['precision'], statistics['covariance'])
            -np.eye(components)))),
        'input_hashes': input_hashes,
        'recipe': vars(args) | {'checkpoint': str(args.checkpoint), 'prepared': str(args.prepared),
                                'targets': str(args.targets), 'raw': str(args.raw),
                                'output': str(args.output)},
        'basis_gradient_statistics': {name: vector_statistics(all_basis[:, index])
                                      for index, name in enumerate(names)},
        'nll_gradient_statistics': vector_statistics(all_nll),
        'selected_radius_relative_weight': selected_weight,
        'leave_one_repeat_out_weights': leave_one_out,
        'comparisons': comparisons,
        'records': records,
    }
    np.savez(args.output/'gradients.npz', basis_gradients=all_basis,
             nll_gradients=all_nll, basis_names=np.asarray(names))
    (args.output/'results.json').write_text(json.dumps(summary, indent=2))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), initial[key], rtol=0, atol=0)
    if input_hashes != {str(path): sha256(path) for path in
                        (args.checkpoint, args.prepared, args.targets, args.raw)}:
        raise ValueError('an audit input changed during execution')
    print('PHYSICAL_COMPONENT_GRADIENT_AUDIT_COMPLETE', selected_weight, flush=True)


if __name__ == '__main__':
    main()
