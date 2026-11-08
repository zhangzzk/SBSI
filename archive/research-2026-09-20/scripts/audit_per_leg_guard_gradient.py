"""Measure corrected-guard gradient stability before optimizing a single flow.

Uses the complete training parent and its own-leg usability, with four frozen
reference integrations and two independent reference banks. No model updates.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import make_guard_cuts, sampled_guard_response_backward
from sbsi.measurement_model import load_measurement_model
from scripts.audit_guard_control_variate import gradient_summary
from scripts.audit_harmonized_usability_response import load_aligned
from scripts.per_leg_guard_response import PerLegGuardResponsePopulation
from scripts.train_fixed_g0_flow import sha256, write_json
from scripts.train_full_domain_guard_flow import MAGNITUDE_THRESHOLDS, RADIUS_THRESHOLDS


def load_population(root, bundle, cases, *, device, smoke=False):
    """Load full parent contexts; no both-usable intersection is taken."""
    manifest = json.loads((root/'manifest.json').read_text())
    arrays = {name: [] for name in ('x0', 'xg', 'y0', 'yg', 'gamma', 'u0', 'ug')}
    provenance = []
    for case in cases:
        legs, sources = load_aligned(root, manifest, case)
        rows = np.arange(len(legs['g0']['usable']))
        if smoke:
            rows = np.sort(np.random.default_rng(2026091810+case).choice(rows, min(512, len(rows)), replace=False))
        for leg, xkey, ykey, ukey in (('g0', 'x0', 'y0', 'u0'), ('g05', 'xg', 'yg', 'ug')):
            path = root/f'usability/{leg}/case{case:03d}.npz'
            if sha256(path) != sources[leg]['usability_sha256']:
                raise ValueError('context changed during loading')
            with np.load(path) as data:
                np.testing.assert_array_equal(data['input_index'], legs[leg]['input_index'])
                frame = pd.DataFrame(data['context'][rows], columns=FLOW_FEATURES)
            arrays[xkey].append(bundle.condition_preprocessor.transform_frame(frame).astype(np.float32))
            arrays[ykey].append(legs[leg]['values'][rows])
            arrays[ukey].append(legs[leg]['usable'][rows])
        arrays['gamma'].append(legs['g05']['gamma'][rows])
        provenance.append({'case': case, 'parents': len(rows), 'sources': sources})
        print(f'PER_LEG_CV_LOADED case={case} parents={len(rows)}', flush=True)
    combined = {key: np.concatenate(value) for key, value in arrays.items()}
    del arrays
    contexts = [torch.as_tensor(combined.pop(key), dtype=torch.float32, device=device) for key in ('x0', 'xg')]
    population = PerLegGuardResponsePopulation(*contexts, combined['y0'], combined['yg'], combined['gamma'],
        combined['u0'], combined['ug'], make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS),
        radius_softness=.05, magnitude_softness=.05, response_scale=.01)
    return population, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-root', type=Path, required=True)
    parser.add_argument('--flow', type=Path, required=True)
    parser.add_argument('--target-bank', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('gradient integration requires Slurm')
    if args.device == 'cpu' and not args.smoke:
        raise ValueError('CPU is reserved for operational smoke')
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = args.domain_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    bank = json.loads(args.target_bank.read_text())
    if (bank['split'] != 'train' or bank['cases'] != manifest['split']['train_cases']
            or bank['domain_manifest_sha256'] != sha256(manifest_path)
            or bank['script_sha256'] != sha256(PerLegGuardResponsePopulation.__init__.__code__.co_filename)):
        raise ValueError('matching full training target bank required')
    flow_hash = sha256(args.flow)
    bundle = load_measurement_model(args.flow, device=torch.device(args.device))
    model = bundle.model.eval()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    cases = bank['cases'][:2] if args.smoke else bank['cases']
    population, provenance = load_population(args.domain_root, bundle, cases, device=args.device, smoke=args.smoke)
    if not args.smoke:
        np.testing.assert_allclose(population.target.cpu(), bank['per_leg']['target'], rtol=0, atol=2e-12)
        if len(population.pairs) != bank['per_leg']['truth_parent_rows']:
            raise ValueError('target bank parent count differs')
    n = len(population.pairs)
    batch, pairs, draws, replicas = (128, 128, 4, 2) if args.smoke else (2048, 65536, 8, 8)
    protocol = {'smoke': args.smoke, 'optimizer_steps': 0, 'flow_sha256': flow_hash,
        'domain_manifest_sha256': sha256(manifest_path), 'target_bank_sha256': sha256(args.target_bank),
        'training_cases': cases, 'pairs': n, 'population': population.statistics['population'],
        'reference_groups': 4, 'reference_draws': 4, 'reference_batch_size': batch,
        'reference_seed': 2026091811000000, 'gradient_seed': 2026091812000000,
        'gradient_pairs_per_group': pairs, 'gradient_draws': draws, 'gradient_replicates': replicas,
        'gradient_chunk_size': min(batch, 1024), 'cuts': population.cuts,
        'response_scale': .01, 'radius_softness': .05, 'magnitude_softness': .05,
        'script_sha256': sha256(__file__), 'guard_adapter_sha256': bank['script_sha256'],
        'gradient_helper_sha256': sha256(sampled_guard_response_backward.__code__.co_filename),
        'label_helper_sha256': sha256(load_aligned.__code__.co_filename), 'provenance': provenance}
    write_json(args.output_root/'protocol.json', protocol)
    means = []
    with torch.no_grad():
        for group in range(4):
            total = torch.zeros_like(population.target)
            for block, start in enumerate(range(0, n, batch)):
                ids = torch.arange(start, min(start+batch, n), device=args.device)
                total += population.contributions(reference, ids, 4,
                    protocol['reference_seed']+group*1000000000+block).sum(0)
                if block % 1000 == 0:
                    print(f'PER_LEG_CV_REFERENCE group={group} parents={min(start+batch,n)}/{n}', flush=True)
            means.append((total/n).cpu().numpy())
    means = np.asarray(means)
    write_json(args.output_root/'reference.json', {'group_means': means, 'mean_response': means.mean(0),
        'integration_se_of_four_group_mean': means.std(0, ddof=1)/2,
        'target': population.target.cpu().numpy(), 'usable_counts': population.statistics['usable_counts']})
    all_gradients, results = {}, {}
    for arm in ('independent', 'reference_bank0', 'reference_bank1'):
        gradients, records = [], []
        options = {} if arm == 'independent' else {'reference_model': reference,
            'reference_group_means': means[:2] if arm.endswith('0') else means[2:]}
        for replica in range(replicas):
            model.zero_grad(set_to_none=True)
            result = sampled_guard_response_backward(model, population, pairs, draws,
                chunk_size=protocol['gradient_chunk_size'], seed=protocol['gradient_seed']+replica*1000003,
                independent_pairs=True, **options)
            gradient = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                for p in model.parameters()]).detach().cpu().numpy()
            if not np.isfinite(gradient).all():
                raise ValueError('nonfinite gradient')
            gradients.append(gradient)
            records.append(result)
            print(f'PER_LEG_CV_GRADIENT arm={arm} replica={replica} loss={result["loss"]:.8g}', flush=True)
        all_gradients[arm] = np.asarray(gradients)
        results[arm] = {'gradient_summary': gradient_summary(gradients), 'records': records}
    left, right = (all_gradients[arm].mean(0) for arm in ('reference_bank0', 'reference_bank1'))
    denominator = float(np.linalg.norm(left)*np.linalg.norm(right))
    cosine = float(np.dot(left, right)/denominator) if denominator else None
    np.savez_compressed(args.output_root/'gradients.npz', **all_gradients)
    if flow_hash != sha256(args.flow):
        raise ValueError('frozen flow changed')
    write_json(args.output_root/'result.json', {'smoke': args.smoke, 'optimizer_steps': 0, 'arms': results,
        'reference_bank_agreement': {'reference_bank_mean_gradient_cosine': cosine,
            'reference_bank_mean_gradient_distance': float(np.linalg.norm(left-right))},
        'protocol_sha256': sha256(args.output_root/'protocol.json'),
        'limitations': ['finite reference integration error; not fitted offsets',
            'training-label forward0.05 response, not coherent joint calibration',
            'operational subset only' if args.smoke else 'complete training parent',
            'no optimizer update, checkpoint selection or acceptance claim']})
    print(f'PER_LEG_CV_COMPLETE reference_bank_cosine={cosine}', flush=True)


if __name__ == '__main__':
    main()
