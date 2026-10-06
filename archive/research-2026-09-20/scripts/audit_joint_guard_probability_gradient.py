"""Read-only feasibility of reducing guard and primary probability error together."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from sbsi.flow_guard_response import sampled_guard_response_backward
from sbsi.measurement_model import load_measurement_model
from scripts import audit_per_leg_guard_gradient as audit
from scripts import guard_selection_mass as mass_helper
from scripts import selection_probability_gradient as gradient_helper
from scripts.audit_guard_control_variate import gradient_summary
from scripts.train_fixed_g0_flow import sha256, write_json


def gradient(model):
    result = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                        for p in model.parameters()]).detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('nonfinite gradient')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('domain-root', 'flow', 'reference-root', 'output-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or (args.device == 'cpu' and not args.smoke):
        raise RuntimeError('full audit requires a scheduled GPU; CPU reserved for smoke')
    reference_protocol = json.loads((args.reference_root/'protocol.json').read_text())
    reference = json.loads((args.reference_root/'reference.json').read_text())
    manifest = json.loads((args.domain_root/'manifest.json').read_text())
    if (reference_protocol['smoke'] != args.smoke or reference_protocol['flow_sha256'] != sha256(args.flow)
            or reference_protocol['domain_manifest_sha256'] != sha256(args.domain_root/'manifest.json')
            or reference_protocol['script_sha256'] != sha256(audit.__file__)):
        raise ValueError('matching per-leg frozen reference required')
    cases = manifest['split']['train_cases'][:2] if args.smoke else manifest['split']['train_cases']
    if reference_protocol['training_cases'] != cases:
        raise ValueError('reference split differs')
    args.output_root.mkdir(parents=True, exist_ok=False)
    bundle = load_measurement_model(args.flow, device=torch.device(args.device))
    model = bundle.model.eval()
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    population, provenance = audit.load_population(args.domain_root, bundle, cases, device=args.device, smoke=args.smoke)
    means = np.asarray(reference['group_means'])
    np.testing.assert_allclose(reference['target'], population.target.cpu(), rtol=0, atol=2e-12)
    if reference_protocol['pairs'] != len(population.pairs) or means.shape != (4, *population.target.shape):
        raise ValueError('full parent reference differs')
    replicas, rows, draws, batch = (2, 128, 4, 128) if args.smoke else (4, 65536, 8, 1024)
    protocol = {'script_sha256': sha256(__file__), 'mass_helper_sha256': sha256(mass_helper.__file__),
                'gradient_helper_sha256': sha256(gradient_helper.__file__), 'smoke': args.smoke,
                'flow_sha256': sha256(args.flow), 'optimizer_steps': 0, 'cases': cases,
                'reference_protocol_sha256': sha256(args.reference_root/'protocol.json'),
                'reference_integral_sha256': sha256(args.reference_root/'reference.json'),
                'replicas': replicas, 'rows_per_replica': rows, 'draws': draws,
                'mass_reference_seed': 2026091818100000, 'gradient_seed': 2026091818200000,
                'provenance': provenance, 'population': 'all training truth parents, own zero-leg U',
                'decision': 'gradient feasibility only; no checkpoint update or ConstGold inputs'}
    write_json(args.output_root/'protocol.json', protocol)
    mass = mass_helper.estimate_primary_mass(model, population, seed=protocol['mass_reference_seed'],
                                             draws=4, groups=2, batch_size=2*batch)
    target = float(population.baseline_mass[mass_helper.primary_cut_index(population)])
    residual = mass['estimate'][0]-target
    if not np.isfinite(residual) or residual == 0:
        raise ValueError('nonzero finite soft mass residual required for signed error gradient')
    write_json(args.output_root/'mass_reference.json', dict(mass, soft_target=target, soft_residual=residual))
    print('JOINT_GRADIENT_MASS_REFERENCE', residual, flush=True)
    gradients = {'mass': [], 'response_bank0': [], 'response_bank1': []}
    for replica in range(replicas):
        seed = protocol['gradient_seed']+replica*1000003
        ids = np.random.default_rng(seed).integers(len(population.pairs), size=rows)
        model.zero_grad(set_to_none=True)
        gradient_helper.sampled_soft_mass_backward(model, population, ids, draws=draws, batch_size=batch, seed=seed)
        gradients['mass'].append(gradient(model)*np.sign(residual))
        for bank in (0, 1):
            model.zero_grad(set_to_none=True)
            sampled_guard_response_backward(model, population, rows, draws, chunk_size=batch,
                seed=seed, independent_pairs=True, reference_model=frozen,
                reference_group_means=means[2*bank:2*bank+2])
            gradients[f'response_bank{bank}'].append(gradient(model))
        print('JOINT_GRADIENT_REPLICA', replica, flush=True)
    gradients = {key: np.asarray(value) for key, value in gradients.items()}
    probability = gradients['mass'].mean(0)
    directions, diagnostics = {}, {}
    for bank in (0, 1):
        name = f'response_bank{bank}'
        direction, report = gradient_helper.common_descent(gradients[name].mean(0), probability)
        diagnostics[name] = report
        if direction is not None:
            directions[name] = direction
    if len(directions) == 2:
        left, right = directions.values()
        diagnostics['common_direction_bank_cosine'] = float(left@right/(np.linalg.norm(left)*np.linalg.norm(right)))
    halves = []
    for bank, take in ((0, slice(0, replicas//2)), (1, slice(replicas//2, None))):
        direction, report = gradient_helper.common_descent(
            gradients[f'response_bank{bank}'][take].mean(0), gradients['mass'][take].mean(0))
        diagnostics[f'independent_half{bank}'] = report
        if direction is not None:
            halves.append(direction)
    if len(halves) == 2:
        left, right = halves
        diagnostics['independent_half_direction_cosine'] = float(left@right/(np.linalg.norm(left)*np.linalg.norm(right)))
    if any(not torch.equal(p, q) for p, q in zip(model.parameters(), frozen.parameters())) or sha256(args.flow) != protocol['flow_sha256']:
        raise ValueError('read-only audit changed model')
    np.savez_compressed(args.output_root/'gradients.npz', **gradients)
    np.savez_compressed(args.output_root/'directions.npz', **directions)
    write_json(args.output_root/'result.json', {'smoke': args.smoke, 'optimizer_steps': 0,
        'soft_mass_residual': residual, 'gradient_summaries': {key: gradient_summary(value) for key, value in gradients.items()},
        'direction_diagnostics': diagnostics, 'protocol_sha256': sha256(args.output_root/'protocol.json'),
        'calibration_claim': False, 'limitations': ['first-order feasibility only, no line search or density guarantee',
            'hard probability gate remains necessary; soft gradient is not the hard-cut derivative',
            'finite row and latent noise; shared mass-gradient estimate across response banks',
            'operational subset' if args.smoke else 'full training mass reference, sampled parameter gradients']})
    print('JOINT_GRADIENT_AUDIT_COMPLETE', diagnostics, flush=True)


if __name__ == '__main__':
    main()
