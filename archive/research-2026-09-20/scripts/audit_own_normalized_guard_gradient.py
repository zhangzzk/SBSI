"""Read-only train-population audit of own-normalized response gradients."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from sbsi.measurement_model import load_measurement_model
from scripts import audit_per_leg_guard_gradient as loader
from scripts import own_normalized_guard as own
from scripts.audit_guard_control_variate import gradient_summary
from scripts.audit_joint_guard_probability_gradient import gradient
from scripts.guard_selection_mass import primary_cut_index
from scripts.selection_probability_gradient import common_descent, sampled_soft_mass_backward
from scripts.train_fixed_g0_flow import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('domain-root', 'flow', 'target-bank', 'output-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or (args.device == 'cpu' and not args.smoke):
        raise RuntimeError('scheduled GPU required for full audit; CPU allowed for smoke')
    manifest_path = args.domain_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    target = json.loads(args.target_bank.read_text())
    if (target['split'] != 'train' or target['cases'] != manifest['split']['train_cases']
            or target['domain_manifest_sha256'] != sha256(manifest_path)
            or target['script_sha256'] != sha256(loader.PerLegGuardResponsePopulation.__init__.__code__.co_filename)):
        raise ValueError('matching full training target bank required')
    args.output_root.mkdir(parents=True, exist_ok=False)
    bundle = load_measurement_model(args.flow, device=torch.device(args.device))
    model = bundle.model.eval()
    parameters = [p.detach().clone() for p in model.parameters()]
    cases = manifest['split']['train_cases'][:2] if args.smoke else manifest['split']['train_cases']
    population, provenance = loader.load_population(args.domain_root, bundle, cases, device=args.device, smoke=args.smoke)
    n = len(population.pairs)
    if not args.smoke:
        np.testing.assert_allclose(population.target.cpu(), target['per_leg']['target'], rtol=0, atol=2e-12)
        if n != target['per_leg']['truth_parent_rows']:
            raise ValueError('training target parent count mismatch')
    batch, rows, draws, replicas = (128, 128, 4, 2) if args.smoke else (2048, 65536, 8, 4)
    source_paths = [Path(__file__), Path(own.__file__), Path(loader.__file__),
                    Path('scripts/per_leg_guard_response.py'), Path('scripts/selection_probability_gradient.py'),
                    Path('sbsi/flow_guard_response.py'), Path('scripts/audit_harmonized_usability_response.py')]
    protocol = {'smoke': args.smoke, 'optimizer_steps': 0, 'flow_sha256': sha256(args.flow),
                'domain_manifest_sha256': sha256(manifest_path), 'target_bank_sha256': sha256(args.target_bank),
                'cases': cases, 'parents': n, 'cuts': population.cuts, 'reference_groups': 4, 'reference_draws': 4,
                'reference_batch_size': batch, 'gradient_rows': rows, 'gradient_draws': draws,
                'gradient_replicas': replicas, 'gradient_batch_size': min(batch, 1024),
                'reference_seed': 2026091819300000, 'gradient_seed': 2026091819400000,
                'source_sha256': {str(path): sha256(path) for path in source_paths}, 'provenance': provenance,
                'objective': 'own-selected-mean response and separate primary probability; no calibration inputs',
                'decision': 'read-only gradient and integration stability; no optimizer or checkpoint selection'}
    write_json(args.output_root/'protocol.json', protocol)
    groups = []
    with torch.no_grad():
        for group in range(4):
            total = torch.zeros((len(population.cuts), 9), dtype=torch.float64, device=args.device)
            for block, start in enumerate(range(0, n, batch)):
                indices = torch.arange(start, min(start+batch, n), device=args.device)
                total += own.moment_contributions(model, population, indices, draws=4,
                          seed=protocol['reference_seed']+group*1000000000+block).sum(0)
                if block % 1000 == 0:
                    print(f'OWN_GUARD_REFERENCE group={group} parents={min(start+batch,n)}/{n}', flush=True)
            groups.append((total/n).cpu().numpy())
    groups = np.asarray(groups)
    responses = np.stack([own.response_from_moments(torch.as_tensor(x)).numpy() for x in groups])
    ki = primary_cut_index(population)
    mass_residual = float(groups[:, ki, 0].mean()-population.baseline_mass[ki])
    if not np.isfinite(mass_residual) or mass_residual == 0:
        raise ValueError('finite nonzero probability residual needed for common-descent audit')
    banks = [groups[2*k:2*k+2].mean(0) for k in (0, 1)]
    write_json(args.output_root/'reference.json', {'group_moments': groups,
        'group_responses': responses, 'mean_group_response': responses.mean(0),
        'response_of_mean_moments': own.response_from_moments(torch.as_tensor(groups.mean(0))).numpy(),
        'integration_group_se': responses.std(0, ddof=1)/2, 'target': population.target.cpu().numpy(),
        'primary_soft_mass_residual': mass_residual, 'primary_cut_index': ki,
        'bank_losses': [own.loss_cotangent(torch.as_tensor(x, device=args.device), population.target,
                                         population.response_scale)[0] for x in banks]})
    print('OWN_GUARD_REFERENCE_COMPLETE', 'primary_mass_residual', mass_residual, flush=True)
    gradients = {'probability': [], 'response_bank0': [], 'response_bank1': []}
    for replica in range(replicas):
        seed = protocol['gradient_seed']+replica*1000003
        indices = np.random.default_rng(seed).integers(n, size=rows)
        model.zero_grad(set_to_none=True)
        sampled_soft_mass_backward(model, population, indices, draws=draws, batch_size=min(batch, 1024), seed=seed)
        gradients['probability'].append(gradient(model)*np.sign(mass_residual))
        for bank in (0, 1):
            model.zero_grad(set_to_none=True)
            own.sampled_moment_backward(model, population, indices, banks[bank], draws=draws,
                                        batch_size=min(batch, 1024), seed=seed)
            gradients[f'response_bank{bank}'].append(gradient(model))
        print('OWN_GUARD_GRADIENT_REPLICA', replica, flush=True)
    gradients = {key: np.asarray(value) for key, value in gradients.items()}
    diagnostics, directions = {}, {}
    for bank in (0, 1):
        direction, report = common_descent(gradients[f'response_bank{bank}'].mean(0), gradients['probability'].mean(0))
        diagnostics[f'bank{bank}'] = report
        if direction is not None:
            directions[f'bank{bank}'] = direction
    halves = []
    for bank, indices in ((0, slice(0, replicas//2)), (1, slice(replicas//2, None))):
        direction, report = common_descent(gradients[f'response_bank{bank}'][indices].mean(0),
                                            gradients['probability'][indices].mean(0))
        diagnostics[f'independent_half{bank}'] = report
        if direction is not None:
            halves.append(direction)
    if len(halves) == 2:
        a, b = halves
        diagnostics['independent_half_direction_cosine'] = float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    for path, expected in protocol['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('audit source changed during execution')
    if (any(not torch.equal(before, after) for before, after in zip(parameters, model.parameters()))
            or sha256(args.flow) != protocol['flow_sha256']):
        raise ValueError('read-only audit changed model')
    np.savez_compressed(args.output_root/'gradients.npz', **gradients)
    np.savez_compressed(args.output_root/'directions.npz', **directions)
    write_json(args.output_root/'result.json', {'smoke': args.smoke, 'optimizer_steps': 0,
        'primary_soft_mass_residual': mass_residual, 'direction_diagnostics': diagnostics,
        'gradient_summaries': {key: gradient_summary(value) for key, value in gradients.items()},
        'protocol_sha256': sha256(args.output_root/'protocol.json'), 'calibration_claim': False,
        'limitations': ['finite reference integration makes a plug-in ratio and cotangent',
                       'no finite parameter update, NLL test, or hard probability guarantee',
                       'soft forward0.05 training response is not coherent joint acceptance',
                       'operational subset only' if args.smoke else 'full training references and sampled gradients']})
    print('OWN_GUARD_AUDIT_COMPLETE', diagnostics, flush=True)


if __name__ == '__main__':
    main()
