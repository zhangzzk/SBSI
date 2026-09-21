"""Density-constrained common descent for response and primary probability.

Reuses independent-row gradient replay and frozen model reference integrals.
Every selected checkpoint must pass full validation NLL, including step zero.
No ConstGold measurements or joint-response estimates are loaded.
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
from sbsi.flow_guard_response import (
    evaluate_guard_response, physical_context_draws, sampled_guard_response_backward, torch_guard_weights,
)
from sbsi.measurement_model import load_measurement_model
from scripts import audit_per_leg_guard_gradient as audit
from scripts import guard_selection_mass
from scripts import audit_joint_guard_probability_gradient as joint_audit
from scripts import selection_probability_gradient as probability_gradient
from scripts import train_fixed_g0_flow as base
from scripts.guard_selection_mass import estimate_primary_mass, primary_cut_index
from scripts.refine_guard_control_variate_flow import DENSITY_TOLERANCE, acceptable_step
from scripts.refine_per_leg_guard_flow import eligible_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('domain-root', 'flow', 'reference-root', 'target-bank-root', 'output-root', 'probability-audit-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('refinement requires Slurm')
    if args.device == 'cpu' and not args.smoke:
        raise ValueError('CPU is reserved for operational smoke')
    reference_protocol = json.loads((args.reference_root/'protocol.json').read_text())
    reference_result = json.loads((args.reference_root/'result.json').read_text())
    reference_integral = json.loads((args.reference_root/'reference.json').read_text())
    probability_protocol = json.loads((args.probability_audit_root/'protocol.json').read_text())
    probability_result = json.loads((args.probability_audit_root/'result.json').read_text())
    probability_reference = json.loads((args.probability_audit_root/'mass_reference.json').read_text())
    manifest_path = args.domain_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    bank_manifest_path = args.target_bank_root/'per_leg_guard_target_banks.json'
    banks = json.loads(bank_manifest_path.read_text())
    flow_hash, domain_hash = base.sha256(args.flow), base.sha256(manifest_path)
    if (probability_protocol['flow_sha256'] != flow_hash
            or probability_protocol['smoke'] != args.smoke or probability_result['smoke'] != args.smoke
            or probability_result['optimizer_steps'] != 0
            or probability_result['protocol_sha256'] != base.sha256(args.probability_audit_root/'protocol.json')
            or probability_protocol['reference_protocol_sha256'] != base.sha256(args.reference_root/'protocol.json')
            or probability_protocol['script_sha256'] != base.sha256(joint_audit.__file__)
            or probability_protocol['mass_helper_sha256'] != base.sha256(guard_selection_mass.__file__)
            or probability_protocol['gradient_helper_sha256'] != base.sha256(probability_gradient.__file__)):
        raise ValueError('matching completed joint-gradient audit required')
    stability = probability_result['direction_diagnostics']['independent_half_direction_cosine']
    if not args.smoke and (not np.isfinite(stability) or stability < .99):
        raise ValueError('independent gradient halves do not establish a stable common direction')
    identities = {'flow_sha256': flow_hash, 'domain_manifest_sha256': domain_hash,
        'script_sha256': base.sha256(audit.__file__),
        'guard_adapter_sha256': base.sha256(audit.PerLegGuardResponsePopulation.__init__.__code__.co_filename),
        'gradient_helper_sha256': base.sha256(sampled_guard_response_backward.__code__.co_filename),
        'label_helper_sha256': base.sha256(audit.load_aligned.__code__.co_filename)}
    if (any(reference_protocol[key] != value for key, value in identities.items())
            or reference_protocol['smoke'] != args.smoke or reference_result['smoke'] != args.smoke
            or reference_result['protocol_sha256'] != base.sha256(args.reference_root/'protocol.json')
            or banks['domain_manifest_sha256'] != domain_hash
            or banks['source_sha256'] != identities['guard_adapter_sha256']):
        raise ValueError('incompatible population-consistent reference and target banks')
    cosine = reference_result['reference_bank_agreement']['reference_bank_mean_gradient_cosine']
    if not args.smoke and (cosine is None or not np.isfinite(cosine) or cosine < .99):
        raise ValueError('independent full reference banks do not establish a stable gradient direction')
    if set(manifest['split']['train_cases']) & set(manifest['split']['validation_cases']):
        raise ValueError('training and validation cases overlap')
    target_banks = {}
    for split in ('train', 'validation'):
        name = f'per_leg_guard_{split}_targets.json'
        path = args.target_bank_root/name
        if base.sha256(path) != banks['files'][name]:
            raise ValueError('changed target bank')
        bank = json.loads(path.read_text())
        if bank['split'] != split or bank['cases'] != manifest['split'][f'{split}_cases']:
            raise ValueError('target bank split mismatch')
        target_banks[split] = bank
    if reference_protocol['target_bank_sha256'] != banks['files']['per_leg_guard_train_targets.json']:
        raise ValueError('reference used another training target bank')
    args.output_root.mkdir(parents=True, exist_ok=False)
    bundle = load_measurement_model(args.flow, device=torch.device(args.device))
    model = bundle.model.eval()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    checkpoint = torch.load(args.flow, map_location='cpu', weights_only=False)
    arrays = {}
    for split, seed in (('train', 753000102), ('validation', 754000102)):
        cases = manifest['split'][f'{split}_cases'][:2] if args.smoke else manifest['split'][f'{split}_cases']
        pop, _ = audit.load_population(args.domain_root, bundle, cases, device=args.device, smoke=args.smoke)
        if not args.smoke:
            np.testing.assert_allclose(pop.target.cpu(), target_banks[split]['per_leg']['target'], rtol=0, atol=2e-12)
        data = base.load_split(args.domain_root, cases)
        rows = np.arange(len(data['target_raw']))
        if args.smoke:
            rows = np.random.default_rng(seed).choice(rows, min(8192, len(rows)), replace=False)
        context = torch.as_tensor(bundle.condition_preprocessor.transform_frame(
            pd.DataFrame(data['context_raw'][rows], columns=FLOW_FEATURES)), dtype=torch.float32, device=args.device)
        values = torch.as_tensor(bundle.target_transform.transform_array(data['target_raw'][rows]),
                                 dtype=torch.float64, device=args.device)
        ids = np.random.default_rng(seed+1).choice(len(values), min(1024 if args.smoke else 500000, len(values)), replace=False)
        arrays[split] = {'population': pop, 'context': context, 'values': values,
                         'ids': torch.as_tensor(ids, dtype=torch.long, device=args.device)}
        del data
    train, validation = arrays['train']['population'], arrays['validation']['population']
    train_mass_target = float(train.baseline_mass[primary_cut_index(train)])
    if (probability_reference['parents'] != len(train.pairs)
            or abs(probability_reference['soft_target']-train_mass_target) > 2e-12):
        raise ValueError('training probability reference population differs')
    train_reference_mass = probability_reference['estimate'][0]
    label_audit_path = args.target_bank_root/'validation_usability_response.json'
    label_audit = json.loads(label_audit_path.read_text())
    if (label_audit['domain_manifest_sha256'] != domain_hash
            or label_audit['script_sha256'] != identities['label_helper_sha256']
            or label_audit['cases'] != manifest['split']['validation_cases']):
        raise ValueError('matching validation label audit required for exact-cut probability')
    if args.smoke:
        from sbsi.flow_guard_response import numpy_guard_weights
        count, hard_count, soft_sum = 0, 0, 0.
        for case in manifest['split']['validation_cases'][:2]:
            legs, _ = audit.load_aligned(args.domain_root, manifest, case)
            rows = np.arange(len(legs['g0']['usable']))
            rows = np.sort(np.random.default_rng(2026091810+case).choice(rows, min(512, len(rows)), replace=False))
            valid = legs['g0']['values'][rows][legs['g0']['usable'][rows]]
            hard_count += int(((valid[:, 2] > 3.) & (30.-2.5*np.log10(valid[:, 3]) < 25.8)).sum())
            soft_sum += numpy_guard_weights(valid, (validation.cuts[primary_cut_index(validation)],),
                radius_softness=.05, magnitude_softness=.05).sum()
            count += len(rows)
        np.testing.assert_allclose(soft_sum/count, float(validation.baseline_mass[primary_cut_index(validation)]), atol=2e-12, rtol=0)
        hard_target = hard_count/count
    else:
        for record, target_record in zip(label_audit['records'], target_banks['validation']['provenance']):
            if record['case'] != target_record['case'] or record['provenance'] != target_record['sources']:
                raise ValueError('exact-cut and soft-cut target source identities differ')
        raw = label_audit['per_case_sufficient']['hard/per_leg']
        count = sum(row['n'] for row in raw)
        hard_target = sum(row['mass_zero'][1] for row in raw)/count
    if count != len(validation.pairs):
        raise ValueError('exact-cut target parent count differs')
    selection_target = np.array([float(validation.baseline_mass[primary_cut_index(validation)]), hard_target])
    groups = np.asarray(reference_integral['group_means'])
    if (groups.shape != (4, *train.target.shape) or reference_protocol['pairs'] != len(train.pairs)
            or reference_protocol['cuts'] != list(train.cuts)):
        raise ValueError('reference integrals do not cover the training population')
    np.testing.assert_allclose(reference_integral['target'], train.target.cpu(), rtol=0, atol=2e-12)
    pairs, line_pairs, batch, draws = (128, 256, 128, 4) if args.smoke else (65536, 262144, 1024, 8)
    protocol = {'smoke': args.smoke, 'initial_flow_sha256': flow_hash, 'domain_manifest_sha256': domain_hash,
        'reference_protocol_sha256': base.sha256(args.reference_root/'protocol.json'),
        'reference_integral_sha256': base.sha256(args.reference_root/'reference.json'),
        'reference_variance_sha256': base.sha256(args.reference_root/'result.json'),
        'target_bank_manifest_sha256': base.sha256(bank_manifest_path),
        'density_tolerance_nats_per_row': DENSITY_TOLERANCE, 'steps': 1 if args.smoke else 6,
        'learning_rates': [1e-5/2**i for i in range(6)],
        'optimizer': 'common descent along angle bisector of response and signed probability gradients; response-gradient norm; backtracking',
        'probability_audit_protocol_sha256': base.sha256(args.probability_audit_root/'protocol.json'),
        'probability_audit_result_sha256': base.sha256(args.probability_audit_root/'result.json'),
        'probability_reference_sha256': base.sha256(args.probability_audit_root/'mass_reference.json'),
        'probability_gradient_helper_sha256': base.sha256(probability_gradient.__file__),
        'training_mass_target': train_mass_target, 'training_reference_mass': train_reference_mass,
        'mass_integration': 'frozen full model reference plus common-random-number sampled model difference; no measured offset',
        'gradient_pairs': pairs, 'gradient_draws': draws, 'gradient_seed': 756000000102,
        'line_search_pairs': line_pairs, 'line_search_draws': 4, 'line_search_seed': 757000000102,
        'validation_seed': 758000000102, 'validation_draws': 4, 'validation_groups': 2,
        'validation_mass_seed': 759000000102, 'validation_mass_draws': 4, 'validation_mass_groups': 2,
        'validation_selection_target': selection_target, 'label_audit_sha256': base.sha256(label_audit_path),
        'selection_mass_helper_sha256': base.sha256(guard_selection_mass.__file__),
        'nll_subset_seeds': [753000103, 754000103], 'script_sha256': base.sha256(__file__),
        'population_loader_sha256': identities['script_sha256'],
        'selection': 'minimum per-leg validation guard with full-NLL gate and no increase in either soft/hard primary selection-mass absolute error; step0 included',
        'likelihood_rows': {split: len(block['values']) for split, block in arrays.items()},
        'guard_parents': {split: len(block['population'].pairs) for split, block in arrays.items()}}
    base.write_json(args.output_root/'protocol.json', protocol)

    @torch.no_grad()
    def nll(split, full=False):
        block = arrays[split]
        ids = slice(None) if full else block['ids']
        return base.nll_pass(model, block['values'][ids], block['context'][ids])['nll']

    def training_score():
        return sampled_guard_response_backward(model, train, line_pairs, 4, chunk_size=batch,
            seed=protocol['line_search_seed'], weight=0, independent_pairs=True,
            reference_model=reference, reference_group_means=groups[:2])

    @torch.no_grad()
    def sampled_mass(evaluated_model, indices, count, seed):
        cut = train.cuts[primary_cut_index(train)]
        total = 0.
        for block, start in enumerate(range(0, len(indices), batch)):
            rows = torch.as_tensor(indices[start:start+batch], device=args.device)
            values = physical_context_draws(evaluated_model, [train.context[train.pairs[rows, 0]]], count, seed+block)[0]
            weights = torch_guard_weights(values, (cut,), radius_softness=train.radius_softness,
                magnitude_softness=train.magnitude_softness, zero_point=train.zero_point)[..., 0].mean(1)
            total += float((weights*train.usable[rows, 0]).double().sum())/len(indices)
        return total

    line_mass_seed = protocol['line_search_seed']+5000000000
    line_mass_rows = np.random.default_rng(line_mass_seed).integers(len(train.pairs), size=line_pairs)
    line_reference_mass = sampled_mass(reference, line_mass_rows, 4, line_mass_seed)

    def training_mass():
        return train_reference_mass+sampled_mass(model, line_mass_rows, 4, line_mass_seed)-line_reference_mass

    def validation_score():
        return evaluate_guard_response(model, validation, np.arange(len(validation.pairs)),
            draws=4, groups=2, batch_size=2*batch, seed=protocol['validation_seed'])

    def validation_mass():
        result = estimate_primary_mass(model, validation, draws=4, groups=2, batch_size=2*batch,
                                       seed=protocol['validation_mass_seed'])
        result['target'] = selection_target
        result['residual'] = result['estimate']-selection_target
        return result

    baseline_train, baseline_subset, baseline_full = nll('train'), nll('validation'), nll('validation', full=True)
    initial_guard = validation_score()
    initial_mass = validation_mass()
    best_loss, best_step, best_full = initial_guard['loss'], 0, baseline_full
    best_mass = initial_mass
    history = [{'step': 0, 'train_nll': baseline_train, 'subset_validation_nll': baseline_subset,
                'full_validation_nll': baseline_full, 'validation_guard': initial_guard,
                'validation_selection_mass': initial_mass}]

    def save(step, score, full_nll):
        base.save_checkpoint(args.output_root/'selected.pt', model, bundle.condition_preprocessor,
            bundle.target_transform, checkpoint['model_config'], {'training_phase': 'joint_guard_probability_common_descent',
                'step': step, 'smoke': args.smoke, 'validation_guard_loss': score,
                'full_validation_nll': full_nll, 'protocol_sha256': base.sha256(args.output_root/'protocol.json'),
                'population': manifest['population']})

    save(0, best_loss, best_full)
    print(f'JOINT_PROB_REFINE_BASELINE nll={baseline_full:.9f} guard={best_loss:.9f}', flush=True)
    for step in range(1, protocol['steps']+1):
        model.zero_grad(set_to_none=True)
        gradient = sampled_guard_response_backward(model, train, pairs, draws, chunk_size=batch,
            seed=protocol['gradient_seed']+step*1000003, independent_pairs=True,
            reference_model=reference, reference_group_means=groups[:2])
        response_gradient = joint_audit.gradient(model)
        probability_seed = protocol['gradient_seed']+step*1000003+5000000000
        probability_rows = np.random.default_rng(probability_seed).integers(len(train.pairs), size=pairs)
        reference_mass = sampled_mass(reference, probability_rows, draws, probability_seed)
        model.zero_grad(set_to_none=True)
        mass_estimate = probability_gradient.sampled_soft_mass_backward(model, train, probability_rows,
            draws=draws, batch_size=batch, seed=probability_seed)
        mass_residual = train_reference_mass+mass_estimate-reference_mass-train_mass_target
        signed_probability_gradient = joint_audit.gradient(model)*np.sign(mass_residual)
        direction, direction_report = probability_gradient.common_descent(response_gradient, signed_probability_gradient)
        if direction is None or not direction_report['strict_common_descent']:
            history.append({'step': step, 'accepted': False, 'reason': 'no strict common descent',
                            'direction': direction_report, 'training_mass_residual': mass_residual})
            break
        gradients, offset = {}, 0
        for name, p in model.named_parameters():
            gradients[name] = torch.as_tensor(-direction[offset:offset+p.numel()], device=p.device, dtype=p.dtype).reshape(p.shape)
            offset += p.numel()
        if offset != len(direction) or any(not bool(torch.isfinite(g).all()) for g in gradients.values()):
            raise ValueError('invalid guard gradient')
        state, before, before_mass, attempts = copy.deepcopy(model.state_dict()), training_score(), training_mass(), []
        accepted = False
        for rate in protocol['learning_rates']:
            model.load_state_dict(state)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in gradients:
                        p.add_(gradients[name], alpha=-rate)
            after, score_nll, after_mass = training_score(), nll('train'), training_mass()
            accepted = bool(acceptable_step(before['loss'], after['loss'], score_nll, baseline_train)
                            and np.isfinite(after_mass)
                            and abs(after_mass-train_mass_target) < abs(before_mass-train_mass_target))
            attempts.append({'learning_rate': rate, 'guard': after, 'nll': score_nll,
                             'training_mass': after_mass, 'accepted': accepted})
            print(f'JOINT_PROB_REFINE_TRIAL step={step} rate={rate:.4g} accepted={accepted}', flush=True)
            if accepted:
                break
        if not accepted:
            model.load_state_dict(state)
            history.append({'step': step, 'accepted': False, 'gradient': gradient, 'attempts': attempts,
                            'direction': direction_report, 'training_mass_residual': mass_residual})
            break
        subset_nll, val_guard = nll('validation'), validation_score()
        full_nll = nll('validation', full=True)
        mass = validation_mass()
        eligible = bool(eligible_selection(val_guard['loss'], best_loss, subset_nll, baseline_subset, full_nll, baseline_full,
            selection_residual=mass['residual'], baseline_selection_residual=initial_mass['residual']))
        if eligible:
            best_loss, best_step, best_full = val_guard['loss'], step, full_nll
            best_mass = mass
            save(step, best_loss, best_full)
        history.append({'step': step, 'accepted': True, 'gradient': gradient, 'attempts': attempts,
            'direction': direction_report, 'training_mass_residual': mass_residual,
            'subset_validation_nll': subset_nll, 'full_validation_nll': full_nll,
            'validation_guard': val_guard, 'validation_selection_mass': mass, 'selected': eligible, 'best_step': best_step})
        base.write_json(args.output_root/'history.json', {'history': history})
        print(f'JOINT_PROB_REFINE_STEP step={step} guard={val_guard["loss"]:.9f} best={best_step}', flush=True)
    model.load_state_dict(torch.load(args.output_root/'selected.pt', map_location=args.device, weights_only=False)['state_dict'])
    final_nll, final_guard = nll('validation', full=True), validation_score()
    final_mass = validation_mass()
    np.testing.assert_allclose(final_mass['residual'], best_mass['residual'], rtol=0, atol=1e-12)
    if np.any(np.abs(final_mass['residual']) > np.abs(initial_mass['residual'])+1e-12):
        raise RuntimeError('primary selected probability degraded')
    np.testing.assert_allclose([final_nll, final_guard['loss']], [best_full, best_loss], rtol=0, atol=1e-8)
    if final_nll > baseline_full+DENSITY_TOLERANCE or base.sha256(args.flow) != flow_hash:
        raise RuntimeError('density constraint or frozen initial checkpoint violated')
    base.write_json(args.output_root/'history.json', {'history': history})
    base.write_json(args.output_root/'completed.json', {'smoke': args.smoke, 'best_step': best_step,
        'eligible_for_joint_validation': not args.smoke and best_step > 0,
        'baseline_full_validation_nll': baseline_full, 'selected_full_validation_nll': final_nll,
        'baseline_guard_loss': initial_guard['loss'], 'selected_guard_loss': final_guard['loss'],
        'baseline_selection_mass': initial_mass, 'selected_selection_mass': final_mass,
        'selected_sha256': base.sha256(args.output_root/'selected.pt'),
        'protocol_sha256': base.sha256(args.output_root/'protocol.json'), 'calibration_claim': False,
        'production_accepted': False, 'limitations': ['soft random-direction forward0.05 guard with per-leg U',
            'fixed reference bank integration error remains',
            'common descent uses Euclidean parameter gradients; finite updates still require all validation gates',
            'soft training probability gradient does not replace the hard primary validation constraint', 'validation scenes are reused',
            'NLL constraint is not joint density or shear-calibration acceptance',
            'small operational subsets only' if args.smoke else 'complete validation NLL checked before selection']})
    print(f'JOINT_PROB_REFINE_COMPLETE best_step={best_step} smoke={args.smoke}', flush=True)


if __name__ == '__main__':
    main()
