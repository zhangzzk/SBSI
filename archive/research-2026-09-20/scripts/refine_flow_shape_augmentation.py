"""Controlled single-flow likelihood continuation with optional shape augmentation.

Both arms use identical rows and optimizer settings. Selection uses unchanged
validation measurements: improve full NLL without increasing the initial guard
loss. The initial checkpoint competes as epoch zero. No ConstGold is read.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS
from sbsi.flow_guard_response import GuardResponsePopulation, evaluate_guard_response, make_guard_cuts
from sbsi.measurement_model import load_measurement_model
from scripts import train_fixed_g0_flow as base
from scripts.train_full_domain_guard_flow import MAGNITUDE_THRESHOLDS, RADIUS_THRESHOLDS


def standardized_sign_action(context, target, signs, context_means, context_scales, target_means, target_scales):
    """Apply physical shape reflections through frozen affine standardizers."""
    if (context.ndim != 2 or target.shape != (len(context), 4)
            or signs.shape != (len(context), 2) or context.shape[1] not in (8, 16)):
        raise ValueError('aligned flow rows and two shape signs required')
    if not bool(((signs == 1) | (signs == -1)).all()):
        raise ValueError('shape actions must be signs')
    if context.shape[1] == 16 and bool((context[:, 8:10] != 0).any()):
        raise ValueError('shape augmentation requires finite observed context shapes')
    transformed_context, transformed_target = context.clone(), target.clone()
    for transformed, mean, scale in ((transformed_context, context_means, context_scales),
                                     (transformed_target, target_means, target_scales)):
        shift = torch.as_tensor(mean[:2], dtype=transformed.dtype, device=transformed.device)
        units = torch.as_tensor(scale[:2], dtype=transformed.dtype, device=transformed.device)
        if not bool(torch.isfinite(shift).all() and torch.isfinite(units).all() and (units > 0).all()):
            raise ValueError('finite positive affine standardizers required')
        action = signs.to(transformed.dtype)
        transformed[:, :2] = action*transformed[:, :2] + (action-1)*shift/units
    return transformed_context, transformed_target


def eligible_checkpoint(nll, guard_loss, best_nll, initial_guard_loss):
    return bool(np.isfinite([nll, guard_loss, best_nll, initial_guard_loss]).all()
                and nll < best_nll-1e-6 and guard_loss <= initial_guard_loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-root', type=Path, required=True)
    parser.add_argument('--flow', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--arm', choices=('control', 'augmented'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true', help='small nonpromotable operational check')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('training requires Slurm')
    if args.device == 'cpu' and not args.smoke:
        raise ValueError('CPU mode is reserved for the bounded smoke check')
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = args.domain_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    initial_hash = base.sha256(args.flow)
    bundle = load_measurement_model(args.flow, device=torch.device(args.device))
    model = bundle.model
    checkpoint = torch.load(args.flow, map_location='cpu', weights_only=False)
    condition, target = bundle.condition_preprocessor, bundle.target_transform
    if (condition.feature_names != list(FLOW_FEATURES) or target.target_names != list(FLOW_TARGETS)
            or any(name in condition.log_features for name in FLOW_FEATURES[:2])):
        raise ValueError('augmentation requires the declared affine shape coordinates')
    epochs, rows_per_epoch = (1, 8192) if args.smoke else (4, 4000000)
    batch_size = 512 if args.smoke else 8192
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    protocol = {'arm': args.arm, 'initial_flow_sha256': initial_hash,
                'domain_manifest_sha256': base.sha256(manifest_path), 'smoke': args.smoke,
                'epochs': epochs, 'rows_per_epoch': rows_per_epoch, 'batch_size': batch_size,
                'optimizer': 'fresh AdamW;lr1e-5;weight_decay0;betas0.9,0.999;eps1e-8;clip_norm5',
                'row_seed': 2026091801, 'augmentation_seed': 2026091802,
                'augmentation': 'uniform independent signs on physical input and output shape; scalar outputs and contexts unchanged',
                'validation_guard_seed': 758000000001, 'validation_guard_draws': 4,
                'validation_guard_groups': 2, 'validation_guard_batch': 2048,
                'validation_nll_subset_seed': 754000001,
                'selection': 'minimum full untransformed validation NLL with guard loss<=initial; epoch0 included',
                'script_sha256': base.sha256(__file__), 'cuts': cuts,
                'limitations': ['augmentation is an experimental regularizer, not proven exact measurement symmetry',
                                'soft forward0.05 response guards are not joint shear-calibration acceptance',
                                'no ensemble is used at evaluation; exactly one checkpoint is selected']}
    base.write_json(args.output_root/'protocol.json', protocol)
    arrays = {}
    for split in ('train', 'validation'):
        cases = manifest['split'][f'{split}_cases'][:2] if args.smoke else manifest['split'][f'{split}_cases']
        print(f'AUGMENT_LOADING arm={args.arm} split={split} cases={len(cases)}', flush=True)
        for case in cases:
            for leg in ('g0', 'g05'):
                path = args.domain_root/f'flow/{leg}/case{case:03d}.npz'
                if base.sha256(path) != manifest['flow'][f'{leg}_cases'][str(case)]['sha256']:
                    raise ValueError('changed prepared training or validation data')
        data = base.load_split(args.domain_root, cases)
        context = torch.as_tensor(condition.transform_frame(pd.DataFrame(data['context_raw'], columns=FLOW_FEATURES)),
                                  dtype=torch.float32, device=args.device)
        values = torch.as_tensor(target.transform_array(data['target_raw']), dtype=torch.float64, device=args.device)
        if bool((context[:, 8:10] != 0).any()):
            raise ValueError('missing physical input shapes')
        arrays[split] = {'context': context, 'values': values, 'cases': cases}
        if split == 'validation':
            population = GuardResponsePopulation(context, data['pairs'], data['pair_zero_target'], data['pair_shear_target'],
                data['gamma'], cuts, radius_softness=.05, magnitude_softness=.05, response_scale=.01)
        del data
    validation = arrays['validation']
    subset = np.random.default_rng(protocol['validation_nll_subset_seed']).choice(
        len(validation['values']), min(1024 if args.smoke else 500000, len(validation['values'])), replace=False)
    subset = torch.as_tensor(subset, device=args.device)
    guard_indices = np.arange(min(256, len(population.pairs))) if args.smoke else np.arange(len(population.pairs))

    @torch.no_grad()
    def nll(full=False):
        rows = slice(None) if full else subset
        return base.nll_pass(model, validation['values'][rows], validation['context'][rows])['nll']

    @torch.no_grad()
    def guard():
        model.eval()
        return evaluate_guard_response(model, population, guard_indices, draws=4, groups=2,
                                       batch_size=2048, seed=protocol['validation_guard_seed'])

    baseline_nll, baseline_subset, baseline_guard = nll(full=True), nll(), guard()
    best_nll, best_epoch = baseline_nll, 0
    history = [{'epoch': 0, 'full_validation_nll': baseline_nll,
                'subset_validation_nll': baseline_subset, 'validation_guard': baseline_guard}]

    def save(epoch, score, response):
        base.save_checkpoint(args.output_root/'selected.pt', model, condition, target, checkpoint['model_config'],
            {'training_phase': 'single_flow_shape_augmentation', 'arm': args.arm, 'epoch': epoch,
             'validation_nll': score, 'validation_guard_loss': response['loss'],
             'protocol_sha256': base.sha256(args.output_root/'protocol.json'), 'smoke': args.smoke,
             'population': manifest['population']})

    save(0, baseline_nll, baseline_guard)
    print(f'AUGMENT_BASELINE arm={args.arm} nll={baseline_nll:.9f} guard={baseline_guard["loss"]:.9f}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
    for epoch in range(1, epochs+1):
        started = time.monotonic()
        train = arrays['train']
        rows = np.random.default_rng(protocol['row_seed']+epoch*1000003).choice(
            len(train['values']), min(rows_per_epoch, len(train['values'])), replace=False)
        generator = torch.Generator(device=args.device).manual_seed(protocol['augmentation_seed']+epoch*1000003)
        model.train()
        total, norms = 0., []
        for step, start in enumerate(range(0, len(rows), batch_size)):
            index = torch.as_tensor(rows[start:start+batch_size], device=args.device)
            context, values = train['context'][index], train['values'][index]
            signs = 2*torch.randint(0, 2, (len(index), 2), generator=generator, device=args.device)-1
            if args.arm == 'augmented':
                context, values = standardized_sign_action(context, values, signs,
                    condition.means, condition.scales, target.means, target.scales)
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prob(values, context).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite likelihood')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(index)
            norms.append(float(norm))
            if step == 0 or (step+1) % 100 == 0:
                print(f'AUGMENT_TRAIN arm={args.arm} epoch={epoch} rows={min(start+batch_size,len(rows))}/{len(rows)}', flush=True)
        score, response = nll(), guard()
        full_score = nll(full=True) if response['loss'] <= baseline_guard['loss'] else None
        accepted = full_score is not None and eligible_checkpoint(full_score, response['loss'], best_nll, baseline_guard['loss'])
        if accepted:
            best_nll, best_epoch = full_score, epoch
            save(epoch, full_score, response)
        history.append({'epoch': epoch, 'training_nll': total/len(rows), 'subset_validation_nll': score,
                        'full_validation_nll': full_score, 'validation_guard': response,
                        'accepted': accepted, 'best_epoch': best_epoch, 'seconds': time.monotonic()-started,
                        'gradient_norm_mean': float(np.mean(norms)), 'gradient_norm_max': float(np.max(norms)),
                        'clipped_step_fraction': float(np.mean(np.asarray(norms)>5))})
        base.write_json(args.output_root/'history.json', {'history': history})
        print(f'AUGMENT_EPOCH arm={args.arm} epoch={epoch} nll={score:.9f} guard={response["loss"]:.9f} best={best_epoch}', flush=True)
    selected = torch.load(args.output_root/'selected.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(selected['state_dict'])
    final_nll, final_guard = nll(full=True), guard()
    np.testing.assert_allclose(final_nll, best_nll, rtol=0, atol=1e-8)
    if final_nll > baseline_nll+1e-8 or final_guard['loss'] > baseline_guard['loss']+1e-8:
        raise RuntimeError('selected checkpoint violates unchanged validation gates')
    if base.sha256(args.flow) != initial_hash:
        raise ValueError('initial checkpoint changed during continuation')
    base.write_json(args.output_root/'history.json', {'history': history})
    base.write_json(args.output_root/'completed.json', {'arm': args.arm, 'smoke': args.smoke,
        'eligible_for_joint_validation': not args.smoke, 'production_accepted': False,
        'best_epoch': best_epoch, 'baseline_full_validation_nll': baseline_nll,
        'selected_full_validation_nll': final_nll, 'baseline_guard_loss': baseline_guard['loss'],
        'selected_guard_loss': final_guard['loss'], 'selected_sha256': base.sha256(args.output_root/'selected.pt'),
        'protocol_sha256': base.sha256(args.output_root/'protocol.json'), 'calibration_claim': False})
    print(f'AUGMENT_COMPLETE arm={args.arm} best_epoch={best_epoch} delta_nll={final_nll-baseline_nll:+.8f}', flush=True)


if __name__ == '__main__':
    main()
