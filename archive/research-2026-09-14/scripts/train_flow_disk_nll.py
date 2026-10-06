"""NLL-only disk-flow initialization, with exact raw-target provenance.

Small learning-rate probes use identical training rows and full validation.
The selected rate is then applied after resetting to the original NLL-only
parent's affine weights. Each subsequent epoch visits every NLL training row.
This establishes a new density baseline, not a response-trained checkpoint.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from sbsi.measurement_model import (
    TargetStandardizer, build_flow, save_measurement_model,
)
from sbsi.selection_model import TabularPreprocessor


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def load_torch(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def nll_pass(model, values, context, batch_size, *, optimizer=None,
             seed=501, max_steps=None, clip=5.):
    """Exact row mean, deterministic shuffled traversal and finite-gradient checks."""
    training = optimizer is not None
    model.train(training)
    order = None
    if training:
        generator = torch.Generator(device=values.device).manual_seed(seed)
        order = torch.randperm(len(values), generator=generator, device=values.device)
    total, rows, norms = 0., 0, []
    started = time.monotonic()
    with torch.set_grad_enabled(training):
        for step, start in enumerate(range(0, len(values), batch_size)):
            if max_steps is not None and step >= max_steps:
                break
            index = slice(start, start + batch_size) if order is None else order[start:start + batch_size]
            y, c = values[index], context[index]
            loss = -model.log_prob(y, c).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite NLL; no checkpoint update accepted')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                norms.append(float(norm))
                optimizer.step()
            total += float(loss.detach()) * len(y)
            rows += len(y)
    if not rows:
        raise ValueError('empty NLL population')
    return dict(nll=total / rows, rows=rows, optimizer_steps=len(norms),
                gradient_norms=norms, clipped_step_fraction=float(np.mean(np.asarray(norms) > clip)) if norms else 0.,
                seconds=time.monotonic() - started)


def fit(model, data, condition, target, config, metadata, output, *, epochs,
        batch_size, learning_rates, probe_steps, seed=501, clip=5., resume=False):
    """Retain every epoch and atomically commit optimizer/history for resumption."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / 'resume.pt'
    contract = dict(batch_size=batch_size, learning_rates=list(learning_rates),
                    probe_steps=probe_steps, seed=seed, clip=clip,
                    model_config=config, provenance={k: v for k, v in metadata.items()
                                                       if k != 'parent_reference'})
    def epoch_path(epoch):
        return output / f'disk_nll_epoch{epoch:03d}.pt'
    def copy_best(epoch):
        temporary = output / 'best_nll.tmp'
        shutil.copyfile(epoch_path(epoch), temporary)
        os.replace(temporary, output / 'best_nll.pt')
    if resume:
        state = load_torch(resume_path)
        if state['contract'] != contract or epochs <= state['epoch']:
            raise ValueError('resume requires unchanged data/recipe and additional epochs')
        model.load_state_dict(state['state_dict'], strict=True)
        chosen_lr = state['chosen_lr']
        optimizer = torch.optim.AdamW(model.parameters(), lr=chosen_lr, weight_decay=0.)
        optimizer.load_state_dict(state['optimizer'])
        history, probes, best_epoch = (state[k] for k in ('history', 'probes', 'best_epoch'))
        first_epoch = state['epoch'] + 1
        copy_best(best_epoch)
    else:
        if any(output.glob('disk_nll_epoch*.pt')) or resume_path.exists() or (output / 'probes.json').exists():
            raise FileExistsError('refusing to overwrite a previous NLL trial')
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        initial_val = nll_pass(model, *data['validation'], batch_size)
        probes = []
        for rate in learning_rates:
            model.load_state_dict(initial, strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=0.)
            try:
                train = nll_pass(model, *data['train'], batch_size, optimizer=optimizer,
                                 seed=seed + 100300, max_steps=probe_steps, clip=clip)
                val = nll_pass(model, *data['validation'], batch_size)
                probes.append(dict(lr=rate, train=train, validation=val, finite=True))
                save_measurement_model(output / f'lr_probe_{len(probes):02d}.pt', model,
                    condition, target, config, {**metadata, 'training_phase': 'NLL_only_LR_probe',
                                                'probe': probes[-1], 'grid_response_weight': 0.})
            except (FloatingPointError, RuntimeError) as error:
                # Only numerical failures are a rejected rate. CUDA/OOM and
                # programming errors remain job failures requiring investigation.
                if not isinstance(error, FloatingPointError) and 'non-finite' not in str(error):
                    raise
                probes.append(dict(lr=rate, finite=False, error=str(error)))
            print(json.dumps({'lr_probe': probes[-1]}), flush=True)
        finite = [p for p in probes if p['finite']]
        if not finite:
            (output / 'probes.json').write_text(json.dumps(probes, indent=2) + '\n')
            raise FloatingPointError('all NLL learning-rate probes failed')
        chosen_lr = min(finite, key=lambda p: p['validation']['nll'])['lr']
        (output / 'probes.json').write_text(json.dumps(dict(
            initial_validation=initial_val, probes=probes, chosen_lr=chosen_lr,
            reset_after_probes=True), indent=2) + '\n')
        model.load_state_dict(initial, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=chosen_lr, weight_decay=0.)
        history, best_epoch, first_epoch = [], 0, 0
    for epoch in range(first_epoch, epochs + 1):
        train = None
        if epoch:
            train = nll_pass(model, *data['train'], batch_size, optimizer=optimizer,
                             seed=seed + epoch * 1000003, clip=clip)
            if train['rows'] != len(data['train'][0]):
                raise ValueError('an NLL epoch did not cover every training row')
        val = nll_pass(model, *data['validation'], batch_size)
        record = dict(epoch=epoch, train=train, validation=val, lr=chosen_lr)
        if not history or val['nll'] < history[best_epoch]['validation']['nll']:
            best_epoch = epoch
        history.append(record)
        checkpoint_metadata = {**metadata, 'epoch': epoch, 'best_nll_epoch': best_epoch,
                               'training_phase': 'NLL_only_disk_refit' if epoch else 'coordinate_changed_initialization',
                               'grid_response_weight': 0., 'nll_weight': 1.,
                               'history': history, 'chosen_lr': chosen_lr}
        path = epoch_path(epoch)
        temporary = path.with_suffix('.tmp')
        save_measurement_model(temporary, model, condition, target, config, checkpoint_metadata)
        os.replace(temporary, path)
        # The epoch file can precede the committed resume state after an
        # interruption. Resume replays from the last committed optimizer.
        state = dict(contract=contract, epoch=epoch, state_dict=model.state_dict(),
                     optimizer=optimizer.state_dict(), history=history, probes=probes,
                     chosen_lr=chosen_lr, best_epoch=best_epoch)
        torch.save(state, resume_path.with_suffix('.tmp'))
        os.replace(resume_path.with_suffix('.tmp'), resume_path)
        copy_best(best_epoch)
        report = dict(history=history, best_nll_epoch=best_epoch, chosen_lr=chosen_lr,
                      response_accuracy_validated=False, physical_size_flux_response_validated=False)
        temporary = output / 'history.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        os.replace(temporary, output / 'history.json')
        print(json.dumps(record), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--raw-root', required=True, help='completed float64 recovery output directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[1.e-5, 1.e-4, 7.e-4])
    parser.add_argument('--probe-steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=501)
    parser.add_argument('--max-grad-norm', type=float, default=5.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (min(args.epochs, args.batch_size, args.probe_steps) < 1 or args.seed < 0
            or not all(math.isfinite(v) and v > 0 for v in [args.max_grad_norm, *args.learning_rates])):
        parser.error('positive finite training options required')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('full-catalogue NLL refit requires scheduled CUDA')
    prepared, raw_root = Path(args.prepared), Path(args.raw_root)
    preparation = json.loads((prepared / 'manifest.json').read_text())
    recovery = json.loads((raw_root / 'results.json').read_text())
    parent_sha = digest(preparation['parent'])
    if parent_sha != preparation['parent_sha256'] or parent_sha != recovery['original_parent_sha256']:
        raise ValueError('original NLL parent changed')
    parent = load_torch(preparation['parent'])
    parent_config, parent_meta = parent['model_config'], parent['metadata']
    if parent_config['flow_type'] != 'affine' or parent_config.get('mean_hidden', 0) != 0:
        raise ValueError('original plain affine NLL parent required')
    response_weights = ('response_weight', 'coupling_weight', 'grid_response_weight', 'ratio_response_weight',
                        'paired_transport_weight', 'distribution_response_weight', 'response_dispersion_weight', 'ra_weight')
    if any(float(parent_meta.get(k, 0) or 0) != 0 for k in response_weights):
        raise ValueError('response-trained parent forbidden')
    target_state = dict(parent['target_transform'], dtype='float64')
    target = TargetStandardizer.from_state(target_state)
    condition = TabularPreprocessor.from_state(parent['condition_preprocessor'])
    coordinate = recovery['splits']['train']['transformed']['radial_atanh']
    config = dict(parent_config, flow_type='disk_affine', disk_map='radial_tanh',
                  disk_shape_means=target.means[:2].tolist(), disk_shape_scales=target.scales[:2].tolist(),
                  disk_coordinate_means=coordinate['component_mean'], disk_coordinate_scales=coordinate['component_std'])
    torch.manual_seed(args.seed)
    model = build_flow(config).cuda()
    model.base.load_state_dict(parent['state_dict'], strict=True)
    data = {}
    parent_reference = None
    source_hashes = {str(prepared / 'manifest.json'): digest(prepared / 'manifest.json'),
                     str(raw_root / 'results.json'): digest(raw_root / 'results.json')}
    for split in ('train', 'validation'):
        info = recovery['splits'][split]
        if not (info['all_cached_targets_and_contexts_exactly_reproduced']
                and info['original_pair_keys_and_shear_vectors_exactly_reproduced']
                and info['raw_shape_minimum_one_minus_norm_squared'] > 0):
            raise ValueError('raw target verification not complete')
        path = prepared / (split + '.pt')
        if digest(path) != preparation['files'][path.name]:
            raise ValueError('prepared population changed')
        raw_path = raw_root / (split + '_raw.npy')
        if digest(raw_path) != recovery['files'][raw_path.name]:
            raise ValueError('raw float64 targets changed')
        source_hashes[str(path)] = preparation['files'][path.name]
        source_hashes[str(raw_path)] = recovery['files'][raw_path.name]
        payload = load_torch(path)
        raw = np.load(raw_path)
        if raw.dtype != np.float64 or raw.shape != (info['nll_rows'], 4) or len(raw) != len(payload['context']):
            raise ValueError('raw target precision or row count changed')
        data[split] = [torch.as_tensor(target.transform_array(raw), device='cuda'), payload['context'].cuda()]
        if split == 'validation':
            original = build_flow(parent_config).cuda()
            original.load_state_dict(parent['state_dict'], strict=True)
            parent_reference = dict(
                legacy_cached=nll_pass(original, payload['values'].cuda(), data[split][1], args.batch_size),
                original_raw_standardized_then_cast_for_float32_base=nll_pass(
                    original, data[split][0].float(), data[split][1], args.batch_size),
            )
            print(json.dumps({'parent_reference': parent_reference}), flush=True)
            del original
        del payload, raw
    metadata = dict(parent=preparation['parent'], parent_sha256=parent_sha,
                    source_hashes=source_hashes, target_features=target.target_names,
                    condition_features=condition.feature_names, seed=args.seed,
                    initialization='Original NLL-only affine weights; disk coordinates change the initial density; refit required.',
                    density_reference_measure='Original parent standardized e1,e2,mag,log-radius; full disk and scale Jacobians included.',
                    parent_reference=parent_reference,
                    empirical_response_calibration=False, source_response_loss_unchanged=True)
    report = fit(model, data, condition, target, config, metadata, args.output,
                 epochs=args.epochs, batch_size=args.batch_size, learning_rates=args.learning_rates,
                 probe_steps=args.probe_steps, seed=args.seed, clip=args.max_grad_norm, resume=args.resume)
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise ValueError('input changed during training: ' + path)
    result = {**report, 'input_hashes_verified_after_training': True,
              'original_parent_sha256': parent_sha, 'parent_reference': parent_reference}
    (Path(args.output) / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('DISK_NLL_REFIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
