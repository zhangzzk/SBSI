"""Smooth state-major-axis crowding classifier; select only by tuning BCE.

Deliberately preserves the static-geometry training recipe, with a new data
contract and explicit inference-coordinate metadata.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from sbsi.selection_model import SelectionMLP, TabularPreprocessor, load_selection_model
from scripts.smooth_scene_classifier_features import FEATURES, FORMAT
from scripts.train_full_parent_classifier import case_metrics, transform, sha256, write_json

SEED = 20260913


def load_cases(root, manifest, cases, dimensions, smoke):
    columns = {key: [] for key in ('x0_raw', 'x1_raw', 'u0', 'u1')}
    slices, offset = [], 0
    for case in cases:
        path = root/f'cases/case{case:03d}.npz'
        if sha256(path) != manifest['case_records'][str(case)]['prepared_sha256']:
            raise ValueError('prepared case changed')
        with np.load(path) as stored:
            n = min(512, len(stored['ids'])) if smoke else len(stored['ids'])
            for i in (0, 1):
                x = stored[f'x{i}'][:n, :dimensions].astype(np.float32)
                u = stored[f'u{i}'][:n].astype(np.float32)
                if not np.isfinite(x).all() or not np.isin(u, [0, 1]).all():
                    raise ValueError('finite features and Bernoulli labels required')
                columns[f'x{i}_raw'].append(x)
                columns[f'u{i}'].append(u)
        slices.append((case, offset, offset+n))
        offset += n
    return {key: np.concatenate(value) for key, value in columns.items()} | {'case_slices': slices}


def fit_preprocessor(data, names):
    total, square = np.zeros(len(names)), np.zeros(len(names))
    n = 0
    for leg in (0, 1):
        x = data[f'x{leg}_raw']
        for start in range(0, len(x), 65536):
            block = x[start:start+65536].astype(float)
            total += block.sum(0)
            square += np.square(block).sum(0)
            n += len(block)
    means = total/n
    scales = np.sqrt(np.maximum((square-n*means**2)/(n-1), 0))
    scales[scales < 1e-6] = 1.
    return TabularPreprocessor(list(names), means.astype(np.float32), means.astype(np.float32),
                               scales.astype(np.float32), False, ())


def initial_model(dimensions):
    """Same initial core in both arms; new input columns begin at zero."""
    torch.manual_seed(SEED)
    config = {'input_dim': 8, 'hidden_dim': 64, 'n_layers': 2, 'dropout': 0., 'activation': 'silu'}
    core = SelectionMLP(**config)
    if dimensions == 8:
        return core, config
    config = config | {'input_dim': dimensions}
    model = SelectionMLP(**config)
    state = core.state_dict()
    first = torch.zeros_like(model.state_dict()['net.0.weight'])
    first[:, :8] = state['net.0.weight']
    state['net.0.weight'] = first
    model.load_state_dict(state)
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--arm', choices=('smooth',), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--frozen-protocol', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('training requires Slurm')
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    frozen = json.loads(args.frozen_protocol.read_text())
    for path, expected in frozen['files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('frozen experiment input changed: '+path)
    manifest_path = args.prepared_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    expected_cases = [40, 100] if args.smoke else list(range(40, 120))
    if (manifest['format_version'] != FORMAT
            or manifest.get('coordinate_convention') != 'state_major_axis' or not manifest['with_corrected_labels']
            or manifest['cases'] != expected_cases or manifest['features'] != list(FEATURES)):
        raise ValueError('frozen corrected-label geometry preparation required')
    train_cases, tune_cases = ([40], [100]) if args.smoke else (list(range(40, 100)), list(range(100, 120)))
    names = list(FEATURES)
    paths = [Path(__file__), Path('scripts/scene_classifier_features.py'), Path('scripts/train_full_parent_classifier.py'),
             Path('sbsi/selection_model.py'), Path('scripts/scene_geometry_coordinates.py'),
             Path('scripts/prepare_smooth_classifier.py'), Path('scripts/smooth_scene_classifier_features.py'),
             Path('scripts/audit_continuous_crowding_residuals.py'), manifest_path]
    pins = frozen['files'] | {str(p.resolve()): sha256(p) for p in [*paths, args.frozen_protocol]}
    args.output_root.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    raw_train = load_cases(args.prepared_root, manifest, train_cases, len(names), args.smoke)
    raw_tune = load_cases(args.prepared_root, manifest, tune_cases, len(names), args.smoke)
    prep = fit_preprocessor(raw_train, names)
    # Retain raw rows only for exported model replay.
    export_rows = raw_tune['x0_raw'][:512].copy()
    data = {'train': transform(raw_train, prep, device), 'tune': transform(raw_tune, prep, device)}
    del raw_train, raw_tune
    model, config = initial_model(len(names))
    model = model.to(device)
    torch.cuda.manual_seed_all(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=6, min_lr=1e-5)
    protocol = {'arm': args.arm, 'smoke': args.smoke, 'seed': SEED, 'features': names,
                'training_cases': train_cases, 'tune_cases': tune_cases, 'model_config': config,
                'max_epochs': 2 if args.smoke else 100, 'patience': 15, 'batch_pairs': 256 if args.smoke else 16384,
                'maximum_pairs_per_epoch': 8000000, 'initial_learning_rate': .001,
                'weight_decay': .0001, 'gradient_clip': 5., 'selection': 'minimum mean-case tune BCE',
                'initialization': 'shared eight-input core; additional geometry weights zero',
                'population': manifest['population'], 'response_read': False, 'pins': pins,
                'geometry_coordinate_convention': 'state_major_axis', 'device': args.device}
    write_json(args.output_root/'protocol.json', protocol)
    best, best_epoch, stale, history = float('inf'), 0, 0, []
    n = len(data['train']['u0'])
    for epoch in range(1, protocol['max_epochs']+1):
        started = time.monotonic()
        sample_n = min(n, protocol['maximum_pairs_per_epoch'])
        order = np.random.default_rng(SEED+epoch*1000003).choice(n, sample_n, replace=False)
        model.train()
        total, clipped = 0., 0
        for indices in torch.as_tensor(order, dtype=torch.long, device=device).split(protocol['batch_pairs']):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(F.binary_cross_entropy_with_logits(model(data['train'][f'x{i}'][indices]),
                       data['train'][f'u{i}'][indices]) for i in (0, 1))/2
            if not torch.isfinite(loss):
                raise ValueError('nonfinite train loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            clipped += float(norm) > 5.
            optimizer.step()
            total += float(loss.detach())*len(indices)
        tuning = case_metrics(model, data['tune'])
        score = float(np.mean([row['bce'] for row in tuning]))
        scheduler.step(score)
        history.append({'epoch': epoch, 'train_pair_bce': total/sample_n, 'tune_mean_case_bce': score,
                        'seconds': time.monotonic()-started, 'gradient_clipped_steps': clipped,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            temporary = args.output_root/'selected.tmp.pt'
            torch.save({'state_dict': model.state_dict(), 'preprocessor': prep.to_state(),
                'model_config': config, 'temperature': 1., 'metadata': {
                    'seed': SEED, 'event': 'per-leg usable measurement U', 'training_cases': train_cases,
                    'tune_cases': tune_cases, 'prepared_manifest_sha256': sha256(manifest_path),
                    'condition_features': names, 'selected_epoch': epoch, 'arm': args.arm,
                    'population': manifest['population'], 'response_used': False,
                    'geometry_coordinate_convention': 'state_major_axis',
                    'prepared_format_version': manifest['format_version'],
                    'protocol_sha256': sha256(args.output_root/'protocol.json')}}, temporary)
            os.replace(temporary, args.output_root/'selected.pt')
        else:
            stale += 1
        write_json(args.output_root/'history.json', {'best_epoch': best_epoch, 'best_tune_mean_case_bce': best, 'history': history})
        print(f'SCENE_CLASSIFIER_EPOCH arm={args.arm} epoch={epoch} tune_bce={score:.9f} best={best:.9f}', flush=True)
        if stale >= protocol['patience']:
            break
    loaded = load_selection_model(args.output_root/'selected.pt', device=device)
    with torch.no_grad():
        raw = torch.as_tensor(export_rows, dtype=torch.float32, device=device)
        actual = loaded.logits_from_raw_tensor(raw)
        expected = loaded.model((raw-torch.as_tensor(prep.means, device=device))/torch.as_tensor(prep.scales, device=device))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for path, expected in pins.items():
        if sha256(Path(path)) != expected:
            raise ValueError('training input/source changed: '+path)
    tune_metrics = case_metrics(loaded.model, data['tune'])
    if abs(np.mean([row['bce'] for row in tune_metrics])-best) > 1e-7:
        raise ValueError('selected classifier does not replay best tune BCE')
    write_json(args.output_root/'completed.json', {'arm': args.arm, 'smoke': args.smoke,
        'best_epoch': best_epoch, 'best_tune_mean_case_bce': best, 'selected_sha256': sha256(args.output_root/'selected.pt'),
        'protocol_sha256': sha256(args.output_root/'protocol.json'), 'train_metrics': case_metrics(loaded.model, data['train']),
        'tune_metrics': tune_metrics, 'parameter_count': sum(p.numel() for p in loaded.model.parameters()),
        'response_read': False, 'production_accepted': False})
    print('SCENE_CLASSIFIER_TRAINING_COMPLETE', args.arm, best_epoch, best, flush=True)


if __name__ == '__main__':
    main()
