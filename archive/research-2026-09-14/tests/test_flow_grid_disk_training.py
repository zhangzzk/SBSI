import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sbsi.flow_disk import radial_tanh
from sbsi.flow_grid_accumulation import sampled_response_backward, response_plan
from sbsi.flow_grid_full_cells import full_cell_response, _cell_mean
from sbsi.flow_grid_response import GridResponsePopulation, independent_grid_score
from sbsi.measurement_model import TargetStandardizer, build_flow, save_measurement_model
from sbsi.selection_model import TabularPreprocessor
from scripts.train_flow_grid_accumulated import nll_initialization, nll_values, main
from scripts.train_flow_grid_response import DEFINITION, digest, load_torch


def disk_fixture(tmp_path):
    torch.manual_seed(125)
    prepared = tmp_path/'prepared'
    raw_root = tmp_path/'raw'
    prepared.mkdir(); raw_root.mkdir()
    target = TargetStandardizer(
        ['measured_ngmix_g1', 'measured_ngmix_g2', 'measured_mag_auto', 'measured_log_flux_radius'],
        np.zeros(4, np.float32), np.array([.3, .4, 1., 1.], np.float32))
    condition = TabularPreprocessor(['x', 'y'], np.zeros(2), np.zeros(2), np.ones(2), False, ())
    config = dict(flow_type='affine', target_dim=4, context_dim=2, hidden_dim=8, n_layers=1, n_flows=2)
    plain = build_flow(config)
    with torch.no_grad():
        for layer in plain.layers:
            layer.net[-1].weight.normal_(std=.03)
    parent = prepared/'parent.pt'
    save_measurement_model(parent, plain, condition, target, config, {'response_weight': 0.})
    parent_sha = digest(parent)
    recovery = dict(original_parent_sha256=parent_sha, splits={}, files={})
    counts = np.array([3, 7, 8])
    payload = dict(pairs=np.arange(36).reshape(18, 2), counts=counts,
                   bin_ids=np.repeat(np.arange(3), counts), design=np.random.default_rng(9).normal(size=(18, 2)),
                   target=np.random.default_rng(91).normal(size=(3, 4))*.1,
                   precision=np.tile(np.eye(4), (3, 1, 1)))
    for split in ('train', 'validation'):
        raw = torch.randn(36, 4).double()*.2
        raw[:, :2] = radial_tanh(raw[:, :2])[0]
        raw[0, :2] = torch.tensor([.999999986571655, 0.], dtype=torch.float64)
        np.save(raw_root/(split+'_raw.npy'), raw.numpy())
        recovery['files'][split+'_raw.npy'] = digest(raw_root/(split+'_raw.npy'))
        recovery['splits'][split] = dict(nll_rows=36, all_cached_targets_and_contexts_exactly_reproduced=True,
            original_pair_keys_and_shear_vectors_exactly_reproduced=True,
            raw_shape_minimum_one_minus_norm_squared=float((1-raw[:, :2].square().sum(-1)).min()))
        torch.save(dict(payload, context=torch.randn(36, 2), values=torch.tensor(target.transform_array(raw.numpy()))),
                    prepared/(split+'.pt'))
    (raw_root/'results.json').write_text(json.dumps(recovery))
    manifest = dict(version=1, definition=DEFINITION, parent=str(parent), parent_sha256=parent_sha, n_cells=3,
                    splits={s: {'nll_rows': 36, 'used_pairs': 18} for s in ('train', 'validation')},
                    files={s+'.pt': digest(prepared/(s+'.pt')) for s in ('train', 'validation')})
    (prepared/'manifest.json').write_text(json.dumps(manifest))
    target = TargetStandardizer.from_state(dict(target.to_state(), dtype='float64'))
    config.update(flow_type='disk_affine', disk_map='radial_tanh', disk_shape_means=target.means[:2].tolist(),
                  disk_shape_scales=target.scales[:2].tolist(), disk_coordinate_means=[.02, -.01],
                  disk_coordinate_scales=[.6, .7])
    model = build_flow(config)
    model.base.load_state_dict(plain.state_dict())
    initial = tmp_path/'disk_nll.pt'
    sources = [prepared/'manifest.json', raw_root/'results.json', raw_root/'train_raw.npy', raw_root/'validation_raw.npy']
    metadata = dict(training_phase='NLL_only_disk_refit', epoch=1, nll_weight=1., grid_response_weight=0.,
                    parent_sha256=parent_sha, source_hashes={str(p): digest(p) for p in sources},
                    test_fixture=True)
    save_measurement_model(initial, model, condition, target, config, metadata)
    args = SimpleNamespace(root=str(prepared), initial_model=str(initial), raw_root=str(raw_root))
    return args, manifest, target, model


def test_disk_inputs_use_original_float64_and_keep_covariance_cache_immutable(tmp_path):
    args, manifest, target, model = disk_fixture(tmp_path)
    original_hashes = {s: digest(Path(args.root)/(s+'.pt')) for s in ('train', 'validation')}
    reference, path, hashes, recovery, _ = nll_initialization(args, manifest)
    assert str(path) == args.initial_model and hashes[str(path)] == digest(path)
    assert reference['metadata']['grid_response_weight'] == 0
    payload = load_torch(Path(args.root)/'train.pt')
    values = nll_values(payload, 'train', target, args.raw_root, recovery)
    assert values.dtype == torch.float64
    np.testing.assert_array_equal(values.numpy(), target.transform_array(np.load(Path(args.raw_root)/'train_raw.npy')))
    assert torch.isfinite(model.log_prob(values, payload['context'])).all()
    # Boundary rounding in the legacy cache must not be passed to the disk.
    old_raw = payload['values'].double() * torch.tensor(target.scales) + torch.tensor(target.means)
    assert old_raw[0, :2].square().sum() >= 1
    with pytest.raises(ValueError, match='cannot use cached float32'):
        nll_values(payload, 'train', target)
    for split, sha in original_hashes.items():
        assert digest(Path(args.root)/(split+'.pt')) == sha
    alias = tmp_path/'trial_prepared'
    alias.symlink_to(Path(args.root), target_is_directory=True)
    aliased = copy.copy(args); aliased.root = str(alias)
    nll_initialization(aliased, manifest)


@pytest.mark.parametrize('change, match', [
    ('response', 'NLL-only parent'), ('initialization', 'coordinate initialization'),
    ('raw_hash', 'verified prepared/raw population'), ('physical', 'physical/context transforms'),
])
def test_disk_initialization_rejects_wrong_lineage_phase_precision_or_targets(tmp_path, change, match):
    args, manifest, _, _ = disk_fixture(tmp_path)
    checkpoint = load_torch(args.initial_model)
    if change == 'response':
        checkpoint['metadata']['grid_response_weight'] = 1.
    elif change == 'initialization':
        checkpoint['metadata']['training_phase'] = 'coordinate_changed_initialization'
        checkpoint['metadata']['epoch'] = 0
    elif change == 'raw_hash':
        checkpoint['metadata']['source_hashes'][str(Path(args.raw_root)/'results.json')] = 'wrong'
    else:
        checkpoint['target_transform']['means'][0] += .1
    torch.save(checkpoint, args.initial_model)
    with pytest.raises(ValueError, match=match):
        nll_initialization(args, manifest)


@pytest.mark.parametrize('mode', ['sampled', 'full'])
def test_disk_two_pass_gradient_matches_direct_normalized_loss_with_nll(tmp_path, mode):
    args, _, target, model = disk_fixture(tmp_path)
    payload = load_torch(Path(args.root)/'train.pt')
    pop = GridResponsePopulation(payload['context'], payload, target.scales[:2])
    values = nll_values(payload, 'train', target, args.raw_root,
                         json.loads((Path(args.raw_root)/'results.json').read_text()))
    if mode == 'sampled':
        cells, local, groups = response_plan(pop, 2, 7, 42)
        means = []
        for stream, rows in enumerate(groups):
            total = torch.zeros((2, 4), dtype=torch.float64)
            for block, start in enumerate(range(0, len(rows), 5)):
                part = pop.contributions(model, rows[start:start+5], 4, seed=42+100000+2*block+stream)
                total = total.index_add(0, local[start:start+5], part)
            means.append(total/7)
    else:
        cells = [2, 0]
        means = [torch.stack([_cell_mean(model, pop, b, np.diff(pop.offsets)[b], 4, 2, 42, stream)
                               for b in cells]) for stream in range(2)]
    score = independent_grid_score(*means, pop.target[cells], pop.precision[cells]).mean()/4
    (-model.log_prob(values, pop.context).mean()+3*score).backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    (-model.log_prob(values, pop.context).mean()).backward()
    if mode == 'sampled':
        actual = sampled_response_backward(model, pop, 2, 7, 4, chunk_size=5, seed=42, weight=3.)
    else:
        actual = full_cell_response(model, pop, cells, 4, chunk_size=2, seed=42, weight=3., backward=True)
    assert actual['loss'] == pytest.approx(float(score.detach()), abs=1.e-13)
    for parameter, gradient in zip(model.parameters(), expected):
        torch.testing.assert_close(parameter.grad, gradient, rtol=1.e-5, atol=2.e-7)


@pytest.mark.parametrize('mode', ['sampled', 'full'])
def test_disk_response_optimizer_resume_and_full_nll_coverage(tmp_path, mode):
    args, manifest, _, _ = disk_fixture(tmp_path)
    options = ['--root', args.root, '--initial-model', args.initial_model, '--raw-root', args.raw_root,
               '--weight', '1', '--mode', mode, '--device', 'cpu', '--nll-batch-size', '13',
               '--nll-chunk-size', '5', '--cells-per-step', '2', '--pairs-per-cell', '7',
               '--response-chunk-size', '5', '--draws', '2', '--validation-draws', '2',
               '--validation-groups', '4', '--validation-batch-size', '5']
    resumed, full = tmp_path/'resumed.pt', tmp_path/'full.pt'
    main(options+['--output', str(resumed), '--epochs', '1'])
    main(options+['--output', str(resumed), '--epochs', '2', '--resume'])
    main(options+['--output', str(full), '--epochs', '2'])
    a, b = [load_torch(tmp_path/(name+'_epoch002.pt')) for name in ('resumed', 'full')]
    for key in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][key], b['state_dict'][key], rtol=0, atol=0)
    for key in ('parent_sha256', 'original_nll_parent_sha256', 'input_hashes'):
        assert a['metadata'][key] == b['metadata'][key]
    assert a['metadata']['parent_sha256'] == digest(args.initial_model)
    assert a['metadata']['original_nll_parent_sha256'] == manifest['parent_sha256']
    assert a['metadata']['nll_target_precision'] == 'float64'
    assert a['metadata']['training_phase'] == 'grid_response_finetune'
    for record in a['metadata']['history'][1:]:
        assert record['nll_rows'] == 36
        assert record['response_pairs_per_group'] == (18 if mode == 'full' else 42)
        assert record['val_total'] == pytest.approx(record['val_nll']+record['val_grid'])
    sa, sb = [load_torch(p.with_suffix('.resume.pt')) for p in (resumed, full)]
    for key, state in sa['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, sb['optimizer']['state'][key][name], rtol=0, atol=0)
    changed_parent = load_torch(args.initial_model)
    changed_parent['state_dict']['base.layers.0.net.0.weight'][0, 0] += .001
    torch.save(changed_parent, args.initial_model)
    with pytest.raises(ValueError, match='NLL initialization or raw inputs changed'):
        main(options+['--output', str(resumed), '--epochs', '3', '--resume'])
