"""Fixed-observation comparison after restoring the supplied bright prior atoms."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_EXPANDED_ROOT'])
MATCHED = Path(os.environ['SBSI_MATCHED_ROOT'])
sys.path.insert(0, str(MATCHED))
import run_v39_matched_inputs as matched  # noqa: E402
from sbsi.selection_normalization import log_mass_derivatives  # noqa: E402

old, study = matched.old, matched.study
np, pd, torch, save = matched.np, matched.pd, matched.torch, matched.save


def protocol():
    return study.read(ROOT/'protocol.json')


def expanded():
    return Path(protocol()['expanded_source'])


def identity():
    record = study.read(ROOT/'execution.json')
    study.verify(__file__, record['driver_sha256'])
    study.verify(ROOT/'protocol.json', record['protocol_sha256'])
    assert matched.identity() == record['matched_execution']
    for path, digest in record['sources'].items():
        study.verify(path, digest)
    return record


def initialize():
    assert not (ROOT/'execution.json').exists()
    gen, p, source = matched.generation(), protocol(), expanded()
    assert gen['n_inputs'] == 20000 and gen['n_observations'] == 16314
    assert gen['center'] == p['center']
    parent_protocol = study.read(source/'protocol.json')
    assert parent_protocol['models'] == study.read(old.SOURCE/'protocol.json')['models']
    support, coords = [study.read(source/f'{name}/manifest.json') for name in ('support', 'assembled')]
    base = study.read(old.SOURCE/'assembled/manifest.json')
    assert support['n_atoms'] == coords['n_atoms'] == p['n_atoms'] == 28012032
    assert coords['n_old'] == base['n_atoms'] == 20231221
    for key in ('metric_center', 'metric_scale', 'floors', 'fractional_flux_floor'):
        assert coords[key] == base[key], key
    paths = [source/'protocol.json', source/'execution.json', source/'support/manifest.json',
             source/'assembled/manifest.json', MATCHED/'mock/manifest.json', MATCHED/'comparison.json']
    for folder, receipt, names in [(source/'support', support, ('zero.npy', 'weights.npy')),
            (source/'assembled', coords, ('proposal/coordinates.npz', 'proposal/manifest.json'))]:
        for name in names:
            study.verify(folder/name, receipt['output_sha256'][name])
    feature_protocol = Path(parent_protocol['feature_source']['root'])/'protocol.json'
    study.verify(feature_protocol, parent_protocol['feature_source']['protocol_sha256'])
    paths.append(feature_protocol)
    for i, receipt in enumerate(support['receipts']):
        path = source/f'features/shard_{i:02d}/manifest.json'
        assert study.read(path) == receipt
        assert receipt['protocol_sha256'] == study.file_hash(feature_protocol)
        paths.append(path)
    assert len(support['receipts']) == 28
    zero = np.load(source/'support/zero.npy', mmap_mode='r')
    weights = np.load(source/'support/weights.npy', mmap_mode='r')
    base_zero = np.load(old.SOURCE/'support/zero.npy', mmap_mode='r')
    base_weights = np.load(old.SOURCE/'support/weights.npy', mmap_mode='r')
    bright_mass = 0.
    bright_count = 0
    for lo in range(0, len(base_zero), 1000000):
        hi = min(lo+1000000, len(base_zero))
        np.testing.assert_array_equal(zero[lo:hi], base_zero[lo:hi])
        bright = base_zero[lo:hi, 3] < 20
        np.testing.assert_array_equal(weights[lo:hi][~bright], base_weights[lo:hi][~bright])
        bright_mass += base_weights[lo:hi][bright].sum()
        bright_count += int(bright.sum())
    expected_weight = bright_mass/(bright_count+len(zero)-len(base_zero))
    for lo in range(0, len(zero), 1000000):
        hi = min(lo+1000000, len(zero))
        bright = zero[lo:hi, 3] < 20
        np.testing.assert_allclose(weights[lo:hi][bright], expected_weight, rtol=1.e-12, atol=0)
        if lo >= len(base_zero):
            assert bright.all()
    assert np.all(zero[len(base_zero):, 3] < 20)
    np.testing.assert_allclose(weights.sum(), base_weights.sum(), rtol=0, atol=1.e-12)
    # Verify cached coordinates, including all old-atom floors, rather than
    # silently refitting the proposal metric on the oversampled bright stratum.
    new_coords = study.ProposalCoordinateTable.load(source/'assembled/proposal')
    old_coords = study.ProposalCoordinateTable.load(old.SOURCE/'assembled/proposal')
    for key in ('values', 'dispersion'):
        np.testing.assert_array_equal(getattr(new_coords, key)[:len(base_zero)], getattr(old_coords, key))
    del new_coords, old_coords
    save(ROOT/'execution.json', dict(driver_sha256=study.file_hash(__file__),
        protocol_sha256=study.file_hash(ROOT/'protocol.json'), matched_execution=matched.identity(),
        sources={str(path): study.file_hash(path) for path in paths}))
    save(ROOT/'prior_audit.json', dict(status='passed', n_atoms=len(zero), n_old=len(base_zero),
        n_added=len(zero)-len(base_zero), n_old_bright=bright_count, bright_mass=float(bright_mass),
        bright_atom_weight=float(expected_weight), total_mass=float(weights.sum()),
        unchanged_nonbright_weights=True, unchanged_old_features_and_coordinates=True,
        frozen_proposal_metric=True, n_retained=gen['n_observations']))
    save(ROOT/'status.json', dict(status='initialized', n_atoms=len(zero), n_retained=16314))
    print('EXPANDED_PRIOR_AUDIT_PASS', flush=True)


def normalize(worker_index):
    source, p = expanded(), protocol()
    flow = study.load_measurement_model(study.V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = study.load_detection_classifier(study.V39_LIKE, device='cuda')
    points = study.stencil(p['center'], .001)
    receipts = study.read(source/'support/manifest.json')['receipts']
    offsets = np.r_[0, np.cumsum([r['n_rows'] for r in receipts])]
    # Existing proposals use the first cached reference point (zero shear).
    # Retain old-atom values and evaluate only the new atoms at that point.
    base_prob = np.load(old.SOURCE/'assembled/probability.npy', mmap_mode='r')[0]
    proposal_center = study.read(old.SOURCE/'assembled/manifest.json')['points'][0]
    for index in range(worker_index, 28, 4):
        folder = source/f'features/shard_{index:02d}'
        rec = receipts[index]
        for name in ('flow_zero.parquet', 'prior_weight.npy'):
            study.verify(folder/name, rec['output_sha256'][name])
        zero, weights = pd.read_parquet(folder/'flow_zero.parquet'), np.load(folder/'prior_weight.npy')
        assert len(zero) == len(weights) == rec['n_rows'] and tuple(zero.columns) == study.FEATURES
        cache = study.V39CatalogueCache(zero, weights, rec['conditions'], detector=detector)
        selection = study.CatalogueSelection(study.output_cut(flow), n_samples=64,
            seed=study.seed(8101, index), row_chunk=8192)
        masses, probabilities = [], []
        for point in points:
            view = cache.get(*point)
            probabilities.append(view.detection_probability)
            masses.append(study.detected_selected_mass_shard(selection, flow, view, weights,
                active_indices=np.arange(len(zero)), random_offset_rows=0))
            cache.discard_views([point])
            print(f'EXPANDED_NORMALIZATION worker={worker_index} shard={index} point={point} mass={masses[-1]:.9g}', flush=True)
        if index < 20:
            local_probability = base_prob[offsets[index]:offsets[index+1]]
        else:
            local_probability = cache.get(*proposal_center).detection_probability
        out = ROOT/f'normalization/shard_{index:02d}'
        out.mkdir(parents=True, exist_ok=False)
        np.save(out/'probability.npy', np.asarray(probabilities))
        np.save(out/'local_probability.npy', local_probability)
        save(out/'manifest.json', dict(status='complete', shard=index, n_rows=len(zero),
            protocol_sha256=study.file_hash(ROOT/'protocol.json'), points=points, masses=masses,
            proposal_center=proposal_center, output_sha256={f.name: study.file_hash(f) for f in out.glob('*.npy')}))
    del flow, detector
    gc.collect()
    torch.cuda.empty_cache()


def assemble():
    p = protocol()
    points = study.stencil(p['center'], .001)
    probability = np.lib.format.open_memmap(ROOT/'probability.npy', mode='w+', dtype=np.float64,
                                          shape=(len(points), p['n_atoms']))
    local_probability = np.lib.format.open_memmap(ROOT/'local_probability.npy', mode='w+',
                                                dtype=np.float64, shape=(p['n_atoms'],))
    offset, masses = 0, np.zeros(len(points))
    for i in range(28):
        folder = ROOT/f'normalization/shard_{i:02d}'
        rec = study.read(folder/'manifest.json')
        assert rec['status'] == 'complete' and rec['shard'] == i
        assert rec['points'] == list(map(list, points))
        assert rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        for name, digest in rec['output_sha256'].items():
            study.verify(folder/name, digest)
        stop = offset+rec['n_rows']
        probability[:, offset:stop] = np.load(folder/'probability.npy', mmap_mode='r')
        local_probability[offset:stop] = np.load(folder/'local_probability.npy', mmap_mode='r')
        masses += rec['masses']
        offset = stop
    assert offset == p['n_atoms'] and np.isfinite(masses).all() and np.all((masses > 0) & (masses <= 1))
    probability.flush()
    local_probability.flush()
    study.ExactPopulationNormalization(dict(zip(points, masses)), .001, identity(),
        dict(method_detail='Expanded weighted prior; 64 CRN QMC draws per atom/node; original shard seeds')
        ).save(ROOT/'normalization.json')
    save(ROOT/'assembled.json', dict(status='complete', points=points, selected_mass=masses.tolist(),
        output_sha256={name: study.file_hash(ROOT/name) for name in
                      ('probability.npy', 'local_probability.npy', 'normalization.json')}))
    print('EXPANDED_NORMALIZATION_ASSEMBLED', flush=True)


def load():
    source, execution = expanded(), identity()
    rec = study.read(ROOT/'assembled.json')
    for name, digest in rec['output_sha256'].items():
        study.verify(ROOT/name, digest)
    support = study.read(source/'support/manifest.json')
    for name, digest in support['output_sha256'].items():
        study.verify(source/'support'/name, digest)
    zero = pd.DataFrame(np.load(source/'support/zero.npy', mmap_mode='r'), columns=study.FEATURES)
    weights = np.load(source/'support/weights.npy')
    probabilities = np.load(ROOT/'probability.npy', mmap_mode='r')
    cache = study.V39CatalogueCache(zero, weights, support['receipts'][0]['conditions'],
        probabilities=dict(zip(map(tuple, rec['points']), probabilities)))
    flow = study.load_measurement_model(study.V39_LIKE.flow_checkpoints[0], device='cuda')
    response = study.response_from_receipts(support['receipts'],
        study.load_disk_response(study.V39_LIKE, device='cuda', nthread=4))
    likelihood = study.DiskCatalogueLikelihood(flow, cache, disk_response=response,
        selection=study.CatalogueSelection(study.output_cut(flow)))
    likelihood.population_normalization = study.ExactPopulationNormalization.from_payload(study.read(ROOT/'normalization.json'))
    assert likelihood.population_normalization.identity == execution
    likelihood.compile_precision_density()
    record = study.read(source/'assembled/manifest.json')
    for name in ('proposal/coordinates.npz', 'proposal/manifest.json'):
        study.verify(source/'assembled'/name, record['output_sha256'][name])
    coords = study.ProposalCoordinateTable.load(source/'assembled/proposal')
    object.__setattr__(coords, '_metric_center', np.asarray(record['metric_center']))
    object.__setattr__(coords, '_metric_scale', np.asarray(record['metric_scale']))
    proposal = study.DefensiveLocalProposal(coords, weights,
        local_base_weights=np.load(ROOT/'local_probability.npy'), score_dtype=torch.float64)
    return likelihood, proposal


def observations():
    gen = matched.generation()
    return dict(mock=study.MockCatalogue.load(MATCHED/'mock/evaluation'),
                actual=study.MockCatalogue.load(matched.actual_root()/'input')), gen


def preflight(likelihood, proposal, samples):
    checks = {}
    for name, sample in samples.items():
        mag = sample.measurements.measured_mag_auto.to_numpy()
        groups = np.unique(np.r_[np.arange(8), np.flatnonzero(mag < 20)[:8]//4])
        rows = (groups[:, None]*4+np.arange(4)).ravel()
        frame = study.MockCatalogue(sample.measurements.iloc[rows], sample.truth.iloc[rows])
        ids = frame.truth.parent_sample_row.to_numpy(np.int64)
        reference = study.run_adaptive_section5(likelihood, frame, proposal,
            center=protocol()['center'], h=.001, draw_ladder=old.LADDER, n_candidates=1024,
            proposal_prefilter_candidates=131072, epsilon=.1, proposal_seed=8701,
            min_ess=32., max_weight_fraction=.5, candidate_backend='torch', estimator_mode='tilted_stratified',
            tilt_delta=.1, retain_full_ladder=True, full_information=True, object_ids=ids, object_chunk=4, atom_chunk=4096)
        checked = matched.evaluate(likelihood, proposal, sample, rows, protocol()['center'], f'expanded_replay_{name}')
        for key in old.KEYS:
            np.testing.assert_array_equal(checked[key], getattr(reference, key), err_msg=f'{name}/{key}')
        checks[name] = dict(n_rows=len(rows), original_parent_rows=ids.tolist(), exact_keys=list(old.KEYS))
    gpu = torch.cuda.get_device_properties(0)
    save(ROOT/'preflight.json', dict(status='passed', execution=identity(), checks=checks,
        gpu=gpu.name, total_memory=gpu.total_memory))
    print('EXPANDED_REPLAY_PASS', flush=True)


def wait(paths):
    started = time.monotonic()
    while not all(path.exists() for path in paths):
        if list(ROOT.glob('failure_*.json')):
            raise RuntimeError('another expanded-prior worker failed')
        if time.monotonic()-started > 3*3600:
            raise TimeoutError('expanded-prior stage barrier timed out')
        time.sleep(10)


def worker(index):
    execution = identity()
    normalize(index)
    wait([ROOT/f'normalization/shard_{i:02d}/manifest.json' for i in range(28)])
    if index == 0:
        assemble()
    wait([ROOT/'assembled.json'])
    likelihood, proposal = load()
    samples, gen = observations()
    if index == 0:
        preflight(likelihood, proposal, samples)
        save(ROOT/'status.json', dict(status='inference_running', n_retained=gen['n_observations']))
    wait([ROOT/'preflight.json'])
    replay = study.read(ROOT/'preflight.json')
    gpu = torch.cuda.get_device_properties(0)
    assert replay['status'] == 'passed' and replay['execution'] == execution
    assert (gpu.name, gpu.total_memory) == (replay['gpu'], replay['total_memory'])
    for name, sample in samples.items():
        assert len(sample.measurements) == 20000
        np.testing.assert_array_equal(sample.truth.parent_sample_row, np.arange(20000))
        folder = ROOT/f'moments/{name}'
        folder.mkdir(parents=True, exist_ok=True)
        lo, hi = 5000*index, 5000*(index+1)
        parts = []
        for start in range(lo, hi, 1000):
            rows = np.arange(start, min(start+1000, hi))
            values = matched.evaluate(likelihood, proposal, sample, rows, gen['center'],
                f'expanded_{name}_worker={index}_start={start}')
            np.savez_compressed(folder/f'worker_{index}_chunk_{start:05d}.npz', object_rows=rows, **values)
            parts.append(values)
            gc.collect()
            torch.cuda.empty_cache()
        values = {key: np.concatenate([part[key] for part in parts],
                  axis=1 if key.startswith(('ladder_', 'weight_')) else 0) for key in old.KEYS}
        path = folder/f'worker_{index}.npz'
        np.savez_compressed(path, object_rows=np.arange(lo, hi), **values)
        save(folder/f'worker_{index}.json', dict(status='complete', execution=execution,
            n_rows=hi-lo, output_sha256=study.file_hash(path)))
        print(f'EXPANDED_WORKER_SAMPLE_COMPLETE worker={index} sample={name}', flush=True)
    # Only worker0 remains for the small reduction; other GPU allocations end.
    if index == 0:
        wait([ROOT/f'moments/{name}/worker_{i}.json' for name in samples for i in range(4)])
        combine()


def combine():
    samples, gen = observations()
    ids = np.asarray(gen['selected_parent_rows'])
    np.testing.assert_array_equal(np.flatnonzero(~samples['mock'].truth.numerical_padding.to_numpy()), ids)
    arrays = {}
    for name in samples:
        parts = []
        for i in range(4):
            folder = ROOT/f'moments/{name}'
            rec = study.read(folder/f'worker_{i}.json')
            assert rec['status'] == 'complete' and rec['execution'] == identity()
            study.verify(folder/f'worker_{i}.npz', rec['output_sha256'])
            with np.load(folder/f'worker_{i}.npz') as data:
                np.testing.assert_array_equal(data['object_rows'], np.arange(5000*i, 5000*(i+1)))
                parts.append({k: data[k] for k in ('ladder_score', 'ladder_information')})
        arrays[name] = tuple(np.concatenate([part[k] for part in parts], axis=1)
                            for k in ('ladder_score', 'ladder_information'))
    arrays = dict(mock=tuple(x[:, ids] for x in arrays['mock']),
        actual_same_ids=tuple(x[:, ids] for x in arrays['actual']), actual_all20k=arrays['actual'])
    baseline = study.read(MATCHED/'comparison.json')
    assert baseline['center'] == gen['center'] and baseline['n_retained'] == len(ids)
    rungs, influences = [], {}
    for j, draws in enumerate(old.LADDER):
        row = dict(draws=draws, original_prior=baseline['rungs'][j])
        assert row['original_prior']['draws'] == draws
        for name, (score, info) in arrays.items():
            result, influences[j, name] = old.summarize(gen['center'], score[j], info[j])
            row[name] = dict(result, n_observations=len(score[j]), score_sum=score[j].sum(0).tolist(),
                information_sum=info[j].sum(0).tolist())
        rungs.append(row)
    ladder_change = {name: old.paired(rungs[-2][name], rungs[-1][name],
        influences[len(old.LADDER)-2, name], influences[len(old.LADDER)-1, name]) for name in arrays}
    _, gradient, hessian = log_mass_derivatives(
        study.ExactPopulationNormalization.from_payload(study.read(ROOT/'normalization.json')), gen['center'])
    save(ROOT/'comparison.json', dict(status='complete', execution=identity(), n_inputs=20000,
        n_retained=len(ids), n_atoms=protocol()['n_atoms'], center=gen['center'], rungs=rungs,
        paired_8192_to_16384=ladder_change, normalization_gradient=gradient.tolist(),
        normalization_hessian=hessian.tolist(),
        same_problem_at_shared_center=rungs[-1]['mock']['status'] == 'non_positive_information',
        limitation='Already selected input cohort; fixed-centre prior comparison, not converged shear or unbiased population closure.'))
    save(ROOT/'status.json', dict(status='comparison_complete', n_retained=len(ids),
        same_problem_at_shared_center=rungs[-1]['mock']['status'] == 'non_positive_information'))
    print('EXPANDED_COMPARISON_COMPLETE', {name: rungs[-1][name] for name in arrays}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize', 'worker', 'combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    try:
        if args.stage == 'worker':
            worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
        else:
            globals()[args.stage]()
    except Exception:
        save(ROOT/f'failure_{os.environ.get("SLURM_JOB_ID", "local")}.json',
             dict(stage=args.stage, traceback=traceback.format_exc()))
        save(ROOT/'status.json', dict(status='failed', stage=args.stage))
        raise
    print(f'EXPANDED_STAGE_COMPLETE stage={args.stage} seconds={time.monotonic()-started:.1f}', flush=True)
