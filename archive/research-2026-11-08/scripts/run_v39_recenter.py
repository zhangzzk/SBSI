"""Frozen four-worker, paired 20k inference diagnostic; no training or new mocks."""
import argparse
import dataclasses
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_RECENTER_ROOT'])
SOURCE = Path(os.environ['SBSI_STUDY_ROOT'])
sys.path.insert(0, str(SOURCE))
import run_v39_sampling_study as study  # noqa: E402
from v39_recenter_math import summarize, paired, next_center  # noqa: E402
import v39_recenter_math as reductions  # noqa: E402

np, torch = study.np, study.torch
KEYS = ('score', 'information', 'draw_counts', 'unique_counts', 'ladder_score',
        'ladder_information', 'weight_ess', 'weight_max_fraction',
        'weight_relative_error', 'weight_pareto_k')
LADDER = (512, 1024, 2048, 4096, 8192, 16384)
ARMS = {'gaussian': 131072, 'distance': 1024}


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    study.write_json(temporary, payload)
    temporary.replace(path)


def identity():
    execution = study.read(ROOT/'execution.json')
    study.verify(__file__, execution['driver_sha256'])
    study.verify(reductions.__file__, execution['reductions_sha256'])
    study.verify(ROOT/'protocol.json', execution['protocol_sha256'])
    assert study.identity() == execution['source_execution']
    for name, digest in execution['source_receipts'].items():
        study.verify(SOURCE/name, digest)
    return execution


def wait_for(paths):
    started = time.monotonic()
    while not all(p.exists() for p in paths):
        failures = list(ROOT.glob('failure_*.json'))
        if failures:
            raise RuntimeError(f'another worker failed: {failures}')
        if time.monotonic()-started > 6*3600:
            raise TimeoutError(f'phase barrier timed out: {paths}')
        time.sleep(10)


class CachedProposal:
    """Precompute retrieval and draws; density calls retain original groups of four."""
    def __init__(self, proposal, observed, ids, arm):
        self.proposal, self.observed, self.ids = proposal, observed.reset_index(drop=True), ids
        self.arm = arm
        self.position = self.draw_position = 0
        candidates, draws = [], []
        # Gaussian pool intermediates are larger than the distance-only ones.
        chunk = 24 if arm == 'gaussian' else 128
        for lo in range(0, len(ids), chunk):
            frame = self.observed.iloc[lo:lo+chunk].reset_index(drop=True)
            local = proposal.candidates(frame, n_candidates=1024,
                prefilter_candidates=ARMS[arm], torch_device='cuda')
            torch.cuda.empty_cache()
            draw = proposal.draw_tilted(local, frame, n_draws=16384, delta=.1,
                seed=8701, object_ids=ids[lo:lo+chunk], device='cuda', object_chunk=24)
            candidates.append(local)
            draws.append(draw)
            torch.cuda.empty_cache()
        self.local = self.join(candidates)
        self.draw = self.join(draws)

    @staticmethod
    def join(values):
        payload = {}
        for field in dataclasses.fields(values[0]):
            data = [getattr(v, field.name) for v in values]
            if isinstance(data[0], np.ndarray):
                payload[field.name] = np.concatenate(data)
            else:
                assert all(x == data[0] for x in data)
                payload[field.name] = data[0]
        return type(values[0])(**payload)

    def __getattr__(self, name):
        return getattr(self.proposal, name)

    def candidates(self, observed, *, n_candidates, prefilter_candidates, torch_device):
        assert n_candidates == 1024 and prefilter_candidates == ARMS[self.arm]
        assert str(torch_device) == 'cuda'
        lo, hi = self.position, self.position+len(observed)
        assert observed.reset_index(drop=True).equals(self.observed.iloc[lo:hi].reset_index(drop=True))
        self.position = hi
        return self.local.take(np.arange(lo, hi))

    def draw_tilted(self, candidates, observed, *, n_draws, delta, seed, object_ids, device, temperature=1.):
        assert (n_draws, delta, seed, temperature) == (16384, .1, 8701, 1.)
        assert str(device) == 'cuda'
        lo, hi = self.draw_position, self.draw_position+len(observed)
        np.testing.assert_array_equal(object_ids, self.ids[lo:hi])
        np.testing.assert_array_equal(candidates.indices, self.local.indices[lo:hi])
        self.draw_position = hi
        return self.draw.take(np.arange(lo, hi))

    def complete(self):
        assert self.position == self.draw_position == len(self.ids)


def evaluate(likelihood, proposal, mock, ids, center, arm, label):
    window = study.MockCatalogue(mock.measurements.iloc[ids].reset_index(drop=True),
                                 mock.truth.iloc[ids].reset_index(drop=True))
    result = study.run_adaptive_section5(likelihood, window, proposal,
        center=center, h=.001, draw_ladder=LADDER, n_candidates=1024,
        proposal_prefilter_candidates=ARMS[arm], epsilon=.1, proposal_seed=8701,
        min_ess=32., max_weight_fraction=.5, candidate_backend='torch',
        estimator_mode='tilted_stratified', tilt_delta=.1, retain_full_ladder=True,
        full_information=True, object_ids=ids, object_chunk=4, atom_chunk=4096,
        progress=lambda done, total, elapsed: print(
            f'INFERENCE {label} arm={arm} rows={done}/{total} seconds={elapsed:.1f}', flush=True))
    return {k: getattr(result, k) for k in KEYS}


def preflight():
    execution = identity()
    likelihood, proposal, mock, receipt = study.load_inference()
    likelihood.compile_precision_density()
    # Same hardware as these saved original groups, plus both disk-boundary groups.
    ids = (np.concatenate((np.arange(3000, 3128), np.arange(86172, 86176), np.arange(112152, 112156)))
           if len(mock.measurements) == 500000 else np.arange(136))
    reports, global_draws = {}, None
    for arm in ARMS:
        torch.cuda.reset_peak_memory_stats()
        reference = evaluate(likelihood, proposal, mock, ids, receipt['center'], arm, 'preflight_original')
        cached = CachedProposal(proposal, mock.measurements.iloc[ids], ids, arm)
        actual = evaluate(likelihood, cached, mock, ids, receipt['center'], arm, 'preflight_batched')
        cached.complete()
        differences = {}
        for key in KEYS:
            np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
            differences[key] = 0.
        for lo in range(0, len(ids), 4):
            frame = mock.measurements.iloc[ids[lo:lo+4]].reset_index(drop=True)
            local = proposal.candidates(frame, n_candidates=1024,
                prefilter_candidates=ARMS[arm], torch_device='cuda')
            draw = proposal.draw_tilted(local, frame, n_draws=16384, delta=.1,
                seed=8701, object_ids=ids[lo:lo+4], device='cuda')
            np.testing.assert_array_equal(local.indices, cached.local.indices[lo:lo+4])
            np.testing.assert_array_equal(draw.indices, cached.draw.indices[lo:lo+4])
            np.testing.assert_array_equal(draw.probability, cached.draw.probability[lo:lo+4])
        if global_draws is None:
            global_draws = (cached.draw.indices.copy(), cached.draw.probability.copy())
        else:
            np.testing.assert_array_equal(global_draws[0], cached.draw.indices)
            np.testing.assert_array_equal(global_draws[1], cached.draw.probability)
        if arm == 'distance' and (SOURCE/'inference/part_03/distance_moments.npz').exists():
            records = {p: dict(np.load(SOURCE/f'inference/part_{p:02d}/distance_moments.npz'))
                       for p in set(ids//1000)}
            for key in KEYS:
                axis = 1 if key.startswith(('ladder_', 'weight_')) else 0
                old = np.stack([records[i//1000][key][:, i%1000] if axis else
                                records[i//1000][key][i%1000] for i in ids], axis=axis)
                np.testing.assert_array_equal(actual[key], old, err_msg=f'original checkpoint {key}')
        gpu = torch.cuda.get_device_properties(0)
        peak = torch.cuda.max_memory_allocated()
        assert peak <= .85*gpu.total_memory, 'insufficient memory headroom'
        reports[arm] = dict(exact_replay=True, differences=differences, peak_allocated=peak)
        del cached
    save(ROOT/'preflight.json', dict(status='passed', execution=execution,
        object_ids=ids.tolist(), reports=reports, same_global_draws=True,
        original_checkpoint_replay=(SOURCE/'inference/part_03/distance_moments.npz').exists(),
        gpu=gpu.name, total_memory=gpu.total_memory))
    print('PREFLIGHT_PASS', flush=True)


def initialize():
    """Bind the newly generated reduced-prior source after its assembly completes."""
    assert not (ROOT/'execution.json').exists()
    source_execution = study.identity()
    assert study.read(SOURCE/'mock/manifest.json')['n_observations'] == 20000
    assert study.read(SOURCE/'assembled/manifest.json')['n_atoms'] == study.read(SOURCE/'protocol.json')['n_atoms']
    save(ROOT/'execution.json', dict(driver_sha256=study.file_hash(__file__),
        reductions_sha256=study.file_hash(reductions.__file__),
        protocol_sha256=study.file_hash(ROOT/'protocol.json'), source_execution=source_execution,
        source_receipts={n: study.file_hash(SOURCE/n) for n in
            ('execution.json', 'mock/manifest.json', 'support/manifest.json', 'assembled/manifest.json')}))
    save(ROOT/'pass_00/center.json', dict(center=study.read(SOURCE/'mock/manifest.json')['center'],
        source='fresh reduced-prior20k raw mean shape; not injected truth'))
    print('INITIALIZED_REDUCED_PRIOR_CHAIN', flush=True)


def normalize(pass_index, worker_index):
    folder = ROOT/f'pass_{pass_index:02d}'
    center = study.read(folder/'center.json')['center']
    points = study.stencil(center, .001)
    flow = study.load_measurement_model(study.V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = study.load_detection_classifier(study.V39_LIKE, device='cuda')
    for shard in range(worker_index, study.read(SOURCE/'protocol.json').get('prior_shards', 28), 4):
        zero, weights, receipt = study.features(shard)
        cache = study.V39CatalogueCache(zero, weights, receipt['conditions'], detector=detector)
        selection = study.CatalogueSelection(study.output_cut(flow), n_samples=64,
            seed=study.seed(8101, shard), row_chunk=8192)
        out = folder/f'normalization/shard_{shard:02d}'
        out.mkdir(parents=True, exist_ok=False)
        mass, probability = [], []
        for point in points:
            view = cache.get(*point)
            probability.append(view.detection_probability)
            mass.append(study.detected_selected_mass_shard(selection, flow, view, weights,
                active_indices=np.arange(len(zero)), random_offset_rows=0))
            cache.discard_views([point])
            print(f'NORMALIZATION pass={pass_index} shard={shard} point={point} mass={mass[-1]:.9g}', flush=True)
        np.save(out/'probability.npy', np.asarray(probability))
        save(out/'manifest.json', dict(status='complete', shard=shard, center=center,
            points=points, mass=mass, n_rows=len(zero),
            protocol_sha256=study.file_hash(ROOT/'protocol.json'),
            probability_sha256=study.file_hash(out/'probability.npy')))


def assemble(pass_index):
    folder = ROOT/f'pass_{pass_index:02d}'
    source_protocol = study.read(SOURCE/'protocol.json')
    records = [study.read(folder/f'normalization/shard_{i:02d}/manifest.json')
               for i in range(source_protocol.get('prior_shards', 28))]
    points = study.stencil(study.read(folder/'center.json')['center'], .001)
    masses = np.zeros(len(points))
    probability = np.lib.format.open_memmap(folder/'probability.npy', mode='w+',
        dtype=np.float64, shape=(len(points), source_protocol['n_atoms']))
    start = 0
    for i, record in enumerate(records):
        assert record['status'] == 'complete' and record['shard'] == i
        assert record['points'] == list(map(list, points))
        assert record['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        path = folder/f'normalization/shard_{i:02d}/probability.npy'
        study.verify(path, record['probability_sha256'])
        stop = start+record['n_rows']
        probability[:, start:stop] = np.load(path, mmap_mode='r')
        masses += record['mass']
        start = stop
    assert start == probability.shape[1] and np.all((masses > 0) & (masses <= 1))
    probability.flush()
    del probability
    study.ExactPopulationNormalization(dict(zip(points, masses)), .001, identity(),
        dict(method_detail='Full weighted prior; 64 QMC draws per atom; fresh nodes and fixed per-shard seeds')
        ).save(folder/'normalization.json')
    save(folder/'assembled.json', dict(status='complete', points=points,
        probability_sha256=study.file_hash(folder/'probability.npy'),
        normalization_sha256=study.file_hash(folder/'normalization.json')))


def load_pass(pass_index):
    likelihood, proposal, mock, receipt = study.load_inference()
    if pass_index:
        folder = ROOT/f'pass_{pass_index:02d}'
        record = study.read(folder/'assembled.json')
        study.verify(folder/'probability.npy', record['probability_sha256'])
        study.verify(folder/'normalization.json', record['normalization_sha256'])
        old = likelihood.cache
        probabilities = np.load(folder/'probability.npy', mmap_mode='r')
        cache = study.V39CatalogueCache(old.zero_flow, old.prior.weights, old.conditions,
            probabilities=dict(zip(map(tuple, record['points']), probabilities)))
        likelihood = study.DiskCatalogueLikelihood(likelihood.flow_model, cache,
            disk_response=likelihood.disk_response, selection=likelihood.selection)
        likelihood.population_normalization = study.ExactPopulationNormalization.from_payload(
            study.read(folder/'normalization.json'))
        assert likelihood.population_normalization.identity == identity()
    likelihood.compile_precision_density()
    return likelihood, proposal, mock


def inference(pass_index, worker_index):
    folder = ROOT/f'pass_{pass_index:02d}'
    center = study.read(folder/'center.json')['center']
    likelihood, proposal, mock = load_pass(pass_index)
    for partition in range(worker_index, 20, 4):
        ids = np.arange(partition*1000, (partition+1)*1000)
        out = folder/f'part_{partition:02d}'
        out.mkdir(exist_ok=False)
        for arm in ARMS:
            tick = time.monotonic()
            cached = CachedProposal(proposal, mock.measurements.iloc[ids], ids, arm)
            arrays = evaluate(likelihood, cached, mock, ids, center, arm,
                              f'pass={pass_index} part={partition}')
            cached.complete()
            del cached
            np.savez_compressed(out/f'{arm}.npz', object_rows=ids, **arrays)
            print(f'CHECKPOINT pass={pass_index} part={partition} arm={arm} seconds={time.monotonic()-tick:.1f}', flush=True)
        save(out/'manifest.json', dict(status='complete', center=center,
            protocol_sha256=study.file_hash(ROOT/'protocol.json'),
            normalization_sha256=study.file_hash(folder/'normalization.json' if pass_index else SOURCE/'assembled/normalization.json'),
            output_sha256={arm: study.file_hash(out/f'{arm}.npz') for arm in ARMS}))


def combine(pass_index):
    folder = ROOT/f'pass_{pass_index:02d}'
    center = study.read(folder/'center.json')['center']
    parts = []
    for partition in range(20):
        out = folder/f'part_{partition:02d}'
        record = study.read(out/'manifest.json')
        assert record['status'] == 'complete' and record['center'] == center
        assert record['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        expected_norm = folder/'normalization.json' if pass_index else SOURCE/'assembled/normalization.json'
        assert record['normalization_sha256'] == study.file_hash(expected_norm)
        data = {}
        for arm in ARMS:
            study.verify(out/f'{arm}.npz', record['output_sha256'][arm])
            data[arm] = dict(np.load(out/f'{arm}.npz'))
            np.testing.assert_array_equal(data[arm]['object_rows'], np.arange(partition*1000, (partition+1)*1000))
        parts.append(data)
    rungs, influences = [], {}
    for i, draws in enumerate(LADDER):
        rung = dict(draws=draws)
        for arm in ARMS:
            score = np.concatenate([p[arm]['ladder_score'][i] for p in parts])
            information = np.concatenate([p[arm]['ladder_information'][i] for p in parts])
            rung[arm], influences[arm, draws] = summarize(center, score, information)
        rung['distance_minus_gaussian'] = paired(rung['gaussian'], rung['distance'],
            influences['gaussian', draws], influences['distance', draws])
        rungs.append(rung)
    draw_change = {arm: paired(rungs[-2][arm], rungs[-1][arm],
        influences[arm, 8192], influences[arm, 16384]) for arm in ARMS}
    primary = rungs[-1]['gaussian']
    result = dict(status='complete', pass_index=pass_index, center=center, n_observations=20000,
        protocol_sha256=study.file_hash(ROOT/'protocol.json'), rungs=rungs,
        paired_draw_change=draw_change, primary_arm='gaussian', primary_draws=16384,
        limitation='Conditional row errors; one mock and seed; no absolute 0.3% acceptance.')
    save(folder/'result.json', result)
    chain = [study.read(ROOT/f'pass_{i:02d}/result.json') for i in range(pass_index+1)]
    # Iteration and draw stability are separate, neither refers to injected shear.
    iteration_stable = len(chain) >= 2 and all(
        c['rungs'][-1]['gaussian'].get('small_step', False) for c in chain[-2:])
    draw_stable = all(d['status'] == 'estimated' and
        np.all(np.abs(d['difference'])+2*np.asarray(d['standard_error']) <= 6.e-5)
        for d in draw_change.values())
    status = 'iteration_stable' if iteration_stable else 'pass_limit_reached' if pass_index == 3 else 'continuing'
    try:
        point = next_center(primary)
    except ValueError as error:
        point, status = None, 'stopped_invalid_newton_step'
        result['stop_reason'] = str(error)
        save(folder/'result.json', result)
    if point is not None and status == 'continuing':
        save(ROOT/f'pass_{pass_index+1:02d}/center.json', dict(center=point,
            source=f'pass_{pass_index:02d}/result.json', source_sha256=study.file_hash(folder/'result.json')))
    save(ROOT/'chain.json', dict(status=status, completed_passes=pass_index+1,
        iteration_stable=iteration_stable, draw_stable=bool(draw_stable),
        accepted_0_3_percent=False, results=[f'pass_{i:02d}/result.json' for i in range(pass_index+1)]))
    # Publish last: workers must not advance before all derived receipts exist.
    save(folder/'decision.json', dict(status=status, next_center=point))
    print(f'PASS_COMPLETE pass={pass_index} status={status} primary={primary} draw_change={draw_change}', flush=True)


def worker(index):
    execution = identity()
    preflight_record = study.read(ROOT/'preflight.json')
    assert preflight_record['status'] == 'passed' and preflight_record['execution'] == execution
    gpu = torch.cuda.get_device_properties(0)
    assert (gpu.name, gpu.total_memory) == (preflight_record['gpu'], preflight_record['total_memory'])
    for pass_index in range(4):
        folder = ROOT/f'pass_{pass_index:02d}'
        wait_for([folder/'center.json'])
        if pass_index:
            normalize(pass_index, index)
            gc.collect()
            torch.cuda.empty_cache()
            wait_for([folder/f'normalization/shard_{i:02d}/manifest.json'
                      for i in range(study.read(SOURCE/'protocol.json').get('prior_shards', 28))])
            if index == 0:
                assemble(pass_index)
            wait_for([folder/'assembled.json'])
        inference(pass_index, index)
        gc.collect()
        torch.cuda.empty_cache()
        wait_for([folder/f'part_{i:02d}/manifest.json' for i in range(20)])
        if index == 0:
            combine(pass_index)
        wait_for([folder/'decision.json'])
        if study.read(folder/'decision.json')['status'] != 'continuing':
            break
    save(ROOT/f'worker_{index}.json', dict(status='complete', job=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize', 'preflight', 'worker'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.stage == 'initialize':
            initialize()
        elif args.stage == 'preflight':
            preflight()
        else:
            worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
    except Exception:
        save(ROOT/f"failure_{os.environ.get('SLURM_JOB_ID', 'local')}.json",
             dict(status='failed', traceback=traceback.format_exc()))
        raise
