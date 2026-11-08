"""Five measured-mag<20 BFGS iterations, carrying all other observations."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_BRIGHT_ROOT'])
OLD = Path(os.environ['SBSI_RECENTER_ROOT'])
sys.path.insert(0, str(OLD))
import run_v39_recenter as old  # noqa: E402
# The continuation's frozen helpers take precedence over unrelated source files.
sys.path.insert(0, str(ROOT))
from combine_hybrid_pass import _bfgs_information, _summarize  # noqa: E402
import combine_hybrid_pass as historical  # noqa: E402
from v39_bright_math import carry  # noqa: E402
import v39_bright_math as bright_math  # noqa: E402
import v39_image_input as image_input  # noqa: E402
from sbsi.selection_normalization import ExactPopulationNormalization, log_mass_derivatives  # noqa: E402

study, np, torch, save = old.study, old.np, old.torch, old.save


def is_image():
    return study.read(ROOT/'protocol.json').get('input_kind') == 'image'


def identity():
    execution = study.read(ROOT/'execution.json')
    for path, key in [(__file__, 'driver_sha256'), (historical.__file__, 'historical_sha256'),
                      (bright_math.__file__, 'math_sha256'), (ROOT/'protocol.json', 'protocol_sha256')]:
        study.verify(path, execution[key])
    assert old.identity() == execution['historical_execution']
    for path, sha in execution['sources'].items():
        study.verify(path, sha)
    if is_image():
        study.verify(image_input.__file__, execution['image_input_sha256'])
        if (ROOT/'base_stage.json').exists():
            stage = study.read(ROOT/'base_stage.json')
            assert stage['execution_sha256'] == study.file_hash(ROOT/'execution.json')
            for path, sha in stage['sources'].items():
                study.verify(path, sha)
    return execution


def normalization(path, center):
    return log_mass_derivatives(ExactPopulationNormalization.from_payload(study.read(path)), center)[1:]


def read_moments(index):
    parts, hashes = [], {}
    for partition in range(20):
        folder = OLD/f'pass_{index:02d}/part_{partition:02d}'
        manifest = study.read(folder/'manifest.json')
        path = folder/'gaussian.npz'
        study.verify(path, manifest['output_sha256']['gaussian'])
        hashes[str(path)] = manifest['output_sha256']['gaussian']
        with np.load(path) as data:
            values = {key: data[key] for key in data.files}
        np.testing.assert_array_equal(values['object_rows'], np.arange(partition*1000, (partition+1)*1000))
        parts.append(values)
    return {key: np.concatenate([p[key] for p in parts], axis=1 if key.startswith(('ladder_', 'weight_')) else 0)
            for key in parts[0]}, hashes


def initialize():
    assert not (ROOT/'execution.json').exists()
    protocol = study.read(ROOT/'protocol.json')
    assert protocol['iterations'] == 5 and protocol['measured_bright_mag_lt'] == 20
    if is_image():
        return initialize_image(protocol)
    base, hashes0 = read_moments(0)
    cached_first, hashes1 = read_moments(1)
    c0, c1 = [np.asarray(study.read(OLD/f'pass_{i:02d}/center.json')['center']) for i in (0, 1)]
    mock = study.MockCatalogue.load(old.SOURCE/'mock')
    mag = 30-2.5*np.log10(mock.measurements['measured_flux_from_mag_auto'].to_numpy())
    bright_ids = np.flatnonzero(mag < 20)
    assert len(mag) == 20000 and 0 < len(bright_ids) < len(mag)
    groups = np.unique(bright_ids//4)
    eval_ids = (groups[:, None]*4+np.arange(4)[None, :]).ravel()
    initial = _summarize(c0, base['score'], base['information'],
                          solver_method='first_full_pass_observed_information')
    np.testing.assert_allclose(initial['estimate'], c1, atol=1.e-14, rtol=0)
    np.savez_compressed(ROOT/'base_moments.npz', **base)
    np.savez_compressed(ROOT/'cached_full_pass01.npz', **cached_first)
    norm_path = old.SOURCE/'assembled/normalization.json'
    grad, hess = normalization(norm_path, c0)
    save(ROOT/'base.json', dict(center=c0.tolist(), normalization=str(norm_path),
        normalization_gradient=grad.tolist(), normalization_hessian=hess.tolist(),
        source='Original first full evaluation, pass_00; fixed reference for every carried row',
        base_moments_sha256=study.file_hash(ROOT/'base_moments.npz')))
    save(ROOT/'selection.json', dict(n_observations=len(mag), bright_count=len(bright_ids),
        measured_mag_lt=20, bright_ids=bright_ids.tolist(), evaluated_group_ids=eval_ids.tolist(),
        evaluated_count=len(eval_ids), padding_count=len(eval_ids)-len(bright_ids),
        meaning='Only measured-mag<20 moments replaced. Other members of original four-row GPU groups are numerical padding.'))
    save(ROOT/'initial_state.json', dict(hybrid=initial, update=dict(method='no_BFGS_update_before_first_bright_iteration')))
    save(ROOT/'pass_00/center.json', dict(center=initial['estimate'], source='initial_state.json'))
    source_files = [norm_path, OLD/'pass_01/normalization.json',
                    ROOT/'base.json', ROOT/'base_moments.npz', ROOT/'cached_full_pass01.npz',
                    ROOT/'selection.json', ROOT/'initial_state.json']
    save(ROOT/'execution.json', dict(driver_sha256=study.file_hash(__file__),
        historical_sha256=study.file_hash(historical.__file__), math_sha256=study.file_hash(bright_math.__file__),
        protocol_sha256=study.file_hash(ROOT/'protocol.json'), historical_execution=old.identity(),
        sources={**hashes0, **hashes1, **{str(p): study.file_hash(p) for p in source_files}}))
    save(ROOT/'chain.json', dict(status='initialized', completed_iterations=0, requested_iterations=5,
        initial_center=initial['estimate'], bright_count=len(bright_ids), accepted_0_3_percent=False))
    print(f'INITIALIZED bright={len(bright_ids)} evaluated_with_padding={len(eval_ids)} center={initial["estimate"]}', flush=True)


def initialize_image(protocol):
    parent = Path(protocol['measurement_parent'])
    study.verify(parent/'image_mock_manifest.json', protocol['measurement_parent_manifest_sha256'])
    manifest = study.read(parent/'image_mock_manifest.json')
    for name, sha in manifest['output_sha256'].items():
        study.verify(parent/name, sha)
    measured, truth = image_input.selected_prefix(study.pd.read_parquet(parent/'measurements.parquet'),
        study.pd.read_parquet(parent/'truth.parquet'), manifest, protocol['n_observations'])
    folder = ROOT/'input'
    folder.mkdir()
    measured.to_parquet(folder/'measurements.parquet', index=False)
    truth.to_parquet(folder/'truth.parquet', index=False)
    audit_keys = ('crossmatch_without_shape', 'shape_without_crossmatch', 'matched_without_truth',
                  'outside_truth_domain', 'invalid_or_unusable', 'selected')
    save(folder/'manifest.json', dict(input_kind='image', catalogue='ConstGold', leg='plus',
        n_observations=len(measured), subset='first 20000 rows of randomly ordered uniform 500k sample',
        source_manifest=str(parent/'image_mock_manifest.json'), source_manifest_sha256=protocol['measurement_parent_manifest_sha256'],
        source_sample_seed=manifest['sample_seed'], truth_property_cuts=None, new_drops=0,
        case_counts={str(k): int(v) for k, v in truth.source_case.value_counts().sort_index().items()},
        original_catalogue_audit={k: sum(p[k] for p in manifest['per_case']) for k in audit_keys},
        shape_log_gap='Derived as log1p(-g1*g1-g2*g2) from recorded strictly interior measured shapes; no clipping',
        adapter_model_hash_provenance_only=manifest['measurement_model_sha256'],
        inference_models='Pinned V3.9 F8/C8/original disk, bound by execution historical source identity',
        output_sha256={p.name: study.file_hash(p) for p in folder.glob('*.parquet')}))
    bright = np.flatnonzero(measured.measured_mag_auto.to_numpy() < 20)
    assert len(measured) == 20000 and 0 < len(bright) < len(measured)
    ids = (np.unique(bright//4)[:, None]*4+np.arange(4)[None, :]).ravel()
    save(ROOT/'selection.json', dict(n_observations=len(measured), bright_count=len(bright),
        measured_mag_lt=20, bright_ids=bright.tolist(), evaluated_group_ids=ids.tolist(),
        evaluated_count=len(ids), padding_count=len(ids)-len(bright),
        meaning='Only measured-mag<20 moments replaced; original four-row groups retained for numerical replay.'))
    center = measured[['measured_ngmix_g1', 'measured_ngmix_g2']].mean().to_numpy()
    save(ROOT/'pass_00/center.json', dict(center=center.tolist(), source='actual measurements raw mean shape'))
    paths = [parent/'image_mock_manifest.json', folder/'manifest.json', folder/'measurements.parquet',
             folder/'truth.parquet', ROOT/'selection.json', ROOT/'pass_00/center.json']
    save(ROOT/'execution.json', dict(driver_sha256=study.file_hash(__file__),
        historical_sha256=study.file_hash(historical.__file__), math_sha256=study.file_hash(bright_math.__file__),
        image_input_sha256=study.file_hash(image_input.__file__), protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        historical_execution=old.identity(), sources={str(p): study.file_hash(p) for p in paths}))
    save(ROOT/'chain.json', dict(status='awaiting_first_full_pass', completed_iterations=0,
        requested_iterations=5, input_kind='image', bright_count=len(bright), accepted_0_3_percent=False))
    print(f'IMAGE_INITIALIZED n={len(measured)} bright={len(bright)} padding={len(ids)-len(bright)} raw_mean={center}', flush=True)


def cached_first():
    """Reuse exactly matching bright rows and normalization at the first estimate."""
    identity()
    assert not is_image(), 'Actual measurements require their own full pass and all five fresh bright updates'
    folder = ROOT/'pass_00'
    center = study.read(folder/'center.json')['center']
    assert center == study.read(OLD/'pass_01/center.json')['center']
    source_norm = OLD/'pass_01/normalization.json'
    # This cache has the historical execution identity; it is used only for
    # this saved-moment combination, not loaded as a new GPU likelihood.
    save(folder/'normalization.json', study.read(source_norm))
    selected = study.read(ROOT/'selection.json')
    groups = np.asarray(selected['evaluated_group_ids']).reshape(-1, 4)
    with np.load(ROOT/'cached_full_pass01.npz') as data:
        for worker in range(4):
            ids = groups[worker::4].ravel()
            arrays = {k: data[k][:, ids] if k.startswith(('ladder_', 'weight_')) else data[k][ids]
                      for k in old.KEYS}
            path = folder/f'worker_{worker}.npz'
            np.savez_compressed(path, object_rows=ids, **arrays)
            save(folder/f'worker_{worker}.json', dict(status='complete', center=center, rows=len(ids),
                output_sha256=study.file_hash(path), normalization_sha256=study.file_hash(folder/'normalization.json'),
                protocol_sha256=study.file_hash(ROOT/'protocol.json'),
                provenance='Reused matching saved full pass; only measured-mag<20 rows replace the carry'))
    combine(0)


def preflight():
    execution = identity()
    if is_image():
        likelihood, proposal, _, receipt = study.load_inference()
        likelihood.compile_precision_density()
        mock = study.MockCatalogue.load(ROOT/'input')
        bright_groups = np.asarray(study.read(ROOT/'selection.json')['evaluated_group_ids'])[:128]
        ids = np.unique(np.concatenate((np.arange(128), bright_groups)))
        reference = old.evaluate(likelihood, proposal, mock, ids, receipt['center'], 'gaussian', 'image_original')
        cached = old.CachedProposal(proposal, mock.measurements.iloc[ids], ids, 'gaussian')
        actual = old.evaluate(likelihood, cached, mock, ids, receipt['center'], 'gaussian', 'image_batched_proposal')
        cached.complete()
        for key in old.KEYS:
            np.testing.assert_array_equal(actual[key], reference[key], err_msg=f'image proposal replay: {key}')
        gpu = torch.cuda.get_device_properties(0)
        save(ROOT/'preflight.json', dict(status='passed', execution=execution, exact_replay_keys=list(old.KEYS),
            n_rows=len(ids), gpu=gpu.name, total_memory=gpu.total_memory,
            purpose='Actual-input numerical replay only at cached reference centre; excluded from scientific estimates'))
        print('IMAGE_PREFLIGHT_PASS', flush=True)
        return
    likelihood, proposal, mock = old.load_pass(1)
    ids = np.asarray(study.read(ROOT/'selection.json')['evaluated_group_ids'])
    center = study.read(OLD/'pass_01/center.json')['center']
    actual = old.evaluate(likelihood, proposal, mock, ids, center, 'gaussian', 'bright20_replay')
    with np.load(ROOT/'cached_full_pass01.npz') as base:
        for key in old.KEYS:
            expected = base[key][:, ids] if key.startswith(('ladder_', 'weight_')) else base[key][ids]
            np.testing.assert_array_equal(actual[key], expected, err_msg=f'bright20 four-row replay: {key}')
    gpu = torch.cuda.get_device_properties(0)
    save(ROOT/'preflight.json', dict(status='passed', execution=execution, exact_replay_keys=list(old.KEYS),
        n_rows=len(ids), gpu=gpu.name, total_memory=gpu.total_memory))
    print('BRIGHT20_PREFLIGHT_PASS', flush=True)


def wait(paths):
    started = time.monotonic()
    while not all(p.exists() for p in paths):
        if list(ROOT.glob('failure_*.json')):
            raise RuntimeError('another continuation worker failed; see failure receipt')
        if time.monotonic()-started > 6*3600:
            raise TimeoutError('continuation barrier exceeded six hours')
        time.sleep(10)


def normalize(index, worker):
    # Reuse the frozen, validated full-prior preparation exactly. Only its output
    # root changes; restore the historical module state before any identity check.
    before = old.ROOT
    try:
        old.ROOT = ROOT
        old.normalize(index, worker)
    finally:
        old.ROOT = before


def assemble(index):
    folder = ROOT/f'pass_{index:02d}'
    center = study.read(folder/'center.json')['center']
    points = study.stencil(center, .001)
    count = study.read(old.SOURCE/'protocol.json')['n_atoms']
    probability = np.lib.format.open_memmap(folder/'probability.npy', mode='w+', dtype=np.float64,
                                           shape=(len(points), count))
    masses, start = np.zeros(len(points)), 0
    for shard in range(20):
        part = folder/f'normalization/shard_{shard:02d}'
        rec = study.read(part/'manifest.json')
        assert rec['status'] == 'complete' and rec['points'] == list(map(list, points))
        assert rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        study.verify(part/'probability.npy', rec['probability_sha256'])
        stop = start+rec['n_rows']
        probability[:, start:stop] = np.load(part/'probability.npy', mmap_mode='r')
        masses += rec['mass']
        start = stop
    assert start == count and np.all((masses > 0) & (masses <= 1))
    probability.flush()
    del probability
    ExactPopulationNormalization(dict(zip(points, masses)), .001, identity(),
        dict(method_detail='Fresh full weighted prior, 64 CRN QMC draws per atom, every bright BFGS centre')).save(folder/'normalization.json')
    save(folder/'assembled.json', dict(status='complete', points=points,
        probability_sha256=study.file_hash(folder/'probability.npy'),
        normalization_sha256=study.file_hash(folder/'normalization.json')))


def evaluate(index, worker):
    folder = ROOT/f'pass_{index:02d}'
    center = study.read(folder/'center.json')['center']
    likelihood, proposal, mock, _ = study.load_inference()
    image_mode = is_image()
    if image_mode:
        mock = study.MockCatalogue.load(ROOT/'input')
    rec = study.read(folder/'assembled.json')
    for name in ('probability.npy', 'normalization.json'):
        study.verify(folder/name, rec[name.split('.')[0]+'_sha256'])
    cache = study.V39CatalogueCache(likelihood.cache.zero_flow, likelihood.cache.prior.weights,
        likelihood.cache.conditions, probabilities=dict(zip(map(tuple, rec['points']),
                                                            np.load(folder/'probability.npy', mmap_mode='r'))))
    likelihood = study.DiskCatalogueLikelihood(likelihood.flow_model, cache,
        disk_response=likelihood.disk_response, selection=likelihood.selection)
    likelihood.population_normalization = ExactPopulationNormalization.from_payload(study.read(folder/'normalization.json'))
    assert likelihood.population_normalization.identity == identity()
    likelihood.compile_precision_density()
    selection = study.read(ROOT/'selection.json')
    if image_mode and index == 0:
        ids = np.arange(worker*5000, (worker+1)*5000)
    else:
        groups = np.asarray(selection['evaluated_group_ids']).reshape(-1, 4)[worker::4]
        ids = groups.ravel()
    if len(ids):
        if image_mode:
            parts = []
            for start in range(0, len(ids), 1000):
                rows = ids[start:start+1000]
                cached = old.CachedProposal(proposal, mock.measurements.iloc[rows], rows, 'gaussian')
                part = old.evaluate(likelihood, cached, mock, rows, center, 'gaussian',
                                    f'image pass={index} worker={worker} start={start}')
                cached.complete()
                np.savez_compressed(folder/f'worker_{worker}_chunk_{start:05d}.npz', object_rows=rows, **part)
                parts.append(part)
                del cached
                torch.cuda.empty_cache()
            data = {k: np.concatenate([p[k] for p in parts], axis=1 if k.startswith(('ladder_', 'weight_')) else 0)
                    for k in old.KEYS}
        else:
            data = old.evaluate(likelihood, proposal, mock, ids, center, 'gaussian', f'bright20 iteration={index+1} worker={worker}')
    else:
        # Keep the worker/barrier protocol well-defined even for a tiny bright set.
        with np.load(ROOT/'base_moments.npz') as base:
            data = {k: base[k][:, :0] if k.startswith(('ladder_', 'weight_')) else base[k][:0] for k in old.KEYS}
    path = folder/f'worker_{worker}.npz'
    np.savez_compressed(path, object_rows=ids, **data)
    save(folder/f'worker_{worker}.json', dict(status='complete', center=center, rows=len(ids),
        output_sha256=study.file_hash(path), normalization_sha256=rec['normalization_sha256'],
        protocol_sha256=study.file_hash(ROOT/'protocol.json')))


def combine_full():
    """Establish the actual sample's fixed carry base and first full-pass estimate."""
    folder = ROOT/'pass_00'
    center = np.asarray(study.read(folder/'center.json')['center'])
    parts = []
    for worker in range(4):
        rec = study.read(folder/f'worker_{worker}.json')
        assert rec['status'] == 'complete' and rec['center'] == center.tolist()
        assert rec['normalization_sha256'] == study.file_hash(folder/'normalization.json')
        assert rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        path = folder/f'worker_{worker}.npz'
        study.verify(path, rec['output_sha256'])
        with np.load(path) as values:
            np.testing.assert_array_equal(values['object_rows'], np.arange(worker*5000, (worker+1)*5000))
            parts.append({k: values[k] for k in old.KEYS})
    base = {k: np.concatenate([p[k] for p in parts], axis=1 if k.startswith(('ladder_', 'weight_')) else 0)
            for k in old.KEYS}
    np.testing.assert_array_equal(base['score'], base['ladder_score'][-1])
    np.testing.assert_array_equal(base['information'], base['ladder_information'][-1])
    initial = _summarize(center, base['score'], base['information'],
                         solver_method='first_full_pass_observed_information')
    point = old.next_center(dict(initial, status='estimated'))
    rungs, influences = [], []
    for j, draws in enumerate(old.LADDER):
        result, influence = old.summarize(center, base['ladder_score'][j], base['ladder_information'][j])
        rungs.append(dict(draws=draws, **result))
        influences.append(influence)
    np.savez_compressed(ROOT/'base_moments.npz', object_rows=np.arange(20000), **base)
    grad, hess = normalization(folder/'normalization.json', center)
    save(ROOT/'base.json', dict(center=center.tolist(), normalization=str(folder/'normalization.json'),
        normalization_gradient=grad.tolist(), normalization_hessian=hess.tolist(),
        source='Actual first full evaluation at actual raw mean; fixed reference for every carried row',
        base_moments_sha256=study.file_hash(ROOT/'base_moments.npz')))
    save(ROOT/'initial_state.json', dict(hybrid=initial, update=dict(method='no_BFGS_update_before_first_bright_iteration')))
    save(folder/'result.json', dict(status='complete', stage='first_full_pass', hybrid=initial, rungs=rungs,
        paired_draw_change=old.paired(rungs[-2], rungs[-1], influences[-2], influences[-1]), n_observations=20000))
    save(ROOT/'base_stage.json', dict(execution_sha256=study.file_hash(ROOT/'execution.json'),
        sources={str(p): study.file_hash(p) for p in [ROOT/'base.json', ROOT/'base_moments.npz',
            ROOT/'initial_state.json', folder/'result.json', folder/'normalization.json']}))
    save(ROOT/'pass_01/center.json', dict(center=point, source='actual first full-pass estimate',
        source_sha256=study.file_hash(folder/'result.json')))
    save(ROOT/'chain.json', dict(status='first_full_pass_complete', completed_iterations=0,
        requested_iterations=5, initial_center=point, input_kind='image', accepted_0_3_percent=False))
    save(folder/'decision.json', dict(status='first_full_pass_complete'))
    print(f'IMAGE_FIRST_FULL_PASS_COMPLETE estimate={point}', flush=True)


def combine(index):
    if is_image() and index == 0:
        return combine_full()
    first_index = 1 if is_image() else 0
    iteration = index-first_index+1
    folder = ROOT/f'pass_{index:02d}'
    center = np.asarray(study.read(folder/'center.json')['center'])
    base_record = study.read(ROOT/'base.json')
    selected = study.read(ROOT/'selection.json')
    bright_ids = np.asarray(selected['bright_ids'])
    parts = []
    for worker in range(4):
        rec = study.read(folder/f'worker_{worker}.json')
        assert rec['status'] == 'complete' and rec['center'] == center.tolist()
        assert rec['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
        assert rec['normalization_sha256'] == study.file_hash(folder/'normalization.json')
        path = folder/f'worker_{worker}.npz'
        study.verify(path, rec['output_sha256'])
        with np.load(path) as data:
            parts.append({k: data[k] for k in ('object_rows', 'ladder_score', 'ladder_information')})
    all_ids = np.concatenate([p['object_rows'] for p in parts])
    order = np.argsort(all_ids)
    np.testing.assert_array_equal(all_ids[order], selected['evaluated_group_ids'])
    positions = np.searchsorted(all_ids[order], bright_ids)
    np.testing.assert_array_equal(all_ids[order][positions], bright_ids)
    measured_score = np.concatenate([p['ladder_score'] for p in parts], axis=1)[:, order][:, positions]
    measured_info = np.concatenate([p['ladder_information'] for p in parts], axis=1)[:, order][:, positions]
    grad, hess = normalization(folder/'normalization.json', center)
    with np.load(ROOT/'base_moments.npz') as base:
        scores, infos = carry(base['ladder_score'], base['ladder_information'], center-base_record['center'],
            base_record['normalization_gradient'], base_record['normalization_hessian'], grad, hess)
    scores[:, bright_ids], infos[:, bright_ids] = measured_score, measured_info
    previous = ROOT/'initial_state.json' if index == first_index else ROOT/f'pass_{index-1:02d}/result.json'
    state = study.read(previous)['hybrid']
    score_sum = scores[-1].sum(0)
    if np.array_equal(center, state['center']) and np.array_equal(score_sum, state['score_sum']):
        solver = np.asarray(state['solver_information_sum'])
        update = dict(method='unchanged_solver_at_identical_center_and_score')
    else:
        solver, update = _bfgs_information(None, None, None, center, score_sum, previous)
    rungs = []
    for j, draws in enumerate(old.LADDER):
        summary = _summarize(center, scores[j], infos[j], solver_information=solver,
                              solver_method='historical_bfgs_bright20_fixed_base_pass00')
        rungs.append(dict(draws=draws, **summary))
    hybrid = rungs[-1]
    point, step = np.asarray(hybrid['estimate']), np.asarray(hybrid['step'])
    if not np.isfinite(point).all() or np.linalg.norm(point) >= 1 or np.linalg.norm(step) > .05:
        raise RuntimeError('unsafe BFGS proposal; inspect without clipping or truth-based repair')
    influences = []
    for j in (-2, -1):
        residual = scores[j]-np.einsum('nij,j->ni', infos[j], rungs[j]['step'])
        influences.append(np.linalg.solve(solver/20000, residual.T).T)
    paired = old.paired(rungs[-2], rungs[-1], influences[0], influences[1])
    small = bool(np.all(abs(step) <= .1*np.asarray(hybrid['robust_standard_error'])))
    result = dict(status='complete', iteration=iteration, center=center.tolist(), hybrid=hybrid,
        bfgs_update=update, rungs=rungs, paired_draw_change=paired, small_step=small,
        n_observations=20000, recomputed_observations=len(bright_ids),
        evaluated_with_padding=selected['evaluated_count'], fixed_carry_base=base_record,
        limitation='Conditional approximation/solver errors only; measured-mag20 carry unvalidated; no bias acceptance')
    save(folder/'result.json', result)
    chain = [study.read(ROOT/f'pass_{j:02d}/result.json') for j in range(first_index, index+1)]
    stable = len(chain) >= 2 and all(p['small_step'] for p in chain[-2:])
    draw_stable = bool(np.all(abs(np.asarray(paired['difference']))+2*np.asarray(paired['standard_error']) <= 6.e-5))
    status = 'five_iterations_complete' if iteration == 5 else 'continuing'
    save(ROOT/'chain.json', dict(status=status, completed_iterations=iteration, requested_iterations=5,
        iteration_stable=stable, draw_stable=draw_stable, accepted_0_3_percent=False,
        results=[f'pass_{j:02d}/result.json' for j in range(first_index, index+1)],
        estimates=[p['hybrid']['estimate'] for p in chain]))
    if iteration < 5:
        save(ROOT/f'pass_{index+1:02d}/center.json', dict(center=point.tolist(),
            source=f'pass_{index:02d}/result.json', source_sha256=study.file_hash(folder/'result.json')))
    save(folder/'decision.json', dict(status=status))
    print(f'ITERATION_COMPLETE iteration={iteration} estimate={point} step={step} small={small}', flush=True)


def worker(worker_index):
    execution = identity()
    record = study.read(ROOT/'preflight.json')
    assert record['status'] == 'passed' and record['execution'] == execution
    gpu = torch.cuda.get_device_properties(0)
    assert (gpu.name, gpu.total_memory) == (record['gpu'], record['total_memory'])
    # Iteration 1 was reconstructed exactly from the matching saved full pass.
    # Only its measured-bright rows were used; perform iterations 2--5 here.
    if is_image():
        passes = range(6)  # Fresh full pass, then exactly five bright-only updates.
    else:
        assert study.read(ROOT/'pass_00/result.json')['iteration'] == 1
        passes = range(1, 5)
    for index in passes:
        folder = ROOT/f'pass_{index:02d}'
        wait([folder/'center.json'])
        normalize(index, worker_index)
        gc.collect()
        torch.cuda.empty_cache()
        wait([folder/f'normalization/shard_{j:02d}/manifest.json' for j in range(20)])
        if worker_index == 0:
            assemble(index)
        wait([folder/'assembled.json'])
        evaluate(index, worker_index)
        gc.collect()
        torch.cuda.empty_cache()
        wait([folder/f'worker_{j}.json' for j in range(4)])
        if worker_index == 0:
            combine(index)
        wait([folder/'decision.json'])
    save(ROOT/f'worker_{worker_index}.json', dict(status='complete', iterations=5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize', 'cached_first', 'preflight', 'worker'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.stage == 'initialize':
            initialize()
        elif args.stage == 'cached_first':
            cached_first()
        elif args.stage == 'preflight':
            preflight()
        else:
            worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
    except Exception:
        save(ROOT/f'failure_{os.environ.get("SLURM_JOB_ID", "local")}.json', dict(traceback=traceback.format_exc()))
        save(ROOT/'chain_status.json', dict(status='stopped_failure', completed_results=len(list(ROOT.glob('pass_*/result.json')))))
        raise
