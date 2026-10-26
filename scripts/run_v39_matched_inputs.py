"""Matched-input forward diagnostic at the failed actual-data expansion point."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ['SBSI_MATCHED_ROOT'])
OLD = Path(os.environ['SBSI_RECENTER_ROOT'])
sys.path.insert(0, str(OLD))
import run_v39_recenter as old  # noqa: E402
sys.path.insert(0, str(ROOT))
import v39_matched_input as inputs  # noqa: E402
from sbsi.disk_precision import LOG_GAP_COLUMN, transport_with_log_gap, validate_shape_encoding  # noqa: E402
from sbsi.output_conditioned_response import TRUTH_FEATURES  # noqa: E402
from sbsi.selection_normalization import log_mass_derivatives  # noqa: E402

study, np, pd, torch, save = old.study, old.np, old.study.pd, old.torch, old.save


def protocol():
    return study.read(ROOT/'protocol.json')


def actual_root():
    return Path(protocol()['actual_root'])


def identity():
    record = study.read(ROOT/'execution.json')
    study.verify(__file__, record['driver_sha256'])
    study.verify(inputs.__file__, record['input_helper_sha256'])
    study.verify(ROOT/'protocol.json', record['protocol_sha256'])
    assert old.identity() == record['source_execution']
    for path, sha in record['sources'].items():
        study.verify(path, sha)
    return record


def prepared():
    record = study.read(ROOT/'prepared/manifest.json')
    assert record['execution'] == identity()
    for name, sha in record['output_sha256'].items():
        study.verify(ROOT/'prepared'/name, sha)
    return record


def generation():
    record = study.read(ROOT/'mock/manifest.json')
    assert record['execution'] == identity()
    study.verify(ROOT/'prepared/manifest.json', record['prepared_sha256'])
    for name, sha in record['output_sha256'].items():
        study.verify(ROOT/'mock'/name, sha)
    return record


def prepare():
    assert not (ROOT/'execution.json').exists()
    p, source = protocol(), actual_root()
    parent = Path(p['parent_manifest'])
    study.verify(parent, p['parent_manifest_sha256'])
    origin = study.read(parent)
    cases = {int(c['case']): c for c in origin['per_case']}
    paths = [source/'execution.json', source/'protocol.json', source/'input/manifest.json',
             source/'input/measurements.parquet', source/'input/truth.parquet',
             source/'pass_00/center.json', source/'pass_00/normalization.json', source/'pass_00/assembled.json', parent]
    for worker in range(4):
        rec = study.read(source/f'pass_00/worker_{worker}.json')
        moment_path = source/f'pass_00/worker_{worker}.npz'
        study.verify(moment_path, rec['output_sha256'])
        paths.extend([source/f'pass_00/worker_{worker}.json', moment_path])
    input_manifest = study.read(source/'input/manifest.json')
    for name, sha in input_manifest['output_sha256'].items():
        study.verify(source/'input'/name, sha)
    save(ROOT/'execution.json', dict(driver_sha256=study.file_hash(__file__),
        input_helper_sha256=study.file_hash(inputs.__file__), protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        source_execution=old.identity(), sources={str(path): study.file_hash(path) for path in paths}))
    truth = pd.read_parquet(source/'input/truth.parquet')
    assert len(truth) == 20000 and not truth.duplicated(['source_case', 'source_input_index']).any()
    np.testing.assert_array_equal(truth.parent_sample_row, np.arange(20000))
    checks = study.read(Path(p['replay_parent'])/'manifest.json')
    zero, row_features, row_secondary, reports = np.empty((20000, 8)), {}, {}, []
    folder = ROOT/'prepared'
    folder.mkdir(exist_ok=False)
    for case in sorted(truth.source_case.unique()):
        rows = np.flatnonzero(truth.source_case.to_numpy() == case)
        ids = truth.source_input_index.to_numpy()[rows]
        raw_path = Path(cases[int(case)]['sources']['truth']['path'])
        study.verify(raw_path, cases[int(case)]['sources']['truth']['sha256'])
        raw = pd.read_feather(raw_path)
        np.testing.assert_allclose(raw[['gamma1_input', 'gamma2_input']],
            np.broadcast_to(p['injected_shear'], (len(raw), 2)), atol=1.e-14, rtol=0)
        frame, pairs, primary = inputs.build_case_inputs(raw, ids, p['conditions'])
        np.testing.assert_array_equal(primary['index'], ids)
        zero[rows] = frame.to_numpy()
        pair_counts = np.bincount(pairs.primary_row.to_numpy(int), minlength=len(rows))
        pair_ptr = np.r_[0, np.cumsum(pair_counts)]
        values = pairs.loc[:, TRUTH_FEATURES].to_numpy(np.float32)
        secondaries = pairs.index_input_s.to_numpy(np.int64)
        for local, row in enumerate(rows):
            row_features[int(row)] = values[pair_ptr[local]:pair_ptr[local+1]]
            row_secondary[int(row)] = secondaries[pair_ptr[local]:pair_ptr[local+1]]
        report = dict(case=int(case), n_inputs=len(rows), unmatched=0, pairs=len(pairs),
            source_truth=str(raw_path), source_sha256=study.file_hash(raw_path),
            target_role_count=int((ids < len(raw)//2).sum()),
            max_abs_shape_difference_from_recorded=float(np.max(np.abs(
                frame.iloc[:, :2].to_numpy()-raw.set_index('index_input').loc[ids, ['e1_input_rot0', 'e2_input_rot0']].to_numpy()))))
        if str(case) in checks['case_reports']:
            path = Path(p['replay_parent'])/f'case{case:03d}.npz'
            study.verify(path, checks['case_reports'][str(case)]['sha256'])
            assert checks['case_reports'][str(case)]['truth_sha256'] == report['source_sha256']
            with np.load(path) as cached:
                positions = pd.Index(cached['input_index']).get_indexer(ids)
                present = positions >= 0
                reference = cached['context'][positions[present]][:, [0, 1, 2, 3, 4, 8, 9, 10]]
            np.testing.assert_allclose(frame.to_numpy()[present], reference, rtol=3.e-5, atol=3.e-5)
            report['feature_replay'] = dict(checked=int(present.sum()),
                max_abs=float(np.max(np.abs(frame.to_numpy()[present]-reference))), sha256=study.file_hash(path))
        reports.append(report)
        print(f'MATCHED_INPUT case={case} rows={len(rows)} pairs={len(pairs)} unmatched=0', flush=True)
    frame = pd.DataFrame(zero, columns=study.FEATURES)
    for name in study.FEATURES[5:]:
        frame[name] = frame[name].astype(np.float32)
    frame.to_parquet(folder/'flow_zero.parquet', index=False)
    counts = np.array([len(row_features[i]) for i in range(20000)])
    np.save(folder/'pair_indptr.npy', np.r_[0, np.cumsum(counts)])
    np.save(folder/'pair_features.npy', np.concatenate([row_features[i] for i in range(20000)]))
    np.save(folder/'pair_secondary.npy', np.concatenate([row_secondary[i] for i in range(20000)]))
    truth.to_parquet(folder/'truth.parquet', index=False)
    save(folder/'manifest.json', dict(status='complete', execution=identity(), n_inputs=20000,
        unmatched=0, truth_cuts=None, features=list(study.FEATURES), conditions=p['conditions'], case_reports=reports,
        output_sha256={f.name: study.file_hash(f) for f in folder.iterdir() if f.is_file()}))
    save(ROOT/'status.json', dict(status='inputs_prepared', n_inputs=20000))


def generate():
    rec, p = prepared(), protocol()
    folder = ROOT/'mock'
    folder.mkdir(exist_ok=False)
    zero = pd.read_parquet(ROOT/'prepared/flow_zero.parquet')
    truth = pd.read_parquet(ROOT/'prepared/truth.parquet')
    g = p['injected_shear']
    context = zero.copy()
    context['e1_input_p'], context['e2_input_p'] = study.apply_shear_to_ellipticity(zero.e1_input_p, zero.e2_input_p, *g)
    flow = study.load_measurement_model(study.V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = study.load_detection_classifier(study.V39_LIKE, device='cuda')
    response = study.CatalogueDiskResponse(np.load(ROOT/'prepared/pair_indptr.npy'),
        np.load(ROOT/'prepared/pair_features.npy'), study.load_disk_response(study.V39_LIKE, device='cuda', nthread=4))
    probability = study.predict_detection_probability(detector, context)
    uniforms = np.random.default_rng(p['mock_seeds']['detection']).random(len(zero))
    usable = np.flatnonzero(uniforms < probability)
    torch.manual_seed(p['mock_seeds']['flow'])
    draws, gaps = flow.sample_with_shape_log_gap(context.iloc[usable].reset_index(drop=True), n_samples=1, batch_size=8192)
    draws, gaps = draws[:, 0].astype(np.float64), gaps[:, 0]
    validate_shape_encoding(draws[:, :2], gaps)
    assert np.isfinite(draws).all() and (draws[:, 2:] > 0).all()
    selected = study.output_cut(flow)(draws)
    selected_rows = usable[selected]
    coefficient, _ = response.predict(usable, draws[:, 2], draws[:, 3])
    base_shape = draws[:, :2].copy()
    shape, observed_gap, logdet = transport_with_log_gap(torch.as_tensor(base_shape),
        torch.as_tensor(gaps), torch.as_tensor(coefficient[:, None]*g))
    inverse, inverse_gap, inverse_logdet = transport_with_log_gap(shape, observed_gap,
        torch.as_tensor(coefficient[:, None]*g), inverse=True)
    np.testing.assert_allclose(inverse, base_shape, atol=1.e-10, rtol=1.e-10)
    np.testing.assert_allclose(inverse_gap, gaps, atol=1.e-12, rtol=1.e-12)
    np.testing.assert_allclose(logdet+inverse_logdet, 0, atol=1.e-10, rtol=0)
    draws[:, :2] = shape.numpy()
    validate_shape_encoding(draws[:, :2], observed_gap.numpy())
    np.testing.assert_array_equal(study.output_cut(flow)(draws), selected)
    measured = pd.DataFrame(draws, columns=flow.target_transform.target_names)
    measured[LOG_GAP_COLUMN] = observed_gap.numpy()
    measured['measured_mag_auto'] = 30-2.5*np.log10(measured.measured_flux_from_mag_auto)
    # All usable draws are saved, including those failing the new measured cuts.
    all_truth = truth.copy()
    all_truth['model_usable'] = uniforms < probability
    all_truth['model_selected'] = False
    all_truth.loc[selected_rows, 'model_selected'] = True
    all_truth['model_usable_probability'] = probability
    all_truth['model_usable_uniform'] = uniforms
    all_truth.to_parquet(folder/'all_input_outcomes.parquet', index=False)
    measured.to_parquet(folder/'usable_measurements.parquet', index=False)
    truth.iloc[usable].to_parquet(folder/'usable_input_ids.parquet', index=False)
    selected_truth = all_truth.iloc[selected_rows].reset_index(drop=True)
    selected_truth['mock_kind'] = 'likelihood'
    selected_truth['true_magnitude'] = zero.r_input_p.to_numpy()[selected_rows]
    selected_truth['disk_coefficient'] = coefficient[selected]
    selected_measurements = measured.loc[selected].reset_index(drop=True)
    study.MockCatalogue(selected_measurements, selected_truth).save(folder)
    assert len(selected_rows) > 0 and np.all(np.diff(selected_rows) > 0)
    original = pd.read_parquet(actual_root()/'input/measurements.parquet')
    evaluation_measured, evaluation_truth = inputs.evaluation_catalogue(original, truth, selected_measurements, selected_truth)
    study.MockCatalogue(evaluation_measured, evaluation_truth).save(folder/'evaluation')
    save(folder/'manifest.json', dict(status='complete', execution=identity(), n_inputs=len(zero),
        n_usable=len(usable), n_observations=len(selected_rows), n_unusable=len(zero)-len(usable),
        n_failed_measurement_cut=int((~selected).sum()), n_bright=int((selected_measurements.measured_mag_auto < 20).sum()),
        matched_actual_bright=int((original.measured_mag_auto.to_numpy()[selected_rows] < 20).sum()),
        selected_parent_rows=selected_rows.tolist(), center=study.read(actual_root()/'pass_00/center.json')['center'],
        mock_raw_mean=draws[selected, :2].mean(0).tolist(), injected_shear=g,
        actual_subset_raw_mean=original.iloc[selected_rows][['measured_ngmix_g1', 'measured_ngmix_g2']].mean().tolist(),
        seeds=p['mock_seeds'], normal_forward_draw=True, no_top_up=True, prepared_sha256=study.file_hash(ROOT/'prepared/manifest.json'),
        prepared_inputs=rec['n_inputs'],
        output_sha256={str(f.relative_to(folder)): study.file_hash(f) for f in folder.rglob('*.parquet')},
        evaluated_with_padding=20000, numerical_padding=20000-len(selected_rows),
        limitation='Input cohort was selected through actual measurements; this is not prior-population closure or bias acceptance.'))
    save(ROOT/'status.json', dict(status='mock_generated', n_inputs=20000, n_usable=len(usable), n_retained=len(selected_rows)))
    print(f'MATCHED_MOCK_GENERATED input=20000 usable={len(usable)} retained={len(selected_rows)}', flush=True)


def load():
    gen = generation()
    likelihood, proposal, _, _ = study.load_inference()
    folder = actual_root()/'pass_00'
    rec = study.read(folder/'assembled.json')
    study.verify(folder/'probability.npy', rec['probability_sha256'])
    study.verify(folder/'normalization.json', rec['normalization_sha256'])
    cache = study.V39CatalogueCache(likelihood.cache.zero_flow, likelihood.cache.prior.weights,
        likelihood.cache.conditions, probabilities=dict(zip(map(tuple, rec['points']),
                                                np.load(folder/'probability.npy', mmap_mode='r'))))
    likelihood = study.DiskCatalogueLikelihood(likelihood.flow_model, cache,
        disk_response=likelihood.disk_response, selection=likelihood.selection)
    likelihood.population_normalization = study.ExactPopulationNormalization.from_payload(study.read(folder/'normalization.json'))
    assert likelihood.population_normalization.identity == study.read(actual_root()/'execution.json')
    likelihood.compile_precision_density()
    return likelihood, proposal, study.MockCatalogue.load(ROOT/'mock/evaluation'), gen


def evaluate(likelihood, proposal, mock, local_rows, center, label):
    frame = mock.measurements.iloc[local_rows].reset_index(drop=True)
    truth = mock.truth.iloc[local_rows].reset_index(drop=True)
    ids = truth.parent_sample_row.to_numpy(np.int64)
    cached = old.CachedProposal(proposal, frame, ids, 'gaussian')
    result = study.run_adaptive_section5(likelihood, study.MockCatalogue(frame, truth), cached,
        center=center, h=.001, draw_ladder=old.LADDER, n_candidates=1024,
        proposal_prefilter_candidates=131072, epsilon=.1, proposal_seed=8701,
        min_ess=32., max_weight_fraction=.5, candidate_backend='torch', estimator_mode='tilted_stratified',
        tilt_delta=.1, retain_full_ladder=True, full_information=True, object_ids=ids,
        object_chunk=4, atom_chunk=4096,
        progress=lambda done, total, elapsed: print(f'MATCHED_INFERENCE {label} rows={done}/{total} seconds={elapsed:.1f}', flush=True))
    cached.complete()
    return {key: getattr(result, key) for key in old.KEYS}


def preflight():
    likelihood, proposal, mock, gen = load()
    # Confirm proposal batching on this newly generated catalogue. Object IDs
    # remain original20k parent rows, not renumbered by the second selection.
    rows = np.arange(min(128, len(mock.measurements)//4*4))
    assert len(rows) >= 4
    original_ids = mock.truth.iloc[rows].parent_sample_row.to_numpy(np.int64)
    frame = study.MockCatalogue(mock.measurements.iloc[rows], mock.truth.iloc[rows])
    result = study.run_adaptive_section5(likelihood, frame, proposal,
        center=gen['center'], h=.001, draw_ladder=old.LADDER, n_candidates=1024,
        proposal_prefilter_candidates=131072, epsilon=.1, proposal_seed=8701, min_ess=32., max_weight_fraction=.5,
        candidate_backend='torch', estimator_mode='tilted_stratified', tilt_delta=.1, retain_full_ladder=True,
        full_information=True, object_ids=original_ids, object_chunk=4, atom_chunk=4096)
    checked = evaluate(likelihood, proposal, mock, rows, gen['center'], 'replay')
    for key in old.KEYS:
        np.testing.assert_array_equal(checked[key], getattr(result, key), err_msg=key)
    gpu = torch.cuda.get_device_properties(0)
    save(ROOT/'preflight.json', dict(status='passed', execution=identity(), n_rows=len(rows),
        keys=list(old.KEYS), gpu=gpu.name, total_memory=gpu.total_memory,
        mock_manifest_sha256=study.file_hash(ROOT/'mock/manifest.json')))
    print('MATCHED_REPLAY_PASS', flush=True)


def worker(worker_index):
    receipt = study.read(ROOT/'preflight.json')
    assert receipt['status'] == 'passed' and receipt['execution'] == identity()
    study.verify(ROOT/'mock/manifest.json', receipt['mock_manifest_sha256'])
    gpu = torch.cuda.get_device_properties(0)
    assert (gpu.name, gpu.total_memory) == (receipt['gpu'], receipt['total_memory'])
    likelihood, proposal, mock, gen = load()
    # Original contiguous groups of four; nonretained rows are numerical padding.
    groups = (len(mock.measurements)+3)//4
    group_start, group_stop = [groups*j//4 for j in (worker_index, worker_index+1)]
    lo, hi = min(4*group_start, len(mock.measurements)), min(4*group_stop, len(mock.measurements))
    folder = ROOT/'moments'
    folder.mkdir(exist_ok=True)
    parts = []
    for start in range(lo, hi, 1000):
        rows = np.arange(start, min(start+1000, hi))
        values = evaluate(likelihood, proposal, mock, rows, gen['center'], f'worker={worker_index} start={start}')
        np.savez_compressed(folder/f'worker_{worker_index}_chunk_{start:05d}.npz',
            object_rows=mock.truth.iloc[rows].parent_sample_row.to_numpy(), **values)
        parts.append(values)
        gc.collect()
        torch.cuda.empty_cache()
    values = {key: np.concatenate([part[key] for part in parts], axis=1 if key.startswith(('ladder_', 'weight_')) else 0)
              for key in old.KEYS}
    path = folder/f'worker_{worker_index}.npz'
    np.savez_compressed(path, object_rows=mock.truth.iloc[lo:hi].parent_sample_row.to_numpy(), **values)
    save(folder/f'worker_{worker_index}.json', dict(status='complete', n_rows=hi-lo, local_range=[lo, hi],
        center=gen['center'], output_sha256=study.file_hash(path), execution=identity(),
        mock_manifest_sha256=study.file_hash(ROOT/'mock/manifest.json')))
    print(f'MATCHED_WORKER_COMPLETE worker={worker_index} rows={hi-lo}', flush=True)


def combine():
    gen = generation()
    parts, actual = [], []
    for worker_index in range(4):
        folder = ROOT/'moments'
        rec = study.read(folder/f'worker_{worker_index}.json')
        assert rec['execution'] == identity() and rec['center'] == gen['center']
        study.verify(ROOT/'mock/manifest.json', rec['mock_manifest_sha256'])
        study.verify(folder/f'worker_{worker_index}.npz', rec['output_sha256'])
        with np.load(folder/f'worker_{worker_index}.npz') as data:
            parts.append({key: data[key] for key in data.files})
        with np.load(actual_root()/f'pass_00/worker_{worker_index}.npz') as data:
            actual.append({key: data[key] for key in data.files})
    ids = np.concatenate([part['object_rows'] for part in parts])
    np.testing.assert_array_equal(ids, np.arange(20000))
    np.testing.assert_array_equal(np.concatenate([part['object_rows'] for part in actual]), np.arange(20000))
    ms, mi = [np.concatenate([part[key] for part in parts], axis=1) for key in ('ladder_score', 'ladder_information')]
    ids = np.asarray(gen['selected_parent_rows'])
    evaluation_truth = pd.read_parquet(ROOT/'mock/evaluation/truth.parquet')
    np.testing.assert_array_equal(np.flatnonzero(~evaluation_truth.numerical_padding.to_numpy()), ids)
    ms, mi = ms[:, ids], mi[:, ids]
    actual_s, actual_i = [np.concatenate([part[key] for part in actual], axis=1) for key in ('ladder_score', 'ladder_information')]
    arrays = dict(mock=(ms, mi), actual_same_ids=(actual_s[:, ids], actual_i[:, ids]), actual_all20k=(actual_s, actual_i))
    rungs = []
    for j, draws in enumerate(old.LADDER):
        record = dict(draws=draws)
        for name, (score, info) in arrays.items():
            value, _ = old.summarize(gen['center'], score[j], info[j])
            record[name] = dict(value, n_observations=len(score[j]),
                score_sum=score[j].sum(0).tolist(), information_sum=info[j].sum(0).tolist())
        rungs.append(record)
    measured_mock = pd.read_parquet(ROOT/'mock/measurements.parquet')
    measured_actual = pd.read_parquet(actual_root()/'input/measurements.parquet').iloc[ids]
    groups = {}
    for name, mask in [('actual_mag_lt20', measured_actual.measured_mag_auto.to_numpy() < 20),
                       ('mock_mag_lt20', measured_mock.measured_mag_auto.to_numpy() < 20)]:
        groups[name] = dict(count=int(mask.sum()),
            actual_information_sum=actual_i[-1, ids[mask]].sum(0).tolist(),
            mock_information_sum=mi[-1, mask].sum(0).tolist())
    _, gradient, hessian = log_mass_derivatives(
        study.ExactPopulationNormalization.from_payload(study.read(actual_root()/'pass_00/normalization.json')), gen['center'])
    save(ROOT/'comparison.json', dict(status='complete', execution=identity(), n_inputs=20000,
        n_retained=len(ids), center=gen['center'], mock_raw_mean=gen['mock_raw_mean'], rungs=rungs,
        groups=groups, normalization_gradient=gradient.tolist(), normalization_hessian=hessian.tolist(),
        same_problem_at_shared_center=rungs[-1]['mock']['status'] == 'non_positive_information',
        limitation='Matched, already image-selected inputs with another model selection; fixed-centre diagnostic, not unbiased population closure or converged shear.'))
    save(ROOT/'status.json', dict(status='comparison_complete', n_retained=len(ids),
        same_problem_at_shared_center=rungs[-1]['mock']['status'] == 'non_positive_information'))
    print('MATCHED_COMPARISON_COMPLETE', rungs[-1], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'generate', 'preflight', 'worker', 'combine'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    try:
        if args.stage == 'worker':
            worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
        else:
            globals()[args.stage]()
    except Exception:
        save(ROOT/f'failure_{os.environ.get("SLURM_JOB_ID", "local")}.json', dict(stage=args.stage, traceback=traceback.format_exc()))
        save(ROOT/'status.json', dict(status='failed', stage=args.stage))
        raise
    print(f'STAGE_COMPLETE {args.stage} seconds={time.monotonic()-started:.1f}', flush=True)
