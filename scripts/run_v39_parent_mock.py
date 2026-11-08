"""One forward draw for a fixed unselected ConstGold input sample, off prior."""
import argparse
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(os.environ['SBSI_PARENT_MOCK_ROOT'])
SOURCE = Path(os.environ['SBSI_EXPANDED_ROOT'])
sys.path.insert(0, str(SOURCE))
import run_v39_expanded_prior as baseline  # noqa: E402

matched, study = baseline.matched, baseline.study
np, pd, torch, save = study.np, study.pd, study.torch, baseline.save


def protocol():
    p = study.read(ROOT/'protocol.json')
    study.verify(__file__, p['script_sha256'])
    return p


def prepare():
    p = protocol()
    assert not (ROOT/'prepared/manifest.json').exists()
    source = Path(matched.protocol()['parent_manifest'])
    study.verify(source, matched.protocol()['parent_manifest_sha256'])
    origin = study.read(source)
    audit = Path(p['population_audit'])
    study.verify(audit, p['population_audit_sha256'])
    reports = study.read(audit)['per_case']
    cases = {int(rec['case']):rec for rec in origin['per_case']}
    counts = np.array([rec['count'] for rec in reports])
    cumulative = np.r_[0, np.cumsum(counts)]
    ranks = np.random.default_rng(p['seeds']['inputs']).choice(int(cumulative[-1]), p['n_inputs'], replace=False)
    positions = np.searchsorted(cumulative[1:], ranks, side='right')
    zero = np.empty((p['n_inputs'], 8))
    truth = pd.DataFrame({'parent_sample_row':np.arange(p['n_inputs']),
        'source_case':np.array([reports[i]['case'] for i in positions]),
        'source_input_index':np.full(p['n_inputs'], -1, dtype=np.int64),
        'injected_g1':p['injected_shear'][0], 'injected_g2':p['injected_shear'][1]})
    pair_rows, case_reports = {}, []
    for position in np.unique(positions):
        case = reports[position]['case']
        selected = np.flatnonzero(positions == position)
        raw_path = Path(cases[case]['sources']['truth']['path'])
        study.verify(raw_path, cases[case]['sources']['truth']['sha256'])
        raw = pd.read_feather(raw_path)
        assert len(raw) == counts[position]
        ids = raw.index_input.to_numpy()[ranks[selected]-cumulative[position]]
        frame, pairs, primary = matched.inputs.build_case_inputs(raw, ids, p['conditions'])
        np.testing.assert_array_equal(primary['index'], ids)
        assert len(frame) == len(selected)
        zero[selected] = frame.to_numpy()
        truth.loc[selected, 'source_input_index'] = ids
        sizes = np.bincount(pairs.primary_row.to_numpy(int), minlength=len(selected))
        indptr = np.r_[0, np.cumsum(sizes)]
        values = pairs.loc[:, matched.TRUTH_FEATURES].to_numpy(np.float32)
        for local, row in enumerate(selected):
            pair_rows[int(row)] = values[indptr[local]:indptr[local+1]]
        case_reports.append(dict(case=case, inputs=len(selected), unmatched=0,
            source_sha256=study.file_hash(raw_path), target_role_count=int((ids < len(raw)//2).sum())))
        print(f'PARENT_INPUTS case={case} rows={len(selected)} pairs={len(pairs)}', flush=True)
    assert np.isfinite(zero).all() and (truth.source_input_index >= 0).all()
    assert not truth.duplicated(['source_case', 'source_input_index']).any()
    folder = ROOT/'prepared'
    folder.mkdir(exist_ok=False)
    frame = pd.DataFrame(zero, columns=study.FEATURES)
    for name in study.FEATURES[5:]:
        frame[name] = frame[name].astype(np.float32)
    frame.to_parquet(folder/'flow_zero.parquet', index=False)
    truth.to_parquet(folder/'truth.parquet', index=False)
    sizes = np.array([len(pair_rows[i]) for i in range(len(truth))])
    np.save(folder/'pair_indptr.npy', np.r_[0, np.cumsum(sizes)])
    np.save(folder/'pair_features.npy', np.concatenate([pair_rows[i] for i in range(len(truth))]))
    existing = pd.read_parquet(matched.actual_root()/'input/truth.parquet')
    overlap = len(truth.merge(existing[['source_case','source_input_index']], on=['source_case','source_input_index']))
    save(folder/'manifest.json', dict(status='complete', protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        baseline_execution=baseline.identity(), n_inputs=len(truth), unselected_parent_rows=int(cumulative[-1]),
        sampling='Uniform rows without replacement from all unselected input cases40–139; no measured or truth cuts',
        selected_actual20k_identity_overlap=overlap, inserted_into_inference_prior=False,
        source_manifest_sha256=study.file_hash(source), case_reports=case_reports,
        output_sha256={path.name:study.file_hash(path) for path in folder.iterdir() if path.is_file()}))
    print('PARENT_INPUTS_COMPLETE', len(truth), 'overlap_selected_actual', overlap, flush=True)


def generate():
    p = protocol()
    prepared = study.read(ROOT/'prepared/manifest.json')
    assert prepared['protocol_sha256'] == study.file_hash(ROOT/'protocol.json')
    assert prepared['baseline_execution'] == baseline.identity()
    for name, sha in prepared['output_sha256'].items():
        study.verify(ROOT/'prepared'/name, sha)
    zero = pd.read_parquet(ROOT/'prepared/flow_zero.parquet')
    truth = pd.read_parquet(ROOT/'prepared/truth.parquet')
    context, g = zero.copy(), p['injected_shear']
    context['e1_input_p'], context['e2_input_p'] = study.apply_shear_to_ellipticity(zero.e1_input_p, zero.e2_input_p, *g)
    flow = study.load_measurement_model(study.V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = study.load_detection_classifier(study.V39_LIKE, device='cuda')
    response = study.CatalogueDiskResponse(np.load(ROOT/'prepared/pair_indptr.npy'),
        np.load(ROOT/'prepared/pair_features.npy'), study.load_disk_response(study.V39_LIKE, device='cuda', nthread=4))
    probability = study.predict_detection_probability(detector, context)
    uniforms = np.random.default_rng(p['seeds']['detection']).random(len(zero))
    usable = np.flatnonzero(uniforms < probability)
    torch.manual_seed(p['seeds']['flow'])
    draws, gaps = flow.sample_with_shape_log_gap(context.iloc[usable].reset_index(drop=True), n_samples=1, batch_size=8192)
    draws, gaps = draws[:, 0].astype(float), gaps[:, 0]
    matched.validate_shape_encoding(draws[:, :2], gaps)
    assert np.isfinite(draws).all() and (draws[:, 2:] > 0).all()
    selected = study.output_cut(flow)(draws)
    coefficient, _ = response.predict(usable, draws[:, 2], draws[:, 3])
    shape, gap, logdet = matched.transport_with_log_gap(torch.as_tensor(draws[:, :2]),
        torch.as_tensor(gaps), torch.as_tensor(coefficient[:, None]*g))
    restored, restored_gap, inverse_logdet = matched.transport_with_log_gap(shape, gap,
        torch.as_tensor(coefficient[:, None]*g), inverse=True)
    np.testing.assert_allclose(restored, draws[:, :2], atol=1.e-10, rtol=1.e-10)
    np.testing.assert_allclose(restored_gap, gaps, atol=1.e-12, rtol=1.e-12)
    np.testing.assert_allclose(logdet+inverse_logdet, 0, atol=1.e-10, rtol=0)
    draws[:, :2] = shape.numpy()
    matched.validate_shape_encoding(draws[:, :2], gap.numpy())
    np.testing.assert_array_equal(selected, study.output_cut(flow)(draws))
    frame = pd.DataFrame(draws, columns=flow.target_transform.target_names)
    frame[matched.LOG_GAP_COLUMN] = gap.numpy()
    frame['measured_mag_auto'] = 30-2.5*np.log10(frame.measured_flux_from_mag_auto)
    outcomes = truth.copy()
    outcomes['model_usable_probability'], outcomes['model_usable_uniform'] = probability, uniforms
    outcomes['model_usable'], outcomes['model_selected'] = uniforms < probability, False
    selected_rows = usable[selected]
    outcomes.loc[selected_rows, 'model_selected'] = True
    selected_truth = outcomes.iloc[selected_rows].reset_index(drop=True)
    selected_truth['mock_kind'] = 'likelihood'
    selected_truth['true_magnitude'] = zero.r_input_p.to_numpy()[selected_rows]
    selected_truth['disk_coefficient'] = coefficient[selected]
    folder = ROOT/'mock'
    folder.mkdir(exist_ok=False)
    study.MockCatalogue(frame.loc[selected].reset_index(drop=True), selected_truth).save(folder)
    outcomes.to_parquet(folder/'all_input_outcomes.parquet', index=False)
    frame.to_parquet(folder/'usable_measurements.parquet', index=False)
    truth.iloc[usable].to_parquet(folder/'usable_input_ids.parquet', index=False)
    save(folder/'manifest.json', dict(status='complete', prepared_sha256=study.file_hash(ROOT/'prepared/manifest.json'),
        n_inputs=len(truth), n_usable=len(usable), n_retained=len(selected_rows),
        n_failed_measurement_cut=int((~selected).sum()), n_bright=int((frame.loc[selected,'measured_mag_auto'] < 20).sum()),
        injected_shear=g, normal_forward_draw=True, no_top_up=True, input_measurement_selection=False,
        inserted_into_inference_prior=False, seeds=p['seeds'],
        output_sha256={path.name:study.file_hash(path) for path in folder.iterdir() if path.suffix == '.parquet'},
        limitation='Independent of inference prior atom identities, but drawn from existing finite ConstGold scenes; no shear acceptance from generation alone.'))
    print(f'PARENT_MOCK_COMPLETE input={len(truth)} usable={len(usable)} retained={len(selected_rows)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','generate'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        globals()[args.stage]()
    except Exception:
        save(ROOT/f'failure_{args.stage}.json', dict(traceback=traceback.format_exc()))
        raise
