"""Protocol-pinned model-closure experiments for exact-atom retrieval.

All expensive stages run under Slurm against a pinned, isolated runtime.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior')
ROOT = Path(os.environ.get('SBSI_STUDY_ROOT', str(BASE/'inference_v39_mock20k_distance_20260925_v1')))
sys.path.insert(0, str(ROOT/'inference_runtime'))
from sbsi.catalogue_closure import MockCatalogue  # noqa: E402
from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood  # noqa: E402
from sbsi.catalogue_disk_response import CatalogueDiskResponse  # noqa: E402
from sbsi.catalogue_likelihood import CatalogueSelection, OutputCut, predict_detection_probability  # noqa: E402
from sbsi.catalogue_null import run_adaptive_section5  # noqa: E402
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable  # noqa: E402
from sbsi.disk_inference_store import ShardedDiskResponse, coordinate_table, file_hash, stencil, write_json  # noqa: E402
from sbsi.disk_response_transport import disk_response_transport  # noqa: E402
from sbsi.measurement_model import load_measurement_model  # noqa: E402
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response  # noqa: E402
from sbsi.selection_normalization import ExactPopulationNormalization, detected_selected_mass_shard  # noqa: E402
from sbsi.shear_map import apply_shear_to_ellipticity  # noqa: E402
from sbsi.v39_inference import FEATURES, V39CatalogueCache  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def verify(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f'artifact hash differs: {path}')


def identity():
    execution = read(ROOT/'execution.json')
    verify(__file__, execution['driver_sha256'])
    verify(ROOT/'protocol.json', execution['protocol_sha256'])
    for name, digest in execution['implementation_sha256'].items():
        verify(ROOT/'inference_runtime'/name, digest)
    V39_LIKE.validate()
    return execution


def features(index):
    root = ROOT/f'features/shard_{index:02d}'
    receipt = read(root/'manifest.json')
    assert receipt['status'] == 'complete' and receipt['shard'] == index
    source = read(ROOT/'protocol.json').get('feature_source')
    source_protocol = ROOT/'protocol.json'
    if source is not None:
        source_protocol = Path(source['root'])/'protocol.json'
        verify(source_protocol, source['protocol_sha256'])
    verify(source_protocol, receipt['protocol_sha256'])
    for name in ('flow_zero.parquet', 'prior_weight.npy'):
        verify(root/name, receipt['output_sha256'][name])
    zero = pd.read_parquet(root/'flow_zero.parquet')
    weights = np.load(root/'prior_weight.npy')
    assert tuple(zero.columns) == FEATURES and len(zero) == len(weights) == receipt['n_rows']
    return zero, weights, receipt


def response_from_receipts(receipts, model):
    result = []
    for receipt in receipts:
        source = Path(receipt['response_root'])
        for name, digest in receipt['response_sha256'].items():
            verify(source/name, digest)
        result.append(CatalogueDiskResponse(np.load(source/'pair_indptr.npy', mmap_mode='r'),
            np.load(source/'pair_features.npy', mmap_mode='r'), model))
    return ShardedDiskResponse(result)


def output_cut(flow):
    return OutputCut.from_specs(flow.target_transform.target_names,
        specs=['measured_flux_radius:>3.0:',
               f'measured_flux_from_mag_auto:>{10**(.4*(30-25.8))}:'])


def seed(base, shard):
    value = int(np.random.SeedSequence([base, shard]).generate_state(1)[0])
    torch.manual_seed(value)
    return value


def preflight():
    execution = identity()
    zero, weights, receipt = features(min(20, read(ROOT/'protocol.json').get('prior_shards', 28)-1))
    zero, weights = zero.iloc[:128].copy(), weights[:128].copy()
    weights /= weights.sum()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    model = load_disk_response(V39_LIKE, device='cuda', nthread=4)
    source = Path(receipt['response_root'])
    ptr = np.load(source/'pair_indptr.npy', mmap_mode='r')[:129]
    response = CatalogueDiskResponse(ptr, np.load(source/'pair_features.npy', mmap_mode='r')[:ptr[-1]], model)
    cache = V39CatalogueCache(zero, weights, receipt['conditions'], detector=detector)
    cut = output_cut(flow)
    likelihood = DiskCatalogueLikelihood(flow, cache, disk_response=response, selection=CatalogueSelection(cut))
    checks = []
    for g in ((.02, 0.), (-.015, .01)):
        view = cache.get(*g)
        torch.manual_seed(27189)
        draws = flow.sample(view.flow, n_samples=1, batch_size=128)[:, 0].astype(float)
        coefficient, _ = response.predict(np.arange(128), draws[:, 2], draws[:, 3])
        shape, determinant = disk_response_transport(torch.as_tensor(draws[:, :2]),
                                                   torch.as_tensor(coefficient[:, None]*g))
        base = view.flow.copy()
        for i, name in enumerate(flow.target_transform.target_names):
            base[name] = draws[:, i]
        direct = flow.log_prob(base)-determinant.numpy()
        draws[:, :2] = shape.numpy()
        selected = np.flatnonzero(cut(draws))
        assert len(selected) > 0
        observed = pd.DataFrame(draws[selected], columns=flow.target_transform.target_names)
        inferred = likelihood.conditional_log_likelihood(observed, *g, atom_indices=selected)
        np.testing.assert_allclose(inferred, direct[selected], rtol=1e-5, atol=1e-3)
        if read(ROOT/'protocol.json').get('disk_shape_encoding') == 'log_gap_v1':
            from sbsi.disk_precision import LOG_GAP_COLUMN, transport_with_log_gap
            torch.manual_seed(27189)
            precise, gap = flow.sample_with_shape_log_gap(view.flow, n_samples=1, batch_size=128)
            np.testing.assert_array_equal(precise[:, 0, 2:], draws[:, 2:])
            _, shifted_gap, _ = transport_with_log_gap(torch.as_tensor(precise[:, 0, :2]),
                torch.as_tensor(gap[:, 0]), torch.as_tensor(coefficient[:, None]*g))
            observed[LOG_GAP_COLUMN] = shifted_gap.numpy()[selected]
            accurate = likelihood.conditional_log_likelihood(observed, *g, atom_indices=selected)
            np.testing.assert_allclose(accurate, inferred, rtol=1e-5, atol=1e-3)
            likelihood.compile_precision_density()
            compiled = likelihood.conditional_log_likelihood(observed, *g, atom_indices=selected)
            np.testing.assert_allclose(compiled, accurate, rtol=1e-5, atol=1e-3)
        np.testing.assert_array_equal(view.flow.iloc[:, 2:].to_numpy(), zero.iloc[:, 2:].to_numpy())
        checks.append(dict(shear=g, selected_test_rows=len(selected),
                           maximum_density_difference=float(np.max(np.abs(inferred-direct[selected])))))
    boundary = np.array([[0., 0., 3., 100.], [0., 0., 4., 10**(.4*(30-25.8))], [0., 0., 4., 100.]])
    np.testing.assert_array_equal(cut(boundary), [False, False, True])
    write_json(ROOT/'preflight.json', dict(status='passed', execution=execution, checks=checks,
        strict_measurement_boundaries=True, unchanged_spin0_context=True))
    print('PREFLIGHT_PASS '+json.dumps(checks), flush=True)


def mock():
    started = time.monotonic()
    execution = identity()
    protocol = read(ROOT/'protocol.json')
    precise_shapes = protocol.get('disk_shape_encoding') == 'log_gap_v1'
    if precise_shapes:
        from sbsi.disk_precision import LOG_GAP_COLUMN, transport_with_log_gap, validate_shape_encoding
    out = ROOT/'mock'
    out.mkdir(exist_ok=False)
    frames, weights, receipts = [], [], []
    for index in range(protocol.get('prior_shards', 28)):
        z, w, receipt = features(index)
        frames.append(z)
        weights.append(w)
        receipts.append(receipt)
    zero = pd.concat(frames, ignore_index=True)
    weights = np.concatenate(weights)
    del frames
    assert len(zero) == protocol['n_atoms']
    np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-9)
    support = ROOT/'support'
    support.mkdir(exist_ok=False)
    np.save(support/'zero.npy', zero.to_numpy())
    np.save(support/'weights.npy', weights)
    write_json(support/'manifest.json', dict(status='complete', receipts=receipts,
        execution=execution, n_atoms=len(zero), output_sha256={n: file_hash(support/n)
            for n in ('zero.npy', 'weights.npy')}))
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    response = response_from_receipts(receipts, load_disk_response(V39_LIKE, device='cuda', nthread=4))
    assert tuple(flow.condition_preprocessor.feature_names) == tuple(detector.preprocessor.feature_names) == FEATURES
    cut = output_cut(flow)
    g = protocol['injected_shear']
    seeds = protocol['mock_seeds']
    atoms_rng, detection_rng = np.random.default_rng(seeds['scene']), np.random.default_rng(seeds['detection'])
    torch.manual_seed(seeds['flow'])
    cdf = np.cumsum(weights/weights.sum())
    cdf[-1] = 1.
    accepted, generated, records, numerical_audit = 0, 0, [], []
    retained_gaps = []
    while accepted < protocol['n_observations']:
        if generated >= 20000000:
            raise RuntimeError('mock selection rate too low; investigate rather than change population')
        atoms = np.searchsorted(cdf, atoms_rng.random(262144), side='right')
        context = zero.iloc[atoms].copy().reset_index(drop=True)
        context['e1_input_p'], context['e2_input_p'] = apply_shear_to_ellipticity(
            context.e1_input_p.to_numpy(), context.e2_input_p.to_numpy(), *g)
        probability = predict_detection_probability(detector, context)
        u = detection_rng.random(len(atoms))
        detected = np.flatnonzero(u < probability)
        if precise_shapes:
            draws, gaps = flow.sample_with_shape_log_gap(context.iloc[detected].reset_index(drop=True),
                                                       n_samples=1, batch_size=8192)
            draws, gaps = draws[:, 0], gaps[:, 0]
        else:
            draws = flow.sample(context.iloc[detected].reset_index(drop=True), n_samples=1, batch_size=8192)[:, 0]
        selected = np.flatnonzero(cut(draws))[:protocol['n_observations']-accepted]
        valid = np.isfinite(draws).all(1) & (draws[:, 2:] > 0).all(1) & (np.square(draws[:, :2]).sum(1) < 1)
        rounded_invalid = ~valid[selected]
        if precise_shapes:
            validate_shape_encoding(draws[selected, :2], gaps[selected])
            valid = np.isfinite(draws).all(1) & (draws[:, 2:] > 0).all(1)
            retained_gaps.append(gaps[selected])
        numerical_audit.append(dict(generated_detected=len(draws), nonfinite=int((~np.isfinite(draws).all(1)).sum()),
            nonpositive_auxiliary=int((~(draws[:, 2:] > 0).all(1)).sum()),
            boundary_or_outside_disk=int((np.square(draws[:, :2]).sum(1) >= 1).sum()),
            rounded_invalid_in_requested_prefix=int(rounded_invalid.sum()),
            invalid_in_requested_prefix=int((~valid[selected]).sum())))
        if protocol.get('replay_failure_root') and generated == 262144:
            previous = np.load(Path(protocol['replay_failure_root'])/'mock/invalid_retained_draws.npz')
            np.testing.assert_array_equal(atoms[detected[selected]], previous['atoms'])
            np.testing.assert_array_equal(draws[selected], previous['draws'])
            print('FAILED_BATCH_REPLAY_PASS atoms and physical draws identical', flush=True)
        # Keep the predeclared measured-selected prefix. Do not reject,
        # clip, replace or redraw a retained invalid sample. An oversized batch
        # may also contain unused draws after the requested stopping point.
        if not valid[selected].all():
            np.savez_compressed(out/'invalid_retained_draws.npz', atoms=atoms[detected[selected]], draws=draws[selected])
            write_json(out/'invalid_retained_draws.json', dict(audit=numerical_audit))
            raise ValueError('invalid retained mock draw; no clipping or resampling')
        records.append((atoms[detected[selected]], draws[selected], u[detected[selected]], probability[detected[selected]]))
        generated += len(atoms)
        accepted += len(selected)
        print(f'MOCK accepted={accepted}/{protocol["n_observations"]} candidates={generated} seconds={time.monotonic()-started:.1f}', flush=True)
    atoms, draws, uniforms, probability = [np.concatenate([r[i] for r in records]) for i in range(4)]
    draws = draws.astype(np.float64)
    coefficient, _ = response.predict(atoms, draws[:, 2], draws[:, 3])
    base_shape = draws[:, :2].copy()
    if precise_shapes:
        base_gaps = np.concatenate(retained_gaps)
        transformed, observed_gaps, logdet = transport_with_log_gap(torch.as_tensor(base_shape),
            torch.as_tensor(base_gaps), torch.as_tensor(coefficient[:, None]*g))
    else:
        transformed, logdet = disk_response_transport(torch.as_tensor(base_shape),
                                                     torch.as_tensor(coefficient[:, None]*g))
    draws[:, :2] = transformed.numpy()
    if precise_shapes:
        inverse, inverse_gaps, inverse_logdet = transport_with_log_gap(transformed, observed_gaps,
            torch.as_tensor(coefficient[:, None]*g), inverse=True)
        np.testing.assert_allclose(inverse_gaps.numpy(), base_gaps, rtol=1e-12, atol=1e-12)
        validate_shape_encoding(draws[:, :2], observed_gaps.numpy())
    else:
        inverse, inverse_logdet = disk_response_transport(transformed,
            torch.as_tensor(coefficient[:, None]*g), inverse=True)
    np.testing.assert_allclose(inverse.numpy(), base_shape, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose((logdet+inverse_logdet).numpy(), 0, atol=1e-10)
    assert cut(draws).all()
    if not precise_shapes:
        assert (np.square(draws[:, :2]).sum(1) < 1).all()
    measurements = pd.DataFrame(draws, columns=flow.target_transform.target_names)
    if precise_shapes:
        measurements[LOG_GAP_COLUMN] = observed_gaps.numpy()
    truth = pd.DataFrame(dict(scene_row=atoms, injected_g1=g[0], injected_g2=g[1], mock_kind='likelihood',
        detection_uniform=uniforms, detection_probability=probability, disk_coefficient=coefficient,
        base_e1=base_shape[:, 0], base_e2=base_shape[:, 1],
        true_magnitude=zero.r_input_p.to_numpy()[atoms], model='V3.9-like'))
    if precise_shapes:
        truth['base_shape_log_disk_gap'] = base_gaps
    MockCatalogue(measurements, truth).save(out)
    center = draws[:, :2].mean(0).tolist()
    write_json(out/'manifest.json', dict(status='complete', execution=execution,
        n_observations=len(draws), candidates=generated, center=center, injected_shear=g,
        seeds=seeds, truth_cuts=None, measured_cuts=protocol['measured_cuts'],
        support_manifest_sha256=file_hash(support/'manifest.json'),
        output_sha256={n: file_hash(out/n) for n in ('measurements.parquet', 'truth.parquet')},
        forward_inverse_max_abs=float(np.max(np.abs(inverse.numpy()-base_shape))),
        numerical_generation_audit=numerical_audit,
        disk_shape_encoding=protocol.get('disk_shape_encoding', 'cartesian_float64'),
        true_r_ge26_selected=int((truth.true_magnitude >= 26).sum()),
        elapsed_seconds=time.monotonic()-started))
    print(f'MOCK_COMPLETE center={center} seconds={time.monotonic()-started:.1f}', flush=True)


def prepare(index):
    started = time.monotonic()
    execution = identity()
    mock_receipt = read(ROOT/'mock/manifest.json')
    assert mock_receipt['execution'] == execution
    zero, weights, feature_receipt = features(index)
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    cache = V39CatalogueCache(zero, weights, feature_receipt['conditions'], detector=detector)
    cache.validate_model_features(flow)
    out = ROOT/f'prepared/shard_{index:02d}'
    out.mkdir(parents=True, exist_ok=False)
    source = read(ROOT/'protocol.json').get('coordinate_source')
    if read(ROOT/'protocol.json').get('reduced_prior_source') is not None:
        for name in ('values.npy', 'dispersion.npy'):
            verify(ROOT/f'coordinate_source/shard_{index:02d}'/name, feature_receipt['coordinate_sha256'][name])
        values = np.load(ROOT/f'coordinate_source/shard_{index:02d}/values.npy')
        dispersion = np.load(ROOT/f'coordinate_source/shard_{index:02d}/dispersion.npy')
        print(f'COORDINATES_REUSED_REDUCED shard={index} rows={len(values)}', flush=True)
    elif source is not None:
        origin = Path(source['root'])
        verify(origin/'execution.json', source['execution_sha256'])
        previous = read(origin/f'prepared/shard_{index:02d}/manifest.json')
        assert previous['execution'] == read(origin/'execution.json')
        verify(ROOT/f'features/shard_{index:02d}/manifest.json', previous['feature_manifest_sha256'])
        for name in ('values.npy', 'dispersion.npy'):
            verify(origin/f'prepared/shard_{index:02d}'/name, previous['output_sha256'][name])
        values = np.load(origin/f'prepared/shard_{index:02d}/values.npy')
        dispersion = np.load(origin/f'prepared/shard_{index:02d}/dispersion.npy')
        print(f'COORDINATES_REUSED shard={index} rows={len(values)}', flush=True)
    else:
        values, dispersion = np.empty((len(zero), 4)), np.empty((len(zero), 4))
        seed(8201, index)
        for lo in range(0, len(zero), 8192):
            hi = min(lo+8192, len(zero))
            draws = flow.sample(zero.iloc[lo:hi], n_samples=128, batch_size=8192, qmc=True)
            values[lo:hi], dispersion[lo:hi] = draws.mean(1), draws.std(1)
            if lo % (8192*32) == 0:
                print(f'COORDINATES shard={index} rows={hi}/{len(zero)} seconds={time.monotonic()-started:.1f}', flush=True)
    assert np.isfinite(values).all() and np.isfinite(dispersion).all()
    np.save(out/'values.npy', values)
    np.save(out/'dispersion.npy', dispersion)
    selection = CatalogueSelection(output_cut(flow), n_samples=64, seed=seed(8101, index), row_chunk=8192)
    points = stencil(mock_receipt['center'], .001)
    masses, probabilities = [], []
    for point in points:
        view = cache.get(*point)
        probabilities.append(view.detection_probability)
        masses.append(detected_selected_mass_shard(selection, flow, view, weights,
            active_indices=np.arange(len(zero)), random_offset_rows=0))
        cache.discard_views([point])
        print(f'NORMALIZATION shard={index} point={point} mass={masses[-1]:.9g} seconds={time.monotonic()-started:.1f}', flush=True)
    np.save(out/'probability.npy', np.array(probabilities))
    write_json(out/'manifest.json', dict(status='complete', execution=execution, shard=index,
        n_rows=len(zero), mock_manifest_sha256=file_hash(ROOT/'mock/manifest.json'),
        feature_manifest_sha256=file_hash(ROOT/f'features/shard_{index:02d}/manifest.json'),
        points=points, selected_mass=masses,
        output_sha256={p.name: file_hash(p) for p in out.glob('*.npy')}, elapsed_seconds=time.monotonic()-started))


def assemble():
    execution = identity()
    receipts, values, rawdisp, probability = [], [], [], []
    for index in range(read(ROOT/'protocol.json').get('prior_shards', 28)):
        root = ROOT/f'prepared/shard_{index:02d}'
        receipt = read(root/'manifest.json')
        assert receipt['status'] == 'complete' and receipt['shard'] == index and receipt['execution'] == execution
        verify(ROOT/'mock/manifest.json', receipt['mock_manifest_sha256'])
        verify(ROOT/f'features/shard_{index:02d}/manifest.json', receipt['feature_manifest_sha256'])
        for name, digest in receipt['output_sha256'].items():
            verify(root/name, digest)
        values.append(np.load(root/'values.npy'))
        rawdisp.append(np.load(root/'dispersion.npy'))
        probability.append(np.load(root/'probability.npy'))
        receipts.append(receipt)
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cpu')
    names = flow.target_transform.target_names
    # Fit global metric/floors on the original-prior portion, as in the
    # controlled bright-extension study, then freeze them for all bright rows.
    n_old = sum(r['n_rows'] for r in receipts[:20])
    values, rawdisp = np.concatenate(values), np.concatenate(rawdisp)
    baseline = coordinate_table(values[:n_old], rawdisp[:n_old], names).with_dispersion_floor(
        percentile=50., fractional_targets=('measured_flux_from_mag_auto',))
    # Fractional scatter flooring and fractional location coordinates are
    # separate settings in this runtime; enable both explicitly.
    baseline = dataclasses.replace(baseline, fractional_targets=('measured_flux_from_mag_auto',))
    floors = baseline.dispersion.min(0)
    flux_floor = np.min(baseline.dispersion[:, 3]/np.maximum(np.abs(baseline.values[:, 3]), np.finfo(float).tiny))
    newfloor = np.broadcast_to(floors, rawdisp[n_old:].shape).copy()
    newfloor[:, 3] = np.where(np.abs(values[n_old:, 3]) > 0, np.abs(values[n_old:, 3])*flux_floor, floors[3])
    coords = dataclasses.replace(baseline, values=values,
        dispersion=np.concatenate([baseline.dispersion, np.maximum(rawdisp[n_old:], newfloor)]))
    object.__setattr__(coords, '_metric_center', baseline._metric_center.copy())
    object.__setattr__(coords, '_metric_scale', baseline._metric_scale.copy())
    np.testing.assert_array_equal(coords.standardize(values[:n_old:10000]), baseline.standardize(values[:n_old:10000]))
    out = ROOT/'assembled'
    out.mkdir(exist_ok=False)
    coords.save(out/'proposal')
    np.save(out/'probability.npy', np.concatenate(probability, axis=1))
    points = receipts[0]['points']
    assert all(r['points'] == points for r in receipts)
    mass = np.sum([r['selected_mass'] for r in receipts], axis=0)
    assert np.isfinite(mass).all() and (mass > 0).all() and (mass <= 1).all()
    ExactPopulationNormalization(dict(zip(map(tuple, points), mass)), .001, execution,
        dict(method_detail='full weighted prior; 64 CRN QMC draws per atom and shear node')).save(out/'normalization.json')
    names = ('proposal/coordinates.npz', 'proposal/manifest.json', 'probability.npy', 'normalization.json')
    write_json(out/'manifest.json', dict(status='complete', execution=execution,
        points=points, receipts=receipts, selected_mass=mass.tolist(), n_old=n_old,
        n_atoms=len(values), metric_center=coords._metric_center.tolist(), metric_scale=coords._metric_scale.tolist(),
        floors=floors.tolist(), fractional_flux_floor=float(flux_floor),
        output_sha256={name: file_hash(out/name) for name in names}))
    print(f'ASSEMBLED atoms={len(values)}', flush=True)


def load_inference():
    execution = identity()
    saved = read(ROOT/'assembled/manifest.json')
    assert saved['execution'] == execution and saved['status'] == 'complete'
    for name, digest in saved['output_sha256'].items():
        verify(ROOT/'assembled'/name, digest)
    support = read(ROOT/'support/manifest.json')
    assert support['execution'] == execution
    for name, digest in support['output_sha256'].items():
        verify(ROOT/'support'/name, digest)
    zero = pd.DataFrame(np.load(ROOT/'support/zero.npy', mmap_mode='r'), columns=FEATURES)
    weights = np.load(ROOT/'support/weights.npy')
    probabilities = np.load(ROOT/'assembled/probability.npy', mmap_mode='r')
    cache = V39CatalogueCache(zero, weights, support['receipts'][0]['conditions'],
        probabilities=dict(zip(map(tuple, saved['points']), probabilities)))
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    response = response_from_receipts(support['receipts'], load_disk_response(V39_LIKE, device='cuda', nthread=4))
    likelihood = DiskCatalogueLikelihood(flow, cache, disk_response=response, selection=CatalogueSelection(output_cut(flow)))
    likelihood.population_normalization = ExactPopulationNormalization.from_payload(read(ROOT/'assembled/normalization.json'))
    assert likelihood.population_normalization.identity == execution
    coords = ProposalCoordinateTable.load(ROOT/'assembled/proposal')
    # Constructor recomputes fractional scales on expanded support; restore
    # the original-prior metric explicitly after every deserialize.
    object.__setattr__(coords, '_metric_center', np.array(saved['metric_center']))
    object.__setattr__(coords, '_metric_scale', np.array(saved['metric_scale']))
    proposal = DefensiveLocalProposal(coords, weights, local_base_weights=probabilities[0], score_dtype=torch.float64)
    receipt = read(ROOT/'mock/manifest.json')
    for name, digest in receipt['output_sha256'].items():
        verify(ROOT/'mock'/name, digest)
    return likelihood, proposal, MockCatalogue.load(ROOT/'mock'), receipt


def run(partition, *, smoke=False):
    started = time.monotonic()
    likelihood, proposal, mock_data, receipt = load_inference()
    start, stop = (0, 8) if smoke else (partition*1000, (partition+1)*1000)
    assert 0 <= start < stop <= len(mock_data.measurements)
    window = MockCatalogue(mock_data.measurements.iloc[start:stop].reset_index(drop=True),
                           mock_data.truth.iloc[start:stop].reset_index(drop=True))
    out = ROOT/('smoke' if smoke else f'inference/part_{partition:02d}')
    out.mkdir(parents=True, exist_ok=False)
    arms = read(ROOT/'protocol.json').get('arms', ['gaussian', 'distance'])
    assert arms and len(arms) == len(set(arms)) and set(arms) <= {'gaussian', 'distance'}
    # Verify identical global mixture draws under both exact-set definitions,
    # including a distance-only run (two probe observations, not a second arm).
    obs = window.measurements.iloc[:2]
    gaussian = proposal.candidates(obs, n_candidates=1024, prefilter_candidates=131072, torch_device='cuda')
    distance = proposal.candidates(obs, n_candidates=1024, prefilter_candidates=1024, torch_device='cuda')
    ids = np.arange(start, start+len(obs))
    a = proposal.draw_tilted(gaussian, obs, n_draws=16384, delta=.1, seed=8701, object_ids=ids, device='cuda')
    b = proposal.draw_tilted(distance, obs, n_draws=16384, delta=.1, seed=8701, object_ids=ids, device='cuda')
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.probability, b.probability)
    del a, b
    if read(ROOT/'protocol.json').get('disk_shape_encoding') == 'log_gap_v1':
        likelihood.compile_precision_density()
    else:
        likelihood.flow_model.compile_log_prob(mode=None, dynamic=True)
    for name, prefilter in (('gaussian', 131072), ('distance', 1024)):
        if name not in arms:
            continue
        print(f'INFERENCE_START part={partition} arm={name} observations={len(window.measurements)} load_seconds={time.monotonic()-started:.1f}', flush=True)
        moments = run_adaptive_section5(likelihood, window, proposal, center=receipt['center'], h=.001,
            draw_ladder=(512, 1024, 2048, 4096, 8192, 16384), n_candidates=1024,
            proposal_prefilter_candidates=prefilter, epsilon=.1, proposal_seed=8701,
            min_ess=32., max_weight_fraction=.5, candidate_backend='torch',
            estimator_mode='tilted_stratified', tilt_delta=.1, retain_full_ladder=True,
            full_information=True, object_id_offset=start, object_chunk=4, atom_chunk=4096,
            progress=lambda done, total, elapsed: print(f'INFERENCE part={partition} arm={name} rows={done}/{total} seconds={elapsed:.2f}', flush=True))
        arrays = {key: getattr(moments, key) for key in ('score', 'information', 'draw_counts',
            'unique_counts', 'ladder_score', 'ladder_information', 'weight_ess', 'weight_max_fraction',
            'weight_relative_error', 'weight_pareto_k')}
        arrays['object_rows'] = np.arange(start, stop)
        np.savez_compressed(out/f'{name}_moments.npz', **arrays)
    write_json(out/'manifest.json', dict(status='complete', smoke=smoke, start=start, stop=stop,
        execution=identity(), mock_manifest_sha256=file_hash(ROOT/'mock/manifest.json'),
        assembled_manifest_sha256=file_hash(ROOT/'assembled/manifest.json'),
        same_draws_check=True, arms=arms, primary_rung=16384,
        output_sha256={p.name: file_hash(p) for p in out.glob('*.npz')}, elapsed_seconds=time.monotonic()-started))
    print(f'INFERENCE_COMPLETE {out} seconds={time.monotonic()-started:.1f}', flush=True)


def summarize_moments(center, score, information):
    hessian = information.sum(0)
    hessian = .5*(hessian+hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        return dict(status='non_positive_information', eigenvalues=eigenvalues.tolist()), None
    step = np.linalg.solve(hessian, score.sum(0))
    influence = np.linalg.solve(hessian/len(score), (score-np.einsum('nij,j->ni', information, step)).T).T
    se = np.sqrt(np.diag(np.cov(influence, rowvar=False, ddof=1))/len(score))
    estimate = np.array(center)+step
    return dict(status='estimated', estimate=estimate.tolist(), robust_standard_error=se.tolist(),
        eigenvalues=eigenvalues.tolist(), g1_relative_error_percent=float(100*(estimate[0]/.02-1)),
        g1_relative_standard_error_pp=float(100*se[0]/.02)), influence


def combine():
    execution = identity()
    protocol = read(ROOT/'protocol.json')
    n_observations = protocol['n_observations']
    assert n_observations > 0 and n_observations % 1000 == 0
    arms = protocol.get('arms', ['gaussian', 'distance'])
    parts = []
    for partition in range(n_observations//1000):
        root = ROOT/f'inference/part_{partition:02d}'
        receipt = read(root/'manifest.json')
        assert receipt['status'] == 'complete' and not receipt['smoke'] and receipt['execution'] == execution
        assert (receipt['start'], receipt['stop']) == (1000*partition, 1000*(partition+1))
        verify(ROOT/'mock/manifest.json', receipt['mock_manifest_sha256'])
        verify(ROOT/'assembled/manifest.json', receipt['assembled_manifest_sha256'])
        data = {}
        for name, digest in receipt['output_sha256'].items():
            verify(root/name, digest)
            data[name.split('_')[0]] = dict(np.load(root/name))
            np.testing.assert_array_equal(data[name.split('_')[0]]['object_rows'], np.arange(receipt['start'], receipt['stop']))
        assert set(data) == set(arms)
        parts.append(data)
    center = read(ROOT/'mock/manifest.json')['center']
    rungs = []
    for i, draws in enumerate((512, 1024, 2048, 4096, 8192, 16384)):
        summaries, influences = {}, {}
        for arm in arms:
            score = np.concatenate([p[arm]['ladder_score'][i] for p in parts])
            info = np.concatenate([p[arm]['ladder_information'][i] for p in parts])
            summaries[arm], influences[arm] = summarize_moments(center, score, info)
        if set(influences) == {'gaussian', 'distance'} and all(x is not None for x in influences.values()):
            diff = influences['distance']-influences['gaussian']
            summaries['paired_difference'] = dict(
                estimate=(np.array(summaries['distance']['estimate'])-summaries['gaussian']['estimate']).tolist(),
                robust_standard_error=np.sqrt(np.diag(np.cov(diff, rowvar=False, ddof=1))/len(diff)).tolist())
        rungs.append(dict(draws=draws, **summaries))
    write_json(ROOT/'result.json', dict(status='complete', execution=execution, n_observations=n_observations,
        arms=arms,
        center=center, injected_shear=[.02, 0.], rungs=rungs, primary_draws=16384,
        interpretation='Model-closure numerical study, not image-simulation calibration or an absolute 0.3% acceptance.',
        limitations=['Single mock realization and proposal seed.', 'One Newton step and 64-draw QMC selection normalizer.',
                     'Finite-draw numerator/log derivative error remains; compare the full draw ladder.',
                     'Non-positive information is reported as a failed solve, never regularized into agreement.']))
    print(json.dumps(rungs[-1], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'mock', 'prepare', 'assemble', 'smoke', 'run', 'combine'])
    parser.add_argument('--shard', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'prepare':
        prepare(args.shard)
    elif args.stage in ('run', 'smoke'):
        run(args.shard, smoke=args.stage == 'smoke')
    else:
        globals()[args.stage]()
