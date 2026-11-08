#!/usr/bin/env python
"""Paired forward response under nested truth populations; no likelihood fit.

The retained population is reproduced by its frozen identities. Expanded
populations are diagnostic arms, never cuts applied to production inference.
No archived module is imported and no checkpoint is fitted or selected here.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.catalogue_likelihood import predict_detection_probability
from sbsi.crowding import FEATURES, FLOW_FEATURES, classifier_features
from sbsi.disk_inference_store import file_hash, write_json
from sbsi.disk_response_transport import disk_response_transport
from sbsi.flow_size_condition import circularized_radius
from sbsi.forward_catalogue import make_pair_catalogue, prepare_flow_inputs, rescale_emulator_pairs
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE, load_detection_classifier, load_disk_response
from sbsi.selection_model import load_selection_model
from sbsi.shear_map import apply_shear_to_ellipticity

CONDITIONS = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224, pixel_rms=.312)
COLUMNS = ('index', 'RA', 'DEC', 'redshift', 'Re', 'axis_ratio', 'position_angle', 'sersic_n', 'r')
GROUPS = ('retained', 'all_bright', 'all', 'added_bright', 'faint')
FLUX_MIN = 10**(.4*(30-25.8))


def selected(values):
    return (values[..., 2] > 3.) & (values[..., 3] > FLUX_MIN)


def group_masks(ids, magnitude, retained_ids):
    ids, magnitude, retained_ids = np.asarray(ids), np.asarray(magnitude), np.asarray(retained_ids)
    if len(np.unique(ids)) != len(ids) or len(np.unique(retained_ids)) != len(retained_ids):
        raise ValueError('duplicate truth identities')
    if magnitude.shape != ids.shape or not np.isfinite(magnitude).all():
        raise ValueError('aligned finite truth magnitudes required')
    old = np.isin(ids, retained_ids)
    if not np.isin(retained_ids, ids).all() or np.any(magnitude[old] >= 26):
        raise ValueError('retained parent absent or outside original magnitude support')
    bright = magnitude < 26
    return dict(retained=old, all_bright=bright, all=np.ones(len(ids), bool),
                added_bright=bright & ~old, faint=~bright)


def measured_leg(root, case, sign, ids):
    base = root/f'case{case}_{sign*.02}'/'real0/catalogues'
    paths = dict(truth=base/'input/gals_info_tile180.0_-0.5.feather',
                 cross=base/'CrossMatch/tile180.0_-0.5_rot0_matched.feather',
                 shapes=base/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather')
    truth = pd.read_feather(paths['truth'])
    cross = pd.read_feather(paths['cross'], columns=['id_detec', 'id_input'])
    shape = pd.read_feather(paths['shapes'], columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO'])
    if (truth.index_input.duplicated().any() or cross.id_input.duplicated().any()
            or cross.id_detec.duplicated().any() or shape.NUMBER.duplicated().any()):
        raise ValueError('duplicate catalogue identities')
    np.testing.assert_allclose(truth[['gamma1_input', 'gamma2_input']],
                               np.tile([sign*.02, 0.], (len(truth), 1)), rtol=0, atol=1e-14)
    if set(truth.index_input) != set(ids):
        raise ValueError('rendered truth parents differ between legs; reject unmatched rows')
    joined = cross.merge(shape, left_on='id_detec', right_on='NUMBER', validate='one_to_one')
    outside = ~joined.id_input.isin(ids)
    joined = joined.loc[~outside].set_index('id_input')
    values = joined.reindex(ids)[['NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO']].to_numpy(float)
    with np.errstate(over='ignore', invalid='ignore'):
        values[:, 3] = 10**(.4*(30-values[:, 3]))
    usable = (np.isfinite(values).all(1) & (values[:, 2:] > 0).all(1)
              & (np.square(values[:, :2]).sum(1) < 1))
    return values, usable, dict(sources={k: dict(path=str(p), sha256=file_hash(p)) for k, p in paths.items()},
        cross_without_shape=len(cross)-len(joined)-int(outside.sum()),
        shape_without_cross=len(shape)-len(joined)-int(outside.sum()),
        matches_outside_truth_dropped=int(outside.sum()),
        parent_without_matched_measurement=int((~np.isin(ids, joined.index)).sum()),
        usable=int(usable.sum()), selected=int((usable & selected(values)).sum())), truth


def moments(values, weight, masks):
    """Unnormalized sufficient sums; denominators must remain separate per leg."""
    good = selected(values)
    if values.ndim == 2:
        w = good*weight
        nums = np.where(w[:, None] > 0, values[:, :2], 0.)*w[:, None]
        den = w
    else:
        w = good*weight[:, None]
        nums = (np.where(w[..., None] > 0, values[..., :2], 0.)*w[..., None]).mean(1)
        den = w.mean(1)
    return {name: np.r_[den[mask].sum(), nums[mask].sum(0)].tolist() for name, mask in masks.items()}


def response(plus, minus, h=.02):
    p, m = np.asarray(plus), np.asarray(minus)
    if p[0] <= 0 or m[0] <= 0:
        raise ValueError('positive selected mass required in both legs')
    return (p[1:]/p[0]-m[1:]/m[0])/(2*h)


def context_and_pairs(primary, scene, pair_query_k=21):
    frame = prepare_flow_inputs(primary, scene, conditions=CONDITIONS)
    frame['e1_input_p'], frame['e2_input_p'] = frame.e1_input_rot0_p, frame.e2_input_rot0_p
    frame['circularized_Re_input_p'] = circularized_radius(frame.Re_input_p, frame.axis_ratio_input_p)
    pairs = make_pair_catalogue(primary, scene, r_max_arcsec=10., k=pair_query_k)
    pairs = pairs.loc[(pairs.distance > 0) & (pairs.distance < 10.)]
    pairs = pairs.sort_values(['primary_row', 'distance'], kind='stable').groupby('primary_row', sort=False).head(20)
    return frame.loc[:, FLOW_FEATURES], rescale_emulator_pairs(pairs, CONDITIONS).reset_index(drop=True)


def paired_draws(model, contexts, draws, seed):
    # Restored antithetic-CRN convention from historical physical_context_draws,
    # evaluated with FP64 neural/input arithmetic. No archived dependency.
    gen = torch.Generator(device=contexts[0].device).manual_seed(seed)
    n = len(contexts[0])
    z = torch.randn(n, draws//2, model.target_dim, dtype=contexts[0].dtype,
                    device=contexts[0].device, generator=gen)
    z = torch.cat((z, -z), 1).reshape(-1, model.target_dim)
    return [model.forward_physical(z, c[:, None].expand(-1, draws, -1).reshape(-1, model.context_dim))[0]
            .reshape(n, draws, model.target_dim) for c in contexts]


def close_crowding_context(scene):
    """Replay Sept24 physical-distance Gaussian features over the full scene.

    Deliberately restored from the newer run's build_close_crowding_lookup;
    no archived imports, neighbour truncation, or measured-property selection.
    """
    from scipy.spatial import cKDTree
    xy = np.column_stack((scene.RA*np.cos(np.deg2rad(scene.DEC.mean())), scene.DEC))*3600.
    flux = 10**(-.4*(scene.r.to_numpy(float)-30.))
    pairs = cKDTree(xy).query_pairs(10., output_type='ndarray')
    i, j = pairs.T
    d2 = ((xy[i]-xy[j])**2).sum(1)
    columns = []
    for scale in (.5, 1., 2.):
        kernel = np.exp(-d2/(2*scale**2))
        total = np.zeros(len(scene))
        np.add.at(total, i, kernel*flux[j]/flux[i])
        np.add.at(total, j, kernel*flux[i]/flux[j])
        columns.append(np.log1p(total))
    return np.column_stack(columns).astype(np.float32)


@torch.no_grad()
def run_case(args, case, models, parent_manifest, reference):
    start = time.monotonic()
    old_path = args.parent/f'case{case:03d}.npz'
    if file_hash(old_path) != parent_manifest['case_reports'][str(case)]['sha256']:
        raise ValueError('retained parent changed')
    with np.load(old_path) as old:
        old_ids, old_context = old['input_index'], old['context']
    plus_path = args.catalogues/f'case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather'
    scene = pd.read_feather(plus_path).rename(columns={f'{c}_input': c for c in COLUMNS})
    scene = scene.loc[:, COLUMNS].sort_values('index').reset_index(drop=True)
    ids = scene['index'].to_numpy()
    groups = group_masks(ids, scene.r.to_numpy(), old_ids)
    result = dict(case=case, population_counts={k: int(v.sum()) for k, v in groups.items()},
                  measured={}, model={}, joins={}, invalid={}, context_replay_max_abs=0.)
    legs = []
    for leg, sign in [('plus', 1), ('minus', -1)]:
        values, usable, report, truth = measured_leg(args.catalogues, case, sign, ids)
        comparable = truth.rename(columns={f'{c}_input': c for c in COLUMNS}).set_index('index').loc[ids, list(COLUMNS[1:])]
        np.testing.assert_allclose(comparable, scene.loc[:, COLUMNS[1:]], rtol=0, atol=0)
        legs.append(usable)
        result['joins'][leg] = report
        result['measured'][leg] = moments(values, usable.astype(float), groups)
        key = 'truth_parent/mag25.8_radius_gt0.60/modeled_smooth'
        ci = reference['cases'].index(case)
        saved = reference['per_case_sufficient'][key]['measured'][leg]
        expected = np.r_[saved['denominator'][ci][0], saved['numerator'][ci][0]]
        np.testing.assert_allclose(result['measured'][leg]['retained'], expected, rtol=1e-12, atol=1e-9)
    result['measured_replay'] = 'passed against retained per-case sufficient sums'
    print(f'MEASURED_REPLAY case={case} counts={result["population_counts"]}', flush=True)
    if args.observed_only:
        return result
    flow, classifier, emulator = models
    extra_context = None
    raw_features = flow.condition_preprocessor.feature_names
    if len(raw_features) == 11:
        if raw_features != [*FLOW_FEATURES, 'nbr_close_050', 'nbr_close_100', 'nbr_close_200']:
            raise ValueError('unexpected 11-input feature order')
        extra_context = close_crowding_context(scene)
        if args.candidate_parent is None:
            raise ValueError('11-input model requires frozen context replay parent')
        candidate_manifest = json.loads((args.candidate_parent/'manifest.json').read_text())
        candidate_path = args.candidate_parent/f'case{case:03d}.npz'
        if file_hash(candidate_path) != candidate_manifest['case_reports'][str(case)]['sha256']:
            raise ValueError('candidate parent changed')
        with np.load(candidate_path) as frozen:
            pos = pd.Index(ids).get_indexer(frozen['input_index'])
            if np.any(pos < 0):
                raise ValueError('candidate replay identity missing')
            np.testing.assert_allclose(extra_context[pos], frozen['context'][:, 8:], rtol=3e-5, atol=3e-5)
            crowd_max_abs = float(np.max(np.abs(extra_context[pos]-frozen['context'][:, 8:])))
        result['close_crowding_replay'] = dict(parent_sha256=file_hash(candidate_path),
            checked=len(pos), max_abs=crowd_max_abs)
        print(f'CLOSE_CROWDING_REPLAY case={case} checked={len(pos)}', flush=True)
    elif list(raw_features) != list(FLOW_FEATURES):
        raise ValueError('unsupported flow context')
    scene['position_angle'] = np.deg2rad(scene.position_angle)
    check_ids = old_ids[:128]
    check_primary = scene.set_index('index', drop=False).loc[check_ids].reset_index(drop=True)
    replay, replay_pairs = context_and_pairs(check_primary, scene, args.pair_query_k)
    np.testing.assert_allclose(replay.to_numpy(), old_context[:128], rtol=3e-5, atol=3e-5)
    if args.candidate_pairs is not None:
        pair_path = args.candidate_pairs/f'case{case:03d}.feather'
        receipt = json.loads(pair_path.with_suffix('.json').read_text())
        if file_hash(pair_path) != receipt['sha256']:
            raise ValueError('frozen candidate pairs changed')
        columns = ['index_input_p', 'index_input_s', *receipt['features']]
        cached = pd.read_feather(pair_path, columns=columns)
        cached = cached.loc[cached.index_input_p.isin(check_ids)].sort_values(columns[:2]).reset_index(drop=True)
        reconstructed = replay_pairs[columns].sort_values(columns[:2]).reset_index(drop=True)
        np.testing.assert_array_equal(cached[columns[:2]], reconstructed[columns[:2]])
        np.testing.assert_allclose(cached[columns[2:]], reconstructed[columns[2:]], rtol=3e-5, atol=3e-5)
        result['pair_replay'] = dict(sha256=receipt['sha256'], checked=len(cached), query_k=args.pair_query_k)
        print(f'PAIR_REPLAY case={case} checked={len(cached)}', flush=True)
    print(f'CONTEXT_REPLAY case={case} checked={len(check_ids)}', flush=True)
    beta, fwhm = CONDITIONS['moffat_beta'], CONDITIONS['psf_fwhm']
    psf = fwhm/2*np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))
    for arm in ('predicted_usable', 'actual_usable'):
        result['model'][arm] = {leg: {g: [0., 0., 0.] for g in GROUPS} for leg in ('plus', 'minus')}
    for start_row in range(0, len(scene), args.context_chunk):
        stop_row = min(start_row+args.context_chunk, len(scene))
        primary = scene.iloc[start_row:stop_row].reset_index(drop=True)
        raw, pairs = context_and_pairs(primary, scene, args.pair_query_k)
        # Independent reconstruction must reproduce the frozen retained contexts.
        old_loc = pd.Index(old_ids).get_indexer(primary['index'])
        matched = old_loc >= 0
        if matched.any():
            delta = np.max(np.abs(raw.to_numpy()[matched]-old_context[old_loc[matched]]))
            result['context_replay_max_abs'] = max(result['context_replay_max_abs'], float(delta))
            np.testing.assert_allclose(raw.to_numpy()[matched], old_context[old_loc[matched]], rtol=3e-5, atol=3e-5)
        for offset in range(0, len(raw), args.batch):
            stop = min(offset+args.batch, len(raw))
            row_ids = primary['index'].to_numpy()[offset:stop]
            raw_block = raw.iloc[offset:stop].to_numpy(float)
            pair_block = pairs.loc[pairs.index_input_p.isin(row_ids)]
            disk = CatalogueDiskResponse.from_pairs(row_ids, pair_block, emulator)
            contexts, probabilities = [], []
            for sign in (1, -1):
                e1, e2 = apply_shear_to_ellipticity(raw_block[:, 0], raw_block[:, 1], sign*.02, 0.)
                state = raw_block.copy()
                state[:, :2] = np.column_stack((e1, e2))
                flow_state = state if extra_context is None else np.column_stack((state,
                    extra_context[start_row+offset:start_row+stop]))
                contexts.append(flow.condition_preprocessor.transform_tensor(torch.tensor(flow_state, device='cuda')))
                features = classifier_features(row_ids, raw_block, state[:, :2], pair_block, psf)
                probabilities.append(predict_detection_probability(classifier, pd.DataFrame(features, columns=FEATURES)))
            global_start, global_stop = start_row+offset, start_row+stop
            generated = paired_draws(flow.model, contexts, args.draws, args.seed+10000019*case+global_start)
            masks = {g: m[global_start:global_stop] for g, m in groups.items()}
            for li, (leg, sign, samples) in enumerate(zip(('plus', 'minus'), (1, -1), generated)):
                arr = samples.cpu().numpy()
                invalid = (~np.isfinite(arr).all(-1)) | (arr[..., 2:] <= 0).any(-1)
                # A finite failing auxiliary bound proves zero selected mass,
                # even when another output underflows. Keep the original draw
                # denominator; never renormalize on numerical validity.
                excluded = ((np.isfinite(arr[..., 2]) & (arr[..., 2] <= 3.))
                            | (np.isfinite(arr[..., 3]) & (arr[..., 3] <= FLUX_MIN)))
                if (invalid & ~excluded).any():
                    rr, dd = np.nonzero(invalid)
                    write_json(args.output/f'case{case}_{leg}_invalid.json', dict(
                        case=case, leg=leg, block_start=global_start,
                        invalid_draws=int(invalid.sum()), affected_objects=int(len(np.unique(rr))),
                        zero_radius=int((arr[..., 2] == 0).sum()), zero_flux=int((arr[..., 3] == 0).sum()),
                        nonfinite=int((~np.isfinite(arr).all(-1)).sum()),
                        invalid_with_selected_auxiliary=int((invalid & selected(arr)).sum()),
                        input_ids=row_ids[np.unique(rr)].tolist(),
                        truth_magnitudes=raw_block[np.unique(rr), 3].tolist(),
                        samples_file=f'case{case}_{leg}_invalid.npz'))
                    np.savez(args.output/f'case{case}_{leg}_invalid.npz',
                             raw=raw_block[rr], values=arr[rr, dd], input_ids=row_ids[rr], draw_ids=dd)
                    raise ValueError(f'case{case} row{global_start} {leg}: invalid physical auxiliary output; no repair')
                before = (arr[..., :2]**2).sum(-1)
                rr, dd = np.nonzero(selected(arr))
                if len(rr):
                    b, _ = disk.predict(rr, arr[rr, dd, 2], arr[rr, dd, 3])
                    shape, _ = disk_response_transport(samples[rr, dd, :2],
                        torch.tensor(b, device='cuda')[:, None]*samples.new_tensor([sign*.02, 0.]))
                    arr[rr, dd, :2] = shape.cpu().numpy()
                if not np.isfinite(arr[rr, dd]).all():
                    raise ValueError('nonfinite transported output')
                after = (arr[..., :2]**2).sum(-1)
                counts = result['invalid'].setdefault(leg, dict(draws=0, disk_before=0, disk_after=0,
                    outside_by_1e12=0, auxiliary_invalid_proven_unselected=0, invalid_by_group={g: 0 for g in GROUPS}))
                counts['draws'] += before.size
                counts['auxiliary_invalid_proven_unselected'] += int(invalid.sum())
                for group, mask in masks.items():
                    counts['invalid_by_group'][group] += int(invalid[mask].sum())
                counts['disk_before'] += int((before >= 1).sum())
                counts['disk_after'] += int((after >= 1).sum())
                counts['outside_by_1e12'] += int((after > 1+1e-12).sum())
                if counts['outside_by_1e12']:
                    raise ValueError('transport leaves disk beyond roundoff')
                # Preserve finite boundary-rounded draws in moment sums and count them;
                # no clipping or silently conditioning on valid generated samples.
                for arm, weight in [('predicted_usable', probabilities[li]),
                                    ('actual_usable', legs[li][global_start:global_stop].astype(float))]:
                    sums = moments(arr, weight, masks)
                    for g, value in sums.items():
                        result['model'][arm][leg][g] = (np.array(result['model'][arm][leg][g])+value).tolist()
            if offset == 0:
                print(f'FORWARD_PROGRESS case={case} rows={global_stop}/{len(scene)} seconds={time.monotonic()-start:.1f}', flush=True)
        write_json(args.output/f'case{case}_progress.json', result)
    result['seconds'] = time.monotonic()-start
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogues', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+', required=True)
    p.add_argument('--observed-only', action='store_true')
    p.add_argument('--draws', type=int, default=64)
    p.add_argument('--seed', type=int, default=7301)
    p.add_argument('--context-chunk', type=int, default=32768)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--candidate', type=Path, help='Frozen JSON with flow/classifier and optional emulator model/metadata paths and hashes')
    p.add_argument('--candidate-parent', type=Path, help='Frozen 11-input replay parent for close-crowding model')
    p.add_argument('--candidate-pairs', type=Path, help='Frozen pair table for feature and identity replay')
    p.add_argument('--pair-query-k', type=int, default=21, help='Neighbour query slots including self; saved Main80 uses 20')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('catalogue work requires scheduler allocation')
    if args.output.exists() or len(set(args.cases)) != len(args.cases) or args.draws < 2 or args.draws % 2:
        raise ValueError('new output, distinct cases, even positive draws required')
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:cpus])
    torch.set_num_threads(cpus)
    args.output.mkdir(parents=True)
    (args.output/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    protocol = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        script_sha256=file_hash(__file__), reference_sha256=file_hash(args.reference),
        parent_manifest_sha256=file_hash(args.parent/'manifest.json'),
        convention='Independent per-leg U, radius>3 pixels, MAG_AUTO<25.8; coherent +/-0.02 g1; antithetic CRN; m=100*(Rmeasured/Rmodel-1).',
        purpose='Development domain diagnostic, not independent validation or production inference.',
        boundary_rounding='Count and retain finite disk-boundary-rounded draws in moments; no clipping. Auxiliary failures proven outside selection contribute zero with the original draw denominator. Abort for any other invalid output or shape outside disk by >1e-12. Transport only selected draws: auxiliary cuts are invariant under shape transport.')
    write_json(args.output/'protocol.json', protocol)
    models = None
    if not args.observed_only:
        V36_LIKE.validate()
        identity = dict(flow=dict(path=str(V36_LIKE.flow_checkpoints[0]), sha256=V36_LIKE.flow_sha256s[0]),
            classifier=dict(path=str(V36_LIKE.detection_classifiers[0]), sha256=V36_LIKE.detection_classifier_sha256s[0]),
            emulator=V36_LIKE.emulator_sha256)
        model_paths = V36_LIKE
        if args.candidate is not None:
            custom = json.loads(args.candidate.read_text())
            for component in ('flow', 'classifier'):
                if file_hash(custom[component]['path']) != custom[component]['sha256']:
                    raise ValueError(f'{component} checkpoint changed')
                identity[component] = custom[component]
            identity['candidate_sha256'] = file_hash(args.candidate)
            if 'emulator' in custom:
                e = custom['emulator']
                model_paths = replace(V36_LIKE, emulator_model=Path(e['path']),
                    emulator_metadata=Path(e['metadata_path']), emulator_sha256=e['sha256'],
                    emulator_metadata_sha256=e['metadata_sha256'])
                model_paths.validate()
                identity['emulator'] = e
        flow = load_measurement_model(identity['flow']['path'], device='cuda')
        flow.model.double().eval()
        classifier = (load_detection_classifier(V36_LIKE, device='cuda') if args.candidate is None else
                      load_selection_model(identity['classifier']['path'], device='cuda'))
        models = (flow, classifier, load_disk_response(model_paths, device='cuda', nthread=cpus))
        write_json(args.output/'model_identity.json', identity)
    parent = json.loads((args.parent/'manifest.json').read_text())
    reference = json.loads(args.reference.read_text())
    for case in args.cases:
        result = run_case(args, case, models, parent, reference)
        write_json(args.output/f'case{case}.json', result)
        print(f'CASE_COMPLETE case={case}', flush=True)
    write_json(args.output/'completed.json', dict(cases=args.cases, status='diagnostic_complete'))


if __name__ == '__main__':
    main()
