"""Rotated-twin partner of a prepared population mock (prepare_galsbi_population.py); run under Slurm.

Every scene of the original mock is drawn again with the same seed and rotated by 90 degrees about its
target. The model sees a scene only through rotation-invariant quantities (magnitudes, sizes, Sersic index,
neighbour distances, crowding) and the target's intrinsic ellipticity, so the rotation is exactly e -> -e of
the target. Shear, the C8 detection draw (same uniforms), the F8 measurement draw, the measured
radius/flux cut and the disk-response transport then act on the twin exactly as in mock_panel.

The twin's F8 base draws are coupled to the original's for the same scene (--coupling): 'antithetic'
negates the two shape coordinates, 'same' reuses all four, 'independent' draws fresh ones. Scenes usable
only in the twin always get fresh draws. Every coupling leaves the twin population distributed exactly like
the original one; the coupling only changes how much noise cancels between the two mocks.

The original chunks are regenerated with the original functions and must reproduce the stored observations
row for row. The twin covers all parents of every complete chunk. Only a partially retained final
chunk stops at the last retained original observation. Twins are kept by their own selection (no pair-completeness requirement, no truth
cuts); pairs.parquet links each scene to its original and twin rows (-1 = not selected).
--test-chunks k: only regenerate k chunks and report, per coupling, how much the pair average of the
measured shapes scatters compared with independent galaxies (no twin catalogue written).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from check_galsbi_continuous_prior import build_prior, digest, save
from check_galsbi_summary_sampler import mock_panel, prepared
from sbsi.catalogue_likelihood import predict_detection_probability
from sbsi.disk_precision import LOG_GAP_COLUMN, transport_with_log_gap
from sbsi.galsbi_scene_prior import GalSBIScenePrior
from sbsi.mock_population import last_retained_parent
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
from sbsi.shear_map import apply_shear_to_ellipticity

BATCH = 4096                      # MeasurementModel.sample_with_shape_log_gap batch size used by mock_panel


def panel(zero, response, flow, detector, *, seed, shear, sign, latent):
    """mock_panel(seed=seed, shear, retain_population=True) with intrinsic shapes times sign and given base
    draws latent (scenes x 4); returns the frame, the scene ids and the usable scenes."""
    context = zero.copy()
    context['e1_input_p'], context['e2_input_p'] = apply_shear_to_ellipticity(
        sign*zero.e1_input_p.to_numpy(), sign*zero.e2_input_p.to_numpy(), *shear)
    probability = predict_detection_probability(detector, context)
    usable = np.flatnonzero(np.random.default_rng(seed).random(len(zero)) < probability)
    draws, gap = [], []
    with torch.no_grad():
        for start in range(0, len(usable), BATCH):
            rows = usable[start:start+BATCH]
            c = flow._context_from_frame(context.iloc[rows])
            z = torch.as_tensor(latent[rows], dtype=c.dtype, device=c.device)
            out = flow.model.forward_physical(z, c, return_shape_log_gap=True)
            draws.append(out[0].cpu().numpy())
            gap.append(out[2].cpu().numpy())
    draws, gap = np.concatenate(draws), np.concatenate(gap)
    keep = np.flatnonzero((draws[:, 2] > 3.) & (draws[:, 3] > 10**(.4*(30-25.8))))
    draws, gap, ids = draws[keep], gap[keep], usable[keep]
    b, _ = response.predict(ids, draws[:, 2], draws[:, 3])
    device = flow.device
    transformed, gap, _ = transport_with_log_gap(torch.as_tensor(draws[:, :2], device=device),
        torch.as_tensor(gap, device=device), torch.as_tensor(b[:, None]*np.array(shear), device=device))
    draws[:, :2] = transformed.cpu().numpy()
    frame = pd.DataFrame(draws, columns=flow.target_transform.target_names)
    frame[LOG_GAP_COLUMN] = gap.cpu().numpy()
    return frame, ids, usable


def original_latent(zero, flow, detector, *, seed, shear):
    """mock_panel's base draws, assigned to the scenes they were used for (NaN for unusable scenes)."""
    context = zero.copy()
    context['e1_input_p'], context['e2_input_p'] = apply_shear_to_ellipticity(
        zero.e1_input_p.to_numpy(), zero.e2_input_p.to_numpy(), *shear)
    probability = predict_detection_probability(detector, context)
    usable = np.flatnonzero(np.random.default_rng(seed).random(len(zero)) < probability)
    torch.manual_seed(seed+1)
    dim = flow.target_transform.dim
    z = [torch.randn(len(usable[s:s+BATCH]), dim, dtype=torch.float32, device=flow.device).cpu().numpy()
         for s in range(0, len(usable), BATCH)]
    latent = np.full((len(zero), dim), np.nan, dtype=np.float32)
    latent[usable] = np.concatenate(z)
    return latent


def coupled(latent, coupling, seed):
    fresh = np.random.default_rng(seed).standard_normal(latent.shape).astype(np.float32)
    if coupling == 'independent':
        return fresh
    twin = latent.copy()
    if coupling == 'antithetic':
        twin[:, :2] *= -1
    missing = ~np.isfinite(twin).all(1)
    twin[missing] = fresh[missing]
    return twin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coupling', choices=('antithetic', 'same', 'independent'), default='antithetic')
    parser.add_argument('--test-chunks', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('fresh output required')
    original = json.loads((args.original/'protocol.json').read_text())
    receipt = json.loads((args.original/'result.json').read_text())
    progress = json.loads((args.original/'progress.json').read_text())['chunks']
    if (original.get('reference_kind') != 'galsbi_population_v1' or original.get('magnitude_limits')
            or not receipt['preparation_checks_pass']
            or receipt['observations_sha256'] != digest(args.original/'observations.parquet')):
        raise ValueError('validated plain population reference required')
    stored = pd.read_parquet(args.original/'observations.parquet')
    shear, conditions = original['shear'], original['conditions']
    args.output.mkdir(parents=True, exist_ok=True)
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    model = load_disk_response(V39_LIKE, device='cpu', nthread=1)
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)

    def distance(z):
        return cosmo.background.dist_ang_a(a=1/(1+z))

    chunks = progress[:args.test_chunks] if args.test_chunks else progress
    frames, pairs, replay, test = [], [], [], {c: [] for c in ('antithetic', 'same', 'independent')}
    cutoff_audit = []
    offset = twin_rows = 0
    for receipt_chunk in chunks:
        chunk, seed, take = receipt_chunk['chunk'], receipt_chunk['seed'], receipt_chunk['selected']
        if seed != original['seed']+10*chunk:
            raise ValueError('original chunk seeds differ from the generator convention')
        zero, response, _ = prepared(prior, distance, conditions, model, n=8192, seed=seed)
        frame, _, ids = mock_panel(zero, response, flow, detector, seed=seed+1, shear=shear, retain_population=True)
        latent = original_latent(zero, flow, detector, seed=seed+1, shear=shear)
        again, again_ids, _ = panel(zero, response, flow, detector, seed=seed+1, shear=shear, sign=1, latent=latent)
        if not (np.array_equal(ids, again_ids) and np.array_equal(frame.to_numpy(), again.to_numpy())):
            raise ValueError(f'chunk {chunk}: explicit base draws do not reproduce mock_panel')
        mine = stored.iloc[offset:offset+take].to_numpy()
        replay.append(float(np.max(np.abs(frame.iloc[:take].to_numpy()-mine))) if take else 0.)
        if take and (len(frame) < take or replay[-1] != 0.):
            raise ValueError(f'chunk {chunk}: regenerated observations differ from the stored mock ({replay[-1]})')
        legacy_last = int(ids[take-1]) if take else -1
        last = last_retained_parent(ids, take, len(zero),
                                    final_chunk=chunk == progress[-1]['chunk'])
        coverage = dict(chunk=chunk, parents=len(zero), generated_selected_originals=len(ids),
                        retained_originals=take, last_retained_parent=last, legacy_last_parent=legacy_last,
                        additional_parents=last-legacy_last, couplings={})
        cutoff_audit.append(coverage)
        for name in (test if args.test_chunks else [args.coupling]):
            twin, twin_ids, _ = panel(zero, response, flow, detector, seed=seed+1, shear=shear, sign=-1,
                                      latent=coupled(latent, name, seed+3))
            inside = twin_ids <= last
            added = inside & (twin_ids > legacy_last)
            final_selected = ((twin.iloc[:, 2].to_numpy() < 100.)
                              & ((twin.iloc[:, :2].to_numpy()**2).sum(1) < .99**2))
            coverage['couplings'][name] = dict(additional_selected_twins=int(added.sum()),
                additional_analysis_selected_twins=int((added & final_selected).sum()),
                additional_selected_parent_ids=twin_ids[added].tolist())
            if args.test_chunks:
                common, a, b = np.intersect1d(ids[:take], twin_ids[inside], return_indices=True)
                test[name].append(dict(original=frame.iloc[a, :4].to_numpy(), twin=twin[inside].iloc[b, :4].to_numpy(),
                                       n_original=take, n_twin=int(inside.sum())))
                continue
            frames.append(twin[inside])
            scenes = np.union1d(ids[:take], twin_ids[inside])
            o = np.full(len(scenes), -1)
            t = np.full(len(scenes), -1)
            o[np.searchsorted(scenes, ids[:take])] = offset+np.arange(take)
            t[np.searchsorted(scenes, twin_ids[inside])] = twin_rows+np.arange(inside.sum())
            pairs.append(pd.DataFrame(dict(chunk=chunk, scene=scenes, original_row=o, twin_row=t)))
            twin_rows += int(inside.sum())
        offset += take
        print('TWIN_PROGRESS', chunk, offset, twin_rows, 'replay', replay[-1], flush=True)
    if args.test_chunks:
        report = {}
        for name, parts in test.items():
            a = np.concatenate([p['original'] for p in parts])
            b = np.concatenate([p['twin'] for p in parts])
            mag = 30-2.5*np.log10(a[:, 3])
            report[name] = dict(pairs=len(a), original=sum(p['n_original'] for p in parts),
                                twin=sum(p['n_twin'] for p in parts))
            for label, m in (('all', np.ones(len(a), bool)), ('mag<23', mag < 23), ('mag>=23', mag >= 23)):
                # variance of the pair mean relative to the mean of two independent galaxies (1 = no gain)
                ratio = [float(np.var((a[m, k]+b[m, k])/2)/((np.var(a[m, k])+np.var(b[m, k]))/4)) for k in (0, 1)]
                corr = [float(np.corrcoef(a[m, k], b[m, k])[0, 1]) for k in (0, 1)]
                report[name][label] = dict(pairs=int(m.sum()), pair_variance_ratio_e1_e2=ratio, correlation_e1_e2=corr,
                                           radius_correlation=float(np.corrcoef(a[m, 2], b[m, 2])[0, 1]))
            print(name, json.dumps(report[name]), flush=True)
        save(args.output/'coupling_test.json', dict(chunks=len(chunks), replay_errors=replay, report=report, parent_cutoff_audit=cutoff_audit))
        return
    if offset != receipt['observation_count']:
        raise ValueError('original chunks do not add up to the stored mock')
    observed = pd.concat(frames, ignore_index=True)
    if not np.isfinite(observed.to_numpy()).all():
        raise ValueError('nonfinite generated measurements')
    pairs = pd.concat(pairs, ignore_index=True)
    observed.to_parquet(args.output/'observations.parquet', index=False)
    pairs.to_parquet(args.output/'pairs.parquet', index=False)
    save(args.output/'protocol.json', dict(original, twin_of=str(args.original.resolve()),
        twin_of_observations_sha256=receipt['observations_sha256'], coupling=args.coupling,
        twin='each original scene rotated by 90 degrees about its target (target intrinsic e -> -e); same scene, '
             'detection and (coupled) measurement random numbers; own selection; same parent scenes as the original',
        parent_coverage='all parents of complete chunks; last retained original only for a partial final chunk',
        preparation_sha256=digest(__file__), parent_coverage_sha256=digest(Path('sbsi/mock_population.py'))))
    save(args.output/'result.json', dict(status='prepared_population', preparation_checks_pass=True,
        observation_count=len(observed), original_replay_max_error=max(replay), parent_cutoff_audit=cutoff_audit,
        pairs=dict(scenes=len(pairs), both=int(((pairs.original_row >= 0) & (pairs.twin_row >= 0)).sum()),
                   original_only=int((pairs.twin_row < 0).sum()), twin_only=int((pairs.original_row < 0).sum())),
        observations_sha256=digest(args.output/'observations.parquet'),
        pairs_sha256=digest(args.output/'pairs.parquet'), protocol_sha256=digest(args.output/'protocol.json'),
        limitation='model-generated population; twin of ' + str(args.original.resolve())))
    print('TWIN_READY', len(observed), flush=True)


if __name__ == '__main__':
    main()
