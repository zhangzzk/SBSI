"""Prepare an independent measured-selected model population; run under Slurm."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_galsbi_continuous_prior import build_prior, digest, save
from check_galsbi_summary_sampler import geometry_replay, mock_panel, prepared
from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood
from sbsi.galsbi_scene_prior import GalSBIScenePrior
from sbsi.galsbi_summary import SummaryBatch, SummaryLikelihood, compress_scenes
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
from sbsi.v39_inference import V39CatalogueCache
from galsbi_bright_math import bright_cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observations', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=2026100301)
    parser.add_argument('--magnitude-limits', type=float, nargs='+',
                        help='optional measured cuts; observations is the requested count per cut')
    args = parser.parse_args()
    if args.observations < 1 or (args.output/'result.json').exists():
        raise ValueError('positive population and fresh output required')
    if args.magnitude_limits and (len(set(args.magnitude_limits)) != len(args.magnitude_limits)
            or not np.isfinite(args.magnitude_limits).all() or max(args.magnitude_limits) > 25.8):
        raise ValueError('distinct finite magnitude limits within the base measured selection required')
    args.output.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.reference/'protocol.json').read_text())
    validation = json.loads((args.reference/'result.json').read_text())
    if not validation['implementation_checks_pass']:
        raise ValueError('original summary/full-likelihood validation failed')
    protocol = dict(old, reference_kind='galsbi_population_v1', seed=args.seed,
        requested_observations=args.observations, generation_chunk=8192,
        selection='first N measured-selected observations in independent generation order; no strata or truth cuts',
        parent_reference=str(args.reference.resolve()),
        parent_protocol_sha256=digest(args.reference/'protocol.json'),
        parent_result_sha256=digest(args.reference/'result.json'),
        preparation_sha256=digest(__file__))
    if args.magnitude_limits:
        protocol.update(magnitude_limits=args.magnitude_limits,
            cohort_count=args.observations,
            cohort_helper_sha256=digest(Path(__file__).with_name('galsbi_bright_math.py')),
            selection='first N per measured magnitude cut, radius>3 pixels, in generation order; no truth cuts')
    save(args.output/'protocol.json', protocol)
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    model = load_disk_response(V39_LIKE, device='cpu', nthread=1)
    likelihood = SummaryLikelihood(flow, detector)
    geometry = geometry_replay()
    if not np.isfinite(geometry) or geometry > 1e-12:
        raise ValueError('renderer geometry replay failed')
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)

    def distance(z):
        return cosmo.background.dist_ang_a(a=1/(1+z))

    frames, receipts, replay_errors = [], [], []
    total = 0
    for chunk in range(1000):
        seed = args.seed+10*chunk
        zero, response, sampling = prepared(prior, distance, old['conditions'], model, n=8192, seed=seed)
        frame, retention, ids = mock_panel(zero, response, flow, detector, seed=seed+1,
            shear=old['shear'], retain_population=True)
        if len(frame) and not replay_errors:
            # Replay the existing full-pair implementation, including order and padding.
            obs = frame.iloc[[0]]
            summary = compress_scenes(zero, response, obs.iloc[0, 2:4].to_numpy())
            kernel = likelihood.log_kernel(summary, obs, *old['shear'])
            cache = V39CatalogueCache(zero, np.ones(len(zero)), {}, detector=detector)
            full = DiskCatalogueLikelihood(flow, cache, disk_response=response)
            density = full.conditional_log_likelihood(obs.iloc[np.zeros(len(zero), dtype=int)].reset_index(drop=True),
                *old['shear'], atom_indices=np.arange(len(zero)), object_chunk=4096)
            probability = detector.predict_proba(cache.get(*old['shear']).flow, batch_size=4096)
            partial = likelihood.log_kernel(SummaryBatch(summary.values[:128], summary.auxiliary[:128]), obs, *old['shear'])
            reverse = likelihood.log_kernel(SummaryBatch(summary.values[::-1], summary.auxiliary[::-1]), obs, *old['shear'])[::-1]
            replay_errors = [float(np.max(np.abs(density+np.log(probability.astype(float))-kernel))),
                float(np.max(np.abs(partial-kernel[:128]))), float(np.max(np.abs(reverse-kernel)))]
            if not np.isfinite(replay_errors).all() or max(replay_errors) > 1e-6:
                save(args.output/'replay_failure.json', dict(errors=replay_errors))
                raise ValueError('population full/batch/order replay failed')
        take = len(frame) if args.magnitude_limits else min(len(frame), args.observations-total)
        frames.append(frame.iloc[:take])
        receipts.append(dict(chunk=chunk, seed=seed, selected=take, sampling=sampling, retention=retention))
        total += take
        save(args.output/'progress.json', dict(observations=total, requested=args.observations, chunks=receipts))
        print('POPULATION_PROGRESS', total, args.observations, flush=True)
        if args.magnitude_limits:
            all_observed = pd.concat(frames, ignore_index=True)
            magnitude = 30-2.5*np.log10(all_observed.iloc[:, 3].to_numpy())
            union, cohorts, complete = bright_cohorts(magnitude, args.magnitude_limits, args.observations)
            if complete:
                break
        elif total == args.observations:
            break
    if (args.magnitude_limits and not complete) or (not args.magnitude_limits and total != args.observations):
        raise ValueError('population generation limit exceeded')
    observed = pd.concat(frames, ignore_index=True)
    if args.magnitude_limits:
        observed = observed.iloc[union].reset_index(drop=True)
        save(args.output/'cohorts.json', dict(cohorts=cohorts,
            generation_selected_indices=union.tolist(), measured_magnitude=magnitude[union].tolist(),
            limitation='Overlapping cut samples are individually representative, not independent cohorts.'))
        total = len(observed)
    if not np.isfinite(observed.to_numpy()).all():
        raise ValueError('nonfinite generated measurements')
    observed.to_parquet(args.output/'observations.parquet', index=False)
    save(args.output/'result.json', dict(status='prepared_population', preparation_checks_pass=True,
        observation_count=total, geometry_replay_error=geometry, full_batch_order_replay_errors=replay_errors,
        observations_sha256=digest(args.output/'observations.parquet'),
        cohorts_sha256=digest(args.output/'cohorts.json') if args.magnitude_limits else None,
        protocol_sha256=digest(args.output/'protocol.json'),
        limitation='model-generated population; no selection-normalized shear inference'))
    print('POPULATION_READY', total, flush=True)


if __name__ == '__main__':
    main()
