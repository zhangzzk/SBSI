"""Controlled component/replay benchmark of the continuous-prior MH workload.

Tiles four saved posterior scenes across batch sizes; this is a throughput
probe, not a new observation sample or a convergence experiment. Each proposal
is scored by the existing full evaluator. An invariant-cache candidate skips
the likelihood only for moves in z/SED, and skips the target prior only for
environment moves. Common proposals and uniforms compare MH decisions.
"""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import resource
import time

import numpy as np
import pandas as pd
import torch

from check_galsbi_continuous_prior import build_prior, digest, save
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import SummaryBatch, SummaryLikelihood, compress_scenes, scene_model_inputs
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
from sbsi.pseudo_marginal import metropolis_accept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--observations', type=int, nargs='+', default=[4, 32, 128, 512, 1024])
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 2 or args.observations[0] != 4 or any(n < 4 or n % 4 for n in args.observations):
        raise ValueError('at least two repeats and batches divisible by four required')
    if (args.output/'result.json').exists():
        raise ValueError('fresh output required')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.reference/'protocol.json').read_text())
    observed_panel = pd.read_parquet(args.reference/'observations.parquet')
    with np.load(args.scenes) as raw:
        base = GalSBISceneBatch(
            {k[7:]: raw[k] for k in raw.files if k.startswith('target_')},
            {k[10:]: raw[k] for k in raw.files if k.startswith('neighbour_')},
            raw['indptr'], raw['offset_arcsec'])
    if len(base.counts) != 16 or len(observed_panel) != 4:
        raise ValueError('four-observation, four-chain panel required')
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device=args.device)
    detector = load_detection_classifier(V39_LIKE, device=args.device)
    model = load_disk_response(V39_LIKE, device='cpu', nthread=1)
    alternatives = {}
    if args.device == 'cuda':
        threads = min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
        alternatives[f'cpu_{threads}_threads'] = load_disk_response(V39_LIKE, device='cpu', nthread=threads)
        alternatives['cuda'] = load_disk_response(V39_LIKE, device='cuda', nthread=1)
    likelihood = SummaryLikelihood(flow, detector)
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)
    def angular(z):
        return cosmo.background.dist_ang_a(a=1/(1+z))
    coords = GalSBITargetCoordinates(galaxy, angular)
    blocks = [('morphology', .3, [1, 6, 7]), ('shape', .3, [8, 9]),
              ('structure', .1, [0, 2, 3, 4, 5]), ('environment_partial', .135, []),
              ('environment_full', .015, []), ('sed', .15, [])]
    sources = [Path(__file__), Path('sbsi/galsbi_summary.py'), Path('sbsi/galsbi_target_coordinates.py'),
               Path('sbsi/galsbi_scene_prior.py'), Path('sbsi/galsbi_prior.py')]
    report = dict(device=args.device, seed=2026100131, chains=4, repeats=args.repeats,
                  source_sha256={str(p): digest(p) for p in sources},
                  scene_sha256=digest(args.scenes), reference=str(args.reference),
                  density_batch_size=4096, no_convergence_or_shear_claim=True,
                  workload='tiled saved states, independent local proposals; no full-history storage',
                  replay_tolerances=dict(summary=1e-10, log_kernel=1e-6), batches=[])
    save(args.output/'protocol.json', report)

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def timed(parts, name, function):
        sync()
        start = time.perf_counter()
        value = function()
        sync()
        parts[name] = time.perf_counter()-start
        return value

    profile = cProfile.Profile()
    for nobs in args.observations:
        rows = np.arange(4*nobs) % len(base.counts)
        scene = base.take(rows)
        observed = observed_panel.iloc[rows//4].reset_index(drop=True)
        aux = observed.iloc[:, 2:4].to_numpy()
        x = coords.encode(scene.targets)
        oldprior, data, _ = coords.log_mixture_prior(x)
        scene = GalSBISceneBatch(data, scene.neighbours, scene.indptr, scene.offset_arcsec)

        def evaluate(candidate, valid, parts):
            logk = np.full(len(rows), -np.inf)
            summaries = np.zeros((len(rows), 9))
            if valid.any():
                selected = timed(parts, 'scene_take', lambda: candidate.take(np.flatnonzero(valid)))
                zero, ptr, features = timed(parts, 'scene_inputs', lambda: scene_model_inputs(
                    selected, prior, angular, protocol['conditions'], response_radius_arcsec=10.,
                    response_neighbours=20))
                summary = timed(parts, 'response', lambda: compress_scenes(
                    zero, CatalogueDiskResponse(ptr, features, model), aux[valid]))
                logk[valid] = timed(parts, 'likelihood', lambda: likelihood.log_kernel(
                    summary, observed.loc[valid], *protocol['shear']))
                summaries[valid] = summary.values
            return logk, summaries

        oldlike, oldsummary = evaluate(scene, np.ones(len(rows), bool), {})
        if not np.isfinite(oldprior+oldlike).all():
            raise ValueError('finite retained starting states required')
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(report['seed'])
        batch = dict(observations=nobs, states=len(rows), blocks=[])
        for name, probability, columns in blocks:
            timing, audit = [], dict(summary_error=0., log_kernel_error=0., acceptance_disagreements=0,
                                     batch_replay_error=0., proposals=0, valid=0)
            for repeat in range(args.repeats+1):
                # First repetition warms each path; subsequent timings are reported.
                if repeat == 0:
                    profile.enable()
                parts = {}
                proposed, correction = x.copy(), np.zeros(len(rows))
                if name == 'sed':
                    proposed, correction = timed(parts, 'proposal', lambda: coords.propose_sed(x, rng))
                elif columns:
                    proposed[:, columns] += rng.normal(size=(len(rows), len(columns)))*.025
                    proposed[:, 9] %= 2*np.pi
                newprior, newdata, _ = timed(parts, 'target_prior', lambda: coords.log_mixture_prior(proposed))
                candidate = GalSBISceneBatch(newdata, scene.neighbours, scene.indptr, scene.offset_arcsec)
                environment = name.startswith('environment')
                if environment:
                    candidate = timed(parts, 'environment_refresh', lambda: prior.refresh(
                        candidate, 1. if name == 'environment_full' else .1,
                        int(rng.integers(0, 2**63))))
                valid = np.isfinite(newprior)
                newlike, newsummary = evaluate(candidate, valid, parts)
                uniforms = rng.uniform(np.finfo(float).tiny, 1., len(rows))
                accepted = metropolis_accept(oldprior+oldlike, newprior+newlike+correction, uniforms)
                timed(parts, 'retained_scene', lambda: scene.retained(candidate, accepted))
                invariant = name in ('structure', 'sed')
                candidate_like = oldlike if invariant else newlike
                candidate_prior = oldprior if environment else newprior
                cached_accepted = metropolis_accept(oldprior+oldlike, candidate_prior+candidate_like+correction,
                                                    uniforms)
                audit['acceptance_disagreements'] += int(np.sum(accepted != cached_accepted))
                audit['proposals'] += len(rows)
                audit['valid'] += int(valid.sum())
                if invariant and valid.any():
                    audit['summary_error'] = max(audit['summary_error'],
                        float(np.max(np.abs(newsummary[valid]-oldsummary[valid]))))
                    audit['log_kernel_error'] = max(audit['log_kernel_error'],
                        float(np.max(np.abs(newlike[valid]-oldlike[valid]))))
                if environment and not np.array_equal(newprior, oldprior):
                    raise ValueError('environment update changed target prior')
                if repeat == 0 and valid.any():
                    check = np.flatnonzero(valid)[:16]
                    replay = likelihood.log_kernel(SummaryBatch(newsummary[check], aux[check]),
                        observed.iloc[check], *protocol['shear'])
                    audit['batch_replay_error'] = float(np.max(np.abs(replay-newlike[check])))
                if repeat == 0:
                    profile.disable()
                if repeat:
                    timing.append(parts)
            stage = {key: float(np.median([t.get(key, 0.) for t in timing]))
                     for key in set().union(*(t.keys() for t in timing))}
            baseline = float(np.median([sum(t.values()) for t in timing]))
            # Component subtraction is an estimate; it is not an end-to-end timing.
            saved = ['scene_take', 'scene_inputs', 'response', 'likelihood'] if invariant else []
            if environment:
                saved += ['target_prior']
            cached_estimate = float(np.median([sum(v for k, v in t.items() if k not in saved) for t in timing]))
            batch['blocks'].append(dict(name=name, probability=probability, stages_seconds=stage,
                baseline_seconds=baseline, cached_component_estimate_seconds=cached_estimate, replay=audit))
            print('BLOCK', nobs, name, baseline, audit, flush=True)
        batch['weighted_baseline_seconds'] = sum(b['probability']*b['baseline_seconds'] for b in batch['blocks'])
        batch['weighted_cached_component_estimate_seconds'] = sum(
            b['probability']*b['cached_component_estimate_seconds'] for b in batch['blocks'])
        # Same artifact, different execution device/thread count. Randomized
        # physical properties avoid benchmarking identical tiled tree paths only.
        batch['response_alternatives'] = {}
        if alternatives:
            proposed = x.copy()
            proposed[:, [1, 6, 7]] += rng.normal(size=(len(rows), 3))*.025
            newprior, newdata, _ = coords.log_mixture_prior(proposed)
            valid = np.isfinite(newprior)
            candidate = GalSBISceneBatch(newdata, scene.neighbours, scene.indptr, scene.offset_arcsec)
            zero, ptr, features = scene_model_inputs(candidate.take(np.flatnonzero(valid)), prior, angular,
                protocol['conditions'], response_radius_arcsec=10., response_neighbours=20)
            reference_summary = compress_scenes(zero, CatalogueDiskResponse(ptr, features, model), aux[valid])
            reference_kernel = likelihood.log_kernel(reference_summary, observed.loc[valid], *protocol['shear'])
            for label, alternative in {'cpu_1_thread': model, **alternatives}.items():
                times = []
                for repeat in range(args.repeats+1):
                    parts = {}
                    candidate_summary = timed(parts, 'response', lambda: compress_scenes(
                        zero, CatalogueDiskResponse(ptr, features, alternative), aux[valid]))
                    if repeat:
                        times.append(parts['response'])
                kernel = likelihood.log_kernel(candidate_summary, observed.loc[valid], *protocol['shear'])
                error = float(np.max(np.abs(kernel-reference_kernel)))
                batch['response_alternatives'][label] = dict(seconds=float(np.median(times)),
                    max_coefficient_error=float(np.max(np.abs(candidate_summary.values[:, 8]
                                                             -reference_summary.values[:, 8]))),
                    max_log_kernel_error=error, replay_pass=error <= 1e-6)
                print('RESPONSE', nobs, label, batch['response_alternatives'][label], flush=True)
        batch['peak_host_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        batch['peak_cuda_allocated_bytes'] = (torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
        batch['replay_pass'] = all(b['replay']['summary_error'] <= 1e-10
            and b['replay']['log_kernel_error'] <= 1e-6 and b['replay']['acceptance_disagreements'] == 0
            and b['replay']['batch_replay_error'] <= 1e-6
            for b in batch['blocks'])
        report['batches'].append(batch)
        save(args.output/'progress.json', report)
        print('BATCH', nobs, batch['weighted_baseline_seconds'], batch['replay_pass'], flush=True)
    profile.dump_stats(args.output/'profile.pstats')
    with (args.output/'profile.txt').open('w') as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(70)
    report['all_replay_checks_pass'] = all(b['replay_pass'] for b in report['batches'])
    save(args.output/'result.json', report)


if __name__ == '__main__':
    main()
