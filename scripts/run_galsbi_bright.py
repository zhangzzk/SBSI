"""Scheduled bright-cohort MH, independent selection integrals and shear profiles."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from check_galsbi_continuous_prior import build_prior, digest, save
from report_galsbi_population import checked_run
from run_galsbi_throughput import run
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.disk_response_transport import disk_response_transport
from sbsi.galsbi_scene_prior import GalSBIScenePrior
from sbsi.galsbi_summary import SummaryBatch, SummaryLikelihood, scene_model_inputs
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
from sbsi.shear_map import apply_shear_to_ellipticity


def selected_reference(root, reference=None):
    """The study's cohort reference, or an explicit population reference (no cohorts file)."""
    cohort = reference is None
    reference = root/'reference' if cohort else Path(reference)
    receipt = json.loads((reference/'result.json').read_text())
    if (not receipt['preparation_checks_pass']
            or receipt['observations_sha256'] != digest(reference/'observations.parquet')
            or (cohort and receipt['cohorts_sha256'] != digest(reference/'cohorts.json'))):
        raise ValueError('bright reference failed validation')
    return reference, receipt


def prepare(root, plan):
    command = [sys.executable, '-u', '-c',
        'import runpy,sys;sys.path[:0]=[".","scripts"];runpy.run_path("scripts/prepare_galsbi_population.py",run_name="__main__")',
        '--reference', plan['parent_reference'], '--output', str(root/'reference'),
        '--observations', str(plan['cohort_count']), '--seed', str(plan['mock_seed']),
        '--magnitude-limits', *map(str, plan['magnitude_limits'])]
    subprocess.run(command, check=True)
    selected_reference(root)


def selection(root, plan, replica, *, smoke=False, reference=None):
    """Independent scenes; common random numbers across shear nodes and nested draws."""
    reference, _ = selected_reference(root, reference)
    protocol = json.loads((reference/'protocol.json').read_text())
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)
    def distance(z):
        return cosmo.background.dist_ang_a(a=1/(1+z))

    # Shear nodes are (g1, g2) pairs from 'shear_nodes', or a g1 grid at g2 = 0.
    grid = np.array([.018, .02, .022]) if smoke else np.asarray(plan.get('shear_nodes', plan.get('g1_grid')))
    nodes = grid if grid.ndim == 2 else np.column_stack((grid, np.zeros(len(grid))))
    chunks = 1 if smoke else plan['selection_chunks']
    depth = plan['selection_flow_draws']
    # Optional cut on measured |e|: the measured shape is the flow draw moved by radial_tanh(b g), with b
    # predicted from that draw's measured radius/flux and the scene's neighbour pairs, as in the likelihood.
    max_shape = plan.get('max_measured_ellipticity')
    if max_shape is not None:
        response_model = load_disk_response(V39_LIKE, device='cpu', nthread=len(os.sched_getaffinity(0)))
    values = np.zeros((chunks, len(grid), len(plan['magnitude_limits']), 2))
    output = root/('smoke_selection' if smoke else f'selection{replica}')
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    for chunk in range(chunks):
        seed = plan['selection_seed_base']+replica*10000+chunk*10+(100000 if smoke else 0)
        scenes, _ = prior.sample(plan['selection_chunk_size'], seed)
        zero, ptr, features = scene_model_inputs(scenes, prior, distance, protocol['conditions'],
            response_radius_arcsec=10., response_neighbours=20)
        if max_shape is not None:
            response = CatalogueDiskResponse(ptr, features, response_model)
            atoms = np.repeat(np.arange(len(zero)), depth).reshape(len(zero), depth)
        for j, g in enumerate(nodes):
            context = zero.copy()
            context['e1_input_p'], context['e2_input_p'] = apply_shear_to_ellipticity(
                zero.e1_input_p.to_numpy(), zero.e2_input_p.to_numpy(), *g)
            q = detector.predict_proba(context, batch_size=4096).astype(float)
            torch.manual_seed(seed+1)
            samples = flow.sample(context, n_samples=depth, batch_size=4096, qmc=True)
            if (not np.isfinite(samples).all() or not np.isfinite(q).all()
                    or np.any((q < 0) | (q > 1))):
                raise ValueError('invalid forward selection draw')
            shaped = True
            if max_shape is not None:
                # b only where the radius/flux are positive; other draws fail the radius cut below anyway
                b, kept = np.zeros(atoms.shape), (samples[:, :, 2] > 3.) & (samples[:, :, 3] > 0)
                b[kept], _ = response.predict(atoms[kept], samples[:, :, 2][kept], samples[:, :, 3][kept])
                measured, _ = disk_response_transport(torch.as_tensor(samples[:, :, :2]),
                                                      torch.as_tensor(b[..., None]*np.asarray(g, float)))
                shaped = measured.square().sum(-1).sqrt().numpy() < max_shape
            for c, limit in enumerate(plan['magnitude_limits']):
                passed = (shaped & (samples[:, :, 2] > 3.) & (samples[:, :, 2] < plan.get('max_radius_pixels', np.inf))
                          & (samples[:, :, 3] > 10**(.4*(30-limit))))
                for d, n in enumerate((depth//2, depth)):
                    values[chunk, j, c, d] = np.mean(q*passed[:, :n].mean(1))
        np.savez_compressed(output/'chunks.npz', values=values[:chunk+1], grid=grid)
        save(output/'progress.json', dict(chunks=chunk+1, requested=chunks,
            elapsed_seconds=time.monotonic()-started))
        print('SELECTION_PROGRESS', replica, chunk+1, chunks, flush=True)
    if np.any(values.mean(0) <= 0):
        raise ValueError('zero estimated selection; increase independent forward population')
    save(output/'result.json', dict(status='completed_selection', replica=replica,
        grid=grid.tolist(), chunks=chunks, scenes_per_chunk=plan['selection_chunk_size'],
        flow_draws=[depth//2, depth], seed_base=plan['selection_seed_base']+replica*10000,
        reference_protocol_sha256=digest(reference/'protocol.json'),
        plan_sha256=digest(root/'plan.json'), data_sha256=digest(output/'chunks.npz'),
        process_seconds=time.monotonic()-started,
        selection='C8 probability times measured radius>3 pixels (and < max_radius_pixels if the plan sets it) '
                  'and flux threshold; measured |e| < max_measured_ellipticity (disk response per draw) if set',
        uncertainty='Independent scene chunks and replicas; CRN over shear nodes; nested conditional draw ladder'))


def profiles(root, plan, replica, *, smoke=False, reference=None, runroot=None):
    reference, receipt = selected_reference(root, reference)
    runroot = runroot or root/('smoke' if smoke else f'runs/replica{replica}')
    protocol, _ = checked_run(runroot)
    observed = pd.read_parquet(reference/'observations.parquet').iloc[protocol['observations']].reset_index(drop=True)
    stride = 1 if smoke else plan['profile_stride']
    grid = np.array([.018, .02, .022]) if smoke else np.asarray(plan['g1_grid'])
    with np.load(runroot/'chain.npz') as archive:
        summary = archive['summaries'][protocol['warmup']::stride].copy()
        kernel = archive['log_kernel'][protocol['warmup']::stride].copy()
    if protocol['environments'] != 1 or protocol.get('ensemble_morphology'):
        raise ValueError('independent persistent-scene chains required')
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    likelihood = SummaryLikelihood(flow, detector)
    ratios = np.empty((len(observed), len(grid), len(summary), 4))
    centre = int(np.flatnonzero(np.isclose(grid, protocol['fixed_shear'][0], atol=1e-12, rtol=0))[0])
    replay = 0.
    started = time.monotonic()
    for i in range(len(observed)):
        rows = slice(4*i, 4*i+4)
        flat = summary[:, rows, 0].reshape(-1, 9)
        obs = observed.iloc[[i]]
        batch = SummaryBatch(flat, np.broadcast_to(obs.iloc[0, 2:4].to_numpy(), (len(flat), 2)))
        for j, g in enumerate(grid):
            shifted = likelihood.log_kernel(batch, obs, g, 0.).reshape(len(summary), 4)
            ratios[i, j] = shifted-kernel[:, rows, 0]
        replay = max(replay, float(np.max(np.abs(ratios[i, centre]))))
        if not np.isfinite(replay) or replay > 1e-6:
            raise ValueError('saved posterior summaries failed fixed-shear replay')
        ratios[i, centre] = 0.
        if np.isnan(ratios[i]).any() or np.isposinf(ratios[i]).any():
            raise ValueError('invalid profile ratios')
        save(runroot/'shear_progress.json', dict(observations=i+1, requested=len(observed),
            elapsed_seconds=time.monotonic()-started))
        if (i+1) % 16 == 0:
            print('PROFILE_PROGRESS', replica, i+1, len(observed), flush=True)
    np.savez_compressed(runroot/'shear_profiles.npz', log_ratios=ratios, grid=grid,
        steps=np.arange(protocol['warmup'], protocol['steps'], stride)+1)
    save(runroot/'shear_profiles.json', dict(status='completed_profiles', max_replay_error=replay,
        protocol_sha256=digest(runroot/'protocol.json'), data_sha256=digest(runroot/'shear_profiles.npz'),
        observation_sha256=receipt['observations_sha256'], plan_sha256=digest(root/'plan.json'),
        stride=stride, process_seconds=time.monotonic()-started,
        interpretation='Conditional g1 likelihood ratios at g2=0; overlap and convergence are checked by the report.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'sample'], required=True)
    parser.add_argument('--replica', type=int, default=0)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root/'plan.json').read_text())
    if args.stage == 'prepare':
        prepare(root, plan)
        return
    reference, receipt = selected_reference(root)
    smoke = args.stage == 'smoke'
    if not smoke:
        if not json.loads((root/'smoke_complete.json').read_text())['passed']:
            raise ValueError('smoke gate failed')
    indices = list(range(min(4, receipt['observation_count']) if smoke else receipt['observation_count']))
    replica = args.replica
    offset = 100 if smoke else replica
    run(root, 'smoke' if smoke else f'runs/replica{replica}', reference, indices,
        plan['mh_seed_base']+offset, plan['initial_bank_seed_base']+offset,
        steps=256 if smoke else plan['steps'], warmup=128 if smoke else plan['warmup'])
    profiles(root, plan, replica, smoke=smoke)
    selection(root, plan, replica, smoke=smoke)
    if smoke:
        save(root/'smoke_complete.json', dict(passed=True,
            limitation='Numerical execution only, no mixing or recovery acceptance.'))


if __name__ == '__main__':
    main()
