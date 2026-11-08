"""Bright-cohort reference preparation and independent selection integrals (the MH stage is retired)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from check_galsbi_continuous_prior import build_prior, digest, save
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.disk_response_transport import disk_response_transport
from sbsi.galsbi_scene_prior import GalSBIScenePrior
from sbsi.galsbi_summary import scene_model_inputs
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'selection'], required=True)
    parser.add_argument('--replica', type=int, default=0)
    args = parser.parse_args()
    plan = json.loads((args.root/'plan.json').read_text())
    if args.stage == 'prepare':
        prepare(args.root, plan)
    else:
        selection(args.root, plan, args.replica)


if __name__ == '__main__':
    main()
