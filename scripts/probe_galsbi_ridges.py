"""Conditional density-path optimization; diagnostic only, never posterior samples."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_galsbi_continuous_prior import build_prior, save
from probe_galsbi_geometry import join_scenes
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import SummaryLikelihood, compress_scenes, scene_model_inputs
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    ids = [177, 80, 225, 209]
    output = args.audit/'ridges'
    output.mkdir(exist_ok=False)
    parts = []
    for r in range(2):
        with np.load(args.source/'runs'/f'replica{r}'/'final_scenes.npz') as z:
            scene = GalSBISceneBatch({k[7:]: z[k] for k in z.files if k.startswith('target_')},
                {k[10:]: z[k] for k in z.files if k.startswith('neighbour_')}, z['indptr'], z['offset_arcsec'])
            parts.append(scene.take((np.array(ids)[:, None]*4+np.arange(4)).ravel()))
    scenes = join_scenes(*parts)
    observed = pd.read_parquet(args.source/'reference/observations.parquet').iloc[ids].reset_index(drop=True)
    protocol = json.loads((args.source/'reference/protocol.json').read_text())
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    response_model = load_disk_response(V39_LIKE, device='cpu', nthread=1)
    likelihood = SummaryLikelihood(flow, detector)
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)
    def distance(z):
        return cosmo.background.dist_ang_a(a=1/(1+z))
    coords = GalSBITargetCoordinates(galaxy, distance)
    endpoints = coords.encode(scenes.targets)
    owner = np.tile(np.repeat(np.arange(4), 4), 2)
    environment = scenes.take(np.arange(4)*4)  # fixed r0/chain0 environment per observation

    def density(x, groups):
        lp, data, _ = coords.log_mixture_prior(x)
        valid = np.flatnonzero(np.isfinite(lp))
        lk = np.full(len(x), -np.inf)
        if len(valid):
            env = environment.take(groups[valid])
            native = GalSBISceneBatch({k: v[valid] for k, v in data.items()}, env.neighbours, env.indptr, env.offset_arcsec)
            zero, ptr, pairs = scene_model_inputs(native, prior, distance, protocol['conditions'],
                response_radius_arcsec=10., response_neighbours=20)
            obs = observed.iloc[groups[valid]].reset_index(drop=True)
            summary = compress_scenes(zero, CatalogueDiskResponse(ptr, pairs, response_model), obs.iloc[:, 2:4].to_numpy())
            lk[valid] = likelihood.log_kernel(summary, obs, .02, 0.)
        return lp, lk

    rng = np.random.default_rng(2026100602)
    knots, population_size, iterations = 17, 16, 200
    movable = [1, 6, 8, 9]
    population, n_grids, lower, upper = [], [], [], []
    for i in range(4):
        candidates = endpoints[owner == i].copy()
        angle = candidates[0, 9]
        candidates[:, 9] = angle+(candidates[:, 9]-angle+np.pi) % (2*np.pi)-np.pi
        ngrid = np.linspace(candidates[:, 7].min()-.1, candidates[:, 7].max()+.1, knots)
        n_grids.append(ngrid)
        lo = candidates[:, movable].min(0)-np.array([2., 2., 4., 1.])
        hi = candidates[:, movable].max(0)+np.array([2., 2., 4., 1.])
        for n in ngrid:
            group = np.tile(candidates[0], (population_size, 1))
            group[:, 7] = n
            group[:, movable] = candidates[np.arange(population_size) % 8][:, movable]
            group[8:, movable] += rng.normal(size=(8, 4))*np.array([.2, .15, .3, .2])
            group[:, movable] = np.clip(group[:, movable], lo, hi)
            population.append(group)
            lower.append(lo)
            upper.append(hi)
    population = np.array(population)
    lower, upper = np.array(lower)[:, None], np.array(upper)[:, None]
    group_owner = np.repeat(np.arange(4), knots)
    groups = np.repeat(group_owner, population_size)
    lp, lk = density(population.reshape(-1, 10), groups)
    values = (lp+lk).reshape(-1, population_size)
    starting = values.max(1).copy()
    group_index = np.arange(len(population))[:, None]
    individual = np.arange(population_size)[None, :]
    for iteration in range(iterations):
        picks = rng.integers(population_size, size=values.shape+(3,))
        while True:
            bad = ((picks == individual[:, :, None]).any(2) | (picks[:, :, 0] == picks[:, :, 1])
                   | (picks[:, :, 0] == picks[:, :, 2]) | (picks[:, :, 1] == picks[:, :, 2]))
            if not bad.any():
                break
            picks[bad] = rng.integers(population_size, size=(bad.sum(), 3))
        mutant = (population[group_index, picks[:, :, 0]]
                  + .7*(population[group_index, picks[:, :, 1]]-population[group_index, picks[:, :, 2]]))
        chosen = rng.random(values.shape+(4,)) < .8
        jrand = rng.integers(4, size=values.shape)
        np.put_along_axis(chosen, jrand[:, :, None], True, axis=2)
        proposed = population.copy()
        proposed[:, :, movable] = np.clip(np.where(chosen, mutant[:, :, movable], population[:, :, movable]), lower, upper)
        np_lp, np_lk = density(proposed.reshape(-1, 10), groups)
        new = (np_lp+np_lk).reshape(values.shape)
        improve = new > values
        population[improve], values[improve] = proposed[improve], new[improve]
        if (iteration+1) % 25 == 0:
            print('RIDGE_OPTIMIZATION', iteration+1, 'median improvement',float(np.median(values.max(1)-starting)), flush=True)
    best = population[np.arange(len(population)), values.argmax(1)].reshape(4, knots, 10)
    curves = []
    for i, obsid in enumerate(ids):
        n = np.linspace(n_grids[i][0], n_grids[i][-1], 129)
        curved = np.tile(best[i, 0], (len(n), 1))
        curved[:, 7] = n
        for col in movable:
            curved[:, col] = np.interp(n, n_grids[i], best[i, :, col])
        t = np.linspace(0, 1, len(n))
        straight = best[i, 0]+t[:, None]*(best[i, -1]-best[i, 0])
        lp_c, lk_c = density(curved, np.full(len(n), i))
        lp_s, lk_s = density(straight, np.full(len(n), i))
        total_c, total_s = lp_c+lk_c, lp_s+lk_s
        if not np.isfinite(total_c).all() or not np.isfinite(total_s).all():
            raise ValueError('optimized path crosses native support; inspect before interpreting barriers')
        curves.append(dict(observation=obsid, sersic_logit=n.tolist(), optimized_knots=best[i].tolist(),
            curved_prior=lp_c.tolist(), curved_kernel=lk_c.tolist(), straight_prior=lp_s.tolist(), straight_kernel=lk_s.tolist(),
            curved_barrier=float(min(total_c[0], total_c[-1])-total_c.min()),
            straight_barrier=float(min(total_s[0], total_s[-1])-total_s.min()),
            maximum_curved_minus_straight=float((total_c-total_s).max())))
    save(output/'result.json', dict(status='completed_conditional_optimization', seed=2026100602,
        observations=ids, iterations=iterations, population_size=population_size, curves=curves,
        limitation='Conditional target-density paths with fixed redshift/SED/environment; optimization is not posterior sampling or a proof of global connectivity. Search bounds surround explored endpoints and are not prior cuts.'))
    print('RIDGE_PROBE_COMPLETE', [(c['observation'],c['straight_barrier'],c['curved_barrier']) for c in curves], flush=True)


if __name__ == '__main__':
    main()
