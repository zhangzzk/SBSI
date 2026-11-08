"""Bounded exact-target replay, cross-chain paths and proposal mobility probes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_galsbi_continuous_prior import build_prior, digest, save
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import SummaryBatch, SummaryLikelihood, compress_scenes, scene_model_inputs
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response


def join_scenes(first, second):
    return GalSBISceneBatch({k: np.concatenate((v, second.targets[k])) for k, v in first.targets.items()},
        {k: np.concatenate((v, second.neighbours[k])) for k, v in first.neighbours.items()},
        np.r_[first.indptr, second.indptr[1:]+first.indptr[-1]],
        np.concatenate((first.offset_arcsec, second.offset_arcsec)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    out = args.audit/'geometry'
    out.mkdir(exist_ok=False)
    panel = json.loads((args.audit/'chains/panel.json').read_text())
    ids = panel['observations']
    observation = pd.read_parquet(args.source/'reference/observations.parquet').iloc[ids].reset_index(drop=True)
    conditions = json.loads((args.source/'reference/protocol.json').read_text())['conditions']
    scene_parts, x_parts, summary_parts, lp_parts, lk_parts, factor_parts = [], [], [], [], [], []
    for r in range(2):
        root = args.source/'runs'/f'replica{r}'
        rows = (np.array(ids)[:, None]*4+np.arange(4)).ravel()
        with np.load(root/'final_scenes.npz') as a:
            scene = GalSBISceneBatch({k[7:]: a[k] for k in a.files if k.startswith('target_')},
                {k[10:]: a[k] for k in a.files if k.startswith('neighbour_')}, a['indptr'], a['offset_arcsec'])
            scene_parts.append(scene.take(rows))
        with np.load(root/'chain.npz') as a:
            x_parts.append(a['target_coordinates'][-1, rows].copy())
            summary_parts.append(a['summaries'][-1, rows, 0].copy())
            lp_parts.append(a['log_prior'][-1, rows].copy())
            lk_parts.append(a['log_kernel'][-1, rows, 0].copy())
        result = json.loads((root/'result.json').read_text())
        factor_parts.append(np.asarray(result['frozen_morphology_cholesky'])[rows]
                            *np.asarray(result['frozen_scale_multipliers'])[rows, 0, None, None])
    scene = join_scenes(*scene_parts)
    x, stored_s, lp, lk, factor = map(np.concatenate, (x_parts, summary_parts, lp_parts, lk_parts, factor_parts))
    owner = np.tile(np.repeat(np.arange(len(ids)), 4), 2)
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    model = load_disk_response(V39_LIKE, device='cpu', nthread=1)
    likelihood = SummaryLikelihood(flow, detector)
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)
    def distance(z):
        return cosmo.background.dist_ang_a(a=1/(1+z))
    coords = GalSBITargetCoordinates(galaxy, distance)

    def evaluate(values, environment_rows):
        prior_value, data, _ = coords.log_mixture_prior(values)
        env = scene.take(environment_rows)
        combined = GalSBISceneBatch(data, env.neighbours, env.indptr, env.offset_arcsec)
        valid = np.flatnonzero(np.isfinite(prior_value))
        kernel = np.full(len(values), -np.inf)
        summaries = np.zeros((len(values), 9))
        if len(valid):
            obs = observation.iloc[owner[environment_rows[valid]]].reset_index(drop=True)
            zero, ptr, pairs = scene_model_inputs(combined.take(valid), prior, distance, conditions,
                response_radius_arcsec=10., response_neighbours=20)
            compressed = compress_scenes(zero, CatalogueDiskResponse(ptr, pairs, model), obs.iloc[:, 2:4].to_numpy())
            kernel[valid] = likelihood.log_kernel(compressed, obs, .02, 0.)
            summaries[valid] = compressed.values
        return prior_value, kernel, summaries

    fresh_p, fresh_k, fresh_s = evaluate(x, np.arange(len(x)))
    replay = dict(prior=float(np.max(np.abs(lp-fresh_p))), kernel=float(np.max(np.abs(lk-fresh_k))),
                  summary=float(np.max(np.abs(stored_s-fresh_s))))
    save(out/'replay.json', replay)
    if replay['prior'] > 1e-9 or replay['kernel'] > 1e-6 or replay['summary'] > 1e-10:
        raise ValueError('final retained state does not reproduce prior/likelihood/summary')
    print('FINAL_STATE_REPLAY', replay, flush=True)

    with np.load(args.audit/'chains/panel_draws.npz') as a:
        history = a['coordinates'].copy()
    columns = [1, 6, 7, 8, 9]
    covariances = []
    for i in range(len(ids)):
        v = history[:, i][:, :, columns].copy()
        angle = np.angle(np.mean(np.exp(1j*v[:, :, -1])))
        v[:, :, -1] = angle+(v[:, :, -1]-angle+np.pi) % (2*np.pi)-np.pi
        # Full explored covariance, explicitly not a converged posterior reference.
        c = np.cov(v.reshape(-1, 5).T)
        c += 1e-8*np.diag(np.maximum(np.diag(c), 1e-12))
        covariances.append(c)
    covariances = np.array(covariances)
    eigen = []
    for row in range(len(x)):
        c = covariances[owner[row], :3, :3]
        whiten = np.linalg.solve(np.linalg.cholesky(c), factor[row])
        eigen.append(np.linalg.eigvalsh(whiten@whiten.T).tolist())
    save(out/'proposal_geometry.json', dict(observations=ids,
        proposal_variance_over_explored_morphology_variance=eigen,
        limitation='Explored covariance can omit modes or include transient drift.'))

    # Crossed targets/environments distinguish conditional incompatibility from target-only barriers.
    paths = []
    for i, obsid in enumerate(ids):
        rows = np.flatnonzero(owner == i)
        targets = np.repeat(rows, len(rows))
        envs = np.tile(rows, len(rows))
        cp, ck, _ = evaluate(x[targets], envs)
        matrix = (cp+ck).reshape(len(rows), len(rows))
        own = lp[rows]+lk[rows]
        a, b = rows[int(np.argmin(x[rows, 7]))], rows[int(np.argmax(x[rows, 7]))]
        t = np.linspace(0, 1, 65)
        delta = x[b]-x[a]
        delta[9] = (delta[9]+np.pi) % (2*np.pi)-np.pi
        values = x[a]+t[:, None]*delta
        line_p, line_k, _ = evaluate(values, np.full(len(t), a))
        logline = line_p+line_k
        paths.append(dict(observation=obsid,
            crossed_target_prior_plus_kernel=np.where(np.isfinite(matrix), matrix, -1e300).tolist(),
            own_prior_plus_kernel=own.tolist(), interpolation_low_high_sersic_rows=[int(a), int(b)],
            interpolation_log_target=np.where(np.isfinite(logline), logline, -1e300).tolist(),
            interpolation_off_support=int((~np.isfinite(logline)).sum()),
            limitation='Environment columns have different prior densities/dimensions; compare targets within one column, not joint posterior heights across columns.'))
    save(out/'crossed_paths.json', dict(t=np.linspace(0, 1, 65).tolist(), paths=paths))

    # Frozen symmetric proposals from final states: exact MH probability and explored-width movement.
    rng = np.random.default_rng(2026100601)
    repeats = 128
    env = np.repeat(np.arange(len(x)), repeats)
    base = x[env]
    noise = rng.normal(size=(len(base), 5))
    arms = [('original_morphology', 3, None)]
    arms += [(f'explored_{d}d_scale_{scale}', d, scale) for d in (3, 5) for scale in (.1, .3, 1.)]
    probes = {}
    for label, dim, scale in arms:
        chosen = columns[:dim]
        if scale is None:
            delta = np.einsum('nij,nj->ni', factor[env], noise[:, :3])
        else:
            c = covariances[owner[env], :dim, :dim]
            delta = scale*2.38/np.sqrt(dim)*np.einsum('nij,nj->ni', np.linalg.cholesky(c), noise[:, :dim])
        proposal = base.copy()
        proposal[:, chosen] += delta
        proposal[:, 9] %= 2*np.pi
        pp, pk, _ = evaluate(proposal, env)
        alpha = np.exp(np.minimum(0., pp+pk-lp[env]-lk[env]))
        whitened = np.linalg.solve(np.linalg.cholesky(covariances[owner[env], :dim, :dim]), delta[:, :, None])[:, :, 0]
        jump = alpha*np.square(whitened).sum(1)
        probes[label] = dict(expected_acceptance_per_state=alpha.reshape(len(x), repeats).mean(1).tolist(),
            expected_standardized_squared_jump_per_state=jump.reshape(len(x), repeats).mean(1).tolist(),
            off_support_fraction=float((~np.isfinite(pp)).mean()))
        print('PROPOSAL_PROBE', label, float(alpha.mean()), float(jump.mean()), flush=True)
    save(out/'proposal_probes.json', dict(observations=ids, seed=2026100601, proposals_per_state=repeats,
        state_owner=owner.tolist(), probes=probes,
        limitation='One-step diagnostics at nonconverged endpoints; no new chain or acceptance claim.'))

    # Smaller shear stencils on identical retained states expose step-size and precision sensitivity.
    observed = observation.iloc[owner].reset_index(drop=True)
    batch = SummaryBatch(fresh_s, observed.iloc[:, 2:4].to_numpy())
    slopes = {}
    for h in (.002, .001, .0005, .00025):
        minus = likelihood.log_kernel(batch, observed, .02-h, 0.)-fresh_k
        plus = likelihood.log_kernel(batch, observed, .02+h, 0.)-fresh_k
        slopes[str(h)] = dict(score=((plus-minus)/(2*h)).tolist(), conditional_information=(-(plus+minus)/h**2).tolist())
    save(out/'shear_stencils.json', slopes)
    save(out/'result.json', dict(status='completed_diagnostic', replay=replay,
        panel_sha256=digest(args.audit/'chains/panel.json'), script_sha256=digest(__file__),
        observations=ids, limitation='Post-hoc conditional geometry diagnostics; scientific target unchanged.'))


if __name__ == '__main__':
    main()
