"""Persistent-scene MH per observed galaxy, with co-moving shear-likelihood readings.

Moves ten continuous target coordinates with the galaxy type summed exactly,
keeps one explicit neighbour scene per chain (refreshed from the prior), and
optionally tempers the likelihood with adjacent-level swaps. Starts are
likelihood-weighted draws from a fresh prior bank, never the generating truth.
After warm-up, every profile-stride step stores the posterior-level states and
the co-moving likelihood ratios at g0 +- 0.002 per component.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from collections import deque

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch

from check_galsbi_continuous_prior import build_prior, digest, save
from check_galsbi_summary_sampler import diagnostics
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.galsbi_prior import ExactMemo
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import (SummaryBatch, SummaryLikelihood, compress_scenes, response_pairs,
                                 scene_model_inputs)
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.measurement_model import load_measurement_model
from sbsi.mcmc_proposals import adjacent_swap_permutation, empirical_block_cholesky, gaussian_block_delta
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
from sbsi.pseudo_marginal import metropolis_accept, retained

# State-independent mixture; the order fixes the random-number stream (block = rng.choice over these).
BLOCKS = dict(morphology=[1, 6, 7], shape=[8, 9], redshift_sed=[0, 2, 3, 4, 5], environment=[], sed_independence=[2, 3, 4, 5])
PROBABILITIES = [.3, .3, .1, .15, .15]
MORPHOLOGY, SHAPE, REDSHIFT_SED, ENVIRONMENT, SED = range(5)
BASE_SCALES = [.15, .08, .15, .15, .15, .15, .08, .2, .2, .2]
REFRESH = dict(fraction=.1, full_probability=.1)
COVARIANCE = dict(window=1024, update_interval=128, shrinkage=.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='galsbi_population_v1 observations')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observation-indices', type=int, nargs='+', required=True)
    parser.add_argument('--shear', type=float, nargs=2, required=True, help='sampling shear g0 = g1 g2')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--initial-bank-seed', type=int, required=True, help='seed of the fresh prior start bank')
    parser.add_argument('--steps', type=int, default=20480)
    parser.add_argument('--warmup', type=int, default=8192)
    parser.add_argument('--profile-stride', type=int, default=16,
                        help='store states and co-moving readings every k-th step after warm-up')
    parser.add_argument('--betas', type=float, nargs='+', default=[1.],
                        help='likelihood-tempering ladder, starting at the posterior (1) and strictly decreasing')
    parser.add_argument('--narrow-step-probability', type=float, default=0.,
                        help='state-independent share of morphology steps at a quarter of the frozen scale')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    betas = np.asarray(args.betas)
    if args.steps <= args.warmup or args.warmup < 1 or args.profile_stride < 1 or args.initial_bank_seed < 0:
        raise ValueError('warmup shorter than run, positive stride and nonnegative bank seed required')
    if betas[0] != 1 or np.any(np.diff(betas) >= 0) or betas[-1] <= 0:
        raise ValueError('betas must decrease from 1 to > 0')
    if not 0 <= args.narrow_step_probability < 1:
        raise ValueError('narrow-step probability must be in [0, 1)')
    if (args.output/'result.json').exists():
        raise ValueError('fresh output required')
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.reference/'protocol.json').read_text())
    results = json.loads((args.reference/'result.json').read_text())
    if (previous.get('reference_kind') != 'galsbi_population_v1' or not results.get('preparation_checks_pass')
            or previous['chains'] != 4):
        raise ValueError('validated four-chain population reference required')
    indices = args.observation_indices
    if len(set(indices)) != len(indices) or any(i < 0 or i >= results['observation_count'] for i in indices):
        raise ValueError('unique in-range reference observation indices required')
    if digest(args.reference/'observations.parquet') != results['observations_sha256']:
        raise ValueError('population observation checksum mismatch')
    observations = pd.read_parquet(args.reference/'observations.parquet').iloc[indices].reset_index(drop=True)
    nobs, nchains = len(observations), 4
    # Level-major rows: level*ncold + 4*observation + chain; level 0 is the posterior.
    levels, ncold = len(betas), nobs*nchains
    nrows = levels*ncold
    beta = np.repeat(betas, ncold)
    cold = slice(0, ncold)
    observed = observations.iloc[np.tile(np.repeat(np.arange(nobs), nchains), levels)].reset_index(drop=True)
    aux = observed.iloc[:, 2:4].to_numpy()
    conditions = previous['conditions']
    g0 = list(args.shear)
    # b enters the kernel only as the transport velocity b*g, so at zero
    # sampling shear the MH target does not depend on b. Then b is computed
    # only for stored posterior-level states after warm-up (and the last
    # warm-up state, so step-to-step differences stay defined), not proposals.
    lazy_b = not any(g0)
    rng = np.random.default_rng(args.seed)
    sources = [Path(__file__), Path('sbsi/pseudo_marginal.py'), Path('sbsi/galsbi_target_coordinates.py'),
        Path('sbsi/galsbi_scene_prior.py'), Path('sbsi/galsbi_summary.py'), Path('sbsi/galsbi_prior.py'),
        Path('scripts/check_galsbi_continuous_prior.py'), Path('scripts/check_galsbi_summary_sampler.py'),
        Path('jobs/job_galsbi_tempered.sh'), Path('configs/models_v3_9_like.json'), Path('sbsi/mcmc_proposals.py')]
    protocol = dict(steps=args.steps, warmup=args.warmup, environments=1, chains_per_observation=nchains,
        seed=args.seed, reference=str(args.reference.resolve()), reference_protocol_sha256=digest(args.reference/'protocol.json'),
        disk_coefficient=('computed for stored posterior-level states from the last warm-up step on (NaN before); '
                          'the kernel at zero shear does not depend on b' if lazy_b else 'computed for every evaluated state'),
        observations=indices, fixed_shear=g0, profile_stride=args.profile_stride,
        diagnostic_draws='profile draws after warm-up',
        profile_shapes='co-moving: eps -> M_node^-1(M_g0(eps)), circularized radius kept, prior and Jacobian '
                       'included, b recomputed',
        target_coordinates=GalSBITargetCoordinates.names,
        environment_update='persistent', refresh_fraction=REFRESH['fraction'],
        refresh_full_probability=REFRESH['full_probability'],
        type='exact red/blue prior sum',
        initial_state=('one state of each prior-bank stream drawn with probability proportional to '
                       'its likelihood (importance resampling); never generating truth'),
        initial_bank_seed=args.initial_bank_seed, initialization='prior-draw',
        proposal_blocks=BLOCKS, proposal_block_probabilities=PROBABILITIES, base_scales=BASE_SCALES,
        adaptation='three block log-scales per chain; warmup only, then frozen',
        morphology_covariance=dict(COVARIANCE, adaptation='retained states including rejects; per-chain; '
                                                          'warmup only; then frozen'),
        sed_proposal='equal red/blue Dirichlet mixture at fixed z, with full ALR Hastings correction',
        invariant_work=('z/SED moves reuse the likelihood, environment moves the target prior; '
                        'every 1024th such move is replayed in full'),
        replay_tolerances=dict(summary_absolute=1e-10, log_kernel_absolute=1e-6),
        exactness='analytic native target density times implemented numerical environment distribution',
        exploration_gates=dict(max_rank_rhat=1.01, min_bulk_ess=400),
        no_population_selection_or_shear_estimate=True,
        implementation_sha256={str(p.resolve()): digest(p) for p in sources})
    if levels > 1:
        protocol['tempering'] = dict(levels=levels, betas=betas.tolist(), target='target prior times kernel**beta',
            swaps='deterministic even/odd adjacent-level exchange after every step',
            adaptation='per level row during warmup, not carried by swapped states; then frozen',
            stored='posterior level only')
    if args.narrow_step_probability:
        protocol['narrow_morphology_steps'] = dict(probability=args.narrow_step_probability, scale=.25,
            choice='independent of state, so the mixture of symmetric kernels stays exact')
    save(args.output/'protocol.json', protocol)
    for source in sources:
        destination = args.output/'source'/source.resolve().relative_to(Path.cwd().resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    started = time.monotonic()
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device=args.device)
    detector = load_detection_classifier(V39_LIKE, device=args.device)
    # Tree prediction parallelizes over rows only, so results do not depend on threads.
    # Raising torch's OpenMP limit is what lets xgboost use them under OMP_NUM_THREADS=1.
    # On cuda the emulator runs on the GPU too (2026-10-13 benchmark: it was ~2/3 of a step's CPU time);
    # GPU and CPU trees agree to float32 rounding (<= 1.2e-7), not bitwise.
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    model = load_disk_response(V39_LIKE, device=args.device, nthread=len(os.sched_getaffinity(0)))
    likelihood = SummaryLikelihood(flow, detector)
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)

    # PyCosmo distances are elementwise (checked batch-independent), so
    # remembered values are exact; unchanged neighbours dominate the calls.
    angular_distance = ExactMemo(lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))

    coords = GalSBITargetCoordinates(galaxy, angular_distance)

    def summarize(scene, auxiliary, with_b=True):
        """Nine summaries per scene; without b, column 8 is 0 (exact only for the kernel at zero shear)."""
        zero, ptr, features = scene_model_inputs(scene, prior, angular_distance, conditions,
            response_radius_arcsec=10., response_neighbours=20, response=with_b)
        if with_b:
            return compress_scenes(zero, CatalogueDiskResponse(ptr, features, model), auxiliary)
        auxiliary = np.broadcast_to(np.asarray(auxiliary, float), (len(zero), 2))
        return SummaryBatch(np.column_stack((zero.to_numpy(), np.zeros(len(zero)))), auxiliary)

    # Starts: per stream, one fresh prior-bank state picked with probability proportional to its likelihood
    # (approximate posterior draw by importance resampling, not the best-fitting state).
    initialization_started = time.monotonic()
    draws = previous['draws_per_chain']
    bank, _ = prior.sample(nchains*draws, args.initial_bank_seed)
    draw_rng = np.random.default_rng([args.initial_bank_seed, 992])
    zero, ptr, features = scene_model_inputs(bank, prior, angular_distance, conditions,
        response_radius_arcsec=10., response_neighbours=20)
    initial_response = CatalogueDiskResponse(ptr, features, model)
    starts, reference_kernels, effective = [], [], []
    for position in range(nobs):
        obs = observations.iloc[[position]]
        summary = compress_scenes(zero, initial_response, obs.iloc[0, 2:4].to_numpy())
        logk = likelihood.log_kernel(summary, obs, *g0).reshape(nchains, draws)
        weights = np.exp(logk-logk.max(axis=1, keepdims=True))
        pick = np.array([draw_rng.choice(len(w), p=w/w.sum()) for w in weights])
        effective.append(dict(effective_draws_per_chain=(weights.sum(1)**2/(weights**2).sum(1)).tolist()))
        starts.extend(pick+np.arange(nchains)*draws)
        reference_kernels.extend(logk[np.arange(nchains), pick])
    save(args.output/'initialization.json', dict(bank_seed=args.initial_bank_seed, bank_size=len(bank.counts),
        source='fresh prior', selected_indices=[int(i) for i in starts], selected_log_kernels=reference_kernels,
        initialization='prior-draw', measurement_details=effective,
        elapsed_seconds=time.monotonic()-initialization_started, uses_generating_truth=False))
    del zero, ptr, features, initial_response, summary
    # Every temperature level of a chain starts from that chain's selected state.
    starts = np.tile(np.asarray(starts), levels)
    reference_kernels = np.tile(np.asarray(reference_kernels), levels)
    x = coords.encode({k: v[starts] for k, v in bank.targets.items()})
    scene = bank.take(starts)
    del bank
    logprior, data, type_probability = coords.log_mixture_prior(x)
    if not np.isfinite(logprior).all():
        raise ValueError('invalid initial target')

    def evaluate_scene(scene, valid, with_b=not lazy_b):
        """Kernels and summaries; without b, column 8 is NaN until fill_b computes it."""
        values = np.zeros((nrows, 9))
        logk = np.full(nrows, -np.inf)
        if valid.any():
            summary = summarize(scene.take(np.flatnonzero(valid)), aux[valid], with_b)
            logk[valid] = likelihood.log_kernel(summary, observed.loc[valid], *g0)
            values[valid] = summary.values
            if not with_b:
                values[valid, 8] = np.nan
        return logk, values

    def fill_b(summaries, scene):
        """Compute the missing b of posterior-level states in place (row-wise, so bitwise the full-evaluation b)."""
        missing = np.flatnonzero(np.isnan(summaries[cold, 8]))
        if len(missing):
            ptr, features = response_pairs(scene.take(missing), prior, angular_distance, conditions,
                                           response_radius_arcsec=10., response_neighbours=20)
            summaries[missing, 8] = CatalogueDiskResponse(ptr, features, model).predict(
                np.arange(len(missing)), aux[missing, 0], aux[missing, 1])[0]

    # Only nodes 0.002 from the sampling shear are reliable for bright galaxies.
    profile_nodes = []
    for axis in (0, 1):
        for delta in (-.002, .002):
            node = list(g0)
            node[axis] = round(node[axis]+delta, 12)
            profile_nodes.append((f'g{axis+1}_{node[axis]:g}', node))

    def comoving_ratios(x, logprior, kernel, scene, nodes=profile_nodes):
        """log[p(x') K(x'; node) |dx'/dx| / (p(x) K(x; g0))] per cold state, x' = coords.comove(x, g0, node).

        Same retained scene (common random numbers); b changes only through
        the target semimajor axis and is recomputed. Off-support x' give -inf.
        """
        n = len(x)
        ratios = np.full((n, len(nodes)), -np.inf)
        for k, (_, node) in enumerate(nodes):
            moved, logj, valid = coords.comove(x, g0, node)
            newprior, newdata, _ = coords.log_mixture_prior(moved)
            valid &= np.isfinite(newprior)
            if valid.any():
                selected = GalSBISceneBatch(newdata, scene.neighbours, scene.indptr,
                                            scene.offset_arcsec).take(np.flatnonzero(valid))
                zero, ptr, features = scene_model_inputs(selected, prior, angular_distance, conditions,
                    response_radius_arcsec=10., response_neighbours=20)
                summary = compress_scenes(zero, CatalogueDiskResponse(ptr, features, model), aux[:n][valid])
                ratios[valid, k] = (newprior[valid]+logj[valid]-logprior[valid]-kernel[valid]
                                    + likelihood.log_kernel(summary, observed.iloc[:n].loc[valid], *node))
        return ratios

    scene = GalSBISceneBatch(data, scene.neighbours, scene.indptr, scene.offset_arcsec)
    loglike, summaries = evaluate_scene(scene, np.ones(nrows, dtype=bool), with_b=True)
    replay_error = float(np.max(np.abs(loglike-reference_kernels)))
    save(args.output/'initial_replay.json', dict(max_abs_error=replay_error, tolerance=1e-6))
    if replay_error > 1e-6:
        raise ValueError(f'initial scene replay failed: {replay_error}')
    np.savez_compressed(args.output/'initial_state.npz', target_coordinates=x, summaries=summaries,
                        log_prior=logprior, log_kernel=loglike, type_probability=type_probability,
                        neighbour_counts=scene.counts)
    log_scales = np.zeros((nrows, 3))
    scales = np.array(BASE_SCALES)
    columns_of = list(BLOCKS.values())
    morphology_columns = columns_of[MORPHOLOGY]
    morphology_factor = np.broadcast_to(np.diag(scales[morphology_columns]),
        (nrows, len(morphology_columns), len(morphology_columns))).copy()
    recent = deque(maxlen=COVARIANCE['window'])
    swap_attempts, swap_accepts = np.zeros((2, max(levels-1, 1))), np.zeros((2, max(levels-1, 1)))
    cold_exchanges = np.zeros(ncold, dtype=int)
    # Posterior-level histories only at the profile draws (every profile-stride step after warm-up).
    trace, summary_trace, type_probability_trace, count_trace, comoving_trace = [], [], [], [], []
    accepted_trace, block_trace, refresh_trace = [], [], []
    comoving_identity_replay = None
    sed_cross_proposals, sed_cross_accepts = np.zeros(ncold, dtype=int), np.zeros(ncold, dtype=int)
    invariant_audit = dict(likelihood_calls_skipped=0, target_prior_calls_skipped=0,
                           checked_proposals=0, maximum_summary_change=0., maximum_log_kernel_change=0.)
    invariant_visits = {REDSHIFT_SED: 0, SED: 0}
    sampling_started = time.monotonic()
    last_progress = time.monotonic()
    for step in range(args.steps):
        block = int(rng.choice(len(PROBABILITIES), p=PROBABILITIES))
        proposal = x.copy()
        log_hastings = np.zeros(nrows)
        if block == MORPHOLOGY:
            multiplier = np.exp(log_scales[:, 0])
            if args.narrow_step_probability:
                multiplier = multiplier*np.where(rng.random(nrows) < args.narrow_step_probability, .25, 1.)
            proposal[:, morphology_columns] += gaussian_block_delta(morphology_factor, multiplier, rng)
        elif block in (SHAPE, REDSHIFT_SED):
            columns = columns_of[block]
            proposal[:, columns] += (rng.normal(size=(nrows, len(columns)))*scales[columns]
                                     *np.exp(log_scales[:, block, None]))
        elif block == SED:
            proposal, log_hastings = coords.propose_sed(x, rng)
        if block < 3:
            proposal[:, 9] %= 2*np.pi
        if block == ENVIRONMENT:
            newprior, newdata, new_type_probability = logprior, scene.targets, type_probability
            invariant_audit['target_prior_calls_skipped'] += 1
        else:
            newprior, newdata, new_type_probability = coords.log_mixture_prior(proposal)
        newscene = GalSBISceneBatch(newdata, scene.neighbours, scene.indptr, scene.offset_arcsec)
        fraction = 0.
        if block == ENVIRONMENT:
            fraction = 1. if rng.random() < REFRESH['full_probability'] else REFRESH['fraction']
            newscene = prior.refresh(newscene, fraction, int(rng.integers(0, 2**63)))
        if block in invariant_visits:
            # At fixed m_r, angular circularized radius, n, a, phase and
            # environment, z/SED affect only the native target prior.
            # Off-support proposals still have -inf prior and are rejected.
            newlike, newsummary = loglike, summaries
            if invariant_visits[block] % 1024 == 0:
                valid = np.isfinite(newprior)
                audit_like, audit_summary = evaluate_scene(newscene, valid, with_b=True)
                invariant_audit['checked_proposals'] += int(valid.sum())
                if valid.any():
                    invariant_audit['maximum_summary_change'] = max(invariant_audit['maximum_summary_change'],
                        float(np.nanmax(np.abs(audit_summary[valid]-summaries[valid]))))
                    invariant_audit['maximum_log_kernel_change'] = max(invariant_audit['maximum_log_kernel_change'],
                        float(np.max(np.abs(audit_like[valid]-loglike[valid]))))
                if (invariant_audit['maximum_summary_change'] > protocol['replay_tolerances']['summary_absolute']
                        or invariant_audit['maximum_log_kernel_change'] > protocol['replay_tolerances']['log_kernel_absolute']):
                    save(args.output/'invariant_replay_failure.json', invariant_audit)
                    raise ValueError('invariant cache disagrees with full likelihood replay')
            else:
                invariant_audit['likelihood_calls_skipped'] += 1
            invariant_visits[block] += 1
        else:
            newlike, newsummary = evaluate_scene(newscene, np.isfinite(newprior))
        accepted = metropolis_accept(logprior+beta*loglike, newprior+beta*newlike+log_hastings,
                                     rng.uniform(np.finfo(float).tiny, 1., nrows))
        if block == SED and step >= args.warmup:
            cross = (np.isfinite(newprior) & ((type_probability[:, 0] >= .5) != (new_type_probability[:, 0] >= .5)))[cold]
            sed_cross_proposals += cross
            sed_cross_accepts += cross & accepted[cold]
        x, logprior, loglike, summaries, type_probability = [retained(old, new, accepted) for old, new in
            ((x, proposal), (logprior, newprior), (loglike, newlike), (summaries, newsummary),
             (type_probability, new_type_probability))]
        if block == ENVIRONMENT:
            scene = scene.retained(newscene, accepted)
        else:
            # Replacing target marks does not replace or copy neighbours.
            scene = GalSBISceneBatch(
                {key: retained(value, newscene.targets[key], accepted) for key, value in scene.targets.items()},
                scene.neighbours, scene.indptr, scene.offset_arcsec)
        if step < args.warmup and block < 3:
            log_scales[:, block] += .05*(accepted-.3)
        if levels > 1:
            parity = step % 2
            perm, exchanged = adjacent_swap_permutation(loglike, betas, ncold, parity,
                rng.uniform(np.finfo(float).tiny, 1., (levels-1, ncold)))
            x, logprior, loglike, summaries, type_probability = [value[perm] for value in
                (x, logprior, loglike, summaries, type_probability)]
            scene = scene.take(perm)
            stage = int(step >= args.warmup)
            swap_attempts[stage, parity::2] += ncold
            swap_accepts[stage] += exchanged.sum(1)
            if stage:
                cold_exchanges += exchanged[0]
        if step < args.warmup:
            recent.append(x[:, morphology_columns].copy())
            if (step+1) % COVARIANCE['update_interval'] == 0:
                morphology_factor = empirical_block_cholesky(np.array(recent), scales[morphology_columns],
                                                             shrinkage=COVARIANCE['shrinkage'])
        if lazy_b and step+1 >= args.warmup:
            fill_b(summaries, scene)
        if step >= args.warmup and (step-args.warmup) % args.profile_stride == 0:
            trace.append(x[cold].copy())
            summary_trace.append(summaries[cold].copy())
            type_probability_trace.append(type_probability[cold].copy())
            count_trace.append(scene.counts[cold])
            state = (x[cold], logprior[cold], loglike[cold], scene.take(np.arange(ncold)))
            if comoving_identity_replay is None:
                # node = g0 must reproduce the stored prior and kernel exactly
                comoving_identity_replay = float(np.max(np.abs(comoving_ratios(*state, nodes=[('g0', g0)]))))
                if not comoving_identity_replay < 1e-6:
                    raise ValueError(f'co-moving identity replay failed: {comoving_identity_replay}')
            comoving_trace.append(comoving_ratios(*state))
        accepted_trace.append(accepted[cold])
        block_trace.append(block)
        refresh_trace.append(fraction)
        if (step+1) % 128 == 0 or time.monotonic()-last_progress > 45:
            report = dict(step=step+1, steps=args.steps, elapsed_seconds=time.monotonic()-started,
                mean_acceptance_last128=float(np.mean(accepted_trace[-128:])),
                median_scale_multipliers=np.median(np.exp(log_scales[cold]), axis=0).tolist(),
                swap_acceptance=(swap_accepts.sum(0)/np.maximum(swap_attempts.sum(0), 1)).tolist()
                if levels > 1 else None)
            save(args.output/'progress.json', report)
            print('LOCAL_PROGRESS', report, flush=True)
            last_progress = time.monotonic()
    sampling_finished = time.monotonic()
    save(args.output/'invariant_replay.json', invariant_audit)
    scene = scene.take(np.arange(ncold))
    np.savez_compressed(args.output/'final_scenes.npz',
        **{'target_'+k: v for k, v in scene.targets.items()},
        **{'neighbour_'+k: v for k, v in scene.neighbours.items()},
        indptr=scene.indptr, offset_arcsec=scene.offset_arcsec)
    histories_saved = time.monotonic()
    trace, summary_trace, comoving_trace = np.array(trace), np.array(summary_trace), np.array(comoving_trace)
    type_probability_trace, count_trace = np.array(type_probability_trace), np.array(count_trace)
    accepted_trace, block_trace = np.array(accepted_trace)[args.warmup:], np.array(block_trace)[args.warmup:]
    refresh_trace = np.array(refresh_trace)[args.warmup:]
    reports, profile_samples = [], {}
    for i in range(nobs):
        rows = slice(i*nchains, (i+1)*nchains)
        # One environment per state, so the stored summaries are the posterior scene moments.
        moments = summary_trace[:, rows]
        checks = {name: diagnostics(moments[:, :, col].T) for name, col in
                  (('e1', 0), ('e2', 1), ('m_r', 3), ('Re_circularized', 4), ('crowding_050', 5), ('b', 8))}
        checks['type'] = diagnostics(type_probability_trace[:, rows, 0].T)
        checks['log_z'] = diagnostics(trace[:, rows, 0].T)
        checks['neighbour_count'] = diagnostics(count_trace[:, rows].T)
        for col in (2, 3, 4, 5, 7, 8):
            checks[coords.names[col]] = diagnostics(trace[:, rows, col].T)
        for col in (6, 7):
            checks[f'crowding_{col}'] = diagnostics(moments[:, :, col].T)
        profile = []
        for k, (name, node) in enumerate(profile_nodes):
            ratios = comoving_trace[:, rows, k]
            profile_samples[f'observation_{indices[i]}_{name}'] = ratios
            values = logsumexp(ratios, axis=0)-np.log(len(ratios))
            checks[f'log_ratio_{name}'] = diagnostics(ratios.T)
            profile.append(dict(g1=node[0], g2=node[1], log_numerator_ratio_per_chain=values.tolist()))
        by_block = {name: accepted_trace[block_trace == b, rows].mean(0).tolist() if (block_trace == b).any() else None
                    for b, name in enumerate(BLOCKS)}
        passing = all(d['rhat'] is not None and d['rhat'] < 1.01 and d['ess_bulk'] > 400
                      for d in checks.values() if not d['constant'])
        reports.append(dict(reference_observation=indices[i],
            measured_magnitude=float(30-2.5*np.log10(observations.iloc[i, 3])),
            diagnostics=checks, acceptance_by_block_per_chain=by_block,
            red_probability_mean_per_chain=type_probability_trace[:, rows, 0].mean(0).tolist(),
            sed_region_switches_per_chain=np.sum(np.diff(type_probability_trace[:, rows, 0] >= .5,
                                                          axis=0) != 0, axis=0).tolist(),
            sed_cross_proposals_per_chain=sed_cross_proposals[rows].tolist(),
            sed_cross_accepts_per_chain=sed_cross_accepts[rows].tolist(),
            environment_acceptance={name: dict(attempts_per_chain=int(select.sum()),
                acceptance_per_chain=accepted_trace[select, rows].mean(0).tolist() if select.any() else None)
                for name, select in (('partial', (refresh_trace > 0) & (refresh_trace < 1)), ('full', refresh_trace == 1))},
            local_numerator_profiles=profile, exploration_checks_pass=passing))
    np.savez_compressed(args.output/'local_profiles.npz', **profile_samples)
    result = dict(status='completed_diagnostic', observations=reports,
        elapsed_seconds=time.monotonic()-started, frozen_scale_multipliers=np.exp(log_scales).tolist(),
        timing_seconds=dict(setup=sampling_started-started, sampling=sampling_finished-sampling_started,
            history_output=histories_saved-sampling_finished, profiles_and_diagnostics=time.monotonic()-histories_saved),
        comoving_identity_replay_max_abs=comoving_identity_replay,
        frozen_morphology_cholesky=morphology_factor.tolist(),
        protocol_sha256=digest(args.output/'protocol.json'),
        all_exploration_checks_pass=all(r['exploration_checks_pass'] for r in reports),
        tempering=dict(betas=betas.tolist(),
            swap_acceptance_warmup=(swap_accepts[0]/np.maximum(swap_attempts[0], 1)).tolist(),
            swap_acceptance_post_warmup=(swap_accepts[1]/np.maximum(swap_attempts[1], 1)).tolist(),
            posterior_level_exchanges_per_chain=cold_exchanges.tolist()) if levels > 1 else None,
        limitations=['No measured-selection normalization or shear estimate.',
            'Native target analytic prior and numerical neighbour generator retain their audited approximation difference.',
            'No learned compressed density; target redshift/SED latents are retained in ten fixed coordinates.',
            'Persistent scenes use deterministic likelihoods and a prior-reversible environment update.',
            'Finite-chain mixing and rare environmental modes still require validation.'])
    save(args.output/'result.json', result)
    print('LOCAL_COMPLETE', result['all_exploration_checks_pass'], result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
