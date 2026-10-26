"""Bounded local MH pilots on the existing bright/faint mock panel.

Moves ten continuous target coordinates, optionally summing over galaxy type.
Either retain explicit neighbours during target moves, with separate partial
refreshes, or use independent pseudo-marginal likelihood averages. Rejected
states remain unchanged. No learned prior or generating-truth starts.
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
from sbsi.galsbi_initialization import measurement_seed_coordinates
from sbsi.galsbi_prior import ExactMemo
from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
from sbsi.galsbi_summary import (SummaryBatch, SummaryLikelihood, compress_scenes, response_pairs,
                                 scene_model_inputs)
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.scene_exchange import propose_native_exchange
from sbsi.gradient_proposals import bounded_langevin_mean, gaussian_log_hastings, ReplicaTransport
from sbsi.measurement_model import load_measurement_model
from sbsi.mcmc_proposals import (adjacent_swap_permutation, complementary_stretch_proposal,
                                empirical_block_cholesky, gaussian_block_delta, pooled_block_cholesky)
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
from sbsi.pseudo_marginal import metropolis_accept, retained


def environment_noise_audit(data, prior, angular_distance, conditions, model, likelihood, observed,
                            *, seed, stored_log_estimate, shear, draws=1024):
    """Independent fixed-target noise probe; does not change the retained chains."""
    target = {k: np.repeat(v, draws, axis=0) for k, v in data.items()}
    scene, _ = prior.sample_given_targets(target, seed)
    zero, ptr, features = scene_model_inputs(scene, prior, angular_distance, conditions,
        response_radius_arcsec=10., response_neighbours=20)
    repeated = observed.iloc[np.repeat(np.arange(len(observed)), draws)].reset_index(drop=True)
    summary = compress_scenes(zero, CatalogueDiskResponse(ptr, features, model), repeated.iloc[:, 2:4].to_numpy())
    logk = likelihood.log_kernel(summary, repeated, *shear).reshape(len(observed), draws)
    mean = logsumexp(logk, axis=1)-np.log(draws)
    weights = np.exp(logk-logsumexp(logk, axis=1, keepdims=True))
    return dict(draws_per_fixed_target=draws, seed=seed,
        log_mean_likelihood=mean.tolist(), stored_minus_audit_log_estimate=(stored_log_estimate-mean).tolist(),
        effective_environments=(1/(weights*weights).sum(1)).tolist(),
        log_estimate_std_by_bank_size={str(m): np.std(logsumexp(logk.reshape(len(observed), -1, m), axis=2)
                                                    -np.log(m), axis=1, ddof=1).tolist() for m in (4, 16, 64)},
        limitation='Independent finite reference at final targets; rare-environment mass can still be missed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=512)
    parser.add_argument('--environments', type=int, default=4)
    parser.add_argument('--marginalize-type', action='store_true')
    parser.add_argument('--environment-update', choices=['independent', 'persistent'], default='independent')
    parser.add_argument('--refresh-fraction', type=float, default=.1)
    parser.add_argument('--scale-adaptation', choices=['per-observation', 'per-chain'], default='per-observation')
    parser.add_argument('--sed-jump', action='store_true')
    parser.add_argument('--joint-morphology', action='store_true')
    parser.add_argument('--joint-shape', action='store_true')
    parser.add_argument('--ensemble-morphology', action='store_true',
                        help='replace half the morphology block probability with complementary stretch moves')
    parser.add_argument('--pooled-morphology', action='store_true',
                        help='pool same-observation warmup geometry; freeze halfway, then tune scales')
    parser.add_argument('--joint-environment', action='store_true',
                        help='occasionally pair a broad morphology step with a full prior environment refresh')
    parser.add_argument('--reuse-invariant-work', action='store_true',
                        help='cache likelihood for z/SED moves and target prior for environment-only moves')
    parser.add_argument('--betas', type=float, nargs='+', default=[1.],
                        help='likelihood-tempering ladder, starting at the posterior (1) and strictly decreasing')
    parser.add_argument('--narrow-step-probability', type=float, default=0.,
                        help='state-independent share of morphology steps at a quarter of the frozen scale')
    parser.add_argument('--mala-morphology-probability', type=float, default=0.,
                        help='experimental state-independent fraction of morphology moves using corrected bounded drift')
    parser.add_argument('--mala-max-drift', type=float, default=1.,
                        help='maximum drift in proposal standard-deviation units')
    parser.add_argument('--scene-exchange-probability', type=float, default=0.,
                        help='experimental probability of a corrected complete target/neighbour mark exchange')
    parser.add_argument('--scene-exchange-radius', type=float, default=1.,
                        help='fixed positional Gaussian scale in arcsec for neighbour selection')
    parser.add_argument('--tempering-diagnostics', action='store_true',
                        help='RNG-free replica round trips and hot-state histories')
    parser.add_argument('--local-positions', action='store_true')
    parser.add_argument('--position-scale', type=float, default=.2)
    parser.add_argument('--observation-indices', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=2026093061)
    parser.add_argument('--initial-bank-seed', type=int,
                        help='draw a new prior bank for initialization while keeping the reference observations')
    parser.add_argument('--initial-bank-draws', type=int,
                        help='prior-bank states per chain for a fresh bank (default: the reference value)')
    parser.add_argument('--initialization', choices=['prior-bank', 'prior-draw', 'measurement'], default='prior-bank',
                        help='starting states only: best prior-bank state, a likelihood-weighted prior-bank draw, or best measurement-centred candidate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--density-batch-size', type=int, default=4096,
                        help='fixed padded F8/C8 batch; rows per step are levels x observations x chains')
    parser.add_argument('--compile-density', nargs='?', const=True, default=False, choices=[True, 'reduce-overhead'],
                        help='torch.compile the log-gap F8 density (fused kernels; not bitwise the eager values); '
                             '"reduce-overhead" also replays CUDA graphs for the fixed padded batch')
    parser.add_argument('--shear', type=float, nargs=2,
                        help='sampling shear g1 g2 (default: the reference protocol shear, i.e. the mock truth)')
    parser.add_argument('--profile-stride', type=int, default=1,
                        help='use every k-th retained step for the saved likelihood-ratio profiles')
    parser.add_argument('--comoving-profiles', action='store_true',
                        help='profile ratios with intrinsic shapes mapped so the sheared shape is unchanged '
                             '(exact change of variables, b recomputed in the retained scene); persistent only')
    parser.add_argument('--no-chain', action='store_true',
                        help='do not write or keep the per-step history (chain.npz); profiles and final scenes are '
                             'unchanged, result.json diagnostics use the profile draws only')
    args = parser.parse_args()
    if args.steps <= args.warmup or args.warmup < 1 or args.environments < 1:
        raise ValueError('positive environment count and warmup shorter than run required')
    if args.initial_bank_seed is not None and args.initial_bank_seed < 0:
        raise ValueError('initial bank seed must be nonnegative')
    if args.initial_bank_draws is not None and (args.initial_bank_seed is None or args.initial_bank_draws < 1):
        raise ValueError('initial bank draws require a fresh bank and must be positive')
    persistent = args.environment_update == 'persistent'
    if args.comoving_profiles and not persistent:
        raise ValueError('co-moving profiles need the retained scenes of persistent environments')
    if args.initialization == 'measurement' and not (
            persistent and args.marginalize_type and args.initial_bank_seed is not None):
        raise ValueError('measurement initialization requires a fresh bank, persistent scenes and type marginalization')
    if persistent and (args.environments != 1 or not 0 < args.refresh_fraction <= 1):
        raise ValueError('persistent scenes require one environment and refresh fraction in (0, 1]')
    if args.sed_jump and not (persistent and args.marginalize_type):
        raise ValueError('SED jump pilot requires persistent neighbours and exact type marginalization')
    if args.joint_morphology and not (args.sed_jump and args.scale_adaptation == 'per-chain'):
        raise ValueError('joint morphology pilot requires SED jumps and per-chain scale adaptation')
    if args.joint_shape and not args.joint_morphology:
        raise ValueError('joint shape requires joint morphology')
    if args.ensemble_morphology and (not args.joint_morphology or args.joint_shape
                                   or args.pooled_morphology or args.joint_environment or args.local_positions):
        raise ValueError('ensemble morphology requires plain three-coordinate joint morphology')
    if args.pooled_morphology and (not args.joint_morphology or args.joint_shape or args.warmup < 2048):
        raise ValueError('pooled morphology requires three-coordinate morphology and at least2048 warmup steps')
    if args.joint_environment and not args.joint_morphology:
        raise ValueError('joint environment requires the persistent joint morphology pilot')
    if args.reuse_invariant_work and not args.joint_morphology:
        raise ValueError('invariant cache requires persistent joint morphology with separate z/SED block')
    if args.local_positions and not (args.sed_jump and np.isfinite(args.position_scale) and args.position_scale > 0):
        raise ValueError('local positions require persistent SED pilot and positive finite position scale')
    betas = np.asarray(args.betas)
    if (betas[0] != 1 or np.any(np.diff(betas) >= 0) or betas[-1] <= 0
            or (len(betas) > 1 and not args.joint_morphology) or (len(betas) > 1 and args.ensemble_morphology)):
        raise ValueError('tempering requires persistent per-chain joint morphology and betas decreasing from 1 to >0')
    if not 0 <= args.narrow_step_probability < 1 or (args.narrow_step_probability and not args.joint_morphology):
        raise ValueError('narrow steps require joint morphology and probability in [0, 1)')
    if (not 0 <= args.mala_morphology_probability <= 1 or not np.isfinite(args.mala_max_drift)
            or args.mala_max_drift <= 0):
        raise ValueError('MALA probability in [0,1] and positive finite drift bound required')
    if args.mala_morphology_probability and (not args.joint_morphology or args.joint_shape
            or args.ensemble_morphology or args.joint_environment or args.narrow_step_probability):
        raise ValueError('MALA pilot requires plain three-coordinate joint morphology')
    if (not 0 <= args.scene_exchange_probability <= .15
            or not np.isfinite(args.scene_exchange_radius) or args.scene_exchange_radius <= 0):
        raise ValueError('exchange probability in [0,.15] and positive finite radius required')
    if args.scene_exchange_probability and (not persistent or not args.marginalize_type
            or not args.sed_jump or args.local_positions or args.ensemble_morphology):
        raise ValueError('exchange requires persistent marginalized-type SED-jump sampler without block-six alternatives')
    if args.tempering_diagnostics and len(betas) < 2:
        raise ValueError('transport diagnostics require multiple temperatures')
    if (args.output/'result.json').exists():
        raise ValueError('fresh output required')
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.reference/'protocol.json').read_text())
    results = json.loads((args.reference/'result.json').read_text())
    population = previous.get('reference_kind') == 'galsbi_population_v1'
    reference_pass = results.get('preparation_checks_pass') if population else results.get('implementation_checks_pass')
    if not reference_pass or previous['chains'] != 4:
        raise ValueError('validated four-chain reference required')
    count = results['observation_count'] if population else len(results['observations'])
    if population and args.initial_bank_seed is None:
        raise ValueError('population reference requires fresh initialization banks')
    indices = args.observation_indices if args.observation_indices is not None else [0, count-1]
    if (len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices)):
        raise ValueError('unique in-range reference observation indices required')
    if population and digest(args.reference/'observations.parquet') != results['observations_sha256']:
        raise ValueError('population observation checksum mismatch')
    observations = pd.read_parquet(args.reference/'observations.parquet').iloc[indices].reset_index(drop=True)
    nobs, nchains, m = len(observations), 4, args.environments
    # Level-major rows: level*ncold + 4*observation + chain; level 0 is the posterior.
    levels, ncold = len(betas), nobs*nchains
    beta = np.repeat(betas, ncold)
    cold = slice(0, ncold)
    owner = np.tile(np.repeat(np.arange(nobs), nchains), levels)
    observed = observations.iloc[owner].reset_index(drop=True)
    repeated = observed.iloc[np.repeat(np.arange(len(owner)), m)].reset_index(drop=True)
    aux = repeated.iloc[:, 2:4].to_numpy()
    conditions = previous['conditions']
    g0 = previous['shear'] if args.shear is None else list(args.shear)
    # b enters the kernel only as the transport velocity b*g, so at zero
    # sampling shear the MH target does not depend on b. Then b is computed
    # only for stored posterior-level states after warm-up (and the last
    # warm-up state, so step-to-step differences stay defined), not proposals.
    lazy_b = persistent and not any(g0)
    rng = np.random.default_rng(args.seed)
    sources = [Path(__file__), Path('sbsi/pseudo_marginal.py'), Path('sbsi/galsbi_target_coordinates.py'),
        Path('sbsi/galsbi_scene_prior.py'), Path('sbsi/galsbi_summary.py'), Path('sbsi/galsbi_prior.py'),
        Path('scripts/check_galsbi_continuous_prior.py'), Path('scripts/check_galsbi_summary_sampler.py'),
        Path('jobs/job_galsbi_local_sampler.sh'), Path('configs/models_v3_9_like.json'),
        Path('sbsi/mcmc_proposals.py'), Path('sbsi/galsbi_initialization.py'),
        Path('sbsi/gradient_proposals.py'), Path('sbsi/summary_gradients.py'), Path('sbsi/scene_exchange.py')]
    protocol = dict(steps=args.steps, warmup=args.warmup, environments=m, chains_per_observation=nchains,
        seed=args.seed, reference=str(args.reference.resolve()), reference_protocol_sha256=digest(args.reference/'protocol.json'),
        disk_coefficient=('computed for stored posterior-level states from the last warm-up step on (NaN before); '
                          'the kernel at zero shear does not depend on b' if lazy_b else 'computed for every evaluated state'),
        observations=indices, fixed_shear=g0, profile_stride=args.profile_stride,
        density_batch_size=args.density_batch_size, compiled_density=args.compile_density,
        chain_saved=not args.no_chain,
        diagnostic_draws='profile draws after warm-up' if args.no_chain else 'every step after warm-up',
        profile_shapes='co-moving: eps -> M_node^-1(M_g0(eps)), circularized radius kept, prior and Jacobian '
                       'included, b recomputed' if args.comoving_profiles else 'fixed intrinsic shapes',
        target_coordinates=GalSBITargetCoordinates.names,
        environment_update=args.environment_update,
        refresh_fraction=args.refresh_fraction if persistent else None,
        refresh_full_probability=.1 if persistent else None,
        environment_initialization='retain reference scenes' if persistent else 'fresh independent banks',
        type='exact red/blue prior sum' if args.marginalize_type else 'binary red/blue',
        initial_state='best of each independent reference prior stream, selected by likelihood; never generating truth',
        initial_bank_seed=args.initial_bank_seed,
        initial_bank_draws=args.initial_bank_draws,
        initialization=args.initialization,
        reuse_invariant_work=args.reuse_invariant_work,
        proposal_blocks=dict(photometry_size=[1, 6], shape=[8, 9], structure=[0, 2, 3, 4, 5, 7],
                             type_flip=[], environment_only=[]),
        proposal_block_probabilities=[.3, .3, .25, 0., .15] if args.marginalize_type else [.3, .3, .2, .05, .15],
        base_scales=[.15, .08, .15, .15, .15, .15, .08, .2, .2, .2],
        adaptation=f'three block log-scales {args.scale_adaptation}; warmup only, then frozen',
        exactness='analytic native target density times implemented numerical environment distribution',
        exploration_gates=dict(max_rank_rhat=1.01, min_bulk_ess=400),
        no_population_selection_or_shear_estimate=True,
        implementation_sha256={str(p.resolve()): digest(p) for p in sources})
    if args.initial_bank_seed is not None:
        protocol['initial_state'] = 'best of each fresh prior-bank stream, selected by likelihood; never generating truth'
    if args.initialization == 'prior-draw':
        protocol['initial_state'] = ('one state of each prior-bank stream drawn with probability proportional to '
                                     'its likelihood (importance resampling); never generating truth')
    if args.initialization == 'measurement':
        protocol['initial_state'] = ('measurement-centred physical coordinates with four Sersic centres; '
            'native redshift/SED and neighbours from fresh bank; maximize target log prior plus full likelihood per stream')
    if args.sed_jump:
        protocol['proposal_blocks']['sed_independence'] = [2, 3, 4, 5]
        protocol['proposal_block_probabilities'] = [.3, .3, .1, 0., .15, .15]
        protocol['sed_proposal'] = 'equal red/blue Dirichlet mixture at fixed z, with full ALR Hastings correction'
        protocol['sed_replay_tolerances'] = dict(summary_absolute=1e-10, log_kernel_absolute=1e-6)
    if args.joint_morphology:
        protocol['proposal_blocks']['photometry_size'] = [1, 6, 7]
        protocol['proposal_blocks']['structure'] = [0, 2, 3, 4, 5]
        protocol['morphology_covariance'] = dict(window=1024, update_interval=128, shrinkage=.05,
            circular_indices=[4] if args.joint_shape else [],
            adaptation='retained states including rejects; per-chain; warmup only; then frozen')
    if args.joint_shape:
        protocol['proposal_blocks']['photometry_size'] = [1, 6, 7, 8, 9]
        protocol['proposal_block_probabilities'] = [.6, 0., .1, 0., .15, .15]
    if args.pooled_morphology:
        protocol['morphology_covariance'] = dict(window=8192, update_interval=1024, shrinkage=.05,
            circular_indices=[], freeze_step=args.warmup//2,
            adaptation='pool same-observation chains (per temperature level); freeze halfway through warmup; '
                       'then scalar tuning only')
    if args.joint_environment:
        protocol['joint_environment'] = dict(probability_given_morphology=.2, morphology_scale=4.,
            refresh_fraction=1., adaptation='use local morphology scale; joint attempts do not tune it',
            acceptance='target prior and likelihood ratio; independent environment-prior draw cancels')
    if args.local_positions:
        protocol['proposal_blocks']['local_positions'] = []
        protocol['proposal_block_probabilities'][5] -= .05
        protocol['proposal_block_probabilities'].append(.05)
        protocol['position_scale_arcsec'] = args.position_scale
    if args.ensemble_morphology:
        protocol['proposal_blocks']['ensemble_morphology'] = [1, 6, 7]
        protocol['proposal_block_probabilities'][0] -= .15
        protocol['proposal_block_probabilities'].append(.15)
        protocol['ensemble_morphology'] = dict(stretch=2., probability=.15,
            active='random half per observation, frozen complementary partners',
            correction='(dimension-1)*log(z)', coupled_walkers=True,
            diagnostics='per-walker Rhat/ESS are descriptive only; use independent ensembles')
    if levels > 1:
        protocol['tempering'] = dict(levels=levels, betas=betas.tolist(), target='target prior times kernel**beta',
            swaps='deterministic even/odd adjacent-level exchange after every step',
            adaptation='per level row during warmup, not carried by swapped states; then frozen',
            stored='posterior level only')
    if args.narrow_step_probability:
        protocol['narrow_morphology_steps'] = dict(probability=args.narrow_step_probability, scale=.25,
            choice='independent of state, so the mixture of symmetric kernels stays exact')
    if args.mala_morphology_probability:
        protocol['mala_morphology'] = dict(probability=args.mala_morphology_probability,
            max_whitened_drift=args.mala_max_drift, prior_difference_step=1e-4,
            columns=[1, 6, 7], gradient='torch summary derivative plus finite-difference analytic prior; '
            'response held fixed at its recomputed value at each endpoint',
            acceptance='unchanged full target and forward/reverse Gaussian densities',
            adaptation='existing morphology covariance and scale during warmup only')
    if args.scene_exchange_probability:
        protocol['proposal_blocks']['scene_exchange'] = list(range(10))
        for index in (4, 5):
            protocol['proposal_block_probabilities'][index] -= args.scene_exchange_probability/2
        protocol['proposal_block_probabilities'].append(args.scene_exchange_probability)
        protocol['scene_exchange'] = dict(probability=args.scene_exchange_probability,
            radius_arcsec=args.scene_exchange_radius,
            selection='position-only Gaussian weights; unchanged by mark exchange',
            type='temporary analytic conditional Gibbs draw, then marginalize',
            acceptance='native mark exchange: numerical/analytic prior correction plus beta*kernel difference',
            environment='same numerical generator law; count and positions fixed')
    protocol['tempering_diagnostics'] = bool(args.tempering_diagnostics)
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
    likelihood = SummaryLikelihood(flow, detector, density_batch_size=args.density_batch_size,
                                   compile_density=args.compile_density)
    galaxy, cosmo, _ = build_prior()
    prior = GalSBIScenePrior(galaxy)

    # PyCosmo distances are elementwise (checked batch-independent), so
    # remembered values are exact; unchanged neighbours dominate the calls.
    angular_distance = ExactMemo(lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))

    coords = GalSBITargetCoordinates(galaxy, angular_distance)
    gradient_evaluator = None
    if args.mala_morphology_probability:
        from sbsi.summary_gradients import SummaryGradients
        gradient_evaluator = SummaryGradients(flow, detector, batch_size=args.density_batch_size)

    def summarize(scene, auxiliary, with_b=True):
        """Nine summaries per scene; without b, column 8 is 0 (exact only for the kernel at zero shear)."""
        zero, ptr, features = scene_model_inputs(scene, prior, angular_distance, conditions,
            response_radius_arcsec=10., response_neighbours=20, response=with_b)
        if with_b:
            return compress_scenes(zero, CatalogueDiskResponse(ptr, features, model), auxiliary)
        auxiliary = np.broadcast_to(np.asarray(auxiliary, float), (len(zero), 2))
        return SummaryBatch(np.column_stack((zero.to_numpy(), np.zeros(len(zero)))), auxiliary)

    # Reproduce only the prior-proposal bank, not the independently generated
    # mock scene. Reference likelihood chooses dispersed starts per stream.
    seeds = previous['seed']+10 if args.initial_bank_seed is None else args.initial_bank_seed
    initialization_started = time.monotonic()
    draws = args.initial_bank_draws or previous['draws_per_chain']
    bank, _ = prior.sample(previous['chains']*draws, seeds)
    starts, reference_kernels = [], []
    draw_rng = np.random.default_rng([seeds, 992])
    measurement_coordinates, initialization_details = [], []
    if args.initialization == 'measurement':
        bank_coordinates = coords.encode(bank.targets)
    elif args.initial_bank_seed is not None:
        zero, ptr, features = scene_model_inputs(bank, prior, angular_distance, conditions,
            response_radius_arcsec=10., response_neighbours=20)
        initial_response = CatalogueDiskResponse(ptr, features, model)
    for position, i in enumerate(protocol['observations']):
        if args.initialization == 'measurement':
            obs = observations.iloc[[position]]
            candidates, details = measurement_seed_coordinates(bank_coordinates, obs.iloc[0, :4].to_numpy(),
                np.repeat(np.arange(nchains), draws), conditions, galaxy.p, g0,
                np.random.default_rng(seeds+991+i))
            candidate_prior, candidate_data, _ = coords.log_mixture_prior(candidates)
            valid = np.isfinite(candidate_prior)
            candidate_scene = GalSBISceneBatch(candidate_data, bank.neighbours, bank.indptr, bank.offset_arcsec)
            summary = summarize(candidate_scene.take(np.flatnonzero(valid)), obs.iloc[0, 2:4].to_numpy(), not lazy_b)
            flat_kernel = np.full(len(bank.counts), -np.inf)
            flat_kernel[valid] = likelihood.log_kernel(summary, obs, *g0)
            scores = (candidate_prior+flat_kernel).reshape(nchains, draws)
            if not np.isfinite(scores).any(axis=1).all():
                raise ValueError('measurement candidate stream has no finite posterior starting state')
            selected = np.argmax(scores, axis=1)+np.arange(nchains)*scores.shape[1]
            measurement_coordinates.append(candidates[selected])
            details.update(valid_candidates=int(valid.sum()), selected_log_prior=candidate_prior[selected].tolist(),
                           selected_log_posterior=(candidate_prior+flat_kernel)[selected].tolist())
            initialization_details.append(details)
            starts.extend(selected)
            reference_kernels.extend(flat_kernel[selected])
            continue
        elif args.initial_bank_seed is None:
            logk = np.load(args.reference/f'observation_{i}.npz')['log_kernel']
        else:
            obs = observations.iloc[[position]]
            summary = compress_scenes(zero, initial_response, obs.iloc[0, 2:4].to_numpy())
            logk = likelihood.log_kernel(summary, obs, *g0).reshape(nchains, draws)
        if args.initialization == 'prior-draw':
            # Bank states are prior draws, so one picked with probability proportional to its likelihood is an
            # approximate posterior draw (importance resampling), not the best-fitting state.
            weights = np.exp(logk-logk.max(axis=1, keepdims=True))
            pick = np.array([draw_rng.choice(len(w), p=w/w.sum()) for w in weights])
            initialization_details.append(dict(effective_draws_per_chain=(weights.sum(1)**2/(weights**2).sum(1)).tolist()))
        else:
            pick = np.argmax(logk, axis=1)
        starts.extend(pick+np.arange(nchains)*logk.shape[1])
        reference_kernels.extend(logk[np.arange(nchains), pick])
    save(args.output/'initialization.json', dict(bank_seed=seeds, bank_size=len(bank.counts),
        source='fresh prior' if args.initial_bank_seed is not None else 'reference prior replay',
        selected_indices=[int(i) for i in starts], selected_log_kernels=reference_kernels,
        initialization=args.initialization, measurement_details=initialization_details,
        elapsed_seconds=time.monotonic()-initialization_started, uses_generating_truth=False))
    if args.initial_bank_seed is not None and args.initialization != 'measurement':
        del zero, ptr, features, initial_response, summary
    # Every temperature level of a chain starts from that chain's selected state.
    starts = np.tile(np.asarray(starts), levels)
    reference_kernels = np.tile(np.asarray(reference_kernels), levels)
    measurement_coordinates = [np.tile(np.concatenate(measurement_coordinates), (levels, 1))] \
        if args.initialization == 'measurement' else measurement_coordinates
    targets = {k: v[starts] for k, v in bank.targets.items()}
    x = np.concatenate(measurement_coordinates) if args.initialization == 'measurement' else coords.encode(targets)
    kinds = targets['type'].copy()
    scene = bank.take(np.array(starts)) if persistent else None
    del bank
    def target_prior(x, kinds):
        if args.marginalize_type:
            return coords.log_mixture_prior(x)
        logprior, data = coords.log_prior(x, kinds)
        return logprior, data, np.eye(2)[kinds]

    logprior, data, type_probability = target_prior(x, kinds)
    if not np.isfinite(logprior).all():
        raise ValueError('invalid initial target')

    def evaluate_scene(scene, valid, with_b=not lazy_b):
        """Kernels and summaries; without b, column 8 is NaN until fill_b computes it."""
        values = np.zeros((len(owner), 1, 9))
        logk = np.full((len(owner), 1), -np.inf)
        if valid.any():
            summary = summarize(scene.take(np.flatnonzero(valid)), aux[valid], with_b)
            logk[valid, 0] = likelihood.log_kernel(summary, observed.loc[valid], *g0)
            values[valid, 0] = summary.values
            if not with_b:
                values[valid, 0, 8] = np.nan
        return logk[:, 0], values, logk

    def fill_b(summaries, scene):
        """Compute the missing b of posterior-level states in place (row-wise, so bitwise the full-evaluation b)."""
        missing = np.flatnonzero(np.isnan(summaries[cold, 0, 8]))
        if len(missing):
            ptr, features = response_pairs(scene.take(missing), prior, angular_distance, conditions,
                                           response_radius_arcsec=10., response_neighbours=20)
            summaries[missing, 0, 8] = CatalogueDiskResponse(ptr, features, model).predict(
                np.arange(len(missing)), aux[missing, 0], aux[missing, 1])[0]

    def change(new, old):
        """Largest summary difference, skipping b where it is not computed."""
        return float(np.nanmax(np.abs(new-old)))

    mala_audit = dict(proposed=0, accepted=0, gradient_seconds=0., prior_boundary_components=0)

    def morphology_gradient(state, state_kinds, state_summary, valid):
        """Deterministic endpoint drift; invalid proposals will be rejected.

        Invalid rows use current retained values solely to keep batch grouping
        fixed. They contribute no reverse correction or acceptance probability.
        At g=0 missing b is replaced by zero, which leaves the kernel unchanged.
        """
        began = time.monotonic()
        safe = np.where(valid[:, None], state, x)
        summaries_safe = np.where(valid[:, None], state_summary[:, 0], summaries[:, 0]).copy()
        if lazy_b:
            summaries_safe[:, 8] = 0.
        _, derivatives = gradient_evaluator.evaluate(summaries_safe, observed, g0)
        gradient = np.empty((len(state), 3))
        gradient[:, 0] = derivatives[:, 3]+(.4*np.log(10.)*derivatives[:, 5:8]
            *(-np.expm1(-summaries_safe[:, 5:8]))).sum(1)
        gradient[:, 1] = derivatives[:, 4]*summaries_safe[:, 4]
        span = galaxy.p['sersic_n_max']-galaxy.p['sersic_n_min']
        fraction = (summaries_safe[:, 2]-galaxy.p['sersic_n_min'])/span
        gradient[:, 2] = derivatives[:, 2]*span*fraction*(1-fraction)
        gradient *= beta[:, None]
        for j, column in enumerate((1, 6, 7)):
            up, down = safe.copy(), safe.copy()
            up[:, column] += 1e-4
            down[:, column] -= 1e-4
            lp, _, _ = target_prior(up, state_kinds)
            lm, _, _ = target_prior(down, state_kinds)
            finite = np.isfinite(lp) & np.isfinite(lm)
            gradient[finite, j] += (lp[finite]-lm[finite])/2e-4
            mala_audit['prior_boundary_components'] += int((~finite & valid).sum())
        mala_audit['gradient_seconds'] += time.monotonic()-began
        return gradient

    def estimate(data, valid):
        values = np.zeros((len(owner), m, 9))
        logk = np.full((len(owner), m), -np.inf)
        if np.any(valid):
            target = {k: np.repeat(v[valid], m, axis=0) for k, v in data.items()}
            scene, _ = prior.sample_given_targets(target, int(rng.integers(0, 2**63)))
            zero, ptr, features = scene_model_inputs(scene, prior, angular_distance, conditions,
                response_radius_arcsec=10., response_neighbours=20)
            response = CatalogueDiskResponse(ptr, features, model)
            selected = np.repeat(valid, m)
            summary = compress_scenes(zero, response, aux[selected])
            logk[valid] = likelihood.log_kernel(summary, repeated.loc[selected], *g0).reshape(-1, m)
            values[valid] = summary.values.reshape(-1, m, 9)
        return logsumexp(logk, axis=1)-np.log(m), values, logk

    # Only nodes 0.002 from the sampling shear are reliable for bright galaxies.
    profile_nodes = []
    for axis in (0, 1):
        for delta in (-.002, .002):
            node = list(g0)
            node[axis] = round(node[axis]+delta, 12)
            profile_nodes.append((f'g{axis+1}_{node[axis]:g}', node))

    def comoving_ratios(x, kinds, logprior, kernel, scene, nodes=profile_nodes):
        """log[p(x') K(x'; node) |dx'/dx| / (p(x) K(x; g0))] per cold state, x' = coords.comove(x, g0, node).

        Same retained scene (common random numbers); b changes only through
        the target semimajor axis and is recomputed. Off-support x' give -inf.
        """
        n = len(x)
        ratios = np.full((n, len(nodes)), -np.inf)
        for k, (_, node) in enumerate(nodes):
            moved, logj, valid = coords.comove(x, g0, node)
            newprior, newdata, _ = target_prior(moved, kinds)
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

    if persistent:
        scene = GalSBISceneBatch(data, scene.neighbours, scene.indptr, scene.offset_arcsec)
        loglike, summaries, kernels = evaluate_scene(scene, np.ones(len(owner), dtype=bool), with_b=True)
        reference_kernels = np.asarray(reference_kernels)
        replay_error = float(np.max(np.abs(loglike-reference_kernels)))
        save(args.output/'initial_replay.json', dict(max_abs_error=replay_error, tolerance=1e-6))
        if replay_error > 1e-6:
            raise ValueError(f'initial scene replay failed: {replay_error}')
    else:
        loglike, summaries, kernels = estimate(data, np.ones(len(owner), dtype=bool))
    if not np.isfinite(loglike).all():
        raise ValueError('initial likelihood estimates must be finite; inspect support')
    np.savez_compressed(args.output/'initial_state.npz', target_coordinates=x, summaries=summaries,
                        log_prior=logprior, log_kernel=loglike, type_probability=type_probability,
                        neighbour_counts=scene.counts if persistent else np.empty(0, dtype=int))
    scale_owner = np.arange(len(owner)) if args.scale_adaptation == 'per-chain' else owner
    log_scales = np.zeros((int(scale_owner.max())+1, 3))
    blocks = tuple(protocol['proposal_blocks'][name] for name in ('photometry_size', 'shape', 'structure'))
    scales = np.array(protocol['base_scales'])
    morphology_factor = np.broadcast_to(np.diag(scales[blocks[0]]),
        (len(owner), len(blocks[0]), len(blocks[0]))).copy()
    recent = deque(maxlen=protocol['morphology_covariance']['window']) if args.joint_morphology else None
    swap_attempts, swap_accepts = np.zeros((2, max(levels-1, 1))), np.zeros((2, max(levels-1, 1)))
    cold_exchanges = np.zeros(ncold, dtype=int)
    trace, type_trace, summary_trace, kernel_trace, accepted_trace, block_trace = [], [], [], [], [], []
    comoving_trace, comoving_identity_replay = [], None
    type_probability_trace = []
    count_trace, refresh_trace = [], []
    joint_environment_trace = []
    logprior_trace, sed_cross_trace = [], []
    sed_audit = dict(valid_proposals=0, maximum_summary_change=0., maximum_log_kernel_change=0.)
    sed_switches = np.zeros(len(owner), dtype=int)
    invariant_audit = dict(likelihood_calls_skipped=0, target_prior_calls_skipped=0,
                           checked_proposals=0, maximum_summary_change=0., maximum_log_kernel_change=0.)
    invariant_visits = {2: 0, 5: 0}
    exchange_audit = dict(attempts=np.zeros((2, levels, ncold), dtype=int),
                          accepts=np.zeros((2, levels, ncold), dtype=int),
                          large_magnitude_accepts=np.zeros((2, levels, ncold), dtype=int))
    transport = None
    transport_trace, transport_steps, transport_states = [], [], []
    # Without a saved chain, per-state histories are kept only at the profile draws (every profile-stride step
    # after warm-up): the profiles use exactly these states, and the diagnostics describe them. Memory then
    # no longer grows with every step (2026-10-14: ~5 GB per process at 16384 steps).
    first, stride = (0, 1) if args.no_chain else (args.warmup, args.profile_stride)
    sampling_started = time.monotonic()
    last_progress = time.monotonic()
    for step in range(args.steps):
        stored = not args.no_chain or (step >= args.warmup and (step-args.warmup) % args.profile_stride == 0)
        # State-independent mixture: local symmetric steps and an optional
        # SED independence step with an explicit reverse/forward correction.
        block = int(rng.choice(len(protocol['proposal_block_probabilities']),
                               p=protocol['proposal_block_probabilities']))
        proposal, newkinds = x.copy(), kinds.copy()
        log_hastings = np.zeros(len(owner))
        active_proposal = np.ones(len(owner), dtype=bool)
        exchange_move = bool(args.scene_exchange_probability and block == 6)
        ensemble_move = args.ensemble_morphology and block == 6
        joint_environment = bool(args.joint_environment and block == 0
                                 and rng.random() < protocol['joint_environment']['probability_given_morphology'])
        mala_move = bool(args.mala_morphology_probability and block == 0
                         and rng.random() < args.mala_morphology_probability)
        if block < 3:
            columns = blocks[block]
            if args.joint_morphology and block == 0:
                multiplier = np.exp(log_scales[:, 0])*(protocol['joint_environment']['morphology_scale']
                                                      if joint_environment else 1.)
                if args.narrow_step_probability:
                    multiplier = multiplier*np.where(rng.random(len(owner)) < args.narrow_step_probability, .25, 1.)
                if mala_move:
                    mala_factor = morphology_factor*multiplier[:, None, None]
                    drift = morphology_gradient(x, kinds, summaries, np.ones(len(x), dtype=bool))
                    mala_forward = bounded_langevin_mean(x[:, columns], drift, mala_factor,
                                                        max_drift=args.mala_max_drift)
                    proposal[:, columns] = mala_forward+gaussian_block_delta(morphology_factor, multiplier, rng)
                else:
                    proposal[:, columns] += gaussian_block_delta(morphology_factor, multiplier, rng)
            else:
                proposal[:, columns] += (rng.normal(size=(len(owner), len(columns)))*scales[columns]
                                         *np.exp(log_scales[scale_owner, block, None]))
            proposal[:, 9] %= 2*np.pi
        elif block == 3:
            newkinds = 1-kinds
        elif block == 5:
            proposal, log_hastings = coords.propose_sed(x, rng)
        elif ensemble_move:
            proposal, log_hastings, active_proposal = complementary_stretch_proposal(
                x, [1, 6, 7], nchains, rng)
        elif exchange_move:
            exchange_scene, exchange_ratio, active_proposal = propose_native_exchange(
                galaxy, scene, type_probability, rng, radius_scale=args.scene_exchange_radius)
            proposal = coords.encode(exchange_scene.targets)
        environment_only = block == 4 or (block == 6 and not ensemble_move and not exchange_move)
        invariant_likelihood = args.reuse_invariant_work and block in (2, 5)
        if args.reuse_invariant_work and environment_only:
            newprior, newdata, new_type_probability = logprior, scene.targets, type_probability
            invariant_audit['target_prior_calls_skipped'] += 1
        else:
            newprior, newdata, new_type_probability = target_prior(proposal, newkinds)
        if exchange_move:
            log_hastings[:] = -np.inf
            valid_exchange = np.isfinite(newprior) & np.isfinite(exchange_ratio)
            log_hastings[valid_exchange] = (exchange_ratio[valid_exchange]
                - (newprior[valid_exchange]-logprior[valid_exchange]))
        fraction = 0.
        if persistent:
            newscene = GalSBISceneBatch(newdata, scene.neighbours, scene.indptr, scene.offset_arcsec)
            if exchange_move:
                newscene = GalSBISceneBatch(newdata, exchange_scene.neighbours,
                                           scene.indptr, scene.offset_arcsec)
            if block == 4 or joint_environment:
                fraction = 1. if joint_environment or rng.random() < .1 else args.refresh_fraction
                newscene = prior.refresh(newscene, fraction, int(rng.integers(0, 2**63)))
            elif block == 6 and not ensemble_move and not exchange_move:
                newscene = prior.move_positions(newscene, args.position_scale, int(rng.integers(0, 2**63)))
            if invariant_likelihood:
                # At fixed m_r, angular circularized radius, n, a, phase and
                # environment, z/SED affect only the native target prior.
                # Off-support proposals still have -inf prior and are rejected.
                newlike, newsummary, newkernel = loglike, summaries, kernels
                check = invariant_visits[block] % 1024 == 0
                invariant_visits[block] += 1
                if check:
                    audit_like, audit_summary, _ = evaluate_scene(newscene, np.isfinite(newprior), with_b=True)
                    valid = np.isfinite(newprior)
                    invariant_audit['checked_proposals'] += int(valid.sum())
                    if valid.any():
                        invariant_audit['maximum_summary_change'] = max(
                            invariant_audit['maximum_summary_change'],
                            change(audit_summary[valid], summaries[valid]))
                        invariant_audit['maximum_log_kernel_change'] = max(
                            invariant_audit['maximum_log_kernel_change'],
                            float(np.max(np.abs(audit_like[valid]-loglike[valid]))))
                    if (invariant_audit['maximum_summary_change'] > 1e-10
                            or invariant_audit['maximum_log_kernel_change'] > 1e-6):
                        save(args.output/'invariant_replay_failure.json', invariant_audit)
                        raise ValueError('invariant cache disagrees with full likelihood replay')
                else:
                    invariant_audit['likelihood_calls_skipped'] += 1
            else:
                newlike, newsummary, newkernel = evaluate_scene(newscene, np.isfinite(newprior))
        else:
            newlike, newsummary, newkernel = estimate(newdata, np.isfinite(newprior))
        sed_cross = np.zeros(len(owner), dtype=bool)
        if block == 5:
            valid = np.isfinite(newprior)
            sed_cross = valid & ((type_probability[:, 0] >= .5) != (new_type_probability[:, 0] >= .5))
            if valid.any() and (not invariant_likelihood or check):
                checked_summary = audit_summary if invariant_likelihood else newsummary
                checked_like = audit_like if invariant_likelihood else newlike
                sed_audit['valid_proposals'] += int(valid.sum())
                sed_audit['maximum_summary_change'] = max(sed_audit['maximum_summary_change'],
                    change(checked_summary[valid], summaries[valid]))
                sed_audit['maximum_log_kernel_change'] = max(sed_audit['maximum_log_kernel_change'],
                    float(np.max(np.abs(checked_like[valid]-loglike[valid]))))
                if (sed_audit['maximum_summary_change'] > protocol['sed_replay_tolerances']['summary_absolute']
                        or sed_audit['maximum_log_kernel_change'] > protocol['sed_replay_tolerances']['log_kernel_absolute']):
                    save(args.output/'sed_replay_failure.json', sed_audit)
                    raise ValueError('SED jump changed fixed physical summaries or likelihood; inspect replay')
        if mala_move:
            valid_mala = np.isfinite(newprior) & np.isfinite(newlike)
            reverse_drift = morphology_gradient(proposal, newkinds, newsummary, valid_mala)
            safe_proposal = np.where(valid_mala[:, None], proposal[:, blocks[0]], x[:, blocks[0]])
            reverse_mean = bounded_langevin_mean(safe_proposal, reverse_drift, mala_factor,
                                                max_drift=args.mala_max_drift)
            correction = gaussian_log_hastings(x[:, blocks[0]], safe_proposal, mala_forward,
                                               reverse_mean, mala_factor)
            log_hastings[valid_mala] += correction[valid_mala]
        accepted = metropolis_accept(logprior+beta*loglike, newprior+beta*newlike+log_hastings,
                                     rng.uniform(np.finfo(float).tiny, 1., len(owner)))
        accepted &= active_proposal
        if mala_move:
            mala_audit['proposed'] += len(accepted)
            mala_audit['accepted'] += int(accepted.sum())
        if exchange_move:
            exchange_audit['attempts'][int(step >= args.warmup)] += active_proposal.reshape(levels, ncold)
            exchange_audit['accepts'][int(step >= args.warmup)] += accepted.reshape(levels, ncold)
            exchange_audit['large_magnitude_accepts'][int(step >= args.warmup)] += (accepted & (np.abs(proposal[:, 1]-x[:, 1]) > 3)).reshape(levels, ncold)
        sed_switches += accepted & sed_cross
        sed_cross_trace.append(sed_cross[cold])
        x, kinds, logprior, loglike, summaries, kernels, type_probability = [retained(old, new, accepted) for old, new in
            ((x, proposal), (kinds, newkinds), (logprior, newprior), (loglike, newlike),
             (summaries, newsummary), (kernels, newkernel), (type_probability, new_type_probability))]
        if persistent:
            if args.reuse_invariant_work and not (environment_only or joint_environment or exchange_move):
                # Replacing target marks does not replace or copy neighbours.
                scene = GalSBISceneBatch(
                    {key: retained(value, newscene.targets[key], accepted) for key, value in scene.targets.items()},
                    scene.neighbours, scene.indptr, scene.offset_arcsec)
            else:
                scene = scene.retained(newscene, accepted)
            if stored:
                count_trace.append(scene.counts[cold])
        refresh_trace.append(fraction)
        joint_environment_trace.append(joint_environment)
        if step < args.warmup and block < 3 and not joint_environment:
            rate = accepted if args.scale_adaptation == 'per-chain' else accepted.reshape(nobs, nchains).mean(1)
            log_scales[:, block] += .05*(rate-.3)
        if levels > 1:
            parity = step % 2
            perm, exchanged = adjacent_swap_permutation(loglike, betas, ncold, parity,
                rng.uniform(np.finfo(float).tiny, 1., (levels-1, ncold)))
            x, kinds, logprior, loglike, summaries, kernels, type_probability = [value[perm] for value in
                (x, kinds, logprior, loglike, summaries, kernels, type_probability)]
            scene = scene.take(perm)
            stage = int(step >= args.warmup)
            swap_attempts[stage, parity::2] += ncold
            swap_accepts[stage] += exchanged.sum(1)
            if stage:
                cold_exchanges += exchanged[0]
                if args.tempering_diagnostics:
                    if transport is None:
                        transport = ReplicaTransport(levels, ncold, 4)
                    transport_values = np.column_stack((x[:, [1, 6, 7]], summaries[:, 0, 7]))
                    transport.update(perm, transport_values)
                    if (step-args.warmup) % args.profile_stride == 0:
                        transport_steps.append(step)
                        transport_trace.append(transport.identity.copy())
                        transport_states.append(transport_values.copy())
        if stored:
            trace.append(x[cold].copy())
        if args.joint_morphology:
            recent.append(x.copy())
            tuning = protocol['morphology_covariance']
            freeze = tuning.get('freeze_step', args.warmup)
            if step < freeze and ((step+1) % tuning['update_interval'] == 0
                                  or (args.pooled_morphology and step+1 == freeze)):
                history = np.array(recent)[:, :, blocks[0]]
                if args.pooled_morphology:
                    morphology_factor = pooled_block_cholesky(history, scales[blocks[0]], nchains,
                                                             shrinkage=tuning['shrinkage'])
                else:
                    morphology_factor = empirical_block_cholesky(history, scales[blocks[0]],
                        shrinkage=tuning['shrinkage'], circular_indices=tuple(tuning['circular_indices']))
        if lazy_b and step+1 >= args.warmup:
            fill_b(summaries, scene)
        if stored:
            type_trace.append(kinds[cold].copy())
            type_probability_trace.append(type_probability[cold].copy())
            logprior_trace.append(logprior[cold].copy())
            summary_trace.append(summaries[cold].copy())
            kernel_trace.append(kernels[cold].copy())
        if args.comoving_profiles and step >= args.warmup and (step-args.warmup) % args.profile_stride == 0:
            state = (x[cold], kinds[cold], logprior[cold], kernels[cold, 0], scene.take(np.arange(ncold)))
            if comoving_identity_replay is None:
                # node = g0 must reproduce the stored prior and kernel exactly
                comoving_identity_replay = float(np.max(np.abs(comoving_ratios(*state, nodes=[('g0', g0)]))))
                if not comoving_identity_replay < 1e-6:
                    raise ValueError(f'co-moving identity replay failed: {comoving_identity_replay}')
            comoving_trace.append(comoving_ratios(*state))
        accepted_trace.append(accepted[cold])
        block_trace.append(block)
        if (step+1) % 128 == 0 or time.monotonic()-last_progress > 45:
            report = dict(step=step+1, steps=args.steps, elapsed_seconds=time.monotonic()-started,
                acceptance_last128=np.mean(accepted_trace[-128:], axis=0).tolist(),
                proposal_scale_multipliers=np.exp(log_scales).tolist(),
                accepted_sed_region_jumps=sed_switches.tolist() if args.sed_jump else None,
                swap_acceptance=(swap_accepts.sum(0)/np.maximum(swap_attempts.sum(0), 1)).tolist()
                if levels > 1 else None)
            save(args.output/'progress.json', report)
            print('LOCAL_PROGRESS', report, flush=True)
            last_progress = time.monotonic()
    sampling_finished = time.monotonic()
    if args.reuse_invariant_work:
        save(args.output/'invariant_replay.json', invariant_audit)
    trace, types = np.array(trace), np.array(type_trace)
    type_probability_trace = np.array(type_probability_trace)
    summary_trace, kernel_trace = np.array(summary_trace), np.array(kernel_trace)
    accepted_trace, block_trace = np.array(accepted_trace), np.array(block_trace)
    if not args.no_chain:
        np.savez_compressed(args.output/'chain.npz', target_coordinates=trace, types=types,
            type_probability=type_probability_trace,
            summaries=summary_trace, log_kernel=kernel_trace, accepted=accepted_trace, block=block_trace,
            neighbour_counts=np.array(count_trace), refresh_fraction=np.array(refresh_trace),
            joint_environment=np.array(joint_environment_trace),
            log_prior=np.array(logprior_trace), sed_cross_proposed=np.array(sed_cross_trace))
    if persistent:
        scene = scene.take(np.arange(ncold))
        np.savez_compressed(args.output/'final_scenes.npz',
            **{'target_'+k: v for k, v in scene.targets.items()},
            **{'neighbour_'+k: v for k, v in scene.neighbours.items()},
            indptr=scene.indptr, offset_arcsec=scene.offset_arcsec)
    histories_saved = time.monotonic()
    comoving_trace = np.array(comoving_trace)
    reports, profile_samples = [], {}
    for i in range(nobs):
        rows = slice(i*nchains, (i+1)*nchains)
        retained_summary = summary_trace[first:, rows]
        retained_kernel = kernel_trace[first:, rows]
        weights = np.exp(retained_kernel-logsumexp(retained_kernel, axis=2, keepdims=True))
        # Environment draws are NOT equally weighted posterior scenes. Under
        # the extended target, K-weighted sums give posterior scene moments.
        moments = (weights[..., None]*retained_summary).sum(2)
        checks = {name: diagnostics(moments[:, :, col].T) for name, col in
                  (('e1', 0), ('e2', 1), ('m_r', 3), ('Re_circularized', 4), ('crowding_050', 5), ('b', 8))}
        checks['type'] = diagnostics(type_probability_trace[first:, rows, 0].T
                                    if args.marginalize_type else types[first:, rows].T)
        checks['log_z'] = diagnostics(trace[first:, rows, 0].T)
        if persistent:
            checks['neighbour_count'] = diagnostics(np.array(count_trace)[first:, rows].T)
            for col in (2, 3, 4, 5, 7, 8):
                checks[coords.names[col]] = diagnostics(trace[first:, rows, col].T)
            for col in (6, 7):
                checks[f'crowding_{col}'] = diagnostics(moments[:, :, col].T)
        profile = []
        flat = retained_summary[::stride].reshape(-1, 9)
        observed_i = observations.iloc[[i]]
        summary = SummaryBatch(flat, np.broadcast_to(observed_i.iloc[0, 2:4].to_numpy(), (len(flat), 2)))
        # Extended-target bridge is sum K_g / sum K_g0 for each retained
        # bank, then average over chain states; not an unweighted bank mean.
        for k, (name, node) in enumerate(profile_nodes):
            if args.comoving_profiles:
                ratios = comoving_trace[:, rows, k]
            else:
                shifted = likelihood.log_kernel(summary, observed_i, *node).reshape(retained_kernel[::stride].shape)
                ratios = logsumexp(shifted, axis=2)-logsumexp(retained_kernel[::stride], axis=2)
            profile_samples[f'observation_{protocol["observations"][i]}_{name}'] = ratios
            values = logsumexp(ratios, axis=0)-np.log(len(ratios))
            checks[f'log_ratio_{name}'] = diagnostics(ratios.T)
            profile.append(dict(g1=node[0], g2=node[1], log_numerator_ratio_per_chain=values.tolist()))
        by_block = {}
        for block in range(len(protocol['proposal_block_probabilities'])):
            select = block_trace[args.warmup:] == block
            by_block[str(block)] = accepted_trace[args.warmup:][select, rows].mean(0).tolist() if select.any() else None
        passing = all(d['rhat'] is not None and d['rhat'] < 1.01 and d['ess_bulk'] > 400
                      for d in checks.values() if not d['constant'])
        # A constant type across all chains does not establish the missing
        # type's posterior probability; flag this explicitly as unresolved.
        reports.append(dict(reference_observation=protocol['observations'][i],
            measured_magnitude=float(30-2.5*np.log10(observed_i.iloc[0, 3])),
            diagnostics=checks, acceptance_by_block_per_chain=by_block,
            type_switches_per_chain=np.sum(np.diff(types[first:, rows], axis=0) != 0, axis=0).tolist(),
            red_probability_mean_per_chain=type_probability_trace[first:, rows, 0].mean(0).tolist(),
            sed_region_switches_per_chain=np.sum(np.diff(type_probability_trace[first:, rows, 0] >= .5,
                                                          axis=0) != 0, axis=0).tolist(),
            sed_cross_proposals_per_chain=np.sum(np.array(sed_cross_trace)[args.warmup:, rows], axis=0).tolist(),
            sed_cross_accepts_per_chain=np.sum(np.array(sed_cross_trace)[args.warmup:, rows]
                                              & accepted_trace[args.warmup:, rows], axis=0).tolist(),
            environment_acceptance={name: dict(attempts_per_chain=int(select.sum()),
                acceptance_per_chain=accepted_trace[args.warmup:][select, rows].mean(0).tolist()
                if select.any() else None) for name, select in
                (('partial', (np.array(refresh_trace[args.warmup:]) > 0) & (np.array(refresh_trace[args.warmup:]) < 1)),
                 ('full', (np.array(refresh_trace[args.warmup:]) == 1) & (block_trace[args.warmup:] == 4)),
                 ('joint_target_full', np.array(joint_environment_trace[args.warmup:]))) } if persistent else None,
            constant_type_requires_additional_check=checks['type']['constant'] and not args.marginalize_type,
            local_numerator_profiles=profile,
            exploration_checks_pass=passing and (args.marginalize_type or not checks['type']['constant'])
                                    and not args.ensemble_morphology))
    np.savez_compressed(args.output/'local_profiles.npz', **profile_samples)
    if transport is not None:
        np.savez_compressed(args.output/'tempering_trace.npz', steps=transport_steps,
                            replica_identity=transport_trace, states=transport_states)
    result = dict(status='completed_diagnostic', observations=reports,
        elapsed_seconds=time.monotonic()-started, frozen_scale_multipliers=np.exp(log_scales).tolist(),
        timing_seconds=dict(setup=sampling_started-started, sampling=sampling_finished-sampling_started,
            history_output=histories_saved-sampling_finished, profiles_and_diagnostics=time.monotonic()-histories_saved),
        coupled_walkers=bool(args.ensemble_morphology), comoving_identity_replay_max_abs=comoving_identity_replay,
        frozen_morphology_cholesky=morphology_factor.tolist() if args.joint_morphology else None,
        protocol_sha256=digest(args.output/'protocol.json'),
        all_exploration_checks_pass=all(r['exploration_checks_pass'] for r in reports),
        sed_replay=sed_audit if args.sed_jump else None,
        mala_morphology=mala_audit if args.mala_morphology_probability else None,
        scene_exchange={k: v.tolist() for k, v in exchange_audit.items()} if args.scene_exchange_probability else None,
        replica_transport=transport.report() if transport is not None else None,
        tempering=dict(betas=betas.tolist(),
            swap_acceptance_warmup=(swap_accepts[0]/np.maximum(swap_attempts[0], 1)).tolist(),
            swap_acceptance_post_warmup=(swap_accepts[1]/np.maximum(swap_attempts[1], 1)).tolist(),
            posterior_level_exchanges_per_chain=cold_exchanges.tolist()) if levels > 1 else None,
        limitations=['Short bright/faint stress panel; no measured-selection normalization or shear estimate.',
            'Native target analytic prior and numerical neighbour generator retain their audited approximation difference.',
            'No learned compressed density; target redshift/SED latents are retained in ten fixed coordinates.',
            ('Persistent scenes use deterministic likelihoods and a prior-reversible environment update.' if persistent
             else 'Pseudo-marginal averages are unbiased on likelihood scale for the implemented environment generator.'),
            'Finite-chain mixing, rare environmental modes and type exploration still require validation.',
            'No production default change, 0.3% claim or automatic large run.'])
    if not persistent:
        final_data, _ = coords.decode(x, kinds)
        result['environment_noise_audit'] = environment_noise_audit(final_data, prior, angular_distance, conditions,
            model, likelihood, observed, seed=args.seed+10000, stored_log_estimate=loglike, shear=g0)
    save(args.output/'result.json', result)
    print('LOCAL_COMPLETE', result['all_exploration_checks_pass'], result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
