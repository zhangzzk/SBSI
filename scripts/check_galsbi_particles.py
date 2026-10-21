"""Replicated fixed-schedule annealed-particle pilot on a small mock panel.

Run through Slurm. Each invocation is one independent population per object.
Resampled particles are correlated; repeated invocations, not nominal weight
ESS or Rhat across particles, provide the uncertainty comparison.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

from check_galsbi_continuous_prior import build_prior, digest, save
from sbsi.annealed_particles import effective_ancestors
from sbsi.particle_populations import (EXTINCTION_SCHEMA, ParticlePopulations,
    masked_rows, survivor_diagnostics)
from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
from sbsi.galsbi_particle_target import GalSBIParticleTarget
from sbsi.galsbi_prior import ExactMemo
from sbsi.galsbi_scene_prior import GalSBIScenePrior
from sbsi.galsbi_summary import SummaryLikelihood
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
from sbsi.summary_gradients import SummaryGradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observation-indices', type=int, nargs='+', default=[58165, 0, 1, 8])
    parser.add_argument('--particles', type=int, default=256)
    parser.add_argument('--stages', type=int, default=64)
    parser.add_argument('--moves', type=int, default=32)
    parser.add_argument('--power', type=float, default=2.)
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--faint-shift', type=float)
    parser.add_argument('--proposal-family', choices=['gaussian', 'student_t'])
    parser.add_argument('--proposal-df', type=float)
    parser.add_argument('--initial-proposal', type=Path,
                        help='Frozen independent-pilot proposal; owns the complete initial law')
    parser.add_argument('--informed-environment', type=Path,
                        help='Optional frozen proposal artifact for one-point initial environment augmentation')
    parser.add_argument('--environment-mixture-probability', type=float,
                        help='Fixed augmentation probability, default0.5; requires --informed-environment')
    parser.add_argument('--physical-geometry', type=Path)
    parser.add_argument('--relocate-exchanges', action='store_true')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--shear', type=float, nargs=2, default=[0., 0.])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--density-batch-size', type=int, default=4096)
    args = parser.parse_args()
    if args.environment_mixture_probability is not None and not args.informed_environment:
        parser.error('--environment-mixture-probability requires --informed-environment')
    environment_probability = .5 if args.environment_mixture_probability is None else args.environment_mixture_probability
    if args.informed_environment and (not np.isfinite(environment_probability) or not 0 < environment_probability < 1):
        parser.error('--environment-mixture-probability must lie strictly between zero and one')
    if args.initial_proposal and any(value is not None for value in
            (args.faint_shift, args.proposal_family, args.proposal_df)):
        parser.error('--initial-proposal cannot be combined with --faint-shift, --proposal-family, or --proposal-df')
    args.faint_shift = 5. if args.faint_shift is None else args.faint_shift
    args.proposal_family = 'gaussian' if args.proposal_family is None else args.proposal_family
    args.proposal_df = 5. if args.proposal_df is None else args.proposal_df
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('fresh output directory required')
    if (args.particles < 4 or args.stages < 2 or args.moves < 1 or not np.isfinite(args.power)
            or args.power <= 0 or not np.isfinite(args.scale) or args.scale <= 0
            or args.seed < 0 or len(set(args.observation_indices)) != len(args.observation_indices)):
        raise ValueError('valid positive effort/scale and distinct observation indices required')
    reference = json.loads((args.reference/'protocol.json').read_text())
    receipt = json.loads((args.reference/'result.json').read_text())
    if (not receipt['preparation_checks_pass']
            or receipt['observations_sha256'] != digest(args.reference/'observations.parquet')):
        raise ValueError('validated unchanged population reference required')
    observations = pd.read_parquet(args.reference/'observations.parquet').iloc[args.observation_indices].reset_index(drop=True)
    nobs, particles = len(observations), args.particles
    owner = np.repeat(np.arange(nobs), particles)
    betas = np.linspace(0., 1., args.stages+1)**args.power
    # Fixed sequence contains local physical/nuisance, structural and global moves.
    # Its scales and schedule do not adapt to the same particle population.
    blocks = ['physical', 'physical', 'nuisance', 'physical', 'environment',
              'physical', 'sed', 'physical', 'positions', 'physical',
              'exchange', 'physical', 'nuisance', 'physical', 'independence', 'physical']
    if args.relocate_exchanges:
        blocks = ['exchange_relocate' if b == 'exchange' else b for b in blocks]
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path('sbsi/annealed_particles.py'),
        Path('sbsi/galsbi_particle_proposal.py'), Path('sbsi/galsbi_particle_target.py'),
        Path('sbsi/scene_exchange.py'), Path('sbsi/galsbi_target_coordinates.py'),
        Path('sbsi/galsbi_summary.py'), Path('sbsi/summary_gradients.py'), Path('configs/models_v3_9_like.json'),
        Path('sbsi/particle_populations.py')]
    protocol = dict(algorithm='fixed-schedule resample-move annealing from normalized measurement proposal',
        particles=particles, stages=args.stages, moves=args.moves, betas=betas.tolist(), scale=args.scale,
        blocks=blocks, seed=args.seed, observations=args.observation_indices, fixed_shear=args.shear,
        reference=str(args.reference.resolve()), reference_protocol_sha256=digest(args.reference/'protocol.json'),
        observation_sha256=receipt['observations_sha256'], density_batch_size=args.density_batch_size,
        bridge='q_environment(E) * q0(x)**(1-beta) * (p_analytic(x)*K(y|x,E))**beta',
        selection='none in fixed-y posterior; no population shear fit', extinction_schema=EXTINCTION_SCHEMA,
        exactness='Known q0; off-prior-support particles retain zero weight; fixed schedules/kernels; '
                  'native target and numerical environment measures unchanged. Finite-particle ratios are not exact posterior expectations.',
        implementation_sha256={str(p.resolve()):digest(p) for p in source_paths})
    geometry = None
    if args.physical_geometry:
        from sbsi.particle_geometry import FrozenParticleGeometry
        geometry = FrozenParticleGeometry.load(args.physical_geometry)
        geometry.validate_production(args.seed, args.observation_indices)
        protocol['physical_geometry'] = dict(path=str(args.physical_geometry.resolve()),
            sha256=digest(args.physical_geometry), metadata=geometry.metadata)
        protocol['implementation_sha256'][str(Path('sbsi/particle_geometry.py').resolve())] = digest('sbsi/particle_geometry.py')
    initial = None
    if args.initial_proposal:
        from sbsi.particle_initial_proposal import FrozenParticleProposal
        initial = FrozenParticleProposal.load(args.initial_proposal)
        initial.validate_production(args.seed, protocol)
        protocol['initial_proposal'] = dict(path=str(args.initial_proposal.resolve()),
            sha256=digest(args.initial_proposal), metadata=initial.metadata,
            effective_law='artifact-owned normalized Student-t/von-Mises mixture')
        protocol['implementation_sha256'][str(Path('sbsi/particle_initial_proposal.py').resolve())] = digest('sbsi/particle_initial_proposal.py')
    environment_source = None
    if args.informed_environment:
        from sbsi.particle_initial_proposal import FrozenParticleProposal
        environment_source = FrozenParticleProposal.load(args.informed_environment)
        # Validate the pilot's scientific target identity before recording the
        # different initial bridge; no existing identity gate is bypassed.
        environment_source.validate_production(args.seed, protocol)
        protocol['informed_environment'] = dict(path=str(args.informed_environment.resolve()),
            sha256=digest(args.informed_environment), source_metadata=environment_source.metadata,
            source_pilot_validation_bridge=protocol['bridge'])
        protocol['bridge'] += ' * R_frozen_environment(E)**(1-beta)'
        protocol['exactness'] += (' Optional initial environment uses the continuous numerical interpolation '
            'density and retains proven off-support raw draws as zero-target initial rows; '
            'unknown numerical failures raise. Clamp atoms and finite-RNG exactness are not implemented.')
        for name in ('particle_initial_proposal.py', 'informed_environment.py', 'poisson_augmentation.py',
                     'numerical_native_intensity.py', 'galsbi_prior.py', 'galsbi_scene_prior.py'):
            path = Path('sbsi')/name
            protocol['implementation_sha256'][str(path.resolve())] = digest(path)
    save(args.output/'protocol.json', protocol)
    started = time.monotonic()
    torch.set_num_threads(min(2, len(os.sched_getaffinity(0))))
    protocol['runtime'] = dict(python=sys.version.split()[0], numpy=np.__version__,
        torch=str(torch.__version__), device=args.device,
        cuda_device_name=torch.cuda.get_device_name() if args.device == 'cuda' else None,
        torch_num_threads=torch.get_num_threads(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_node_list=os.environ.get('SLURM_JOB_NODELIST'))
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device=args.device)
    detector = load_detection_classifier(V39_LIKE, device=args.device)
    response_model = load_disk_response(V39_LIKE, device=args.device, nthread=torch.get_num_threads())
    likelihood = SummaryLikelihood(flow, detector, density_batch_size=args.density_batch_size)
    galaxy, cosmo, _ = build_prior()
    distance = ExactMemo(lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))
    prior = GalSBIScenePrior(galaxy)
    coordinates = GalSBITargetCoordinates(galaxy, distance)
    informed = None
    if environment_source is not None:
        from sbsi.informed_environment import InformedEnvironmentProposal
        informed = InformedEnvironmentProposal(prior, coordinates, environment_source,
            protocol['informed_environment']['sha256'], mixture_probability=environment_probability)
        protocol['informed_environment']['metadata'] = informed.metadata
    proposals = (list(initial.proposals) if initial is not None else [MeasurementParticleProposal.from_measurement(row.iloc[:4].to_numpy(),
        reference['conditions'], galaxy.p, args.shear, faint_magnitude_shift=args.faint_shift,
        component_family=args.proposal_family, student_df=args.proposal_df)
        for _, row in observations.iterrows()])
    protocol['proposals'] = [p.metadata for p in proposals]
    save(args.output/'protocol.json', protocol)
    target_options = dict(physical_geometry=geometry)
    if informed is not None:
        target_options['informed_environment'] = informed
    target = GalSBIParticleTarget(prior, coordinates, likelihood, response_model, distance,
        reference['conditions'], observations, proposals, owner, args.shear, **target_options)
    rng = np.random.default_rng(args.seed)
    initialization_started = time.monotonic()
    state = target.initialize(rng)
    initial_valid = np.isfinite(state.potential).reshape(nobs, particles).sum(1)
    initial_environment = None
    if informed is not None:
        payload = state.environment_payload
        # Preserve every raw draw before ordinary weighting/resampling removes
        # zero-target rows. Final surviving ancestry cannot reconstruct this.
        path = args.output/'initial_environment.npz'
        np.savez_compressed(path, **vars(payload), target_coordinates=state.x,
                            log_potential=state.potential)
        reasons = payload.classification.reshape(nobs, particles)
        initial_environment = dict(path=path.name, sha256=digest(path),
            original_particles_per_observation=particles,
            augmented=payload.augmented.reshape(nobs, particles).sum(1).tolist(),
            supported_environment=payload.valid.reshape(nobs, particles).sum(1).tolist(),
            zero_target_environment=(~payload.valid).reshape(nobs, particles).sum(1).tolist(),
            classifications=[{str(k):int(v) for k,v in zip(*np.unique(row,return_counts=True))} for row in reasons])
    populations = ParticlePopulations(target, nobs, particles)
    logz, labels = populations.logz, populations.labels
    history = []
    attempted = {b:np.zeros(nobs, int) for b in set(blocks)}
    accepted = {b:np.zeros(nobs, int) for b in set(blocks)}
    initialization_seconds = time.monotonic()-initialization_started
    sampling_started = time.monotonic()
    measured_magnitude = reference['conditions']['zero_point']-2.5*np.log10(observations.iloc[:, 3].to_numpy())
    for stage, beta in enumerate(betas[1:], 1):
        state, ess = populations.reweight_resample(state, beta-betas[stage-1], stage, rng)
        stage_attempted = {b:np.zeros(nobs, int) for b in attempted}
        stage_accepted = {b:np.zeros(nobs, int) for b in attempted}
        for move in range(args.moves):
            block = blocks[((stage-1)*args.moves+move) % len(blocks)]
            state, decisions, active = populations.mutate(state, beta, rng, block=block, scale=args.scale,
                check_reuse=(stage == 1 and move < len(blocks)))
            attempted[block] += active.reshape(nobs, particles).sum(1)
            accepted[block] += decisions.reshape(nobs, particles).sum(1)
            stage_attempted[block] += active.reshape(nobs, particles).sum(1)
            stage_accepted[block] += decisions.reshape(nobs, particles).sum(1)
        region = state.x[:, 1].reshape(nobs, particles)-measured_magnitude[:, None]
        entry = dict(stage=stage, beta=float(beta), pre_resampling_ess=ess, **populations.receipt(),
            initial_ancestor_ess=[float(effective_ancestors(v)) if alive else 0.
                                  for v,alive in zip(labels, populations.alive)],
            distinct_initial_ancestors=[int(len(np.unique(v))) if alive else 0
                                        for v,alive in zip(labels, populations.alive)],
            magnitude_quantiles=masked_rows(np.quantile(state.x[:, 1].reshape(nobs, particles), [.1, .5, .9], axis=1).T, populations.alive),
            target_within_two_magnitudes=masked_rows((np.abs(region) < 2).mean(1), populations.alive),
            target_over_three_magnitudes_fainter=masked_rows((region > 3).mean(1), populations.alive),
            acceptance={b:dict(attempted=stage_attempted[b].tolist(), accepted=stage_accepted[b].tolist())
                        for b in sorted(attempted)},
            elapsed_seconds=time.monotonic()-sampling_started)
        history.append(entry)
        save(args.output/'progress.json', dict(stages=history, requested=args.stages))
        if stage == 1 or stage % 8 == 0 or stage == args.stages:
            print('PARTICLE_STAGE', stage, float(beta), 'ESS', ess, 'ancestors',
                  entry['distinct_initial_ancestors'], 'seconds', round(entry['elapsed_seconds'], 2), flush=True)
    sampling_seconds = time.monotonic()-sampling_started
    def evaluate_scores(full, observed):
        # Called only after finite surviving-state likelihood replay passes.
        flow.model.double()
        for member in getattr(detector, 'members', (detector,)):
            member.model.double()
        return SummaryGradients(flow, detector, batch_size=4096).evaluate(
            full, observed, args.shear, wrt='shear')[1]
    full, score, replay_error = survivor_diagnostics(target, state, populations.alive,
        particles, likelihood, evaluate_scores)
    score = score.reshape(nobs, particles, 2)
    environment_arrays = {}
    if state.environment_payload is not None:
        environment_arrays = {'environment_'+name:value.reshape((nobs, particles)+value.shape[1:])
                              for name,value in vars(state.environment_payload).items()}
    np.savez_compressed(args.output/'particles.npz', coordinates=state.x.reshape(nobs, particles, 10),
        summaries=full.reshape(nobs, particles, 9), log_prior=state.log_prior.reshape(nobs, particles),
        log_kernel=state.log_kernel.reshape(nobs, particles), scores=score, ancestor_labels=labels,
        extinct=~populations.alive, log_normalizer=logz, **environment_arrays)
    np.savez_compressed(args.output/'scenes.npz', indptr=state.scene.indptr,
        offset_arcsec=state.scene.offset_arcsec,
        **{'target_'+k:v for k,v in state.scene.targets.items()},
        **{'neighbour_'+k:v for k,v in state.scene.neighbours.items()})
    score_mean = masked_rows(score.mean(1), populations.alive)
    save(args.output/'result.json', dict(status='completed' if populations.alive.all() else 'completed_with_extinction',
        observations=args.observation_indices,
        particles=particles, score_mean=score_mean,
        score_variance=masked_rows(score.var(1, ddof=1), populations.alive),
        **populations.receipt(), initial_valid=initial_valid.tolist(),
        **({} if initial_environment is None else {'informed_environment_initialization':initial_environment}),
        timing_seconds=dict(initialization=initialization_seconds, sampling=sampling_seconds,
                            total=time.monotonic()-started),
        acceptance={b:dict(attempted=attempted[b].tolist(), accepted=accepted[b].tolist()) for b in sorted(attempted)},
        final_stage=history[-1], audit=target.audit, final_state_replay_max=replay_error,
        protocol_sha256=digest(args.output/'protocol.json'), particles_sha256=digest(args.output/'particles.npz'),
        scenes_sha256=digest(args.output/'scenes.npz'),
        limitations=['Independent full runs are needed to assess uncertainty; particles share ancestry.',
            'Extinct observations retain zero normalizing mass and their seed; binary score/summary placeholders are undefined under the extinction mask.',
            'Initial-ancestor ESS measures genealogy, not post-mutation score ESS.',
            'F64 score diagnostic is evaluated on the unchanged F32 posterior target.',
            'No reference correctness, full-chain convergence or population shear claim.']))
    print('PARTICLE_READY', args.output, 'scores', score_mean, flush=True)


if __name__ == '__main__':
    main()
