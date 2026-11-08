"""Compare 1024/4096 saved particle rows without changing density batch shape.

Preparation only: schedule explicitly on one GPU after resource coordination.
This is a duplicated-state throughput benchmark, never an inference population
or evidence of improved effective sample size. No adaptive settings are fitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026102210)
    parser.add_argument('--require-gpu-model', default='V100')
    parser.add_argument('--check-inputs-only', action='store_true',
                        help='stdlib-only identity preflight; no model imports or GPU use')
    args = parser.parse_args()
    if args.repetitions < 2 or args.warmup < 1 or args.output.exists():
        raise ValueError('fresh output, >=2 repetitions and >=1 warmup required')
    args.source = args.source.resolve()
    protocol = json.loads((args.run/'protocol.json').read_text())
    result = json.loads((args.run/'result.json').read_text())
    if (result['status'] != 'completed' or protocol['particles'] != 256
            or len(protocol['observations']) != 4 or protocol['density_batch_size'] != 4096
            or protocol.get('initial_proposal') or protocol.get('relocate_exchanges')):
        raise ValueError('benchmark requires the completed original 4x256 v5 protocol')
    source_hashes = {}
    for old, expected in protocol['implementation_sha256'].items():
        parts = Path(old).parts
        indexes = [i for i, part in enumerate(parts) if part in ('scripts', 'sbsi', 'configs')]
        if not indexes:
            raise ValueError(f'unknown source identity path: {old}')
        path = args.source.joinpath(*parts[indexes[-1]:])
        if digest(path) != expected:
            raise ValueError(f'benchmark source differs from saved population: {path}')
        source_hashes[str(path)] = expected
    input_hashes = {str(args.run/name): digest(args.run/name)
                    for name in ('protocol.json','result.json','particles.npz','scenes.npz')}
    for name in ('protocol', 'particles', 'scenes'):
        suffix = 'npz' if name != 'protocol' else 'json'
        if digest(args.run/f'{name}.{suffix}') != result[f'{name}_sha256']:
            raise ValueError(f'changed saved {name}')
    if args.check_inputs_only:
        print('INPUT_IDENTITIES_PASS', args.source, args.run, flush=True)
        return
    # Explicitly pin this diagnostic to the immutable source requested by the
    # launcher. No editable-install or PYTHONPATH changes are made.
    sys.path[:0] = [str(args.source), str(args.source/'scripts')]
    import numpy as np
    import pandas as pd
    import torch
    import sbsi
    from check_galsbi_continuous_prior import build_prior
    from sbsi.galsbi_prior import ExactMemo
    from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
    from sbsi.galsbi_particle_target import GalSBIParticleTarget
    from sbsi.galsbi_scene_prior import GalSBISceneBatch, GalSBIScenePrior
    from sbsi.galsbi_summary import SummaryLikelihood
    from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
    from sbsi.measurement_model import load_measurement_model
    from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
    from sbsi.particle_geometry import FrozenParticleGeometry
    assert Path(sbsi.__file__).resolve().parent.parent == args.source
    if not torch.cuda.is_available():
        raise ValueError('scheduled GPU required; do not benchmark on the login node')
    gpu_name = torch.cuda.get_device_name()
    if args.require_gpu_model not in gpu_name:
        raise ValueError(f'expected {args.require_gpu_model}, got {gpu_name}')
    torch.set_num_threads(min(2, len(os.sched_getaffinity(0))))
    started = time.monotonic()
    reference = Path(protocol['reference'])
    reference_protocol = json.loads((reference/'protocol.json').read_text())
    if digest(reference/'observations.parquet') != protocol['observation_sha256']:
        raise ValueError('changed observed population')
    observations = pd.read_parquet(reference/'observations.parquet').iloc[
        protocol['observations']].reset_index(drop=True)
    with np.load(args.run/'particles.npz') as saved:
        x = saved['coordinates'].reshape(-1, 10).copy()
        saved_kernel = saved['log_kernel'].reshape(-1).copy()
        saved_prior = saved['log_prior'].reshape(-1).copy()
    with np.load(args.run/'scenes.npz') as saved:
        scene = GalSBISceneBatch(
            {k[7:]: saved[k].copy() for k in saved.files if k.startswith('target_')},
            {k[10:]: saved[k].copy() for k in saved.files if k.startswith('neighbour_')},
            saved['indptr'].copy(), saved['offset_arcsec'].copy())
    V39_LIKE.validate()
    flow = load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(V39_LIKE, device='cuda')
    response = load_disk_response(V39_LIKE, device='cuda', nthread=torch.get_num_threads())
    likelihood = SummaryLikelihood(flow, detector, density_batch_size=4096)
    galaxy, cosmo, _ = build_prior()
    distance = ExactMemo(lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))
    prior = GalSBIScenePrior(galaxy)
    coordinates = GalSBITargetCoordinates(galaxy, distance)
    proposals = [MeasurementParticleProposal(item['means'], item['scales'], item['weights'],
        component_family=item['component_family'], student_df=item.get('student_df') or 5.)
        for item in protocol['proposals']]
    geometry_receipt = protocol['physical_geometry']
    if digest(geometry_receipt['path']) != geometry_receipt['sha256']:
        raise ValueError('changed frozen geometry')
    geometry = FrozenParticleGeometry.load(geometry_receipt['path'])
    conditions = reference_protocol['conditions']
    base_owner = np.repeat(np.arange(4), 256)
    targets, states = {}, {}
    for factor in (1, 4):
        rows = np.tile(np.arange(1024), factor)
        owner = base_owner[rows]
        target = GalSBIParticleTarget(prior, coordinates, likelihood, response, distance,
            conditions, observations, proposals, owner, protocol['fixed_shear'], geometry)
        state = target.evaluate(x[rows], scene.take(rows))
        if not np.isfinite(state.log_kernel).all():
            raise ValueError('saved support changed during replay')
        targets[factor], states[factor] = target, state
    error = float(np.max(np.abs(states[1].log_kernel-saved_kernel)))
    prior_error = float(np.max(np.abs(states[1].log_prior-saved_prior)))
    repeated_kernel = np.tile(states[1].log_kernel, 4)
    batch_error = float(np.max(np.abs(states[4].log_kernel-repeated_kernel)))
    batch_summary_error = float(np.max(np.abs(states[4].summaries-np.tile(states[1].summaries, (4,1)))))
    batch_bitwise = bool(np.array_equal(states[4].log_kernel, repeated_kernel))
    if error > 1e-6 or prior_error > 1e-9 or batch_error > 1e-6 or batch_summary_error > 1e-10:
        raise ValueError(f'likelihood/prior replay failed: {error}, {prior_error}, {batch_error}, {batch_summary_error}')
    print('REPLAY_PASS', error, prior_error, batch_error, batch_bitwise, flush=True)
    def fingerprint(state):
        hasher = hashlib.sha256()
        arrays = [state.x, state.log_prior, state.log_kernel, state.log_proposal,
                  state.type_probability, state.summaries, state.scene.indptr, state.scene.offset_arcsec]
        for fields in (state.scene.targets, state.scene.neighbours):
            arrays.extend(fields[key] for key in sorted(fields))
        for array in arrays:
            hasher.update(array.tobytes())
        return hasher.hexdigest()
    retained_hashes = {factor:fingerprint(state) for factor,state in states.items()}
    blocks = ('physical', 'nuisance', 'environment', 'positions', 'exchange')
    benchmark = {}
    # Each timed call starts from exactly the same retained scenes. No chain is
    # advanced and no convergence result can be inferred from these acceptances.
    for block_index, block in enumerate(blocks):
        for factor in (1, 4):
            for warm in range(args.warmup):
                targets[factor].mutate(states[factor], 1.,
                    np.random.default_rng(args.seed+1000*block_index+warm),
                    block=block, scale=protocol['scale'])
                torch.cuda.synchronize()
        timings, rates, attempted = {1:[],4:[]}, {1:[],4:[]}, {1:[],4:[]}
        for repetition in range(args.repetitions):
            for factor in ((1,4) if repetition % 2 == 0 else (4,1)):
                rng = np.random.default_rng(args.seed+1000*block_index+100+repetition)
                torch.cuda.synchronize()
                before = time.perf_counter()
                moved, accepted, active = targets[factor].mutate(states[factor], 1., rng,
                    block=block, scale=protocol['scale'])
                torch.cuda.synchronize()
                timings[factor].append(time.perf_counter()-before)
                if not np.isfinite(moved.log_kernel).all():
                    raise ValueError('retained mutation state left likelihood support')
                rates[factor].append(float(accepted.sum()/max(active.sum(), 1)))
                attempted[factor].append(int(active.sum()))
        median = {factor:float(np.median(timings[factor])) for factor in (1,4)}
        benchmark[block] = dict(
            rows_1024=dict(seconds=timings[1], median_seconds=median[1], acceptance=rates[1], active_rows=attempted[1]),
            rows_4096=dict(seconds=timings[4], median_seconds=median[4], acceptance=rates[4], active_rows=attempted[4]),
            per_row_throughput_ratio=4*median[1]/median[4], call_time_ratio=median[4]/median[1])
        print('BLOCK_TIME', block, median, flush=True)
    assert np.array_equal(states[1].x, x)
    assert np.array_equal(states[4].x, np.tile(x, (4,1)))
    assert retained_hashes == {factor:fingerprint(state) for factor,state in states.items()}
    assert source_hashes == {path:digest(path) for path in source_hashes}
    assert input_hashes == {path:digest(path) for path in input_hashes}
    payload = dict(status='completed', diagnostic_only=True, source=str(args.source),
        source_sha256=source_hashes, input_sha256=input_hashes, script_sha256=digest(__file__),
        gpu_name=gpu_name, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_node_list=os.environ.get('SLURM_JOB_NODELIST'),
        torch_version=str(torch.__version__), python_version=sys.version.split()[0],
        density_batch_size=4096, repeated_population_factors=[1,4],
        repetitions=args.repetitions, warmup=args.warmup, beta=1., benchmark_seed=args.seed,
        replay=dict(saved_kernel_max_error=error, saved_prior_max_error=prior_error,
            replicated_kernel_max_error=batch_error, replicated_kernel_bitwise_equal=batch_bitwise,
            replicated_summary_max_error=batch_summary_error),
        blocks=benchmark, elapsed_seconds=time.monotonic()-started,
        limitations=['Repeated saved states, not sixteen distinct observed galaxies.',
            'Fixed density shape stays 4096; no larger numerical batch was introduced.',
            'Wall time includes native prior, full scene handling, GPU synchronization, and retention.',
            'Only final beta=1 kernels timed; early-stage invalid proposals and extinct observations are not represented.',
            'No independent population, normalizer, score precision or efficiency gate was evaluated.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print('BATCH_BENCHMARK_READY', args.output, flush=True)


if __name__ == '__main__':
    main()
