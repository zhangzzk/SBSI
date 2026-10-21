"""Prepared GPU replay/cost harness; submit only after resource coordination.

Based on benchmark_particle_batch_20261021.py, whose SHA256 is
9bff2aca28e450248b79f51397b46f6100ac96ec27f6dbb8c4364a7af618ae46.
Uses one frozen source and the corrected 55f1220 helper as control. Saved v5
states are read-only benchmark inputs, not a new inference population. No
source-v5b snapshot or numerical density batch shape is modified.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifest(source, manifest):
    hashes = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = (source/relative.lstrip('*')).resolve()
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError(f'frozen source manifest mismatch: {path}')
        hashes[str(path)] = expected
    required = ['sbsi/scene_exchange.py', 'sbsi/galsbi_particle_target.py', 'sbsi/galsbi_prior.py',
        'sbsi/galsbi_scene_prior.py', 'sbsi/galsbi_summary.py', 'sbsi/galsbi_target_coordinates.py',
        'sbsi/galsbi_particle_proposal.py', 'sbsi/particle_geometry.py',
        'doc/evidence/scene_exchange_baseline_55f1220.py', 'scripts/benchmark_exchange_trained.py']
    if any(str(source/name) not in hashes for name in required):
        raise ValueError('manifest is missing a required implementation file')
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--seed', type=int, default=2026103300)
    parser.add_argument('--require-gpu-model', default='V100')
    parser.add_argument('--check-inputs-only', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.exists() or args.repetitions < 2:
        raise ValueError('fresh output and at least two repetitions required')
    source_hashes = check_manifest(source, args.manifest)
    baseline_path = source/'doc/evidence/scene_exchange_baseline_55f1220.py'
    if digest(baseline_path) != 'c5102579e3dfd1cd3c3e9ae5a267cf8ac0df9a6a82d60417fd2772e56aebd5ad':
        raise ValueError('corrected 55f1220 baseline changed')
    protocol = json.loads((args.run/'protocol.json').read_text())
    saved_result = json.loads((args.run/'result.json').read_text())
    if (saved_result['status'] != 'completed' or protocol['particles'] != 256
            or len(protocol['observations']) != 4 or protocol['density_batch_size'] != 4096
            or protocol.get('initial_proposal') or protocol.get('relocate_exchanges')):
        raise ValueError('requires original completed four-observation v5 population')
    input_hashes = {str(args.run/name):digest(args.run/name)
        for name in ('protocol.json','result.json','particles.npz','scenes.npz')}
    for name in ('protocol','particles','scenes'):
        path = args.run/(name+('.json' if name == 'protocol' else '.npz'))
        if digest(path) != saved_result[name+'_sha256']:
            raise ValueError(f'changed saved {name}')
    if args.check_inputs_only:
        print('FROZEN_INPUTS_PASS', source, args.run, flush=True)
        return
    sys.path[:0] = [str(source), str(source/'scripts')]
    import numpy as np
    import pandas as pd
    import torch
    import sbsi
    import sbsi.galsbi_particle_target as target_module
    from check_galsbi_continuous_prior import build_prior
    from sbsi.galsbi_prior import ExactMemo
    from sbsi.galsbi_scene_prior import GalSBIScenePrior, GalSBISceneBatch
    from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
    from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
    from sbsi.particle_geometry import FrozenParticleGeometry
    from sbsi.galsbi_summary import SummaryLikelihood
    from sbsi.measurement_model import load_measurement_model
    from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response
    from sbsi.scene_exchange import propose_native_exchange
    assert Path(sbsi.__file__).resolve().parent.parent == source
    if not torch.cuda.is_available() or args.require_gpu_model not in torch.cuda.get_device_name():
        raise ValueError('matching scheduled GPU required')
    torch.set_num_threads(min(2, len(os.sched_getaffinity(0))))
    spec = importlib.util.spec_from_file_location('sbsi._exchange_corrected_control', baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    functions = dict(baseline=baseline.propose_native_exchange, candidate=propose_native_exchange)
    reference = Path(protocol['reference'])
    reference_protocol = json.loads((reference/'protocol.json').read_text())
    if digest(reference/'observations.parquet') != protocol['observation_sha256']:
        raise ValueError('changed observations')
    observed = pd.read_parquet(reference/'observations.parquet').iloc[protocol['observations']].reset_index(drop=True)
    with np.load(args.run/'particles.npz') as saved:
        x = saved['coordinates'].reshape(-1,10).copy()
        saved_kernel, saved_prior = saved['log_kernel'].reshape(-1).copy(), saved['log_prior'].reshape(-1).copy()
    with np.load(args.run/'scenes.npz') as saved:
        scene = GalSBISceneBatch({k[7:]:saved[k].copy() for k in saved.files if k.startswith('target_')},
            {k[10:]:saved[k].copy() for k in saved.files if k.startswith('neighbour_')},
            saved['indptr'].copy(), saved['offset_arcsec'].copy())
    V39_LIKE.validate()
    likelihood = SummaryLikelihood(load_measurement_model(V39_LIKE.flow_checkpoints[0], device='cuda'),
        load_detection_classifier(V39_LIKE, device='cuda'), density_batch_size=4096)
    response = load_disk_response(V39_LIKE, device='cuda', nthread=torch.get_num_threads())
    galaxy, cosmo, _ = build_prior()
    distance = ExactMemo(lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))
    prior, coordinates = GalSBIScenePrior(galaxy), GalSBITargetCoordinates(galaxy, distance)
    proposals = [MeasurementParticleProposal(p['means'],p['scales'],p['weights'],
        component_family=p['component_family'],student_df=p.get('student_df') or 5.) for p in protocol['proposals']]
    geometry = protocol['physical_geometry']
    if digest(geometry['path']) != geometry['sha256']:
        raise ValueError('changed geometry')
    target = target_module.GalSBIParticleTarget(prior, coordinates, likelihood, response, distance,
        reference_protocol['conditions'], observed, proposals, np.repeat(np.arange(4),256),
        protocol['fixed_shear'], FrozenParticleGeometry.load(geometry['path']))
    state = target.evaluate(x,scene)
    if len(state.x) != 1024 or not np.isfinite(state.log_kernel).all():
        raise ValueError('retained saved-state support mismatch')
    errors = dict(kernel=float(np.max(np.abs(state.log_kernel-saved_kernel))),
                  prior=float(np.max(np.abs(state.log_prior-saved_prior))))
    if errors['kernel'] > 1e-6 or errors['prior'] > 1e-9:
        raise ValueError(f'saved-state replay failed: {errors}')

    def fingerprint(value):
        h = hashlib.sha256()
        def update(obj):
            if isinstance(obj, np.ndarray):
                h.update(str((obj.dtype.str,obj.shape)).encode()); h.update(obj.tobytes())
            elif isinstance(obj, dict):
                for key in sorted(obj):
                    h.update(key.encode()); update(obj[key])
            elif obj is not None:
                update(vars(obj))
        update(value)
        return h.hexdigest()

    frozen_state = fingerprint(state)
    # Paired complete trajectories, including all caches and payload fields.
    states = dict(baseline=state, candidate=state)
    generators = {name:np.random.default_rng(args.seed) for name in functions}
    blocks = ('exchange','exchange_relocate','physical','positions','nuisance','sed')
    for step in range(18):
        outputs = {}
        for name in functions:
            target_module.propose_native_exchange = functions[name]
            outputs[name] = target.mutate(states[name], (.001,.4,1.)[step%3],
                generators[name], block=blocks[step%len(blocks)], scale=protocol['scale'])
            states[name] = outputs[name][0]
        a,b = outputs['baseline'],outputs['candidate']
        assert fingerprint(a[0]) == fingerprint(b[0]), ('trajectory',step)
        assert a[1].tobytes() == b[1].tobytes() and a[2].tobytes() == b[2].tobytes()
        assert pickle.dumps(generators['baseline'].bit_generator.state) == pickle.dumps(generators['candidate'].bit_generator.state)
    timings = []
    for beta in (.001,1.):
        for block in ('exchange','exchange_relocate'):
            elapsed = {name:[] for name in functions}
            for repetition in range(-1,args.repetitions):
                outputs, rngs = {}, {}
                for name in (('baseline','candidate') if repetition%2 else ('candidate','baseline')):
                    target_module.propose_native_exchange = functions[name]
                    rngs[name] = np.random.default_rng(args.seed+100+repetition)
                    torch.cuda.synchronize(); before = time.perf_counter()
                    outputs[name] = target.mutate(state,beta,rngs[name],block=block,scale=protocol['scale'])
                    torch.cuda.synchronize()
                    if repetition >= 0:
                        elapsed[name].append(time.perf_counter()-before)
                a,b = outputs['baseline'],outputs['candidate']
                assert fingerprint(a[0]) == fingerprint(b[0])
                assert a[1].tobytes() == b[1].tobytes() and a[2].tobytes() == b[2].tobytes()
                assert pickle.dumps(rngs['baseline'].bit_generator.state) == pickle.dumps(rngs['candidate'].bit_generator.state)
            medians = {name:float(np.median(v)) for name,v in elapsed.items()}
            timings.append(dict(beta=beta,block=block,seconds=elapsed,median_seconds=medians,
                speedup=medians['baseline']/medians['candidate']))
    target_module.propose_native_exchange = propose_native_exchange
    assert fingerprint(state) == frozen_state
    assert source_hashes == check_manifest(source,args.manifest)
    assert input_hashes == {path:digest(path) for path in input_hashes}
    report = dict(status='completed',source=str(source),source_sha256=source_hashes,
        input_sha256=input_hashes,script_sha256=digest(__file__),density_batch_size=4096,
        rows=1024,observations=4,seed=args.seed,repetitions=args.repetitions,
        gpu=torch.cuda.get_device_name(),numpy_version=np.__version__,saved_replay_errors=errors,
        trajectory_steps=18,bitwise_replay_pass=True,timings=timings,
        no_convergence_normalizer_or_scientific_efficiency_claim=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
