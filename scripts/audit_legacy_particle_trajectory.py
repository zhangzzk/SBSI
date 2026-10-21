"""Replay selected frozen development populations with observational support audits.

--plan reads startup configuration only. Actual replay requires a Slurm job
with the original device/version/thread configuration; this script submits none.
"""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import sys
import traceback


LEGACY_EXCHANGE_SHA256 = 'c47763c55228257721135811b68f800945f64513c052945331974bb976b67c37'
FIXED_EXCHANGE_SHA256 = 'c5102579e3dfd1cd3c3e9ae5a267cf8ac0df9a6a82d60417fd2772e56aebd5ad'
MANIFESTS = dict(source_v7='ffc1ed784c4eecb3152a1009183032cb037b5a32f267ac5fba9824431cee1b3b',
                 source_v8='7edf32b90bb75d906d1ce3040fc50fa31fb99f4ffb2adafd5340fbbf609b1fce')
BLOCKS = ['physical', 'physical', 'nuisance', 'physical', 'environment', 'physical',
          'sed', 'physical', 'positions', 'physical', 'exchange', 'physical',
          'nuisance', 'physical', 'independence', 'physical']


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_snapshot(source):
    manifest = source/'SHA256SUMS'
    if digest(manifest) != MANIFESTS[source.name]:
        raise ValueError('unknown or changed frozen source manifest')
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if digest(source/name.lstrip('*')) != expected:
            raise ValueError(f'frozen source changed: {name}')


def replay_plan(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if not re.fullmatch(r'dev_(?:v7|v9|v8_baseline|v8_informed)_[0-7]', run.name):
        raise ValueError('only declared v7/v9/v8 development populations are supported; no heldout runs')
    protocol = json.loads((run/'protocol.json').read_text())
    driver_paths = [Path(path) for path in protocol['implementation_sha256']
                    if path.endswith('/scripts/check_galsbi_particles.py')]
    if len(driver_paths) != 1:
        raise ValueError('one recorded frozen driver required')
    source = driver_paths[0].parent.parent
    expected_source = 'source_v8' if run.name.startswith('dev_v8_') else 'source_v7'
    if source != run.parent/expected_source:
        raise ValueError('run/source snapshot convention differs from audited legacy study')
    if protocol['stages'] != 64 or protocol['betas'] != [(i/64)**2 for i in range(65)]:
        raise ValueError('only the recorded fixed 64-stage power-2 schedule is supported')
    blocks = protocol['blocks']
    relocated = ['exchange_relocate' if x == 'exchange' else x for x in BLOCKS]
    if blocks not in (BLOCKS, relocated):
        raise ValueError('unknown legacy mutation schedule')
    if not protocol.get('initial_proposal') or not protocol.get('physical_geometry'):
        raise ValueError('declared frozen initial proposal and geometry are required')
    informed = protocol.get('informed_environment')
    if bool(informed) != run.name.startswith('dev_v8_informed_'):
        raise ValueError('informed law disagrees with declared arm')
    runtime = protocol['runtime']
    arguments = ['--reference', protocol['reference'], '--output', str(output/'replay'),
        '--particles', str(protocol['particles']), '--stages', str(protocol['stages']),
        '--moves', str(protocol['moves']), '--power', '2', '--scale', str(protocol['scale']),
        '--seed', str(protocol['seed']), '--density-batch-size', str(protocol['density_batch_size']),
        '--device', runtime['device'], '--observation-indices', *map(str, protocol['observations']),
        '--shear', *map(str, protocol['fixed_shear']),
        '--initial-proposal', protocol['initial_proposal']['path'],
        '--physical-geometry', protocol['physical_geometry']['path']]
    if blocks == relocated:
        arguments.append('--relocate-exchanges')
    if informed:
        arguments.extend(['--informed-environment', informed['path'],
            '--environment-mixture-probability', str(informed['metadata']['mixture_probability'])])
    exchanges = sum(blocks[i % len(blocks)] in ('exchange','exchange_relocate')
                    for i in range(protocol['stages']*protocol['moves']))
    return dict(original_run=str(run), output=str(output), source=str(source),
        driver=str(driver_paths[0]), driver_arguments=arguments,
        protocol_sha256=digest(run/'protocol.json'), protocol=protocol,
        required_runtime=runtime,
        cost=dict(full_inference_replays=1, stages=protocol['stages'],
            mutation_calls=protocol['stages']*protocol['moves'],
            scheduled_exchange_mutations=exchanges,
            diagnostic_native_evaluations_upper_bound=2*exchanges,
            diagnostic_native_rows_upper_bound=2*exchanges*protocol['particles']*len(protocol['observations']),
            extra_likelihood_calls=0,
            added_work='Two diagnostic native-prior/support evaluations per nonempty exchange proposal, complete artifact hashing/comparison; unbenchmarked wall-clock overhead'),
        historical_limit='Original drivers do not persist complete intermediate states or final RNG; all saved arrays and every saved stage diagnostic must match.')


def load_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', action='store_true', help='Only inspect startup settings; do not read scientific outcomes or run inference')
    args = parser.parse_args()
    plan = replay_plan(args.run, args.output)
    if args.plan:
        print(json.dumps({k:v for k,v in plan.items() if k != 'protocol'}, indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('actual trajectory replay requires a scheduler allocation')
    output, source, original = Path(plan['output']), Path(plan['source']), Path(plan['original_run'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('fresh separate audit output directory required')
    if output == original or original in output.parents or source == output or source in output.parents:
        raise ValueError('audit outputs must not modify original runs or source snapshots')
    worktree = Path(__file__).resolve().parent.parent
    fixed_path = worktree/'sbsi/scene_exchange.py'
    if digest(fixed_path) != FIXED_EXCHANGE_SHA256:
        raise ValueError('fixed diagnostic helper changed; independent re-review required')
    if digest(source/'sbsi/scene_exchange.py') != LEGACY_EXCHANGE_SHA256:
        raise ValueError('legacy correction changed; pairing/identity review required')
    verify_snapshot(source)
    for path, expected in plan['protocol']['implementation_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'recorded source changed: {path}')
    for field in ('initial_proposal', 'physical_geometry', 'informed_environment'):
        artifact = plan['protocol'].get(field)
        if artifact and digest(artifact['path']) != artifact['sha256']:
            raise ValueError(f'changed {field} artifact')
    output.mkdir(parents=True, exist_ok=True)
    # Reproduce the frozen launcher's artifact roots, without changing snapshots.
    os.environ['SBSI_CACHE_DIR'] = '/project/ls-gruen/users/zekang.zhang/sbsi_caches'
    os.environ['BLENDEMU_RUNS_DIR'] = '/project/ls-gruen/users/zekang.zhang/blendemu_runs'
    sys.pycache_prefix = f'/tmp/legacy-exchange-audit-{os.environ["SLURM_JOB_ID"]}'
    sys.path[:0] = [str(source), str(source/'scripts')]
    if 'sbsi' in sys.modules:
        raise ValueError('fresh interpreter required before importing the frozen package')
    import sbsi
    if Path(sbsi.__file__).resolve().parent.parent != source:
        raise ValueError('frozen package import escaped its snapshot')
    import numpy as np
    import torch
    runtime = plan['required_runtime']
    actual = dict(python=sys.version.split()[0], numpy=np.__version__, torch=str(torch.__version__),
        device=runtime['device'], torch_num_threads=min(2, len(os.sched_getaffinity(0))),
        cuda_device_name=torch.cuda.get_device_name() if runtime['device'] == 'cuda' else None)
    if any(actual[k] != runtime[k] for k in actual):
        raise ValueError(f'replay runtime differs: required {runtime}, actual {actual}')
    import sbsi.galsbi_particle_target as target_module
    import sbsi.scene_exchange as exchange_module
    from sbsi.particle_populations import ParticlePopulations
    fixed = load_path(fixed_path, 'sbsi._trajectory_fixed_support')
    audit_module = load_path(worktree/'sbsi/legacy_exchange_audit.py', '_legacy_exchange_audit')
    audit = audit_module.LegacyExchangeAudit(plan['protocol']['observations'], fixed.numerical_over_analytic)
    if digest(original/'protocol.json') != plan['protocol_sha256']:
        raise ValueError('original startup protocol changed during preflight')
    # Check completeness, checksums and schema before spending a full inference
    # replay. This is reached only by explicit scheduled execution, never plan.
    audit_module.compare_complete_outputs(original, original)
    originals = {p.name:digest(p) for p in original.iterdir() if p.is_file()}
    original_result = audit_module.read_json(original/'result.json')
    if original_result.get('status') not in ('completed', 'completed_with_extinction'):
        raise ValueError('cannot replay an incomplete or unknown failed original run')
    receipt = dict(schema='legacy-exchange-full-replay-audit-v1', status='running',
        plan={k:v for k,v in plan.items() if k != 'protocol'},
        diagnostic_sources={str(path):digest(path) for path in
            (Path(__file__), fixed_path, worktree/'sbsi/legacy_exchange_audit.py')},
        old_ratios_returned_unchanged=True, counts_trusted=False)
    receipt_path = output/'audit_receipt.json'
    def save_receipt():
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    save_receipt()
    os.chdir(source)
    sys.argv = [plan['driver'], *plan['driver_arguments']]
    try:
        with audit.instrument(target_module, exchange_module, ParticlePopulations):
            runpy.run_path(plan['driver'], run_name='__main__')
        replay = output/'replay'
        audit.validate_progress(audit_module.read_json(replay/'progress.json'))
        comparison = audit_module.compare_complete_outputs(original, replay)
        if originals != {p.name:digest(p) for p in original.iterdir() if p.is_file()}:
            raise ValueError('original artifacts changed during replay')
        verify_snapshot(source)
        if any(digest(path) != value for path,value in receipt['diagnostic_sources'].items()):
            raise ValueError('audit implementation changed during replay')
        if audit.last_state is None or audit.last_rng is None:
            raise ValueError('complete final population/RNG capture missing')
        # Extra receipts stay outside the replay directory and are not presented
        # as RNG/intermediate-state evidence that existed in the original run.
        state, populations = audit.last_state, audit.last_populations
        np.savez_compressed(output/'unpersisted_final_cache.npz', log_proposal=state.log_proposal,
            type_probability=state.type_probability, cached_summaries=state.summaries,
            owner=populations.target.owner, log_normalizer=populations.logz,
            ancestor_labels=populations.labels, alive=populations.alive)
        (output/'replayed_final_rng.json').write_text(json.dumps(audit.last_rng.bit_generator.state,
            indent=2, allow_nan=False)+'\n')
        receipt.update(status='verified_full_replay', counts_trusted=True,
            comparison=comparison, support_counts=audit.records(),
            proposal_calls=audit.proposal_calls, support_calls=audit.support_calls,
            replay_only_artifacts={name:digest(output/name) for name in
                ('unpersisted_final_cache.npz', 'replayed_final_rng.json')})
        save_receipt()
    except BaseException as error:
        receipt.update(status='replay_failed_or_mismatched', counts_trusted=False,
            support_counts_provisional=audit.records(), error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc())
        save_receipt()
        raise
    print('LEGACY_TRAJECTORY_AUDIT_VERIFIED', receipt_path, flush=True)


if __name__ == '__main__':
    main()
