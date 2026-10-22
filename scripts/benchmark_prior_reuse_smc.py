"""Four frozen paired full-SMC replay/cost tests of target-prior reuse.

Only three predeclared runtime files may differ. No retries, omitted pairs,
algorithm changes, automatic adoption, or posterior-convergence inference.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import socket
import statistics
import sys
import time


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


_exact = module('_prior_replay_exact', 'benchmark_exchange_full_smc.py')
_cpu = module('_prior_replay_cpu', 'benchmark_smc_concurrency.py')
_provenance = module('_prior_replay_provenance', 'prior_reuse_proposal_provenance.py')
digest, load, save = _exact.digest, _exact.load, _exact.save
compare_json, compare_npz = _exact.compare_json, _exact.compare_npz
BASE_MANIFEST = 'fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1'
SOURCE_COUNT = 320
HASHES = {
    'sbsi/galsbi_particle_target.py': (
        '9de76a6089b31f73d632e6c3cf768df7667e96f1608e20dff43ee8ab9ce7e5ec',
        '7ba7ac7f95ff096fbc5339b149ae0526f381ee0d6da4bafd07ac9fd9a85a9316'),
    'sbsi/galsbi_prior.py': (
        'f6612ec614f966993946e3beb94221e6e6ad3adcf1782393f18cb8facc782652',
        'be00a6278f27bed4fa2a2625f9cd8df2823a0fed4a9bbb32b158dfa4086288f7'),
    'sbsi/galsbi_target_coordinates.py': (
        '519c9cd4930a2a0a22d9989bad55b8dc436fe16760f8ed6d5dff3d8160047209',
        '2305ce98548f807b6acba8f1d4869a47f619639d80cae5fd0bcf76eda4bfabd8'),
}
SEEDS = [2026103900 + i for i in range(4)]
OBSERVATIONS = [58165, 0, 1, 8]
SETTINGS = dict(particles=256, stages=64, moves=64, power=2., scale=1.,
                shear=[0., 0.], density_batch_size=4096)
OUTPUTS = ('result.json', 'protocol.json', 'progress.json', 'particles.npz', 'scenes.npz')


def normalized_protocol(protocol, source, sources):
    # The shared path normalizer keeps hashes exact. Replace only this experiment's
    # known, individually checked hashes; never broaden the shared comparator.
    p = _cpu.normalize_protocol(protocol, source, sources)
    for name, checksum in list(p['implementation_sha256'].items()):
        if name in HASHES:
            if checksum not in HASHES[name]:
                raise ValueError('unknown prior-reuse implementation hash')
            p['implementation_sha256'][name] = 'validated prior-reuse arm: ' + name
    return p


def expected_protocol(d, sources, replica):
    if type(replica) is not int or replica not in range(4):
        raise ValueError('one of four original replicas required')
    t = d['protocol_template']
    if digest(t['path']) != t['sha256']:
        raise ValueError('changed audited protocol template')
    p = normalized_protocol(load(t['path']), t['source'], sources['baseline'])
    fixed = dict(observations=OBSERVATIONS, particles=256, stages=64, moves=64,
        scale=1., fixed_shear=[0., 0.], density_batch_size=4096,
        betas=[(i / 64) ** 2 for i in range(65)])
    for key, value in fixed.items():
        compare_json(p[key], value, 'protocol')
    compare_json(d['settings'], SETTINGS, 'protocol')
    if (p['reference'] != d['reference'] or
            p['initial_proposal']['path'] != d['initial_proposal'] or
            p['physical_geometry']['path'] != d['physical_geometry']):
        raise ValueError('inputs differ from audited development recipe')
    runtime = {k: v for k, v in p['runtime'].items() if k != 'slurm_job_id'}
    compare_json(runtime, d['expected_runtime'], 'protocol')
    if (runtime['device'] != 'cuda' or runtime['torch_num_threads'] != 2 or
            runtime['cuda_device_name'] != 'Tesla V100-SXM2-32GB' or
            runtime['slurm_node_list'] != 'th-cl-nv02'):
        raise ValueError('frozen two-thread V100 runtime required')
    p['seed'] = SEEDS[replica]
    return p


def verify(d, require_preflight=True):
    if (type(d['schema_version']) is not int or d['schema_version'] not in (1, 2) or
            d['seeds'] != SEEDS or d['observations'] != OBSERVATIONS or
            set(d['arms']) != {'baseline', 'optimized'}):
        raise ValueError('unknown prior-reuse declaration')
    if Path(d['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('declared interpreter must execute the harness')
    _cpu.nonnegative(d.get('source_preparation_seconds', 0.), 'source preparation seconds')
    sources = {}
    for arm, item in d['arms'].items():
        root = Path(item['source'])
        if digest(root / 'SHA256SUMS') != item['manifest_sha256']:
            raise ValueError('changed source manifest')
        sources[arm] = _exact.manifest(root)
    baseline, optimized = sources['baseline'], sources['optimized']
    if (d['arms']['baseline']['manifest_sha256'] != BASE_MANIFEST or
            len(baseline) != SOURCE_COUNT or set(baseline) != set(optimized) or
            {name for name in baseline if baseline[name] != optimized[name]} != set(HASHES)):
        raise ValueError('exactly the three declared files may differ in the complete source')
    for name, (old, new) in HASHES.items():
        if baseline[name] != old or optimized[name] != new:
            raise ValueError('unknown baseline or prior-reuse candidate: ' + name)
    required = {str(Path(__file__).resolve()), str(Path(_exact.__file__).resolve()),
                str(Path(_cpu.__file__).resolve())}
    if not required <= d['harness_files'].keys():
        raise ValueError('all three executed harness sources must be pinned')
    inputs = {d['initial_proposal'], d['physical_geometry'],
        str(Path(d['reference']) / 'protocol.json'),
        str(Path(d['reference']) / 'observations.parquet')}
    if not inputs <= d['external_files'].keys():
        raise ValueError('missing external input identities')
    for mapping in ('external_files', 'harness_files'):
        for path, checksum in d[mapping].items():
            if not Path(path).is_absolute() or digest(path) != checksum:
                raise ValueError('changed ' + mapping + ': ' + path)
    expected_protocol(d, sources, 0)
    if d['schema_version'] == 2:
        required_v2 = {str(Path(_provenance.__file__).resolve()),
                       str(Path(__file__).with_name('check_prior_reuse_startup.py').resolve())}
        if not required_v2 <= d['harness_files'].keys():
            raise ValueError('provenance migration and actual startup preflight must be pinned')
        _provenance.validate_migration(d, sources, HASHES)
        _provenance.validate_failed_startup(d['previous_failed_startup'])
        for key in ('receipt', 'log', 'cost'):
            item = d['previous_failed_startup'][key]
            if d['external_files'].get(item['path']) != item['sha256']:
                raise ValueError('all original failed-startup evidence must remain pinned')
        for path in (d['proposal_migration']['derived'], d['proposal_migration']['native_replay']):
            if d['external_files'].get(path) != digest(path):
                raise ValueError('derived proposal and native evidence must be pinned')
        if require_preflight:
            preflight = module('_prior_replay_startup', 'check_prior_reuse_startup.py')
            preflight.validate_receipt(d, sources)
    return sources


def driver_command(d, arm, seed, output):
    adapted = dict(d, source=dict(path=d['arms'][arm]['source']),
        initial_proposal=d['initial_proposals'][arm] if d['schema_version'] == 2 else d['initial_proposal'])
    return _cpu.driver_command(adapted, seed, output)


def resources():
    if (not os.environ.get('SLURM_JOB_ID') or
            os.environ.get('SLURM_JOB_NODELIST') != 'th-cl-nv02' or
            os.environ.get('SLURM_CPUS_PER_TASK') != '2'):
        raise ValueError('scheduled th-cl-nv02 two-core allocation required')
    affinity = sorted(os.sched_getaffinity(0))
    topology = _cpu.physical_topology(affinity)
    selected = _cpu.select_physical_cores(affinity, topology, count=2)
    return dict(slurm_job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
        slurm_node_list=os.environ['SLURM_JOB_NODELIST'], affinity=affinity,
        selected_cpus=selected, cpu_topology={str(k): list(v) for k, v in topology.items()},
        gpu=_cpu.gpu_snapshot(), memory_info=Path('/proc/meminfo').read_text(),
        thread_environment={k: os.environ.get(k) for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def allocation_identity(resource):
    identity = {k: resource[k] for k in
        ('slurm_job_id', 'node', 'slurm_node_list', 'selected_cpus', 'thread_environment')}
    identity['gpu'] = {k: resource['gpu'][k] for k in
        ('uuid', 'name', 'cuda_visible_devices', 'memory_total_mib')}
    return identity


def validate_previous(root, replica, declaration_sha256, resource):
    if type(replica) is not int or replica not in range(4):
        raise ValueError('undeclared pair')
    if replica:
        previous = load(root / f'pair_{replica - 1}' / 'receipt.json')
        if (previous['status'] != 'completed' or
                previous['declaration_sha256'] != declaration_sha256 or
                allocation_identity(previous['resources']) != allocation_identity(resource)):
            raise ValueError('previous pair must complete in the same original allocation')
    if any((root / f'pair_{i}').exists() for i in range(replica, 4)):
        raise FileExistsError('existing pair cannot be overwritten or reordered')


def saved_run(d, sources, arm, folder, replica):
    result, protocol, progress = (load(folder / name) for name in OUTPUTS[:3])
    if result['status'] != 'completed':
        raise ValueError('all original populations must complete without extinction')
    for key in _exact.ARTIFACT_HASH_KEYS:
        name = key[:-7] + ('.json' if key == 'protocol_sha256' else '.npz')
        if result[key] != digest(folder / name):
            raise ValueError('changed saved artifact: ' + name)
        del result[key]
    if d['schema_version'] == 2:
        protocol = _provenance.normalize_initial_protocol(protocol, d, arm, sources, HASHES)
    protocol = normalized_protocol(protocol, d['arms'][arm]['source'], sources[arm])
    _exact.validate_saved_run(result, protocol, progress, expected_protocol(d, sources, replica))
    if any(any(stage.get('extinct', [])) for stage in progress['stages']):
        raise ValueError('extinct population cannot pass the performance gate')
    return result, protocol, progress


def compare_pair(d, sources, directory, replica):
    left = saved_run(d, sources, 'baseline', directory / 'baseline', replica)
    right = saved_run(d, sources, 'optimized', directory / 'optimized', replica)
    for a, b, kind in zip(left, right, ('result', 'protocol', 'progress')):
        compare_json(a, b, kind)
    return {name: compare_npz(directory / 'baseline' / name, directory / 'optimized' / name)
            for name in ('particles.npz', 'scenes.npz')}


def validate_workers(d, receipt, directory):
    if set(receipt['arms']) != {'baseline', 'optimized'}:
        raise ValueError('both original arms required')
    cpus = receipt['resources']['selected_cpus']
    topology = receipt['resources']['cpu_topology']
    if len(cpus) != 2 or len({tuple(topology[str(c)]) for c in cpus}) != 2:
        raise ValueError('two distinct physical cores required')
    for arm, item in receipt['arms'].items():
        if set(item['workers']) != {'0'}:
            raise ValueError('exactly one process per arm required')
        worker = item['workers']['0']
        saved = load(directory / (arm + '_runtime') / 'worker_0.json')
        expected = driver_command(d, arm, SEEDS[receipt['replica']], directory / arm)
        if (worker.get('returncode') != 0 or worker['cpus'] != cpus or
                worker['command'] != expected or saved['command'] != expected or
                saved['affinity'] != cpus or saved['pid'] != worker['pid'] or
                saved['node'] != receipt['resources']['node'] or
                saved['cuda_visible_devices'] != receipt['resources']['gpu']['cuda_visible_devices']):
            raise ValueError('worker failed or binding/command identity differs')
        _cpu.nonnegative(worker['process_seconds'], 'child process seconds')
        _cpu.nonnegative(item['wall_seconds'], 'arm wall seconds')


def run_pair(declaration_path, replica):
    started = time.monotonic()
    d = load(declaration_path)
    if d['schema_version'] != 2:
        raise ValueError('new GPU runs require explicit v2 proposal provenance and startup preflight')
    root = Path(d['output'])
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.running'
    lock.mkdir()
    receipt = None
    try:
        sources = verify(d)
        resource = resources()
        declaration_sha256 = digest(declaration_path)
        validate_previous(root, replica, declaration_sha256, resource)
        directory = root / f'pair_{replica}'
        directory.mkdir()
        receipt_path = directory / 'receipt.json'
        order = ['baseline', 'optimized'] if replica % 2 == 0 else ['optimized', 'baseline']
        receipt = dict(status='started', replica=replica, seed=SEEDS[replica], order=order,
            declaration_sha256=declaration_sha256, resources=resource, arms={})
        save(receipt_path, receipt)
        for arm in order:
            runtime = directory / (arm + '_runtime')
            runtime.mkdir()
            receipt['arms'][arm] = dict(workers={})
            def update(records):
                receipt['arms'][arm]['workers'] = records
                save(receipt_path, receipt)
            command = driver_command(d, arm, SEEDS[replica], directory / arm)
            records, wall = _cpu.run_workers([command], [resource['selected_cpus']], runtime,
                d['python'], d['arms'][arm]['source'], update)
            receipt['arms'][arm].update(workers=records, wall_seconds=wall,
                output_sha256={name: digest(directory / arm / name) for name in OUTPUTS})
            save(receipt_path, receipt)
        validate_workers(d, receipt, directory)
        receipt['arrays'] = compare_pair(d, sources, directory, replica)
        verify(d)
        receipt['gpu_after'] = _cpu.gpu_snapshot()
        receipt.update(status='completed', bitwise_saved_output_replay=True)
    except BaseException as error:
        if receipt is not None:
            receipt.update(status='failed', error=repr(error), bitwise_saved_output_replay=False)
        raise
    finally:
        if receipt is not None:
            receipt['pair_internal_wall_seconds'] = time.monotonic() - started
            save(receipt_path, receipt)
        lock.rmdir()


def cost_receipt(path):
    cost = load(path)
    if type(cost['exit_code']) is not int or cost['exit_code'] != 0:
        raise ValueError('successful original process cost required')
    _cpu.nonnegative(cost['wall_seconds'], 'process wall seconds')
    return cost


def report(declaration_path, output):
    if not os.environ.get('SLURM_JOB_ID') or Path(output).exists():
        raise ValueError('scheduled report and new output required')
    start = time.monotonic()
    d = load(declaration_path)
    sources = verify(d)
    pairs, allocation = [], None
    for replica in range(4):
        directory = Path(d['output']) / f'pair_{replica}'
        r = load(directory / 'receipt.json')
        expected_order = ['baseline', 'optimized'] if replica % 2 == 0 else ['optimized', 'baseline']
        if (r['status'] != 'completed' or r['replica'] != replica or r['seed'] != SEEDS[replica] or
                r['order'] != expected_order or r['declaration_sha256'] != digest(declaration_path) or
                r['bitwise_saved_output_replay'] is not True):
            raise ValueError('all four original pairs must complete and replay')
        identity = allocation_identity(r['resources'])
        if allocation is not None and identity != allocation:
            raise ValueError('allocation changed between original pairs')
        allocation = identity
        validate_workers(d, r, directory)
        for arm in ('baseline', 'optimized'):
            if r['arms'][arm]['output_sha256'] != {name: digest(directory / arm / name) for name in OUTPUTS}:
                raise ValueError('changed output after pair completion')
        compare_pair(d, sources, directory, replica)
        process = cost_receipt(Path(d['output']) / f'pair_{replica}_process_cost.json')
        _cpu.nonnegative(r['pair_internal_wall_seconds'], 'pair internal wall seconds')
        arm_total = sum(r['arms'][a]['wall_seconds'] for a in ('baseline', 'optimized'))
        # GNU time rounds to hundredths. Permit only that rounding uncertainty;
        # a materially shorter outer timer cannot silently become zero overhead.
        if process['wall_seconds'] + .011 < r['pair_internal_wall_seconds'] or process['wall_seconds'] + .011 < arm_total:
            raise ValueError('outer process cost is shorter than recorded work')
        common = max(0., process['wall_seconds'] - arm_total)
        charged = {a: r['arms'][a]['wall_seconds'] + common / 2 for a in ('baseline', 'optimized')}
        if any(value <= 0 for value in charged.values()):
            raise ValueError('positive measured arm costs required')
        pairs.append(dict(replica=replica, seed=SEEDS[replica], order=expected_order,
            arms=r['arms'], process_cost=process, common_overhead_seconds=common,
            charged_seconds=charged, baseline_to_optimized_ratio=charged['baseline'] / charged['optimized']))
    ratios = [p['baseline_to_optimized_ratio'] for p in pairs]
    save(output, dict(status='pending_report_process_cost', declaration_sha256=digest(declaration_path),
        all_four_saved_output_replay=True, pairs=pairs,
        previous_failed_startup=d.get('previous_failed_startup'),
        failed_startup_included_in_pair_speedups=False,
        ratio_range=[min(ratios), max(ratios)], ratio_mean=statistics.mean(ratios),
        ratio_mean_standard_error=statistics.stdev(ratios) / 2,
        source_preparation_seconds=d.get('source_preparation_seconds', 0.),
        internal_report_seconds=time.monotonic() - start,
        limitations=['Four fixed development pairs on one V100; timing variability is descriptive.',
            'Whole populations share fixed saved-output replay, not posterior-convergence acceptance.',
            'All saved JSON science, 64 stage histories and particle/scene array bytes must agree.',
            'Unrecorded mutation states and terminal RNG are not checked by this driver.',
            'Existing pilot generation/fitting are excluded; shared new verification/report/preparation costs are split equally.',
            'No automatic adoption. Finalization serializes JSON only and is outside the timed report.']))


def finalize(report_path, cost_path):
    r, cost = load(report_path), cost_receipt(cost_path)
    if r['status'] != 'pending_report_process_cost':
        raise ValueError('original pending report required')
    if cost['wall_seconds'] + .011 < r['internal_report_seconds']:
        raise ValueError('outer report timer is shorter than recorded work')
    total = {arm: sum(p['charged_seconds'][arm] for p in r['pairs']) + cost['wall_seconds'] / 2
             + r['source_preparation_seconds'] / 2 for arm in ('baseline', 'optimized')}
    r.update(status='completed', report_process_cost=cost, new_work_seconds_by_arm=total,
        total_baseline_to_optimized_ratio=total['baseline'] / total['optimized'],
        cost_scope='Original pair and report process wall times plus declared source preparation, conditional on existing fits. Pair verification and other shared work split equally. Scheduler queueing, launcher setup and JSON-only finalization excluded.')
    save(report_path, r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run-pair', 'report', 'finalize'])
    parser.add_argument('--declaration', type=Path)
    parser.add_argument('--replica', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cost', type=Path)
    a = parser.parse_args()
    def terminate(signum, frame):
        raise RuntimeError(f'harness interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    if a.mode == 'run-pair':
        run_pair(a.declaration, a.replica)
    elif a.mode == 'report':
        report(a.declaration, a.output)
    else:
        finalize(a.output, a.cost)


if __name__ == '__main__':
    main()
