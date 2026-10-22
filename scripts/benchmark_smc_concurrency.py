"""Frozen SMC replay at 1, 2 and 4 CPU workers sharing one allocated GPU.

No algorithm changes, retries, missing-population exclusions, or automatic adoption.
The six settings must run in order in the same one-GPU/eight-core allocation.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

_HELPER = Path(__file__).with_name('benchmark_exchange_full_smc.py').resolve()
_SPEC = importlib.util.spec_from_file_location('_smc_exact_replay', _HELPER)
_exact = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_exact)
digest, load, save = _exact.digest, _exact.load, _exact.save
compare_json, compare_npz = _exact.compare_json, _exact.compare_npz
SEEDS = [2026103700 + i for i in range(4)]
ORDER = [1, 2, 4, 4, 2, 1]
OBSERVATIONS = [58165, 0, 1, 8]
SETTINGS = dict(particles=256, stages=64, moves=64, power=2., scale=1.,
                shear=[0., 0.], density_batch_size=4096)
OUTPUTS = ('result.json', 'protocol.json', 'progress.json', 'particles.npz', 'scenes.npz')


def nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid nonnegative finite ' + name)
    return value


def normalize_protocol(protocol, source, sources):
    """Only checked absolute implementation paths become relative; hashes stay exact."""
    protocol = copy.deepcopy(protocol)
    source = Path(source).resolve()
    mapped = {}
    for path, checksum in protocol['implementation_sha256'].items():
        relative = Path(path).resolve().relative_to(source).as_posix()
        if relative in mapped or sources.get(relative) != checksum:
            raise ValueError('unexpected or changed protocol implementation: ' + relative)
        mapped[relative] = checksum
    if not {'scripts/check_galsbi_particles.py', 'sbsi/scene_exchange.py'} <= mapped.keys():
        raise ValueError('missing protocol implementation')
    protocol['implementation_sha256'] = mapped
    return protocol


def expected_protocol(d, sources, replica):
    if type(replica) is not int or replica not in range(4):
        raise ValueError('one of four original replicas required')
    template = d['protocol_template']
    if digest(template['path']) != template['sha256']:
        raise ValueError('changed protocol template')
    p = normalize_protocol(load(template['path']), template['source'], sources)
    fixed = dict(observations=OBSERVATIONS, particles=256, stages=64, moves=64,
        scale=1., fixed_shear=[0., 0.], density_batch_size=4096,
        betas=[(i / 64) ** 2 for i in range(65)])
    for key, value in fixed.items():
        compare_json(p[key], value, 'protocol')
    compare_json(d['settings'], SETTINGS, 'protocol')
    if (p['reference'] != d['reference'] or
            p['initial_proposal']['path'] != d['initial_proposal'] or
            p['physical_geometry']['path'] != d['physical_geometry']):
        raise ValueError('inputs differ from frozen development recipe')
    runtime = {k: v for k, v in p['runtime'].items() if k != 'slurm_job_id'}
    compare_json(runtime, d['expected_runtime'], 'protocol')
    if (runtime['device'] != 'cuda' or runtime['torch_num_threads'] != 2 or
            runtime['cuda_device_name'] != 'Tesla V100-SXM2-32GB' or
            runtime['slurm_node_list'] != 'th-cl-nv02'):
        raise ValueError('frozen two-thread V100 runtime required')
    p['seed'] = SEEDS[replica]
    return p


def verify(d):
    if (type(d['schema_version']) is not int or d['schema_version'] != 1 or
            d['seeds'] != SEEDS or d['observations'] != OBSERVATIONS or d['order'] != ORDER or
            type(d['threads_per_worker']) is not int or d['threads_per_worker'] != 2 or
            type(d['required_allocated_cpus']) is not int or d['required_allocated_cpus'] != 8):
        raise ValueError('unknown concurrency declaration')
    if Path(d['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('declared interpreter must execute the harness')
    nonnegative(d.get('source_preparation_seconds', 0.), 'source preparation seconds')
    source = Path(d['source']['path']).resolve()
    if digest(source / 'SHA256SUMS') != d['source']['manifest_sha256']:
        raise ValueError('changed source manifest')
    sources = _exact.manifest(source)
    required_harness = {str(Path(__file__).resolve()), str(_HELPER)}
    if not required_harness <= d['harness_files'].keys():
        raise ValueError('both executed harness sources must be pinned')
    required_inputs = {d['initial_proposal'], d['physical_geometry'],
        str(Path(d['reference']) / 'protocol.json'),
        str(Path(d['reference']) / 'observations.parquet')}
    if not required_inputs <= d['external_files'].keys():
        raise ValueError('missing frozen external input identities')
    for mapping in ('external_files', 'harness_files'):
        for path, checksum in d[mapping].items():
            if not Path(path).is_absolute() or digest(path) != checksum:
                raise ValueError('changed ' + mapping + ': ' + path)
    expected_protocol(d, sources, 0)
    return sources


def physical_topology(cpus):
    result = {}
    for cpu in cpus:
        root = Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
        result[cpu] = (int((root / 'physical_package_id').read_text()),
                       int((root / 'core_id').read_text()))
    return result


def select_physical_cores(affinity, topology, count=8):
    selected, seen = [], set()
    for cpu in sorted(affinity):
        core = tuple(topology[cpu])
        if core not in seen:
            selected.append(cpu)
            seen.add(core)
        if len(selected) == count:
            return selected
    raise ValueError(f'{count} distinct allocated physical cores required')


def gpu_snapshot():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(visible.split(',')) != 1 or not visible or visible in ('-1', 'NoDevFiles'):
        raise ValueError('exactly one visible CUDA device required')
    # CUDA ordinals can be remapped by Slurm; nvidia-smi physical indices cannot.
    # Resolve ordinal zero in an isolated context that exits before worker timing.
    probe = ('import json,torch; n=torch.cuda.device_count(); '
             'assert n==1, ("one CUDA device required", n); '
             'p=torch.cuda.get_device_properties(0); '
             'print(json.dumps(dict(count=n,uuid=str(p.uuid),name=p.name)))')
    resolved = subprocess.run([sys.executable, '-c', probe], check=True, text=True,
        capture_output=True, timeout=30)
    device = json.loads(resolved.stdout)
    if device['count'] != 1 or device['name'] != 'Tesla V100-SXM2-32GB':
        raise ValueError('one CUDA-visible V100 required')
    cuda_uuid = device['uuid']
    if not cuda_uuid.startswith('GPU-'):
        cuda_uuid = 'GPU-' + cuda_uuid
    command = ['nvidia-smi', '-i', cuda_uuid,
        '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits']
    run = subprocess.run(command, check=True, text=True, capture_output=True, timeout=15)
    rows = [line.strip().split(', ') for line in run.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError('one physical GPU telemetry row required')
    uuid, name, total, used, utilization = rows[0]
    if name != 'Tesla V100-SXM2-32GB' or uuid.lower() != cuda_uuid.lower():
        raise ValueError('CUDA-visible V100 and telemetry UUID must agree')
    return dict(cuda_visible_devices=visible, uuid=uuid, name=name,
                memory_total_mib=int(total), memory_used_mib=int(used),
                utilization_percent=int(utilization))


def resources():
    if (not os.environ.get('SLURM_JOB_ID') or
            os.environ.get('SLURM_JOB_NODELIST') != 'th-cl-nv02' or
            os.environ.get('SLURM_CPUS_PER_TASK') != '8'):
        raise ValueError('one scheduled th-cl-nv02 eight-core allocation required')
    affinity = sorted(os.sched_getaffinity(0))
    topology = physical_topology(affinity)
    selected = select_physical_cores(affinity, topology)
    return dict(slurm_job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
        slurm_node_list=os.environ['SLURM_JOB_NODELIST'], affinity=affinity,
        cpu_topology={str(k): list(v) for k, v in topology.items()}, selected_cpus=selected,
        gpu=gpu_snapshot(), memory_info=Path('/proc/meminfo').read_text(),
        cgroup=Path('/proc/self/cgroup').read_text(),
        thread_environment={k: os.environ.get(k) for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def validate_predecessor(root, index, declaration_sha256, resource):
    if type(index) is not int or index not in range(len(ORDER)):
        raise ValueError('undeclared setting index')
    if index:
        previous = load(root / f'setting_{index - 1}' / 'receipt.json')
        if (previous['status'] != 'completed' or
                previous['declaration_sha256'] != declaration_sha256):
            raise ValueError('previous original setting must complete first')
        old = previous['resources']
        for key in ('slurm_job_id', 'node', 'slurm_node_list', 'selected_cpus', 'thread_environment'):
            if old[key] != resource[key]:
                raise ValueError('allocation or CPU recipe changed between settings: ' + key)
        for key in ('uuid', 'name', 'cuda_visible_devices', 'memory_total_mib'):
            if old['gpu'][key] != resource['gpu'][key]:
                raise ValueError('physical GPU changed between settings')
    if any((root / f'setting_{i}').exists() for i in range(index, len(ORDER))):
        raise FileExistsError('existing setting cannot be overwritten or reordered')


def driver_command(d, seed, output):
    bootstrap = ('import runpy,sys;from pathlib import Path;'
        's=Path(sys.argv.pop(1)).resolve();sys.path[:0]=[str(s),str(s/"scripts")];'
        'import sbsi;assert Path(sbsi.__file__).resolve().parent==s/"sbsi";'
        'runpy.run_path(str(s/"scripts/check_galsbi_particles.py"),run_name="__main__")')
    return [d['python'], '-u', '-c', bootstrap, d['source']['path'],
        '--reference', d['reference'], '--output', str(output),
        '--observation-indices', *map(str, OBSERVATIONS), '--particles', '256',
        '--stages', '64', '--moves', '64', '--power', '2', '--scale', '1',
        '--seed', str(seed), '--shear', '0', '0', '--device', 'cuda',
        '--density-batch-size', '4096', '--physical-geometry', d['physical_geometry'],
        '--initial-proposal', d['initial_proposal']]


def affinity_worker(cpus, receipt_path, command):
    """Pin in a fresh process, then exec the driver; no threaded preexec_fn."""
    if len(cpus) != 2 or len(set(cpus)) != 2 or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('two distinct allocated worker CPUs required')
    if len(set(physical_topology(cpus).values())) != 2:
        raise ValueError('worker CPUs must be on distinct physical cores')
    os.sched_setaffinity(0, cpus)
    if sorted(os.sched_getaffinity(0)) != sorted(cpus):
        raise ValueError('worker CPU binding failed')
    path = Path(receipt_path)
    with path.open('x') as stream:
        json.dump(dict(pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)),
            node=socket.gethostname(), command=command,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), stream)
    os.execvpe(command[0], command, os.environ.copy())


def run_workers(commands, slots, directory, python, cwd, on_update):
    """Bounded isolated children; any failure terminates and reaps every active group."""
    if not slots or any(len(s) != 2 for s in slots):
        raise ValueError('two CPUs per slot required')
    flat = [cpu for slot in slots for cpu in slot]
    if len(set(flat)) != len(flat):
        raise ValueError('worker affinity slots overlap')
    active, records, next_replica = {}, {}, 0
    start = time.monotonic()
    failure = None
    try:
        while next_replica < len(commands) or active:
            for slot_index, cpus in enumerate(slots):
                if slot_index in active or next_replica == len(commands):
                    continue
                replica = next_replica
                next_replica += 1
                log = (directory / f'population_{replica}.log').open('x')
                record = dict(replica=replica, cpus=cpus, command=commands[replica],
                              start_offset_seconds=time.monotonic() - start)
                records[str(replica)] = record
                wrapper = [python, '-u', str(Path(__file__).resolve()), 'affinity-worker',
                    '--cpus', ','.join(map(str, cpus)), '--worker-receipt',
                    str(directory / f'worker_{replica}.json'), '--', *commands[replica]]
                child_start = time.monotonic()
                try:
                    process = subprocess.Popen(wrapper, cwd=cwd, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    record['launch_failed'] = True
                    raise
                record['pid'] = process.pid
                active[slot_index] = (process, log, record, child_start)
                on_update(records)
            for slot_index, (process, log, record, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                record.update(returncode=code, process_seconds=time.monotonic() - started,
                              end_offset_seconds=time.monotonic() - start)
                log.close()
                del active[slot_index]
                on_update(records)
                if code:
                    raise RuntimeError(f'original population {record["replica"]} failed ({code})')
            if active:
                time.sleep(.05)
        return records, time.monotonic() - start
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        for process, _, _, _ in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, log, record, started in active.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
            record.update(returncode=process.returncode, terminated_due_to_failure=failure,
                process_seconds=time.monotonic() - started,
                end_offset_seconds=time.monotonic() - start)
        on_update(records)


def saved_run(d, sources, folder, replica):
    result, protocol, progress = (load(folder / name) for name in OUTPUTS[:3])
    if result['status'] != 'completed':
        raise ValueError('all original populations must complete without extinction')
    for key in _exact.ARTIFACT_HASH_KEYS:
        name = key[:-7] + ('.json' if key == 'protocol_sha256' else '.npz')
        if result[key] != digest(folder / name):
            raise ValueError('changed saved artifact: ' + name)
        del result[key]
    protocol = normalize_protocol(protocol, d['source']['path'], sources)
    _exact.validate_saved_run(result, protocol, progress, expected_protocol(d, sources, replica))
    for stage in progress['stages']:
        if any(stage.get('extinct', [])):
            raise ValueError('extinct population is not a successful performance replicate')
    return result, protocol, progress


def compare_population(d, sources, first, second, replica):
    left = saved_run(d, sources, first, replica)
    right = saved_run(d, sources, second, replica)
    for a, b, kind in zip(left, right, ('result', 'protocol', 'progress')):
        compare_json(a, b, kind)
    return {name: compare_npz(first / name, second / name)
            for name in ('particles.npz', 'scenes.npz')}


def check_workers(receipt, directory):
    if set(receipt['workers']) != {str(i) for i in range(4)}:
        raise ValueError('all four original worker receipts required')
    selected = receipt['resources']['selected_cpus']
    topology = receipt['resources']['cpu_topology']
    if (len(selected) != 8 or len(set(selected)) != 8 or
            len({tuple(topology[str(cpu)]) for cpu in selected}) != 8):
        raise ValueError('eight distinct selected physical cores required')
    allowed_slots = [selected[2 * i:2 * i + 2] for i in range(receipt['concurrency'])]
    for key, record in receipt['workers'].items():
        if record.get('returncode') != 0:
            raise ValueError('failed or missing original worker')
        nonnegative(record['process_seconds'], 'worker process seconds')
        nonnegative(record['start_offset_seconds'], 'worker start offset')
        nonnegative(record['end_offset_seconds'], 'worker end offset')
        if (record['end_offset_seconds'] < record['start_offset_seconds'] or
                record['replica'] != int(key) or record['cpus'] not in allowed_slots):
            raise ValueError('invalid worker interval, identity or declared CPU slot')
        worker = load(directory / f'worker_{key}.json')
        if (worker['affinity'] != sorted(record['cpus']) or
                worker['command'] != record['command'] or worker['pid'] != record['pid'] or
                worker['node'] != receipt['resources']['node'] or
                worker['cuda_visible_devices'] != receipt['resources']['gpu']['cuda_visible_devices'] or
                len(record['cpus']) != 2 or not set(record['cpus']) <= set(selected)):
            raise ValueError('worker binding or command identity mismatch')
    records = list(receipt['workers'].values())
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            overlap = max(a['start_offset_seconds'], b['start_offset_seconds']) < min(
                a['end_offset_seconds'], b['end_offset_seconds'])
            if overlap and set(a['cpus']) & set(b['cpus']):
                raise ValueError('overlapping workers share CPU affinity')


def run_setting(declaration_path, index):
    start = time.monotonic()
    d = load(declaration_path)
    root = Path(d['output'])
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.running'
    lock.mkdir()  # Atomic guard against simultaneous settings or automatic retries.
    receipt = None
    try:
        source_start = time.monotonic()
        sources = verify(d)
        source_seconds = time.monotonic() - source_start
        resource = resources()
        declaration_sha256 = digest(declaration_path)
        validate_predecessor(root, index, declaration_sha256, resource)
        directory = root / f'setting_{index}'
        directory.mkdir()
        receipt_path = directory / 'receipt.json'
        concurrency = ORDER[index]
        receipt = dict(status='started', setting_index=index, concurrency=concurrency,
            repetition=0 if index < 3 else 1, seeds=SEEDS,
            declaration_sha256=declaration_sha256, resources=resource,
            source_verification_before_seconds=source_seconds, workers={})
        save(receipt_path, receipt)
        cpus = resource['selected_cpus']
        slots = [cpus[2 * i:2 * i + 2] for i in range(concurrency)]
        commands = [driver_command(d, seed, directory / f'population_{i}') for i, seed in enumerate(SEEDS)]
        def update(records):
            receipt['workers'] = records
            save(receipt_path, receipt)
        records, wall = run_workers(commands, slots, directory, d['python'],
                                    d['source']['path'], update)
        receipt.update(workers=records, execution_wall_seconds=wall)
        check_workers(receipt, directory)
        validation_start = time.monotonic()
        baseline = root / 'setting_0'
        receipt['replay_arrays'] = {}
        for replica in range(4):
            folder = directory / f'population_{replica}'
            receipt['replay_arrays'][str(replica)] = compare_population(d, sources,
                baseline / f'population_{replica}', folder, replica)
            receipt['workers'][str(replica)]['output_sha256'] = {
                name: digest(folder / name) for name in OUTPUTS}
        receipt['validation_seconds'] = time.monotonic() - validation_start
        source_start = time.monotonic()
        verify(d)
        receipt['source_verification_after_seconds'] = time.monotonic() - source_start
        receipt['gpu_after'] = gpu_snapshot()
        receipt.update(status='completed', bitwise_saved_output_replay=True)
    except BaseException as error:
        if receipt is not None:
            receipt.update(status='failed', error=repr(error), bitwise_saved_output_replay=False)
        raise
    finally:
        if receipt is not None:
            receipt['setting_wall_seconds'] = time.monotonic() - start
            save(receipt_path, receipt)
        lock.rmdir()


def report(declaration_path, output):
    if not os.environ.get('SLURM_JOB_ID') or Path(output).exists():
        raise ValueError('scheduled report and new output required')
    start = time.monotonic()
    d = load(declaration_path)
    sources = verify(d)
    root = Path(d['output'])
    receipts, allocation = [], None
    for index, concurrency in enumerate(ORDER):
        directory = root / f'setting_{index}'
        r = load(directory / 'receipt.json')
        if (r['status'] != 'completed' or r['setting_index'] != index or
                r['concurrency'] != concurrency or r['seeds'] != SEEDS or
                r['repetition'] != (0 if index < 3 else 1) or
                r['declaration_sha256'] != digest(declaration_path) or
                r['bitwise_saved_output_replay'] is not True):
            raise ValueError('all six original settings must complete and replay')
        current = {k: r['resources'][k] for k in
            ('slurm_job_id', 'node', 'slurm_node_list', 'selected_cpus', 'cpu_topology', 'thread_environment')}
        current['gpu_uuid'] = r['resources']['gpu']['uuid']
        if allocation is not None and current != allocation:
            raise ValueError('benchmark settings changed allocation')
        allocation = current
        check_workers(r, directory)
        for replica in range(4):
            folder = directory / f'population_{replica}'
            worker = r['workers'][str(replica)]
            if worker['command'] != driver_command(d, SEEDS[replica], folder):
                raise ValueError('saved command differs from declared recipe')
            if worker['output_sha256'] != {name: digest(folder / name) for name in OUTPUTS}:
                raise ValueError('changed output after setting completion')
            compare_population(d, sources, root / 'setting_0' / f'population_{replica}', folder, replica)
        for key in ('execution_wall_seconds', 'setting_wall_seconds'):
            nonnegative(r[key], key)
            if r[key] == 0:
                raise ValueError('positive benchmark times required')
        receipts.append(r)
    ratios = []
    for repetition, indices in enumerate(([0, 1, 2], [5, 4, 3])):
        baseline = receipts[indices[0]]
        for index in indices[1:]:
            r = receipts[index]
            ratios.append(dict(repetition=repetition, concurrency=r['concurrency'],
                execution_speedup=baseline['execution_wall_seconds'] / r['execution_wall_seconds'],
                setting_inclusive_speedup=baseline['setting_wall_seconds'] / r['setting_wall_seconds']))
    report_seconds = time.monotonic() - start
    save(output, dict(status='completed', declaration_sha256=digest(declaration_path),
        all_24_saved_output_replay=True, setting_order=ORDER, ratios=ratios, settings=receipts,
        source_preparation_seconds=d.get('source_preparation_seconds', 0.),
        internal_report_seconds=report_seconds,
        total_setting_wall_seconds=sum(r['setting_wall_seconds'] for r in receipts),
        cost_scope='Setting wall includes verification, resource checks, all child startup/run/save time and replay validation. Preparation and report overhead are separate; launcher process startup and final JSON serialization are outside this internal timer.',
        limitations=['Repeated identical seeds across settings and repetitions are timing replicates, not independent science.',
            'One fixed development group, one V100, two CPU threads per worker; no 500-core extrapolation.',
            'Only saved histories/states are replayed; the driver does not save terminal RNG or every mutation.',
            'Existing pilot/geometry/proposal preparation is excluded and remains scientifically necessary.',
            'Reports all settings without automatic adoption or posterior-accuracy acceptance.']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run-setting', 'report', 'affinity-worker'])
    parser.add_argument('--declaration', type=Path)
    parser.add_argument('--setting-index', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpus')
    parser.add_argument('--worker-receipt', type=Path)
    args, rest = parser.parse_known_args()
    if args.mode == 'affinity-worker':
        if not rest or rest[0] != '--' or len(rest) < 2:
            parser.error('worker requires -- followed by its command')
        affinity_worker([int(x) for x in args.cpus.split(',')], args.worker_receipt, rest[1:])
    else:
        if rest:
            parser.error('unexpected arguments')
        # Let the supervisor reap child process groups on scheduler termination.
        def terminate(signum, frame):
            raise RuntimeError(f'harness interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, terminate)
        if args.mode == 'run-setting':
            run_setting(args.declaration, args.setting_index)
        else:
            report(args.declaration, args.output)


if __name__ == '__main__':
    main()
