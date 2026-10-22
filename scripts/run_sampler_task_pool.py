"""Bounded local execution of immutable sampler tasks; no scientific acceptance.

Four independent processes share one V100 and eight distinct physical cores.
Every invocation is preserved. There are no retries, merged observations or
replacement tasks. Use the shell launcher to capture complete outer CLI cost.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location('_task_pool_cpu', HERE.with_name('benchmark_smc_concurrency.py'))
_cpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_cpu)
digest = _cpu.digest
THREADS = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
SCHEMA = 'sbsi-sampler-task-pool-v1'
BOOTSTRAP = ('import runpy,sys;from pathlib import Path;'
    's=Path(sys.argv.pop(1)).resolve();e=Path(sys.argv.pop(1)).resolve();'
    'assert e.is_relative_to(s);sys.path[:0]=[str(s),str(e.parent)];'
    'import sbsi;assert Path(sbsi.__file__).resolve().parent==s/"sbsi";'
    'runpy.run_path(str(e),run_name="__main__")')


def command(python, source, entrypoint, arguments):
    """Preparation helper: freeze this complete argv, never rewrite it at launch."""
    return [python, '-u', '-c', BOOTSTRAP, source, entrypoint, *arguments]


def fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError('unexpected or missing ' + name + ' fields')


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON ' + value)))


def save(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def absolute(value):
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise ValueError('canonical absolute path required: ' + value)
    return path


def relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') for p in path.parts) or path.as_posix() != value:
        raise ValueError('canonical relative path required: ' + value)
    return path


def checksum(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('SHA256 required')
    return value


def tree_hashes(root):
    """Complete regular-file tree. Symlinks and special files are not snapshots."""
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in immutable tree: ' + str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
        elif not path.is_dir():
            raise ValueError('special file in immutable tree: ' + str(path))
    return result


def source_hashes(item):
    fields(item, 'path manifest_sha256', 'source')
    root = absolute(item['path'])
    if not root.is_dir() or digest(root / 'SHA256SUMS') != checksum(item['manifest_sha256']):
        raise ValueError('changed source manifest')
    expected = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        value, name = line.split('  ', 1)
        relative(name)
        if name in expected or name == 'SHA256SUMS':
            raise ValueError('duplicate or recursive source manifest')
        expected[name] = checksum(value)
    actual = tree_hashes(root)
    del actual['SHA256SUMS']
    if not expected or actual != expected:
        raise ValueError('changed or incomplete source snapshot: ' + str(root))
    return actual


def verify(manifest, expected_digest, fresh=True):
    """Verify every task/input before any process starts; no path rewriting."""
    manifest = absolute(str(Path(manifest).resolve()))
    if digest(manifest) != checksum(expected_digest):
        raise ValueError('changed task declaration')
    d = load(manifest)
    fields(d, 'schema workers cores_per_worker output_root python environment sources harness_files input_files tasks', 'manifest')
    if d['schema'] != SCHEMA or type(d['workers']) is not int or d['workers'] != 4 or d['cores_per_worker'] != 2:
        raise ValueError('four workers/two cores required')
    root = absolute(d['output_root'])
    if fresh and root.exists():
        raise FileExistsError('fresh output root required')
    if root == manifest or manifest.is_relative_to(root):
        raise ValueError('manifest must be outside output root')
    python = absolute(d['python'])
    if python.resolve() != Path(sys.executable).resolve():
        raise ValueError('declared Python must execute the pool')
    if d['environment'].get('OMP_NUM_THREADS') != '2' or any(d['environment'].get(k) != v for k, v in THREADS.items()):
        raise ValueError('fixed two-thread environment required')
    if any(not isinstance(k, str) or not isinstance(v, str) or '\x00' in k + v or '=' in k for k, v in d['environment'].items()):
        raise ValueError('invalid environment')
    if set(d['environment']) & {'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX', 'LD_PRELOAD'}:
        raise ValueError('environment may not change device or import authority')
    if d['environment'].get('PYTHONDONTWRITEBYTECODE') != '1':
        raise ValueError('immutable sources require PYTHONDONTWRITEBYTECODE=1')
    if not d['sources']:
        raise ValueError('at least one complete source snapshot required')
    sources = {key: source_hashes(item) for key, item in d['sources'].items()}
    required = {str(HERE), str(Path(_cpu.__file__).resolve()), str(_cpu._HELPER), str(HERE.with_suffix('.sh'))}
    if not required <= d['harness_files'].keys() or str(python) not in d['input_files']:
        raise ValueError('all executed harnesses and Python executable must be pinned')
    pins = {}
    for mapping in ('harness_files', 'input_files'):
        for name, value in d[mapping].items():
            path = absolute(name)
            if path.is_relative_to(root) or digest(path) != checksum(value):
                raise ValueError('changed or output-overlapping ' + mapping + ': ' + name)
            if name in pins and pins[name] != value:
                raise ValueError('conflicting input pins')
            pins[name] = value
    if not isinstance(d['tasks'], list) or not d['tasks']:
        raise ValueError('nonempty ordered task list required')
    ids, outputs, invocations = set(), [], set()
    for task in d['tasks']:
        fields(task, 'id source cwd output argv identity argument_indices expected_outputs entrypoint', 'task')
        name = task['id']
        if not isinstance(name, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name) or name in ids:
            raise ValueError('unique safe task IDs required')
        ids.add(name)
        source = absolute(d['sources'][task['source']]['path'])
        cwd, output = absolute(task['cwd']), absolute(task['output'])
        if not cwd.is_dir() or not cwd.is_relative_to(source):
            raise ValueError('cwd must be inside the declared source')
        if source.is_relative_to(root) or root.is_relative_to(source):
            raise ValueError('source and output trees must be disjoint')
        if not output.is_relative_to(root / 'results') or output == root / 'results' or output.exists():
            if fresh or not output.is_relative_to(root / 'results') or output == root / 'results':
                raise ValueError('fresh task output beneath results/ required')
        if any(output.is_relative_to(other) or other.is_relative_to(output) for other in outputs):
            raise ValueError('task outputs overlap')
        outputs.append(output)
        argv = task['argv']
        if (not isinstance(argv, list) or not argv or argv[0] != str(python) or
                any(not isinstance(arg, str) or '\x00' in arg for arg in argv)):
            raise ValueError('exact Python argv required')
        entrypoint = absolute(task['entrypoint'])
        if (not entrypoint.is_relative_to(source) or entrypoint.relative_to(source).as_posix() not in sources[task['source']] or
                argv[:6] != command(str(python), str(source), str(entrypoint), [])):
            raise ValueError('exact source-authority bootstrap and pinned entrypoint required')
        identity = task['identity']
        fields(identity, 'seed observations recipe target_law', 'task identity')
        observations = identity['observations']
        if (type(identity['seed']) is not int or identity['seed'] < 0 or not isinstance(observations, list) or
                len(observations) != 4 or len(set(observations)) != 4 or any(type(x) is not int or x < 0 for x in observations)):
            raise ValueError('original seed and ordered four-observation group required')
        for key in ('recipe', 'target_law'):
            item = identity[key]
            fields(item, 'path sha256', key)
            if pins.get(item['path']) != item['sha256']:
                raise ValueError('recipe and target-law artifacts must be pinned')
        binding = task['argument_indices']
        fields(binding, 'seed output observations', 'argument index')
        indices = [binding['seed'], binding['output'], *binding['observations']]
        if (len(binding['observations']) != 4 or any(type(i) is not int or i < 6 or i >= len(argv) for i in indices) or
                len(set(indices)) != 6 or argv[binding['seed']] != str(identity['seed']) or
                argv[binding['output']] != str(output) or
                [argv[i] for i in binding['observations']] != list(map(str, observations))):
            raise ValueError('argv seed/output/group bindings differ from task identity')
        # Reject identical invocations even if task IDs/output labels were changed.
        logical = list(argv)
        logical[binding['output']] = '<TASK_OUTPUT>'
        invocation = canonical([logical, task['cwd'], task['source']])
        if invocation in invocations:
            raise ValueError('duplicate original task invocation')
        invocations.add(invocation)
        expected = task['expected_outputs']
        if not isinstance(expected, list) or not expected or len(set(expected)) != len(expected):
            raise ValueError('unique expected output files required')
        for value in expected:
            relative(value)
    return d, dict(manifest_sha256=expected_digest, source_files=sources, input_files=pins)


def child_environment(d):
    # Keep allocation/loader details recorded, but remove ambient Python injection.
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX', 'LD_PRELOAD'):
        env.pop(name, None)
    env.update(d['environment'])
    return env


def validate_slots(slots):
    flat = [cpu for slot in slots for cpu in slot]
    if len(slots) != 4 or any(len(slot) != 2 for slot in slots) or len(set(flat)) != 8:
        raise ValueError('four nonoverlapping two-core slots required')
    if not set(flat) <= os.sched_getaffinity(0):
        raise ValueError('worker cores exceed allocation')
    if len(set(_cpu.physical_topology(flat).values())) != 8:
        raise ValueError('eight distinct physical cores required')


def usage(value):
    return dict(user_seconds=value.ru_utime, system_seconds=value.ru_stime,
                max_rss_kib=value.ru_maxrss, minor_faults=value.ru_minflt,
                major_faults=value.ru_majflt, voluntary_context_switches=value.ru_nvcsw,
                involuntary_context_switches=value.ru_nivcsw)


def reap(process, block=False):
    pid, status, cost = os.wait4(process.pid, 0 if block else os.WNOHANG)
    if not pid:
        return None
    process.returncode = os.waitstatus_to_exitcode(status)
    return process.returncode, usage(cost)


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def finish_task(task, record, runtime):
    worker = load(runtime / 'worker.json')
    if (worker['command'] != task['argv'] or worker['affinity'] != record['cpus'] or
            worker['pid'] != record['pid'] or worker['cuda_visible_devices'] != os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('worker did not preserve command/device/affinity')
    record['worker'] = worker
    output = Path(task['output'])
    if not output.is_dir() or output.is_symlink():
        raise ValueError('task output directory missing or symlinked')
    hashes = tree_hashes(output)
    if not set(task['expected_outputs']) <= hashes.keys():
        raise ValueError('task missing expected output files')
    record['output_sha256'] = hashes
    record['status'] = 'completed_execution'


def execute_tasks(d, slots, root, update):
    """Scheduling kernel; tests supply tiny processes and real physical affinity."""
    validate_slots(slots)
    started = time.monotonic()
    records = [dict(index=i, id=t['id'], status='unstarted', task=t) for i, t in enumerate(d['tasks'])]
    active, next_index, failure = {}, 0, None
    update(records)

    def finish(slot, result):
        process, stream, record, launched, runtime = active.pop(slot)
        code, cost = result
        now = time.monotonic()
        stream.close()
        record.update(returncode=code, cost=cost, process_seconds=now - launched,
                      end_offset_seconds=now - started)
        # A sampler task must reap its own children. Never leave descendants
        # running after a leader exits, even when that leader returned zero.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            terminate_group(process)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            record.update(status='failed', error='task left live descendants')
            raise RuntimeError('task left live descendants')
        if code:
            record['status'] = 'failed'
            raise RuntimeError(f'task {record["id"]} exited {code}')
        try:
            finish_task(record['task'], record, runtime)
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            raise
        update(records)

    try:
        while next_index < len(records) or active:
            # Reap all completed processes before launching any replacement. A
            # simultaneous failure cannot be hidden by an earlier free slot.
            for slot, (process, *_rest) in list(active.items()):
                result = reap(process)
                if result is not None:
                    finish(slot, result)
            for slot, cpus in enumerate(slots):
                if slot in active or next_index == len(records):
                    continue
                record = records[next_index]
                next_index += 1
                task = record['task']
                runtime = root / 'tasks' / task['id']
                runtime.mkdir(parents=True, exist_ok=False)
                output = Path(task['output'])
                if output.exists():
                    record.update(status='failed', error='output collision before launch')
                    raise FileExistsError('task output collision before launch')
                output.parent.mkdir(parents=True, exist_ok=True)
                wrapper = [d['python'], '-u', str(Path(_cpu.__file__).resolve()), 'affinity-worker',
                           '--cpus', ','.join(map(str, cpus)), '--worker-receipt', str(runtime / 'worker.json'),
                           '--', *task['argv']]
                stream = (runtime / 'stdout.log').open('x')
                launch_time = time.monotonic()
                record.update(status='starting', cpus=cpus, start_offset_seconds=launch_time - started)
                try:
                    process = subprocess.Popen(wrapper, cwd=task['cwd'], env=child_environment(d),
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException as error:
                    stream.close()
                    record.update(status='failed', error=repr(error), launch_failed=True)
                    raise
                record.update(status='running', pid=process.pid)
                active[slot] = (process, stream, record, launch_time, runtime)
                update(records)
            if active:
                time.sleep(.025)
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        for process, *_ in active.values():
            terminate_group(process)
        deadline = time.monotonic() + 3
        for slot, (process, stream, record, launch_time, runtime) in list(active.items()):
            result = reap(process)
            while result is None and time.monotonic() < deadline:
                time.sleep(.025)
                result = reap(process)
            # Kill remaining process-group members even if their leader exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if result is None:
                result = reap(process, block=True)
            stream.close()
            record.update(status='terminated', returncode=result[0], cost=result[1],
                process_seconds=time.monotonic() - launch_time,
                end_offset_seconds=time.monotonic() - started, termination_reason=failure)
        for record in records:
            runtime = root / 'tasks' / record['id']
            if runtime.is_dir():
                record['runtime_sha256'] = tree_hashes(runtime)
            output = Path(record['task']['output'])
            if record['status'] != 'completed_execution' and output.is_dir():
                try:
                    record['partial_output_sha256'] = tree_hashes(output)
                except ValueError as error:
                    record['partial_output_error'] = repr(error)
            if record['status'] == 'unstarted':
                record['not_started_reason'] = failure
        update(records)
    return records


def run(manifest, expected_digest):
    started = time.monotonic()
    before_self = resource.getrusage(resource.RUSAGE_SELF)
    before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    d, before = verify(manifest, expected_digest)
    root = Path(d['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'manifest.json', d)
    receipt = dict(schema=SCHEMA, status='starting', success_scope='execution only; no convergence or scientific acceptance',
        manifest_path=str(Path(manifest).resolve()), manifest_sha256=expected_digest, hashes_before=before,
        tasks=[dict(index=i, id=t['id'], status='unstarted', task=t) for i, t in enumerate(d['tasks'])])
    def update(records):
        receipt['tasks'] = records
        save(root / 'receipt.json', receipt)
    def interrupted(signum, _frame):
        raise RuntimeError('pool interrupted by signal ' + str(signum))
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        receipt['resources_before'] = _cpu.resources()
        receipt['effective_environment'] = child_environment(d)
        # Environment can include site credentials; persist only names and hashes
        # outside the explicitly declared operational values.
        receipt['effective_environment'] = {key: (value if key in d['environment'] or key in ('SLURM_JOB_ID', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK') or
            key in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'PATH') else {'sha256': canonical(value)})
            for key, value in receipt['effective_environment'].items()}
        selected = receipt['resources_before']['selected_cpus']
        slots = [selected[2 * i:2 * i + 2] for i in range(4)]
        receipt['status'] = 'running'
        execute_tasks(d, slots, root, update)
        receipt['status'] = 'completed_execution'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        # Diagnostics cannot turn a failed run into success, and failure must
        # still leave every original task and the source recheck on disk.
        try:
            _, after = verify(manifest, expected_digest, fresh=False)
            receipt['hashes_after'] = after
            receipt['hashes_unchanged'] = after == before
        except BaseException as error:
            receipt.update(status='failed', post_hash_error=repr(error), hashes_unchanged=False)
        try:
            receipt['resources_after'] = _cpu.resources()
            if 'resources_before' in receipt:
                for key in ('uuid', 'name', 'cuda_visible_devices'):
                    if receipt['resources_after']['gpu'][key] != receipt['resources_before']['gpu'][key]:
                        raise ValueError('GPU identity changed')
        except BaseException as error:
            receipt.update(status='failed', post_resource_error=repr(error))
        own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt['internal_cost'] = dict(wall_seconds=time.monotonic() - started,
            user_seconds=own.ru_utime - before_self.ru_utime + children.ru_utime - before_children.ru_utime,
            system_seconds=own.ru_stime - before_self.ru_stime + children.ru_stime - before_children.ru_stime,
            coordinator_max_rss_kib=own.ru_maxrss, maximum_child_rss_kib=children.ru_maxrss,
            scope='verification, dispatch, complete child processes, output hashing and final source/resource rechecks; excludes imports and final receipt serialization')
        save(root / 'receipt.json', receipt)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if receipt['status'] != 'completed_execution':
        raise RuntimeError('post-execution identity checks failed')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'run'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.mode == 'validate':
        declaration, checked = verify(args.manifest, args.manifest_sha256)
        print(json.dumps(dict(status='validated_only', tasks=len(declaration['tasks']), hashes=checked)))
    else:
        run(args.manifest, args.manifest_sha256)
