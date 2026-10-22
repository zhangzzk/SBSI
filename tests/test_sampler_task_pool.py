"""Synthetic subprocess checks only; no sampler, model, GPU or convergence claim."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import pytest

SPEC = importlib.util.spec_from_file_location('sampler_task_pool',
    Path(__file__).resolve().parents[1] / 'scripts/run_sampler_task_pool.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

DRIVER = '''import json, os, random, signal, sys, time
from pathlib import Path
seed, output, a, b, c, d, mode = sys.argv[1:]
if mode == 'fail':
    time.sleep(.1)
    raise SystemExit(7)
if mode == 'slow':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)
if mode == 'mutate':
    Path(__file__).with_name('payload.txt').write_text('changed')
if mode == 'descendant':
    import subprocess
    subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    raise SystemExit(0)
time.sleep(.04)
root = Path(output)
root.mkdir()
if mode != 'missing':
    (root / 'data.json').write_text(json.dumps(dict(seed=int(seed),
        observations=list(map(int, [a,b,c,d])), draw=random.Random(int(seed)).random(),
        argv=sys.argv[1:], cwd=os.getcwd(), cpus=sorted(os.sched_getaffinity(0)),
        threads=os.environ['OMP_NUM_THREADS'])))
'''


def make_fixture(tmp_path, count=9, modes=None):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'sbsi').mkdir()
    (source / 'sbsi/__init__.py').write_text('')
    (source / 'driver.py').write_text(DRIVER)
    (source / 'payload.txt').write_text('original')
    entries = m.tree_hashes(source)
    (source / 'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name, value in entries.items()))
    recipe, target = tmp_path / 'recipe.json', tmp_path / 'target.json'
    recipe.write_text('{"particles":256,"density_batch_size":4096}')
    target.write_text('{"law":"synthetic-process-test"}')
    python = str(Path(sys.executable).resolve())
    root = tmp_path / 'output'
    tasks = []
    for index in range(count):
        seed, observations = 2026104000 + index, [index * 4 + i for i in (3, 0, 2, 1)]
        output = root / 'results' / str(index)
        args = [str(seed), str(output), *map(str, observations), (modes or {}).get(index, 'ok')]
        tasks.append(dict(id=f'task_{index}', source='original', cwd=str(source),
            entrypoint=str(source / 'driver.py'), output=str(output),
            argv=m.command(python, str(source), str(source / 'driver.py'), args),
            identity=dict(seed=seed, observations=observations,
                recipe=dict(path=str(recipe), sha256=m.digest(recipe)),
                target_law=dict(path=str(target), sha256=m.digest(target))),
            argument_indices=dict(seed=6, output=7, observations=[8, 9, 10, 11]),
            expected_outputs=['data.json']))
    harness = [m.HERE, Path(m._cpu.__file__).resolve(), m._cpu._HELPER, m.HERE.with_suffix('.sh')]
    d = dict(schema=m.SCHEMA, workers=4, cores_per_worker=2, output_root=str(root),
        python=python, environment=dict(m.THREADS, PYTHONDONTWRITEBYTECODE='1'),
        sources={'original': dict(path=str(source), manifest_sha256=m.digest(source / 'SHA256SUMS'))},
        harness_files={str(p): m.digest(p) for p in harness},
        input_files={str(p): m.digest(p) for p in [Path(python), recipe, target]}, tasks=tasks)
    manifest = tmp_path / 'tasks.json'
    m.save(manifest, d)
    return d, manifest


def cpus():
    affinity = os.sched_getaffinity(0)
    try:
        selected = m._cpu.select_physical_cores(affinity, m._cpu.physical_topology(affinity))
    except ValueError:
        pytest.skip('eight physical cores needed for tiny synthetic affinity test')
    return [selected[i:i+2] for i in range(0, 8, 2)]


def resource_fixture(monkeypatch, slots):
    selected = sum(slots, [])
    def resources():
        return dict(slurm_job_id='synthetic-test', selected_cpus=selected,
            cpu_topology={str(k): list(v) for k, v in m._cpu.physical_topology(selected).items()},
            gpu=dict(uuid='synthetic-no-gpu', name='synthetic-no-gpu', cuda_visible_devices=None))
    monkeypatch.setattr(m._cpu, 'resources', resources)


def test_multiwave_keeps_order_commands_seeds_groups_affinity_and_bounded_concurrency(tmp_path):
    d, manifest = make_fixture(tmp_path)
    m.verify(manifest, m.digest(manifest))
    root, slots = Path(d['output_root']), cpus()
    root.mkdir()
    history = []
    records = m.execute_tasks(d, slots, root, lambda rows: history.append(copy.deepcopy(rows)))
    assert [row['id'] for row in records] == [task['id'] for task in d['tasks']]
    assert max(sum(r['status'] == 'running' for r in rows) for rows in history) == 4
    assert all(row['status'] == 'completed_execution' for row in records)
    for record, task in zip(records, d['tasks']):
        output = m.load(Path(task['output']) / 'data.json')
        assert output['seed'] == task['identity']['seed']
        assert output['observations'] == task['identity']['observations']
        assert output['draw'] == random.Random(task['identity']['seed']).random()
        assert output['argv'] == task['argv'][6:]
        assert output['cwd'] == task['cwd']
        assert output['cpus'] == record['cpus'] == record['worker']['affinity']
        assert output['threads'] == '2'
        assert record['worker']['command'] == task['argv']
        assert record['cost']['user_seconds'] >= 0 and record['process_seconds'] > 0
        assert record['output_sha256']['data.json'] == m.digest(Path(task['output']) / 'data.json')
    events = [(row['start_offset_seconds'], 1) for row in records] + [(row['end_offset_seconds'], -1) for row in records]
    active = 0
    for _, delta in sorted(events):
        active += delta
        assert 0 <= active <= 4
    assert records[4]['start_offset_seconds'] >= min(row['end_offset_seconds'] for row in records[:4])
    assert records[8]['start_offset_seconds'] >= min(row['end_offset_seconds'] for row in records[4:8])


def test_failure_terminates_siblings_and_retains_all_unstarted_tasks(tmp_path):
    d, manifest = make_fixture(tmp_path, modes={0: 'fail', 1: 'slow', 2: 'slow', 3: 'slow'})
    m.verify(manifest, m.digest(manifest))
    root = Path(d['output_root']); root.mkdir()
    saved = []
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='exited 7'):
        m.execute_tasks(d, cpus(), root, lambda rows: saved.__setitem__(slice(None), copy.deepcopy(rows)))
    assert time.monotonic() - started < 10
    assert len(saved) == 9 and saved[0]['status'] == 'failed' and saved[0]['returncode'] == 7
    assert all(row['status'] == 'terminated' for row in saved[1:4])
    assert all(row['status'] == 'unstarted' and row['not_started_reason'] for row in saved[4:])
    for row in saved[1:4]:
        with pytest.raises(ProcessLookupError):
            os.kill(row['pid'], 0)
    assert not any(Path(task['output']).exists() for task in d['tasks'][4:])


@pytest.mark.parametrize('case', ['seed', 'group', 'source_argv', 'bootstrap', 'recipe', 'cwd', 'duplicate_id',
    'duplicate_invocation', 'overlap', 'outside_output', 'expected_traversal', 'harness', 'python', 'threads', 'unlisted_source',
    'changed_source', 'changed_input', 'unknown_field'])
def test_preflight_rejects_mutation_before_launch(tmp_path, case):
    d, manifest = make_fixture(tmp_path, count=2)
    task = d['tasks'][0]
    if case == 'seed': task['argv'][6] = '999'
    elif case == 'group': task['argv'][8:12] = task['argv'][8:12][::-1]
    elif case == 'source_argv': task['argv'][4] = str(tmp_path)
    elif case == 'bootstrap': task['argv'][3] = 'print("wrong import authority")'
    elif case == 'recipe': task['identity']['recipe']['sha256'] = '0' * 64
    elif case == 'cwd': task['cwd'] = str(tmp_path)
    elif case == 'duplicate_id': d['tasks'][1]['id'] = task['id']
    elif case == 'duplicate_invocation':
        other = copy.deepcopy(task)
        other.update(id='second', output=d['tasks'][1]['output'])
        other['argv'][7] = other['output']
        d['tasks'][1] = other
    elif case == 'overlap':
        task['output'] = d['tasks'][1]['output'] + '/inside'
        task['argv'][7] = task['output']
    elif case == 'outside_output': task['output'] = str(tmp_path / 'elsewhere')
    elif case == 'expected_traversal': task['expected_outputs'] = ['../data.json']
    elif case == 'harness': d['harness_files'].pop(str(m._cpu._HELPER))
    elif case == 'python': d['input_files'].pop(d['python'])
    elif case == 'threads': d['environment']['OMP_NUM_THREADS'] = '8'
    elif case == 'unlisted_source': (Path(task['cwd']) / 'extra.py').write_text('')
    elif case == 'changed_source': (Path(task['cwd']) / 'payload.txt').write_text('changed')
    elif case == 'changed_input': Path(task['identity']['recipe']['path']).write_text('changed')
    elif case == 'unknown_field': task['seed_override'] = 42
    m.save(manifest, d)
    with pytest.raises((ValueError, FileExistsError)):
        m.verify(manifest, m.digest(manifest))
    assert not Path(d['output_root']).exists()


def test_manifest_digest_and_duplicate_json_keys_are_not_ignored(tmp_path):
    d, manifest = make_fixture(tmp_path, count=1)
    checksum = m.digest(manifest)
    manifest.write_text(manifest.read_text() + ' ')
    with pytest.raises(ValueError, match='changed task declaration'):
        m.verify(manifest, checksum)
    manifest.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ValueError, match='duplicate JSON'):
        m.load(manifest)


def test_overlapping_logical_or_physical_slots_fail(tmp_path, monkeypatch):
    slots = cpus()
    with pytest.raises(ValueError, match='nonoverlapping'):
        m.validate_slots([slots[0]] * 4)
    monkeypatch.setattr(m._cpu, 'physical_topology', lambda selected: {cpu: (0, i % 4) for i, cpu in enumerate(selected)})
    with pytest.raises(ValueError, match='distinct physical'):
        m.validate_slots(slots)


def test_collision_refuses_existing_output_and_no_retry(tmp_path, monkeypatch):
    d, manifest = make_fixture(tmp_path, count=1)
    root = Path(d['output_root']); root.mkdir()
    with pytest.raises(FileExistsError, match='fresh output'):
        m.verify(manifest, m.digest(manifest))
    Path(d['tasks'][0]['output']).mkdir(parents=True)
    saved = []
    with pytest.raises(FileExistsError, match='collision'):
        m.execute_tasks(d, cpus(), root, lambda rows: saved.__setitem__(slice(None), copy.deepcopy(rows)))
    assert saved[0]['status'] == 'failed' and 'pid' not in saved[0]


@pytest.mark.parametrize('mode', ['missing', 'mutate', 'descendant'])
def test_post_checks_or_missing_outputs_fail_and_leave_receipt(tmp_path, monkeypatch, mode):
    d, manifest = make_fixture(tmp_path, count=1, modes={0: mode})
    resource_fixture(monkeypatch, cpus())
    with pytest.raises((RuntimeError, ValueError)):
        m.run(manifest, m.digest(manifest))
    receipt = m.load(Path(d['output_root']) / 'receipt.json')
    assert receipt['status'] == 'failed'
    assert len(receipt['tasks']) == 1 and receipt['internal_cost']['wall_seconds'] > 0
    if mode == 'mutate':
        assert receipt['hashes_unchanged'] is False and 'post_hash_error' in receipt
    else:
        assert receipt['tasks'][0]['status'] == 'failed'
        assert receipt['hashes_unchanged'] is True


def test_success_preserves_hashes_and_captures_all_costs(tmp_path, monkeypatch):
    d, manifest = make_fixture(tmp_path, count=2)
    resource_fixture(monkeypatch, cpus())
    receipt = m.run(manifest, m.digest(manifest))
    assert receipt['status'] == 'completed_execution'
    assert receipt['hashes_before'] == receipt['hashes_after']
    assert receipt['hashes_unchanged'] is True
    assert receipt['success_scope'].startswith('execution only')
    assert receipt['internal_cost']['user_seconds'] >= sum(row['cost']['user_seconds'] for row in receipt['tasks'])
    assert receipt['resources_before']['gpu'] == receipt['resources_after']['gpu']
    assert m.load(Path(d['output_root']) / 'manifest.json') == d
    with pytest.raises(FileExistsError):
        m.run(manifest, m.digest(manifest))


def test_interruption_cleans_live_processes_and_preserves_unstarted(tmp_path):
    d, _ = make_fixture(tmp_path, count=6, modes={i: 'slow' for i in range(6)})
    root = Path(d['output_root']); root.mkdir()
    saved, interrupted = [], False
    def update(rows):
        nonlocal interrupted
        saved[:] = copy.deepcopy(rows)
        if not interrupted and sum(row['status'] == 'running' for row in rows) == 2:
            interrupted = True
            raise KeyboardInterrupt('synthetic cancellation')
    with pytest.raises(KeyboardInterrupt):
        m.execute_tasks(d, cpus(), root, update)
    assert all(row['status'] == 'terminated' for row in saved[:2])
    assert all(row['status'] == 'unstarted' for row in saved[2:])
    for row in saved[:2]:
        with pytest.raises(ProcessLookupError): os.kill(row['pid'], 0)


def test_launch_error_marks_failed_task_and_preserves_remaining(tmp_path, monkeypatch):
    d, _ = make_fixture(tmp_path, count=3)
    root = Path(d['output_root']); root.mkdir()
    def fail(*args, **kwargs): raise OSError('synthetic launch failure')
    monkeypatch.setattr(m.subprocess, 'Popen', fail)
    saved = []
    with pytest.raises(OSError, match='launch failure'):
        m.execute_tasks(d, cpus(), root, lambda rows: saved.__setitem__(slice(None), copy.deepcopy(rows)))
    assert saved[0]['status'] == 'failed' and saved[0]['launch_failed']
    assert all(row['status'] == 'unstarted' for row in saved[1:])


def test_interrupted_receipt_save_preserves_previous_and_allows_final_save(tmp_path, monkeypatch):
    path = tmp_path / 'receipt.json'
    m.save(path, {'status': 'before'})
    original = m.json.dump
    def interrupt(*args, **kwargs): raise KeyboardInterrupt('during receipt write')
    monkeypatch.setattr(m.json, 'dump', interrupt)
    with pytest.raises(KeyboardInterrupt): m.save(path, {'status': 'during'})
    assert m.load(path) == {'status': 'before'}
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(m.json, 'dump', original)
    m.save(path, {'status': 'final'})
    assert m.load(path) == {'status': 'final'}
