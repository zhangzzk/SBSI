import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('smc_concurrency',
    Path(__file__).resolve().parents[1] / 'scripts/benchmark_smc_concurrency.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture(tmp_path):
    source = tmp_path / 'source'
    files = ['scripts/check_galsbi_particles.py', 'sbsi/scene_exchange.py']
    for name in files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen ' + name)
    (source / 'SHA256SUMS').write_text(''.join(m.digest(source / name) + '  ' + name + '\n' for name in files))
    sources = m._exact.manifest(source)
    reference = tmp_path / 'reference'
    reference.mkdir()
    external = [tmp_path / 'initial.json', tmp_path / 'geometry.json',
                reference / 'protocol.json', reference / 'observations.parquet']
    for path in external:
        path.write_text('{}')
    runtime = dict(python='3.10.14', numpy='2.2.6', torch='2.9.1+cu126', device='cuda',
        cuda_device_name='Tesla V100-SXM2-32GB', torch_num_threads=2,
        slurm_job_id='123', slurm_node_list='th-cl-nv02')
    protocol = dict(seed=m.SEEDS[0], particles=256, stages=64, moves=64, scale=1.,
        observations=m.OBSERVATIONS, fixed_shear=[0., 0.], density_batch_size=4096,
        betas=[(i / 64) ** 2 for i in range(65)], reference=str(reference),
        initial_proposal=dict(path=str(external[0])), physical_geometry=dict(path=str(external[1])),
        implementation_sha256={str(source / name): checksum for name, checksum in sources.items()},
        runtime=runtime)
    template = tmp_path / 'template.json'
    m.save(template, protocol)
    d = dict(schema_version=1, source=dict(path=str(source), manifest_sha256=m.digest(source / 'SHA256SUMS')),
        protocol_template=dict(path=str(template), sha256=m.digest(template), source=str(source)),
        external_files={str(p): m.digest(p) for p in external},
        harness_files={str(p): m.digest(p) for p in [Path(m.__file__).resolve(), m._HELPER]},
        python=sys.executable, output=str(tmp_path / 'runs'), reference=str(reference),
        physical_geometry=str(external[1]), initial_proposal=str(external[0]),
        observations=m.OBSERVATIONS, seeds=m.SEEDS, order=m.ORDER, threads_per_worker=2,
        required_allocated_cpus=8, settings=m.SETTINGS,
        expected_runtime={k: v for k, v in runtime.items() if k != 'slurm_job_id'})
    return d, sources, protocol


def save_population(folder, protocol):
    folder.mkdir(parents=True)
    stages = [dict(stage=i, beta=protocol['betas'][i], elapsed_seconds=float(i),
                   extinct=[False] * 4, score=[1., 0.]) for i in range(1, 65)]
    m.save(folder / 'protocol.json', protocol)
    m.save(folder / 'progress.json', dict(requested=64, stages=stages))
    np.savez(folder / 'particles.npz', x=np.array([1., -0.]), dtype_test=np.array([1], dtype='i8'))
    np.savez(folder / 'scenes.npz', indptr=np.array([0, 0], dtype='i8'), marks=np.empty((0, 2)))
    result = dict(status='completed', particles=256, observations=m.OBSERVATIONS,
                  final_stage=stages[-1], timing_seconds=dict(initialization=1., sampling=2., total=3.))
    for key in m._exact.ARTIFACT_HASH_KEYS:
        name = key[:-7] + ('.json' if key == 'protocol_sha256' else '.npz')
        result[key] = m.digest(folder / name)
    m.save(folder / 'result.json', result)


def test_fixed_declaration_verifies_and_paths_only_normalize(tmp_path):
    d, sources, protocol = fixture(tmp_path)
    assert m.verify(d) == sources
    p = m.expected_protocol(d, sources, 3)
    assert p['seed'] == m.SEEDS[3]
    assert p['implementation_sha256'] == sources
    other = tmp_path / 'relocated'
    shifted = copy.deepcopy(protocol)
    shifted['implementation_sha256'] = {str(other / k): v for k, v in sources.items()}
    assert m.normalize_protocol(shifted, other, sources) == m.normalize_protocol(protocol, d['source']['path'], sources)
    shifted['implementation_sha256'][str(other / 'sbsi/scene_exchange.py')] = 'different'
    with pytest.raises(ValueError, match='changed protocol'):
        m.normalize_protocol(shifted, other, sources)


@pytest.mark.parametrize('key,value', [
    ('seeds', [1, 2, 3, 4]), ('order', [1, 4, 2, 4, 2, 1]),
    ('observations', [58165, 0, 1, 9]), ('threads_per_worker', True),
    ('required_allocated_cpus', 16), ('schema_version', True),
    ('source_preparation_seconds', -1), ('source_preparation_seconds', float('nan')),
    ('source_preparation_seconds', True)])
def test_wrong_declaration_rejected(tmp_path, key, value):
    d, _, _ = fixture(tmp_path)
    d[key] = value
    with pytest.raises(ValueError):
        m.verify(d)


def test_changed_external_or_harness_is_rejected(tmp_path):
    d, _, _ = fixture(tmp_path)
    Path(d['initial_proposal']).write_text('changed')
    with pytest.raises(ValueError, match='changed external_files'):
        m.verify(d)
    del d['harness_files'][str(m._HELPER)]
    with pytest.raises(ValueError, match='both executed harness'):
        m.verify(d)


def test_physical_core_selection_avoids_smt_siblings():
    topology = {i: (0, i % 8) for i in range(16)}
    assert m.select_physical_cores(range(16), topology) == list(range(8))
    with pytest.raises(ValueError, match='distinct allocated physical cores'):
        m.select_physical_cores(range(8), {i: (0, i % 4) for i in range(8)})


def test_command_preserves_original_recipe(tmp_path):
    d, _, _ = fixture(tmp_path)
    command = m.driver_command(d, m.SEEDS[0], tmp_path / 'out')
    for option, expected in [('particles', '256'), ('moves', '64'), ('stages', '64'),
                             ('density-batch-size', '4096'), ('device', 'cuda')]:
        assert command[command.index('--' + option) + 1] == expected
    assert '--informed-environment' not in command and '--relocate-exchanges' not in command


@pytest.mark.parametrize('case', ['missing_stage', 'wrong_beta', 'runtime', 'science', 'array', 'artifact_hash'])
def test_replay_rejects_corruption(tmp_path, case):
    d, sources, protocol = fixture(tmp_path)
    first, second = tmp_path / 'a', tmp_path / 'b'
    save_population(first, protocol)
    save_population(second, protocol)
    assert m.compare_population(d, sources, first, second, 0)
    if case in ('missing_stage', 'wrong_beta', 'science'):
        p = m.load(second / 'progress.json')
        if case == 'missing_stage':
            p['stages'].pop(10)
        elif case == 'wrong_beta':
            p['stages'][10]['beta'] += .001
        else:
            p['stages'][10]['score'][0] += .001
        m.save(second / 'progress.json', p)
    elif case == 'runtime':
        p = m.load(second / 'protocol.json')
        p['runtime']['torch_num_threads'] = 4
        m.save(second / 'protocol.json', p)
        r = m.load(second / 'result.json')
        r['protocol_sha256'] = m.digest(second / 'protocol.json')
        m.save(second / 'result.json', r)
    elif case == 'array':
        np.savez(second / 'particles.npz', x=np.array([1., 0.]), dtype_test=np.array([1], dtype='i8'))
        r = m.load(second / 'result.json')
        r['particles_sha256'] = m.digest(second / 'particles.npz')
        m.save(second / 'result.json', r)
    else:
        with (second / 'scenes.npz').open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError):
        m.compare_population(d, sources, first, second, 0)


def test_replay_allows_only_named_timings_and_job_id(tmp_path):
    d, sources, protocol = fixture(tmp_path)
    first, second = tmp_path / 'a', tmp_path / 'b'
    save_population(first, protocol)
    protocol['runtime']['slurm_job_id'] = 'other'
    save_population(second, protocol)
    p = m.load(second / 'progress.json')
    for stage in p['stages']:
        stage['elapsed_seconds'] += 2.
    m.save(second / 'progress.json', p)
    r = m.load(second / 'result.json')
    r['final_stage'] = p['stages'][-1]
    r['timing_seconds']['total'] += 3.
    m.save(second / 'result.json', r)
    assert m.compare_population(d, sources, first, second, 0)
    r['new_timing'] = 1.
    m.save(second / 'result.json', r)
    with pytest.raises(ValueError):
        m.compare_population(d, sources, first, second, 0)


def test_predecessor_enforces_order_allocation_and_no_overwrite(tmp_path):
    resource = dict(slurm_job_id='1', node='n', slurm_node_list='th-cl-nv02',
        selected_cpus=list(range(8)), thread_environment={},
        gpu=dict(uuid='id', name='V100', cuda_visible_devices='0', memory_total_mib=32000))
    m.validate_predecessor(tmp_path, 0, 'hash', resource)
    previous = tmp_path / 'setting_0'
    previous.mkdir()
    m.save(previous / 'receipt.json', dict(status='completed', declaration_sha256='hash', resources=resource))
    m.validate_predecessor(tmp_path, 1, 'hash', resource)
    wrong = copy.deepcopy(resource)
    wrong['gpu']['uuid'] = 'different'
    with pytest.raises(ValueError, match='physical GPU'):
        m.validate_predecessor(tmp_path, 1, 'hash', wrong)
    with pytest.raises(FileExistsError):
        m.validate_predecessor(tmp_path, 0, 'hash', resource)
    with pytest.raises(FileNotFoundError):
        m.validate_predecessor(tmp_path, 3, 'hash', resource)


def usable_cpus():
    affinity = os.sched_getaffinity(0)
    try:
        return m.select_physical_cores(affinity, m.physical_topology(affinity), count=4)
    except ValueError:
        pytest.skip('four distinct physical cores needed for tiny affinity child test')


def test_real_worker_pool_is_bounded_and_binds_children(tmp_path):
    cpus = usable_cpus()
    commands = [[sys.executable, '-c', 'import time;time.sleep(.05)'] for _ in range(4)]
    records, wall = m.run_workers(commands, [cpus[:2], cpus[2:]], tmp_path,
        sys.executable, tmp_path, lambda _: None)
    assert set(records) == {'0', '1', '2', '3'} and wall > 0
    for key, record in records.items():
        worker = m.load(tmp_path / f'worker_{key}.json')
        assert worker['affinity'] == record['cpus']
        assert worker['command'] == commands[int(key)] and record['returncode'] == 0
    assert records['2']['start_offset_seconds'] >= min(records['0']['end_offset_seconds'], records['1']['end_offset_seconds'])
    with pytest.raises(ValueError, match='overlap'):
        m.run_workers(commands, [cpus[:2], cpus[:2]], tmp_path, sys.executable, tmp_path, lambda _: None)


def test_failed_worker_stops_queue_and_reaps_sibling(tmp_path):
    cpus = usable_cpus()
    commands = [[sys.executable, '-c', 'raise SystemExit(7)'],
                [sys.executable, '-c', 'import time;time.sleep(30)'],
                [sys.executable, '-c', 'raise AssertionError("must not launch")']]
    saved = {}
    def update(records):
        saved.update(copy.deepcopy(records))
    with pytest.raises(RuntimeError, match='failed'):
        m.run_workers(commands, [cpus[:2], cpus[2:]], tmp_path, sys.executable, tmp_path, update)
    assert set(saved) == {'0', '1'}
    assert saved['0']['returncode'] == 7
    assert saved['1']['returncode'] < 0
    with pytest.raises(ProcessLookupError):
        os.kill(saved['1']['pid'], 0)


def test_report_refuses_missing_original_settings(tmp_path, monkeypatch):
    d, _, _ = fixture(tmp_path)
    declaration = tmp_path / 'declaration.json'
    m.save(declaration, d)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    with pytest.raises(FileNotFoundError):
        m.report(declaration, tmp_path / 'report.json')
    assert not (tmp_path / 'report.json').exists()


def complete_setting_fixture(tmp_path):
    d, sources, protocol = fixture(tmp_path)
    declaration = tmp_path / 'declaration.json'
    m.save(declaration, d)
    root = Path(d['output'])
    costs = [100., 60., 40., 45., 65., 110.]
    for index, concurrency in enumerate(m.ORDER):
        directory = root / f'setting_{index}'
        directory.mkdir(parents=True)
        resource = dict(slurm_job_id='1', node='n', slurm_node_list='th-cl-nv02',
            selected_cpus=list(range(8)), cpu_topology={str(i): [0, i] for i in range(8)},
            thread_environment={}, gpu=dict(uuid='gpu1', cuda_visible_devices='0'))
        workers = {}
        for replica, seed in enumerate(m.SEEDS):
            out = directory / f'population_{replica}'
            protocol['seed'] = seed
            save_population(out, protocol)
            slot = replica % concurrency
            cpus = [2 * slot, 2 * slot + 1]
            command = m.driver_command(d, seed, out)
            workers[str(replica)] = dict(replica=replica, returncode=0, cpus=cpus,
                pid=100 + replica, command=command, process_seconds=10.,
                start_offset_seconds=float(10 * (replica // concurrency)),
                end_offset_seconds=float(10 * (replica // concurrency + 1)),
                output_sha256={name: m.digest(out / name) for name in m.OUTPUTS})
            m.save(directory / f'worker_{replica}.json', dict(affinity=cpus,
                pid=100 + replica, command=command, node='n', cuda_visible_devices='0'))
        m.save(directory / 'receipt.json', dict(status='completed', setting_index=index,
            concurrency=concurrency, repetition=0 if index < 3 else 1,
            seeds=m.SEEDS, declaration_sha256=m.digest(declaration), resources=resource,
            workers=workers, bitwise_saved_output_replay=True,
            execution_wall_seconds=costs[index], setting_wall_seconds=costs[index] + 10.))
    return d, declaration


def test_complete_report_matches_repetitions_and_includes_overhead(tmp_path, monkeypatch):
    _, declaration = complete_setting_fixture(tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    output = tmp_path / 'report.json'
    m.report(declaration, output)
    report = m.load(output)
    assert report['all_24_saved_output_replay'] is True
    assert report['total_setting_wall_seconds'] == 480.
    ratios = report['ratios']
    assert [(x['repetition'], x['concurrency']) for x in ratios] == [(0, 2), (0, 4), (1, 2), (1, 4)]
    assert ratios[0]['execution_speedup'] == 100. / 60.
    assert ratios[2]['execution_speedup'] == 110. / 65.
    assert ratios[3]['setting_inclusive_speedup'] == 120. / 55.
    with pytest.raises(ValueError, match='new output'):
        m.report(declaration, output)


@pytest.mark.parametrize('case', ['failed_worker', 'missing_worker', 'shared_gpu_changed',
    'command', 'output_identity', 'overlapping_cpu', 'same_physical_core', 'wrong_slot'])
def test_report_refuses_operational_corruption(tmp_path, monkeypatch, case):
    d, declaration = complete_setting_fixture(tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    path = Path(d['output']) / 'setting_2' / 'receipt.json'
    receipt = m.load(path)
    if case == 'failed_worker':
        receipt['workers']['0']['returncode'] = 7
    elif case == 'missing_worker':
        del receipt['workers']['0']
    elif case == 'shared_gpu_changed':
        receipt['resources']['gpu']['uuid'] = 'gpu2'
    elif case == 'command':
        receipt['workers']['0']['command'] = ['wrong']
    elif case == 'output_identity':
        receipt['workers']['0']['output_sha256']['result.json'] = 'wrong'
    elif case == 'overlapping_cpu':
        receipt['workers']['1']['cpus'] = [0, 1]
    elif case == 'same_physical_core':
        receipt['resources']['cpu_topology']['1'] = [0, 0]
    else:
        receipt['workers']['1']['cpus'] = [1, 2]
    m.save(path, receipt)
    with pytest.raises(ValueError):
        m.report(declaration, tmp_path / 'report.json')


def test_gpu_identity_resolves_remapped_cuda_ordinal_to_uuid(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    calls = []
    class Result:
        def __init__(self, text): self.stdout = text
    uuid = 'GPU-12345678-1234-5678-1234-567812345678'
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == sys.executable:
            return Result(json.dumps(dict(count=1, uuid=uuid[4:], name='Tesla V100-SXM2-32GB')))
        assert command[:3] == ['nvidia-smi', '-i', uuid]
        return Result(f'{uuid}, Tesla V100-SXM2-32GB, 32768, 20, 0\n')
    monkeypatch.setattr(m.subprocess, 'run', run)
    assert m.gpu_snapshot()['uuid'] == uuid
    assert len(calls) == 2
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(ValueError, match='exactly one visible'):
        m.gpu_snapshot()


def test_gpu_telemetry_must_match_cuda_uuid(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    class Result:
        def __init__(self, text): self.stdout = text
    def run(command, **kwargs):
        if command[0] == sys.executable:
            return Result(json.dumps(dict(count=1, uuid='GPU-correct', name='Tesla V100-SXM2-32GB')))
        return Result('GPU-wrong, Tesla V100-SXM2-32GB, 32768, 20, 0\n')
    monkeypatch.setattr(m.subprocess, 'run', run)
    with pytest.raises(ValueError, match='UUID must agree'):
        m.gpu_snapshot()
