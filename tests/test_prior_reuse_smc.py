import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('prior_reuse_smc',
    Path(__file__).resolve().parents[1] / 'scripts/benchmark_prior_reuse_smc.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture(tmp_path, monkeypatch):
    names = list(m.HASHES) + ['scripts/check_galsbi_particles.py', 'sbsi/scene_exchange.py']
    arms, sources, hashes = {}, {}, {}
    for arm in ('baseline', 'optimized'):
        root = tmp_path / arm
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + (arm if name in m.HASHES else 'identical'))
        (root / 'SHA256SUMS').write_text(''.join(m.digest(root / name) + '  ' + name + '\n' for name in names))
        arms[arm] = dict(source=str(root), manifest_sha256=m.digest(root / 'SHA256SUMS'))
        sources[arm] = m._exact.manifest(root)
    for name in m.HASHES:
        hashes[name] = tuple(sources[a][name] for a in ('baseline', 'optimized'))
    monkeypatch.setattr(m, 'HASHES', hashes)
    monkeypatch.setattr(m, 'SOURCE_COUNT', len(names))
    monkeypatch.setattr(m, 'BASE_MANIFEST', arms['baseline']['manifest_sha256'])
    reference = tmp_path / 'reference'
    reference.mkdir()
    external = [tmp_path / 'initial.json', tmp_path / 'geometry.json',
                reference / 'protocol.json', reference / 'observations.parquet']
    for p in external:
        p.write_text('{}')
    runtime = dict(python='3.10.14', numpy='2.2.6', torch='2.9.1+cu126', device='cuda',
        cuda_device_name='Tesla V100-SXM2-32GB', torch_num_threads=2,
        slurm_job_id='123', slurm_node_list='th-cl-nv02')
    protocol = dict(seed=m.SEEDS[0], particles=256, stages=64, moves=64, scale=1.,
        observations=m.OBSERVATIONS, fixed_shear=[0., 0.], density_batch_size=4096,
        betas=[(i / 64) ** 2 for i in range(65)], reference=str(reference),
        initial_proposal=dict(path=str(external[0])), physical_geometry=dict(path=str(external[1])),
        implementation_sha256={str(Path(arms['baseline']['source']) / n): c for n, c in sources['baseline'].items()},
        runtime=runtime)
    template = tmp_path / 'template.json'
    m.save(template, protocol)
    d = dict(schema_version=1, arms=arms,
        protocol_template=dict(path=str(template), sha256=m.digest(template), source=arms['baseline']['source']),
        external_files={str(p): m.digest(p) for p in external},
        harness_files={str(Path(p).resolve()): m.digest(p) for p in [m.__file__, m._cpu.__file__, m._exact.__file__]},
        python=sys.executable, output=str(tmp_path / 'runs'), reference=str(reference),
        initial_proposal=str(external[0]), physical_geometry=str(external[1]),
        settings=m.SETTINGS, expected_runtime={k: v for k, v in runtime.items() if k != 'slurm_job_id'},
        seeds=m.SEEDS, observations=m.OBSERVATIONS, source_preparation_seconds=6.)
    return d, sources, protocol


def rewrite_manifest(d, arm):
    root = Path(d['arms'][arm]['source'])
    old = (root / 'SHA256SUMS').read_text().splitlines()
    (root / 'SHA256SUMS').write_text(''.join(m.digest(root / line.split('  ', 1)[1]) + '  ' + line.split('  ', 1)[1] + '\n' for line in old))
    d['arms'][arm]['manifest_sha256'] = m.digest(root / 'SHA256SUMS')


def test_exact_three_file_identity_gate(tmp_path, monkeypatch):
    d, sources, _ = fixture(tmp_path, monkeypatch)
    assert m.verify(d) == sources
    p = Path(d['arms']['optimized']['source']) / 'sbsi/scene_exchange.py'
    p.write_text('unsolicited fourth change')
    rewrite_manifest(d, 'optimized')
    with pytest.raises(ValueError, match='exactly the three'):
        m.verify(d)


@pytest.mark.parametrize('name', list(m.HASHES))
def test_each_candidate_hash_is_pinned(tmp_path, monkeypatch, name):
    d, _, _ = fixture(tmp_path, monkeypatch)
    (Path(d['arms']['optimized']['source']) / name).write_text('different optimization')
    rewrite_manifest(d, 'optimized')
    with pytest.raises(ValueError, match='unknown baseline or prior-reuse'):
        m.verify(d)


def test_missing_candidate_change_rejected(tmp_path, monkeypatch):
    d, _, _ = fixture(tmp_path, monkeypatch)
    name = 'sbsi/galsbi_prior.py'
    (Path(d['arms']['optimized']['source']) / name).write_bytes((Path(d['arms']['baseline']['source']) / name).read_bytes())
    rewrite_manifest(d, 'optimized')
    with pytest.raises(ValueError, match='exactly the three'):
        m.verify(d)


@pytest.mark.parametrize('case', ['seed', 'observation', 'schema', 'preparation', 'harness', 'external', 'runtime', 'settings'])
def test_declaration_and_protocol_guards(tmp_path, monkeypatch, case):
    d, _, _ = fixture(tmp_path, monkeypatch)
    if case == 'seed': d['seeds'] = [1, 2, 3, 4]
    elif case == 'observation': d['observations'] = [0, 1, 2, 3]
    elif case == 'schema': d['schema_version'] = True
    elif case == 'preparation': d['source_preparation_seconds'] = float('nan')
    elif case == 'harness': del d['harness_files'][str(Path(m._cpu.__file__).resolve())]
    elif case == 'external': Path(d['initial_proposal']).write_text('changed')
    elif case == 'runtime': d['expected_runtime']['torch_num_threads'] = 4
    else: d['settings'] = dict(m.SETTINGS, moves=32)
    with pytest.raises(ValueError): m.verify(d)


def population(folder, protocol):
    folder.mkdir(parents=True)
    stages = [dict(stage=i, beta=protocol['betas'][i], elapsed_seconds=float(i),
                   extinct=[False] * 4, score=[1., 0.]) for i in range(1, 65)]
    m.save(folder / 'protocol.json', protocol)
    m.save(folder / 'progress.json', dict(requested=64, stages=stages))
    np.savez(folder / 'particles.npz', x=np.array([1., -0.]))
    np.savez(folder / 'scenes.npz', indptr=np.array([0, 0], dtype='i8'), marks=np.empty((0, 2)))
    r = dict(status='completed', particles=256, observations=m.OBSERVATIONS,
        final_stage=stages[-1], timing_seconds=dict(initialization=1., sampling=2., total=3.))
    for key in m._exact.ARTIFACT_HASH_KEYS:
        name = key[:-7] + ('.json' if key == 'protocol_sha256' else '.npz')
        r[key] = m.digest(folder / name)
    m.save(folder / 'result.json', r)


def pair_fixture(tmp_path, monkeypatch):
    d, sources, protocol = fixture(tmp_path, monkeypatch)
    pair = Path(d['output']) / 'pair_0'
    for arm in ('baseline', 'optimized'):
        p = copy.deepcopy(protocol)
        p['implementation_sha256'] = {str(Path(d['arms'][arm]['source']) / n): c for n, c in sources[arm].items()}
        population(pair / arm, p)
    return d, sources, pair


def test_verified_hash_normalization_preserves_exact_science(tmp_path, monkeypatch):
    d, sources, pair = pair_fixture(tmp_path, monkeypatch)
    assert m.compare_pair(d, sources, pair, 0)
    # Do not let the new experiment alter shared exchange exclusions or hashes.
    assert m._exact.BASE_HASH == 'c5102579e3dfd1cd3c3e9ae5a267cf8ac0df9a6a82d60417fd2772e56aebd5ad'
    assert ('runtime', 'torch_num_threads') not in m._exact.TIMINGS['protocol']


@pytest.mark.parametrize('case', ['score', 'missing_stage', 'beta', 'runtime', 'array', 'archive', 'extinction'])
def test_full_output_replay_rejects_corruption(tmp_path, monkeypatch, case):
    d, sources, pair = pair_fixture(tmp_path, monkeypatch)
    folder = pair / 'optimized'
    if case in ('score', 'missing_stage', 'beta', 'extinction'):
        p = m.load(folder / 'progress.json')
        if case == 'score': p['stages'][10]['score'][0] += .01
        elif case == 'missing_stage': p['stages'].pop(10)
        elif case == 'beta': p['stages'][10]['beta'] += .01
        else: p['stages'][10]['extinct'][0] = True
        m.save(folder / 'progress.json', p)
    elif case == 'runtime':
        p = m.load(folder / 'protocol.json')
        p['runtime']['torch_num_threads'] = 4
        m.save(folder / 'protocol.json', p)
        r = m.load(folder / 'result.json')
        r['protocol_sha256'] = m.digest(folder / 'protocol.json')
        m.save(folder / 'result.json', r)
    elif case == 'array':
        np.savez(folder / 'particles.npz', x=np.array([1., 0.]))
        r = m.load(folder / 'result.json')
        r['particles_sha256'] = m.digest(folder / 'particles.npz')
        m.save(folder / 'result.json', r)
    else:
        with (folder / 'scenes.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): m.compare_pair(d, sources, pair, 0)


def full_report_fixture(tmp_path, monkeypatch):
    d, sources, protocol = fixture(tmp_path, monkeypatch)
    declaration = tmp_path / 'declaration.json'
    m.save(declaration, d)
    root = Path(d['output'])
    for replica in range(4):
        directory = root / f'pair_{replica}'
        arms = {}
        for arm, wall in [('baseline', 10.), ('optimized', 8.)]:
            p = copy.deepcopy(protocol)
            p['seed'] = m.SEEDS[replica]
            p['implementation_sha256'] = {str(Path(d['arms'][arm]['source']) / n): c for n, c in sources[arm].items()}
            population(directory / arm, p)
            command = m.driver_command(d, arm, m.SEEDS[replica], directory / arm)
            runtime = directory / (arm + '_runtime')
            runtime.mkdir()
            m.save(runtime / 'worker_0.json', dict(command=command, affinity=[0, 1], pid=100,
                node='n', cuda_visible_devices='0'))
            worker = dict(command=command, cpus=[0, 1], pid=100, returncode=0, process_seconds=wall)
            arms[arm] = dict(workers={'0': worker}, wall_seconds=wall,
                output_sha256={name: m.digest(directory / arm / name) for name in m.OUTPUTS})
        resource = dict(slurm_job_id='1', node='n', slurm_node_list='th-cl-nv02', selected_cpus=[0, 1],
            thread_environment={}, cpu_topology={'0': [0, 0], '1': [0, 1]},
            gpu=dict(uuid='gpu', name='Tesla V100-SXM2-32GB', cuda_visible_devices='0', memory_total_mib=32000))
        m.save(directory / 'receipt.json', dict(status='completed', replica=replica, seed=m.SEEDS[replica],
            order=['baseline', 'optimized'] if replica % 2 == 0 else ['optimized', 'baseline'],
            declaration_sha256=m.digest(declaration), resources=resource, arms=arms,
            bitwise_saved_output_replay=True, pair_internal_wall_seconds=19.))
        m.save(root / f'pair_{replica}_process_cost.json', dict(wall_seconds=20., exit_code=0))
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    return d, declaration


def test_full_report_and_finalization_charge_all_shared_costs(tmp_path, monkeypatch):
    _, declaration = full_report_fixture(tmp_path, monkeypatch)
    output, cost = tmp_path / 'report.json', tmp_path / 'report_process_cost.json'
    m.report(declaration, output)
    r = m.load(output)
    assert r['status'] == 'pending_report_process_cost' and r['all_four_saved_output_replay'] is True
    assert r['pairs'][0]['charged_seconds'] == dict(baseline=11., optimized=9.)
    m.save(cost, dict(wall_seconds=r['internal_report_seconds'] + 2., exit_code=0))
    m.finalize(output, cost)
    r = m.load(output)
    shared = m.load(cost)['wall_seconds'] / 2 + 3.
    assert r['new_work_seconds_by_arm'] == dict(baseline=44. + shared, optimized=36. + shared)
    assert r['status'] == 'completed'
    with pytest.raises(ValueError): m.finalize(output, cost)


@pytest.mark.parametrize('case', ['failed', 'missing', 'order', 'gpu', 'cpu', 'output_hash', 'cost_short', 'cost_failed'])
def test_report_rejects_failed_incomplete_or_unmatched_costs(tmp_path, monkeypatch, case):
    d, declaration = full_report_fixture(tmp_path, monkeypatch)
    pair = Path(d['output']) / 'pair_2'
    r = m.load(pair / 'receipt.json')
    if case == 'failed': r['arms']['baseline']['workers']['0']['returncode'] = 7
    elif case == 'missing': del r['arms']['baseline']
    elif case == 'order': r['order'].reverse()
    elif case == 'gpu': r['resources']['gpu']['uuid'] = 'different'
    elif case == 'cpu': r['resources']['cpu_topology']['1'] = [0, 0]
    elif case == 'output_hash': r['arms']['optimized']['output_sha256']['result.json'] = 'wrong'
    else:
        cost = dict(wall_seconds=10. if case == 'cost_short' else 20., exit_code=7 if case == 'cost_failed' else 0)
        m.save(Path(d['output']) / 'pair_2_process_cost.json', cost)
    m.save(pair / 'receipt.json', r)
    with pytest.raises(ValueError): m.report(declaration, tmp_path / 'report.json')


def test_no_overwrite_reorder_or_changed_allocation(tmp_path, monkeypatch):
    d, declaration = full_report_fixture(tmp_path, monkeypatch)
    root = Path(d['output'])
    r = m.load(root / 'pair_0/receipt.json')
    with pytest.raises(FileExistsError): m.validate_previous(root, 0, m.digest(declaration), r['resources'])
    resource = copy.deepcopy(r['resources'])
    resource['gpu']['uuid'] = 'other'
    with pytest.raises(ValueError, match='same original allocation'):
        m.validate_previous(root, 1, m.digest(declaration), resource)
