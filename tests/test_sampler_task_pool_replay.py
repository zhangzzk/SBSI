import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('task_pool_replay',
    Path(__file__).resolve().parents[1] / 'scripts/benchmark_sampler_task_pool_replay.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def phases_fixture(tmp_path):
    declaration = dict(python=str(Path(sys.executable).resolve()), source=dict(path=str(tmp_path / 'source')),
        reference='/original/reference', initial_proposal='/original/proposal.json',
        physical_geometry='/original/geometry.json',
        protocol_template=dict(path='/original/protocol.json', sha256='a' * 64))
    return {phase: dict(declaration=copy.deepcopy(declaration)) for phase in m.PHASES}


def test_eight_original_tasks_preserve_all_scientific_arguments_and_two_waves(tmp_path):
    phases = phases_fixture(tmp_path)
    source, output = tmp_path / 'source', tmp_path / 'output'
    python = str(Path(sys.executable).resolve())
    tasks = m.tasks_for(phases, source, output, python)
    assert [task['id'] for task in tasks] == [f'{phase}_{i}' for phase in m.PHASES for i in range(4)]
    assert [task['identity']['seed'] for task in tasks] == list(range(2026103700, 2026103704)) + list(range(2026103800, 2026103804))
    for task in tasks:
        phase = task['id'].rsplit('_', 1)[0]
        original = m.pool._cpu.driver_command(phases[phase]['declaration'], task['identity']['seed'], task['output'])
        assert task['argv'][6:] == original[5:]
        assert task['identity']['observations'] == [58165, 0, 1, 8]
        for key, value in [('particles', '256'), ('stages', '64'), ('moves', '64'), ('density-batch-size', '4096')]:
            assert task['argv'][task['argv'].index('--' + key) + 1] == value
        assert task['expected_outputs'] == list(m.FILES)


@pytest.mark.parametrize('case', ['missing_task', 'changed_seed', 'changed_recipe', 'wrong_source', 'missing_evidence'])
def test_replay_gate_rejects_partial_or_changed_original_manifest(tmp_path, monkeypatch, case):
    phases = phases_fixture(tmp_path)
    source, output = tmp_path / 'source', tmp_path / 'output'
    python = str(Path(sys.executable).resolve())
    evidence = {'/original/proof': 'e' * 64}
    tasks = m.tasks_for(phases, source, output, python)
    manifest = dict(tasks=tasks, sources={'original': dict(path=str(source), manifest_sha256=m.SOURCE_SHA)},
                    input_files=dict(evidence), output_root=str(output), python=python)
    declaration = dict(schema='sbsi-sampler-task-pool-replay-v1', manifest=str(tmp_path / 'tasks.json'),
        manifest_sha256='a' * 64, root=str(tmp_path), evidence_sha256=evidence, preparation_seconds=1.)
    path = tmp_path / 'replay.json'; m.pool.save(path, declaration)
    monkeypatch.setattr(m, 'imported', lambda *_: SimpleNamespace(verify=lambda *_, **__: (manifest, {})))
    monkeypatch.setattr(m, 'original_evidence', lambda *_: (phases, evidence))
    assert m.verify_replay(path)[1] == manifest
    if case == 'missing_task': manifest['tasks'].pop()
    elif case == 'changed_seed': manifest['tasks'][0]['identity']['seed'] += 10
    elif case == 'changed_recipe': manifest['tasks'][0]['argv'].append('--different-law')
    elif case == 'wrong_source': manifest['sources']['original']['manifest_sha256'] = 'b' * 64
    else: manifest['input_files'].clear()
    with pytest.raises(ValueError): m.verify_replay(path)


def test_preparation_requires_scheduler_and_fresh_destination(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(ValueError, match='scheduler'): m.prepare(tmp_path, tmp_path / 'new')
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic')
    with pytest.raises(FileExistsError): m.prepare(tmp_path, tmp_path)


def finalization_fixture(tmp_path):
    root, prepared = tmp_path / 'runs', tmp_path / 'prepared'
    root.mkdir(); prepared.mkdir()
    report, timer, pool_timer = tmp_path / 'report.json', tmp_path / 'audit_cost.json', tmp_path / 'pool_cost.json'
    manifest, declaration = prepared / 'tasks.json', prepared / 'replay.json'
    m.pool.save(manifest, dict(output_root=str(root)))
    d = dict(schema='sbsi-sampler-task-pool-replay-v1', manifest=str(manifest),
        manifest_sha256=m.pool.digest(manifest), report=str(report), audit_process_cost=str(timer),
        pool_process_cost=str(pool_timer), preparation_seconds=2.)
    m.pool.save(declaration, d)
    receipt = root / 'receipt.json'
    m.pool.save(receipt, dict(status='completed_execution', hashes_unchanged=True,
        manifest_sha256=d['manifest_sha256'], internal_cost=dict(wall_seconds=99.)))
    pool_cost = dict(exit_code=0, wall_seconds=100., user_seconds=90., system_seconds=2., max_rss_kib=100)
    m.pool.save(pool_timer, pool_cost)
    cost = dict(exit_code=0, wall_seconds=1.2, user_seconds=.5, system_seconds=.1, max_rss_kib=10)
    m.pool.save(timer, cost)
    rows = [dict(id=f'{phase}_{i}', seed=seed, all_scientific_json_equal=True, complete_64_stages=True)
            for phase, definition in m.PHASES.items() for i, seed in enumerate(definition['seeds'])]
    result = dict(status='pending_audit_process_cost', all_eight_original_saved_outputs_equal=True,
        comparisons=rows, audit_internal_seconds=1., preparation_seconds=2., pool_process_cost=pool_cost,
        replay_declaration=dict(path=str(declaration), sha256=m.pool.digest(declaration)),
        proof_sha256={str(path): m.pool.digest(path) for path in [declaration, receipt, pool_timer]})
    m.pool.save(report, result)
    return report, timer, d, result


@pytest.mark.parametrize('case', ['failed_process', 'negative_time', 'undercount', 'missing_task', 'false_replay'])
def test_final_gate_refuses_failed_undercounted_or_partial_evidence(tmp_path, case):
    report, timer, _, result = finalization_fixture(tmp_path)
    cost = m.pool.load(timer)
    if case == 'failed_process': cost['exit_code'] = 1
    elif case == 'negative_time': cost['system_seconds'] = -1
    elif case == 'undercount': cost['wall_seconds'] = .5
    elif case == 'missing_task': result['comparisons'].pop()
    else: result['all_eight_original_saved_outputs_equal'] = False
    m.pool.save(report, result); m.pool.save(timer, cost)
    with pytest.raises(ValueError): m.finalize(report, timer)
    assert m.pool.load(report)['status'] == 'pending_audit_process_cost'


@pytest.mark.parametrize('changed', ['declaration', 'receipt', 'pool_cost', 'manifest'])
def test_finalization_rechecks_audited_proof_and_declared_manifest(tmp_path, changed):
    report, timer, d, result = finalization_fixture(tmp_path)
    paths = dict(declaration=result['replay_declaration']['path'], receipt=str(tmp_path / 'runs/receipt.json'),
                 pool_cost=d['pool_process_cost'], manifest=d['manifest'])
    path = Path(paths[changed])
    # Even a semantically equivalent changed file must fail its original hash.
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='changed'): m.finalize(report, timer)
    assert m.pool.load(report)['status'] == 'pending_audit_process_cost'


@pytest.mark.parametrize('case', ['missing_proof', 'extra_proof', 'wrong_hash', 'wrong_declaration_pin',
                                  'pool_cost_content', 'preparation_cost', 'wrong_output', 'swapped_cost'])
def test_finalization_rejects_incomplete_proof_and_swapped_paths_or_costs(tmp_path, case):
    report, timer, d, result = finalization_fixture(tmp_path)
    actual_output, actual_cost = report, timer
    if case == 'missing_proof': result['proof_sha256'].pop(d['pool_process_cost'])
    elif case == 'extra_proof': result['proof_sha256'][str(timer)] = m.pool.digest(timer)
    elif case == 'wrong_hash': result['proof_sha256'][d['pool_process_cost']] = 'a' * 64
    elif case == 'wrong_declaration_pin': result['replay_declaration']['sha256'] = 'b' * 64
    elif case == 'pool_cost_content': result['pool_process_cost']['wall_seconds'] = 50.
    elif case == 'preparation_cost': result['preparation_seconds'] = 0.
    elif case == 'wrong_output': actual_output = tmp_path / 'copied_report.json'
    elif case == 'swapped_cost':
        actual_cost = tmp_path / 'copied_cost.json'
        actual_cost.write_bytes(timer.read_bytes())
    m.pool.save(report, result)
    if actual_output != report: actual_output.write_bytes(report.read_bytes())
    with pytest.raises(ValueError): m.finalize(actual_output, actual_cost)
    assert m.pool.load(actual_output)['status'] == 'pending_audit_process_cost'


def test_finalization_preserves_proof_and_charges_small_validation_work(tmp_path):
    report, timer, _, before = finalization_fixture(tmp_path)
    m.finalize(report, timer)
    result = m.pool.load(report)
    assert result['status'] == 'completed' and result['generic_pool_replay_passed'] is True
    assert result['proof_sha256'] == before['proof_sha256']
    assert result['replay_declaration'] == before['replay_declaration']
    assert result['audit_process_cost'] == m.pool.load(timer)
    assert result['audit_process_cost_sha256'] == m.pool.digest(timer)
    assert result['complete_new_work_seconds'] == 103.2 + result['finalization_validation_seconds']
    with pytest.raises(ValueError, match='pending'): m.finalize(report, timer)


def test_replay_intervals_require_four_workers_and_disjoint_live_slots():
    rows = [dict(start_offset_seconds=i // 4 * 2, end_offset_seconds=i // 4 * 2 + 1,
                 cpus=[2 * (i % 4), 2 * (i % 4) + 1]) for i in range(8)]
    assert m.check_intervals(rows) == 4
    changed = copy.deepcopy(rows); changed[1]['cpus'] = changed[0]['cpus']
    with pytest.raises(ValueError, match='overlap'): m.check_intervals(changed)
    changed = copy.deepcopy(rows); changed[4]['start_offset_seconds'] = .5
    with pytest.raises(ValueError): m.check_intervals(changed)
    with pytest.raises(ValueError, match='four-worker'): m.check_intervals(rows[:2])


def test_audit_rejects_undeclared_report_path_before_replay(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic')
    monkeypatch.setattr(m, 'verify_replay', lambda _: (dict(report=str(tmp_path / 'declared.json')), {}, {}, {}))
    with pytest.raises(ValueError, match='report path'):
        m.audit(tmp_path / 'replay.json', tmp_path / 'wrong.json')
    assert not (tmp_path / 'wrong.json').exists()
