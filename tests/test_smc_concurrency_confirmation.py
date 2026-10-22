import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = import_file('concurrency_confirmation', ROOT / 'scripts/benchmark_smc_concurrency_confirmation.py')
fixtures = import_file('confirmation_development_fixtures', ROOT / 'tests/test_smc_concurrency.py')


def ratio_fixture(k2=(1.5, 1.6), k4=(3.5, 3.4)):
    return [dict(repetition=rep, concurrency=k, setting_inclusive_speedup=values[rep])
            for rep in range(2) for k, values in [(2, k2), (4, k4)]]


def test_original_selection_rule_and_five_percent_tie():
    assert m.choose_level(ratio_fixture())['selected_workers'] == 4
    assert m.choose_level(ratio_fixture((3.3, 3.4), (3.4, 3.45)))['selected_workers'] == 2
    assert m.choose_level(ratio_fixture((1.24, 1.8), (1.25, 1.25)))['selected_workers'] == 4
    with pytest.raises(ValueError, match='no eligible'):
        m.choose_level(ratio_fixture((1.24, 2.), (1.1, 3.)))
    with pytest.raises(ValueError, match='duplicate'):
        m.choose_level(ratio_fixture()[:-1] + [ratio_fixture()[0]])


def development_fixture(tmp_path, monkeypatch):
    d, declaration = fixtures.complete_setting_fixture(tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    source = Path(d['source']['path'])
    for i in range(318):
        (source / f'extra_{i}.txt').write_text(str(i))
    names = sorted([p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'])
    (source / 'SHA256SUMS').write_text(''.join(m.digest(source / name) + '  ' + name + '\n' for name in names))
    d['source']['manifest_sha256'] = m.digest(source / 'SHA256SUMS')
    d['selection_rule'] = dict(future_test_seeds=m.SEEDS)
    m.save(declaration, d)
    root = Path(d['output'])
    for i in range(6):
        path = root / f'setting_{i}/receipt.json'
        r = m.load(path)
        r['declaration_sha256'] = m.digest(declaration)
        m.save(path, r)
        m.save(root / f'setting_{i}_process_cost.json', dict(wall_seconds=r['setting_wall_seconds'] + 1., exit_code=0))
    report_path = root / 'report.json'
    m._base.report(declaration, report_path)
    report = m.load(report_path)
    report_cost = dict(wall_seconds=report['internal_report_seconds'] + 1., exit_code=0)
    m.save(root / 'report_process_cost.json', report_cost)
    parent_path = tmp_path / 'parent_audit.json'
    pins = {str(report_path): m.digest(report_path), str(root / 'report_process_cost.json'): m.digest(root / 'report_process_cost.json')}
    pins.update({str(root / f'setting_{i}_process_cost.json'): m.digest(root / f'setting_{i}_process_cost.json') for i in range(6)})
    m.save(parent_path, dict(status='passed', all_24_saved_output_replayed_again=True,
        strict_replayed_report_equal_except_its_elapsed_time=True, original_report_sha256=m.digest(report_path),
        selected_concurrency=4, ratios=report['ratios'], report_process_cost=report_cost,
        process_costs=[m.load(root / f'setting_{i}_process_cost.json') for i in range(6)],
        pinned_reports_and_costs=pins))
    monkeypatch.setattr(m, 'SOURCE_SHA', d['source']['manifest_sha256'])
    monkeypatch.setattr(m, 'DEVELOPMENT_DECLARATION_SHA', m.digest(declaration))
    monkeypatch.setattr(m, 'DEVELOPMENT_HARNESS_SHA', m.digest(m._base.__file__))
    return d, declaration, report_path, parent_path


def confirmation_fixture(tmp_path, monkeypatch):
    old, declaration, report_path, parent_path = development_fixture(tmp_path, monkeypatch)
    selection = m.audit_development(declaration, report_path, parent_path, replay=True)
    selection_path = tmp_path / 'selection.json'
    m.save(selection_path, selection)
    d = dict(old, phase=m.PHASE, selected_workers=4, seeds=m.SEEDS, order=m.ORDER,
        output=str(tmp_path / 'confirmation'), source_preparation_seconds=2.,
        external_files=dict(selection['evidence_sha256']), harness_files=dict(old['harness_files']),
        development_selection=dict(path=str(selection_path), sha256=m.digest(selection_path)))
    d['external_files'][str(selection_path)] = m.digest(selection_path)
    d['harness_files'][str(Path(m.__file__).resolve())] = m.digest(m.__file__)
    path = tmp_path / 'confirmation.json'
    m.save(path, d)
    return d, path, selection


def test_development_evidence_replay_and_all_seven_costs_bound(tmp_path, monkeypatch):
    d, declaration, report, parent = development_fixture(tmp_path, monkeypatch)
    audit = m.audit_development(declaration, report, parent, replay=True)
    assert audit['all_24_saved_output_replay_verified'] is True
    assert audit['selection']['selected_workers'] == 4
    assert len([p for p in audit['evidence_sha256'] if p.endswith('_process_cost.json')]) == 7
    assert str(parent) in audit['evidence_sha256']


@pytest.mark.parametrize('case', ['cost', 'failed_cost', 'report_claim', 'wrong_selection', 'parent', 'output'])
def test_development_corruption_rejected(tmp_path, monkeypatch, case):
    d, declaration, report_path, parent = development_fixture(tmp_path, monkeypatch)
    root = Path(d['output'])
    if case in ('cost', 'failed_cost'):
        path = root / 'setting_2_process_cost.json'
        m.save(path, dict(wall_seconds=1. if case == 'cost' else 100., exit_code=1 if case == 'failed_cost' else 0))
    elif case in ('report_claim', 'wrong_selection'):
        r = m.load(report_path)
        if case == 'report_claim': r['all_24_saved_output_replay'] = False
        else: r['ratios'][1]['setting_inclusive_speedup'] = 99.
        m.save(report_path, r)
    elif case == 'parent':
        r = m.load(parent); r['all_24_saved_output_replayed_again'] = False; m.save(parent, r)
    else:
        with (root / 'setting_3/population_1/scenes.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): m.audit_development(declaration, report_path, parent)


def test_confirmation_strict_phase_and_original_validation_unchanged(tmp_path, monkeypatch):
    d, _, _ = confirmation_fixture(tmp_path, monkeypatch)
    sources = m.verify(d)
    assert m.expected_protocol(d, sources, 0)['seed'] == 2026103800
    with pytest.raises(ValueError, match='unknown concurrency declaration'):
        m._base.verify(d)
    assert m._base.SEEDS == [2026103700, 2026103701, 2026103702, 2026103703]
    assert m._base.ORDER == [1, 2, 4, 4, 2, 1]
    for key, value in [('selected_workers', 2), ('order', [1, 4, 1, 4]), ('seeds', m._base.SEEDS), ('phase', 'other')]:
        bad = dict(d); bad[key] = value
        with pytest.raises(ValueError): m.verify(bad)


def test_confirmation_cannot_drop_development_cost_pins(tmp_path, monkeypatch):
    d, _, _ = confirmation_fixture(tmp_path, monkeypatch)
    target = next(p for p in d['external_files'] if p.endswith('setting_2_process_cost.json'))
    del d['external_files'][target]
    with pytest.raises(ValueError, match='all original development evidence'):
        m.verify(d)


def complete_confirmation_fixture(tmp_path, monkeypatch, slow=False, outer_slow=False):
    d, declaration, _ = confirmation_fixture(tmp_path, monkeypatch)
    sources = m.verify(d)
    root = Path(d['output'])
    internal = [100., 30., 35. if not slow else 95., 110.]
    for index, concurrency in enumerate(m.ORDER):
        folder = root / f'setting_{index}'
        folder.mkdir(parents=True)
        resource = dict(slurm_job_id='1', node='n', selected_cpus=list(range(8)),
            cpu_topology={str(i): [0, i] for i in range(8)}, thread_environment={},
            gpu=dict(uuid='gpu', cuda_visible_devices='0'))
        workers = {}
        for replica, seed in enumerate(m.SEEDS):
            output = folder / f'population_{replica}'
            protocol = m.expected_protocol(d, sources, replica)
            protocol['implementation_sha256'] = {str(Path(d['source']['path']) / n): v
                for n, v in protocol['implementation_sha256'].items()}
            fixtures.save_population(output, protocol)
            slot = replica % concurrency
            cpus = [2 * slot, 2 * slot + 1]
            command = m._base.driver_command(d, seed, output)
            workers[str(replica)] = dict(replica=replica, cpus=cpus, command=command,
                pid=100 + replica, returncode=0, process_seconds=1.,
                start_offset_seconds=float(replica // concurrency),
                end_offset_seconds=float(replica // concurrency + 1),
                output_sha256={name: m.digest(output / name) for name in m._base.OUTPUTS})
            m.save(folder / f'worker_{replica}.json', dict(pid=100 + replica,
                affinity=cpus, command=command, node='n', cuda_visible_devices='0'))
        m.save(folder / 'receipt.json', dict(status='completed', phase=m.PHASE, setting_index=index,
            concurrency=concurrency, repetition=index // 2, seeds=m.SEEDS,
            declaration_sha256=m.digest(declaration), resources=resource, workers=workers,
            bitwise_saved_output_replay=True, setting_wall_seconds=internal[index]))
        wall = 100. if outer_slow and index == 2 else internal[index] + 1.
        m.save(root / f'setting_{index}_process_cost.json', dict(wall_seconds=wall, exit_code=0))
    return d, declaration


@pytest.mark.parametrize('slow,outer_slow,expected', [(False, False, True), (True, False, False), (False, True, False)])
def test_all16_replay_and_both_inclusive_and_outer_gates(tmp_path, monkeypatch, slow, outer_slow, expected):
    d, declaration = complete_confirmation_fixture(tmp_path, monkeypatch, slow=slow, outer_slow=outer_slow)
    output, report_cost = tmp_path / 'report.json', tmp_path / 'report_cost.json'
    m.report(declaration, output)
    report = m.load(output)
    assert report['all_16_saved_output_replay'] is True
    assert report['confirmation_passed'] is expected
    assert report['ratios'][1]['setting_inclusive_speedup'] == 110. / (95. if slow else 35.)
    m.save(report_cost, dict(wall_seconds=report['internal_report_seconds'] + 1., exit_code=0))
    m.finalize(output, report_cost)
    report = m.load(output)
    assert report['status'] == 'completed' and report['scientific_accuracy_accepted'] is False
    assert report['confirmation_passed'] is expected
    with pytest.raises(ValueError): m.finalize(output, report_cost)


@pytest.mark.parametrize('case', ['missing', 'failed', 'seed', 'stage', 'array', 'cost', 'allocation'])
def test_confirmation_refuses_partial_or_changed_original_runs(tmp_path, monkeypatch, case):
    d, declaration = complete_confirmation_fixture(tmp_path, monkeypatch)
    root = Path(d['output'])
    path = root / 'setting_2/receipt.json'
    r = m.load(path)
    if case == 'missing': del r['workers']['2']
    elif case == 'failed': r['workers']['2']['returncode'] = 7
    elif case == 'seed': r['seeds'] = m._base.SEEDS
    elif case == 'allocation': r['resources']['gpu']['uuid'] = 'wrong'
    elif case == 'cost': m.save(root / 'setting_2_process_cost.json', dict(wall_seconds=1., exit_code=0))
    elif case == 'array':
        with (root / 'setting_2/population_0/particles.npz').open('ab') as stream: stream.write(b'changed')
    else:
        output = root / 'setting_2/population_0/progress.json'
        progress = m.load(output); progress['stages'].pop(10); m.save(output, progress)
        r['workers']['0']['output_sha256']['progress.json'] = m.digest(output)
    m.save(path, r)
    with pytest.raises(ValueError): m.report(declaration, tmp_path / 'report.json')
