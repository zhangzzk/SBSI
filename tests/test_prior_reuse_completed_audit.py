import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location('completed_prior_reuse_audit',
    Path(__file__).resolve().parents[1] / 'scripts/audit_prior_reuse_completed.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def receipt(replica=0):
    gpu = dict(uuid='gpu-original', name='V100', cuda_visible_devices='0', memory_total_mib=32768)
    return dict(status='completed', replica=replica, seed=m.SEEDS[replica],
        order=['baseline', 'optimized'] if replica % 2 == 0 else ['optimized', 'baseline'],
        declaration_sha256=m.DECLARATION_SHA256, bitwise_saved_output_replay=True,
        resources=dict(gpu=gpu), gpu_after=copy.deepcopy(gpu),
        pair_internal_wall_seconds=195., arrays={'particles.npz': {'field': 'exact'}},
        arms={arm: dict(wall_seconds=wall, workers={'0': dict(start_offset_seconds=.01,
            end_offset_seconds=wall - .1, process_seconds=wall - .2)})
            for arm, wall in [('baseline', 100.), ('optimized', 80.)]})


def cost(wall=200.):
    return dict(exit_code=0, wall_seconds=wall, user_seconds=150., system_seconds=10., max_rss_kib=1000)


def test_independent_cost_recomputation_includes_shared_work_and_all_four_pairs():
    pairs = [m.pair_cost(i, receipt(i), cost(200. + i)) for i in range(4)]
    assert pairs[0]['charged_seconds'] == dict(baseline=110., optimized=90.)
    assert pairs[0]['common_overhead_seconds'] == 20.
    assert pairs[0]['baseline_to_optimized_ratio'] == 110. / 90.
    summary = m.cost_summary(pairs, cost(12.), 8.)
    assert summary['new_work_seconds_by_arm'] == dict(baseline=453., optimized=373.)
    assert summary['total_baseline_to_optimized_ratio'] == 453. / 373.
    ratios = [110. / 90., 110.5 / 90.5, 111. / 91., 111.5 / 91.5]
    mean = sum(ratios) / 4
    expected_se = math.sqrt(sum((ratio - mean) ** 2 for ratio in ratios) / 3) / 2
    assert summary['ratio_mean_standard_error'] == pytest.approx(expected_se, abs=1e-15)
    for wrong in (pairs[:3], [pairs[1], pairs[0], *pairs[2:]]):
        with pytest.raises(ValueError, match='four ordered'):
            m.cost_summary(wrong, cost(12.), 8.)


@pytest.mark.parametrize('case', ['negative', 'nan', 'boolean', 'failed', 'underreported', 'negative_minimum'])
def test_process_cost_fails_closed(tmp_path, case):
    value, minimum = cost(), 195.
    if case == 'negative': value['user_seconds'] = -1
    elif case == 'nan': value['system_seconds'] = float('nan')
    elif case == 'boolean': value['wall_seconds'] = True
    elif case == 'failed': value['exit_code'] = 1
    elif case == 'underreported': value['wall_seconds'] = 190.
    else: minimum = -1
    path = tmp_path / 'cost.json'; save(path, value)
    with pytest.raises(ValueError): m.process_cost(path, minimum)


def test_pair_cost_checks_inner_and_outer_time_without_hiding_rounding():
    r = receipt(); r['pair_internal_wall_seconds'] = 180.
    result = m.pair_cost(0, r, cost(179.995))
    assert result['common_overhead_seconds'] == 0.
    assert result['charged_seconds'] == dict(baseline=100., optimized=80.)
    with pytest.raises(ValueError, match='undercounts'): m.pair_cost(0, r, cost(179.98))
    r['arms']['baseline']['wall_seconds'] = -1
    with pytest.raises(ValueError): m.pair_cost(0, r, cost())


@pytest.mark.parametrize('status', [None, 'pending_report_process_cost', 'failed'])
def test_final_report_barrier_does_not_open_pair_outputs(tmp_path, monkeypatch, status):
    d = dict(output=str(tmp_path))
    if status is not None:
        save(tmp_path / 'report.json', dict(status=status))
    original = m.load
    opened = []
    def observed(path):
        opened.append(Path(path).name)
        assert Path(path).name == 'report.json'
        return original(path)
    monkeypatch.setattr(m, 'load', observed)
    with pytest.raises(ValueError, match='finalized'): m.completed_report(d, m.DECLARATION_SHA256, {})
    assert opened == ([] if status is None else ['report.json'])


def test_completed_report_still_requires_all_original_receipts(tmp_path):
    save(tmp_path / 'report.json', dict(status='completed', declaration_sha256=m.DECLARATION_SHA256,
        all_four_saved_output_replay=True, pairs=[{}] * 4))
    with pytest.raises(ValueError, match='all original'):
        m.completed_report(dict(output=str(tmp_path)), m.DECLARATION_SHA256, {})


@pytest.mark.parametrize('case', ['valid', 'changed_gpu', 'changed_output', 'wrong_order', 'wrong_seed', 'bad_interval', 'changed_arrays'])
def test_pair_audit_enforces_identity_outputs_and_recorded_replay(tmp_path, case):
    folder = tmp_path / 'pair_0'
    r = receipt()
    for arm in m.ARMS:
        for name in m.OUTPUTS:
            path = folder / arm / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixed output')
        r['arms'][arm]['output_sha256'] = {name: m.digest(folder / arm / name) for name in m.OUTPUTS}
    save(tmp_path / 'pair_0_process_cost.json', cost())
    calls = []
    def equal(left, right, _kind):
        if left != right: raise ValueError('array comparison receipt differs')
    helper = SimpleNamespace(validate_workers=lambda *_: calls.append('workers'),
        compare_pair=lambda *_: copy.deepcopy(receipt()['arrays']), compare_json=equal)
    if case == 'changed_gpu': r['gpu_after']['uuid'] = 'different'
    elif case == 'changed_output': (folder / 'optimized/particles.npz').write_bytes(b'changed')
    elif case == 'wrong_order': r['order'].reverse()
    elif case == 'wrong_seed': r['seed'] += 1
    elif case == 'bad_interval': r['arms']['baseline']['workers']['0']['end_offset_seconds'] = 101.
    elif case == 'changed_arrays': r['arrays']['particles.npz']['field'] = 'changed'
    save(folder / 'receipt.json', r)
    if case == 'valid':
        actual, pair, arrays = m.validate_pair(helper, dict(output=str(tmp_path)), {}, 0)
        assert actual == r and arrays == r['arrays'] and pair['charged_seconds']['baseline'] == 110.
        assert calls == ['workers']
    else:
        with pytest.raises(ValueError): m.validate_pair(helper, dict(output=str(tmp_path)), {}, 0)


def test_pin_rejects_change_and_strict_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / 'proof.json'; save(path, {'a': 1})
    pins = {}; m.pin(path, pins)
    save(path, {'a': 2})
    with pytest.raises(ValueError, match='changed evidence'): m.pin(path, pins)
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match='duplicate'): m.load(path)


def test_audit_requires_scheduler_and_new_output_before_reading_inputs(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(ValueError, match='scheduler'): m.audit(tmp_path / 'missing.json', tmp_path / 'new.json')
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic')
    output = tmp_path / 'existing.json'; output.write_text('preserve')
    with pytest.raises(FileExistsError): m.audit(tmp_path / 'missing.json', output)
    assert output.read_text() == 'preserve'
