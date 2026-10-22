"""Predeclared fresh-seed confirmation of the selected fixed SMC worker count.

The original development entrypoint is unchanged. Only seeds and concurrency
order differ; unchanged generic source/protocol/process/replay helpers are reused.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import time

_SPEC = importlib.util.spec_from_file_location('_confirmation_development',
    Path(__file__).with_name('benchmark_smc_concurrency.py'))
_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)
digest, load, save = _base.digest, _base.load, _base.save
compare_json, compare_npz = _base.compare_json, _base.compare_npz
SEEDS = [2026103800 + i for i in range(4)]
ORDER = [1, 4, 4, 1]
SOURCE_SHA = 'fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1'
DEVELOPMENT_DECLARATION_SHA = '34383474bb5c5bdce06be6a711ae9b088b1ee5e2d5b48a14a1545892951ff185'
DEVELOPMENT_HARNESS_SHA = '6455d3c8d4175f5abe6a8a079dc812befa29cef3c18a01c7b25b89c6ade1d915'
PHASE = 'predeclared-fresh-seed-concurrency-confirmation-v1'


def cost(path, minimum=0.):
    value = load(path)
    if type(value['exit_code']) is not int or value['exit_code'] != 0:
        raise ValueError('successful original process cost required')
    _base.nonnegative(value['wall_seconds'], 'outer process wall seconds')
    if value['wall_seconds'] + .011 < minimum:
        raise ValueError('outer timer shorter than recorded work')
    return value


def choose_level(ratios):
    """Original development rule, using complete internal setting wall times."""
    if len(ratios) != 4:
        raise ValueError('both repetitions of both original candidate levels required')
    by_level = {2: [], 4: []}
    seen = set()
    for row in ratios:
        key = (row['repetition'], row['concurrency'])
        if key in seen or key not in {(0, 2), (0, 4), (1, 2), (1, 4)}:
            raise ValueError('duplicate or unknown development ratio')
        seen.add(key)
        by_level[key[1]].append(_base.nonnegative(row['setting_inclusive_speedup'], 'speedup'))
    eligible = {level: min(values) for level, values in by_level.items() if min(values) >= 1.25}
    if not eligible:
        raise ValueError('no eligible development concurrency')
    best = max(eligible.values())
    # A lower-worker candidate within 5% of the best worst repetition wins.
    selected = min(level for level, score in eligible.items() if score >= .95 * best)
    return dict(selected_workers=selected, minimum_speedup_by_level={str(k): v for k, v in eligible.items()},
                rule='both inclusive repeats >=1.25; largest minimum; within5% prefer fewer workers')


def audit_development(declaration_path, report_path, parent_audit_path, replay=False):
    """Bind original completed evidence and all seven full process-cost receipts."""
    declaration_path, report_path = Path(declaration_path), Path(report_path)
    if digest(declaration_path) != DEVELOPMENT_DECLARATION_SHA:
        raise ValueError('only the completed predeclared development experiment is eligible')
    d, report = load(declaration_path), load(report_path)
    if (d['seeds'] != _base.SEEDS or d['order'] != _base.ORDER or
            d['source']['manifest_sha256'] != SOURCE_SHA or
            d['selection_rule']['future_test_seeds'] != SEEDS or
            report['status'] != 'completed' or report['all_24_saved_output_replay'] is not True or
            report['setting_order'] != _base.ORDER or
            report['declaration_sha256'] != digest(declaration_path) or len(report['settings']) != 6):
        raise ValueError('incomplete or changed original development experiment')
    source = Path(d['source']['path'])
    if digest(source / 'SHA256SUMS') != SOURCE_SHA:
        raise ValueError('development source manifest changed')
    sources = _base._exact.manifest(source)
    if len(sources) != 320:
        raise ValueError('complete development source required')
    _base.expected_protocol(d, sources, 0)
    files = {str(declaration_path.resolve()): digest(declaration_path),
             str(report_path.resolve()): digest(report_path),
             str((source / 'SHA256SUMS').resolve()): SOURCE_SHA}
    files.update({str(source / name): checksum for name, checksum in sources.items()})
    files.update(d['external_files'])
    files.update(d['harness_files'])
    harnesses = [p for p in d['harness_files'] if Path(p).name == 'benchmark_smc_concurrency.py']
    if len(harnesses) != 1 or d['harness_files'][harnesses[0]] != DEVELOPMENT_HARNESS_SHA:
        raise ValueError('unknown development harness')
    root = Path(d['output'])
    settings, process_costs, allocation = [], [], None
    for index, concurrency in enumerate(_base.ORDER):
        folder = root / f'setting_{index}'
        receipt_path = folder / 'receipt.json'
        r = load(receipt_path)
        compare_json(r, report['settings'][index], 'protocol')
        if (r['status'] != 'completed' or r['setting_index'] != index or r['concurrency'] != concurrency or
                r['repetition'] != (0 if index < 3 else 1) or r['seeds'] != _base.SEEDS or
                r['declaration_sha256'] != digest(declaration_path) or
                r['bitwise_saved_output_replay'] is not True):
            raise ValueError('all six original settings required')
        identity = {k: r['resources'][k] for k in
            ('slurm_job_id', 'node', 'selected_cpus', 'cpu_topology', 'thread_environment')}
        identity['gpu_uuid'] = r['resources']['gpu']['uuid']
        if allocation is not None and identity != allocation:
            raise ValueError('development allocation changed')
        allocation = identity
        _base.check_workers(r, folder)
        files[str(receipt_path)] = digest(receipt_path)
        for replica in range(4):
            worker_receipt = folder / f'worker_{replica}.json'
            files[str(worker_receipt)] = digest(worker_receipt)
            population = folder / f'population_{replica}'
            worker = r['workers'][str(replica)]
            if worker['command'] != _base.driver_command(d, _base.SEEDS[replica], population):
                raise ValueError('development command differs from frozen recipe')
            for name in _base.OUTPUTS:
                path = population / name
                if digest(path) != worker['output_sha256'][name]:
                    raise ValueError('development saved output changed')
                files[str(path)] = worker['output_sha256'][name]
            if replay:
                _base.compare_population(d, sources, root / 'setting_0' / f'population_{replica}', population, replica)
        _base.nonnegative(r['setting_wall_seconds'], 'setting wall seconds')
        if r['setting_wall_seconds'] <= 0:
            raise ValueError('positive setting time required')
        cost_path = root / f'setting_{index}_process_cost.json'
        process_costs.append(cost(cost_path, r['setting_wall_seconds']))
        files[str(cost_path)] = digest(cost_path)
        settings.append(r)
    report_cost_path = root / 'report_process_cost.json'
    report_cost = cost(report_cost_path, report['internal_report_seconds'])
    files[str(report_cost_path)] = digest(report_cost_path)
    ratios = []
    for repetition, indices in enumerate(([0, 1, 2], [5, 4, 3])):
        baseline = settings[indices[0]]
        for index in indices[1:]:
            r = settings[index]
            ratios.append(dict(repetition=repetition, concurrency=r['concurrency'],
                execution_speedup=baseline['execution_wall_seconds'] / r['execution_wall_seconds'],
                setting_inclusive_speedup=baseline['setting_wall_seconds'] / r['setting_wall_seconds']))
    compare_json(ratios, report['ratios'], 'protocol')
    selected = choose_level(ratios)
    if selected['selected_workers'] != 4:
        raise ValueError('this confirmation is restricted to the actually selected four-worker level')
    parent_audit_path = Path(parent_audit_path).resolve()
    parent = load(parent_audit_path)
    if (parent['status'] != 'passed' or parent['all_24_saved_output_replayed_again'] is not True or
            parent['strict_replayed_report_equal_except_its_elapsed_time'] is not True or
            parent['original_report_sha256'] != digest(report_path) or parent['selected_concurrency'] != 4):
        raise ValueError('successful independent development audit required')
    for key, value in [('ratios', ratios), ('process_costs', process_costs), ('report_process_cost', report_cost)]:
        compare_json(parent[key], value, 'protocol')
    expected_parent_pins = {str(report_path.resolve()): digest(report_path)}
    expected_parent_pins.update({str(root / f'setting_{i}_process_cost.json'): digest(root / f'setting_{i}_process_cost.json') for i in range(6)})
    expected_parent_pins[str(report_cost_path)] = digest(report_cost_path)
    if parent['pinned_reports_and_costs'] != expected_parent_pins:
        raise ValueError('independent audit must pin original report and all seven process costs')
    files[str(parent_audit_path)] = digest(parent_audit_path)
    for path, checksum in files.items():
        if digest(path) != checksum:
            raise ValueError('changed development evidence: ' + path)
    return dict(status='passed', all_24_saved_output_replay_verified=bool(replay),
        declaration=str(declaration_path.resolve()), report=str(report_path.resolve()),
        parent_audit=str(parent_audit_path),
        selection=selected, ratios=ratios, process_costs=process_costs,
        report_process_cost=report_cost, evidence_sha256=files)


def development_projection(d):
    """Project only phase fields for unchanged common source/recipe validation."""
    projected = dict(d)
    projected['seeds'], projected['order'] = _base.SEEDS, _base.ORDER
    return projected


def verify(d):
    if (d.get('phase') != PHASE or d['seeds'] != SEEDS or d['order'] != ORDER or
            type(d.get('selected_workers')) is not int or d['selected_workers'] != 4 or
            d['source']['manifest_sha256'] != SOURCE_SHA):
        raise ValueError('unknown confirmation phase, source, seeds or worker order')
    selection_path = Path(d['development_selection']['path'])
    if digest(selection_path) != d['development_selection']['sha256']:
        raise ValueError('changed development selection receipt')
    selection = load(selection_path)
    if (selection['status'] != 'passed' or selection['all_24_saved_output_replay_verified'] is not True or
            selection['selection']['selected_workers'] != 4):
        raise ValueError('full completed development replay and selected four workers required')
    required = dict(selection['evidence_sha256'])
    required[str(selection_path)] = d['development_selection']['sha256']
    if any(d['external_files'].get(p) != checksum for p, checksum in required.items()):
        raise ValueError('all original development evidence and cost receipts must remain pinned')
    if str(Path(__file__).resolve()) not in d['harness_files']:
        raise ValueError('confirmation harness must be pinned')
    # Fresh seeds have already been checked. This call validates every common
    # science/runtime/input field with the unchanged development implementation.
    sources = _base.verify(development_projection(d))
    current = audit_development(selection['declaration'], selection['report'], selection['parent_audit'], replay=False)
    for key in ('selection', 'ratios', 'process_costs', 'report_process_cost', 'evidence_sha256'):
        compare_json(selection[key], current[key], 'protocol')
    return sources


def expected_protocol(d, sources, replica):
    p = _base.expected_protocol(development_projection(d), sources, replica)
    p['seed'] = SEEDS[replica]
    return p


def saved_run(d, sources, folder, replica):
    result, protocol, progress = (load(folder / name) for name in _base.OUTPUTS[:3])
    if result['status'] != 'completed':
        raise ValueError('all original confirmation populations must complete without extinction')
    for key in _base._exact.ARTIFACT_HASH_KEYS:
        name = key[:-7] + ('.json' if key == 'protocol_sha256' else '.npz')
        if result[key] != digest(folder / name):
            raise ValueError('changed confirmation saved artifact')
        del result[key]
    protocol = _base.normalize_protocol(protocol, d['source']['path'], sources)
    _base._exact.validate_saved_run(result, protocol, progress, expected_protocol(d, sources, replica))
    if any(any(stage.get('extinct', [])) for stage in progress['stages']):
        raise ValueError('extinction fails the confirmation')
    return result, protocol, progress


def compare_population(d, sources, first, second, replica):
    a, b = saved_run(d, sources, first, replica), saved_run(d, sources, second, replica)
    for left, right, kind in zip(a, b, ('result', 'protocol', 'progress')):
        compare_json(left, right, kind)
    return {name: compare_npz(first / name, second / name)
            for name in ('particles.npz', 'scenes.npz')}


def run_setting(declaration_path, index):
    start = time.monotonic()
    if type(index) is not int or index not in range(4):
        raise ValueError('one of four original confirmation settings required')
    d = load(declaration_path)
    root = Path(d['output'])
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.running'
    lock.mkdir()
    receipt = None
    try:
        sources = verify(d)
        resource = _base.resources()
        _base.validate_predecessor(root, index, digest(declaration_path), resource)
        directory = root / f'setting_{index}'
        directory.mkdir()
        receipt_path = directory / 'receipt.json'
        concurrency = ORDER[index]
        receipt = dict(status='started', phase=PHASE, setting_index=index,
            concurrency=concurrency, repetition=index // 2, seeds=SEEDS,
            declaration_sha256=digest(declaration_path), resources=resource, workers={})
        save(receipt_path, receipt)
        cpus = resource['selected_cpus']
        slots = [cpus[2 * i:2 * i + 2] for i in range(concurrency)]
        commands = [_base.driver_command(d, seed, directory / f'population_{i}') for i, seed in enumerate(SEEDS)]
        def update(records):
            receipt['workers'] = records
            save(receipt_path, receipt)
        records, wall = _base.run_workers(commands, slots, directory, d['python'], d['source']['path'], update)
        receipt.update(workers=records, execution_wall_seconds=wall)
        _base.check_workers(receipt, directory)
        receipt['replay_arrays'] = {}
        for replica in range(4):
            folder = directory / f'population_{replica}'
            receipt['replay_arrays'][str(replica)] = compare_population(d, sources,
                root / 'setting_0' / f'population_{replica}', folder, replica)
            receipt['workers'][str(replica)]['output_sha256'] = {name: digest(folder / name) for name in _base.OUTPUTS}
        verify(d)
        receipt['gpu_after'] = _base.gpu_snapshot()
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
    settings, costs, allocation = [], [], None
    for index, concurrency in enumerate(ORDER):
        folder = root / f'setting_{index}'
        r = load(folder / 'receipt.json')
        if (r['status'] != 'completed' or r['phase'] != PHASE or r['setting_index'] != index or
                r['concurrency'] != concurrency or r['repetition'] != index // 2 or r['seeds'] != SEEDS or
                r['declaration_sha256'] != digest(declaration_path) or r['bitwise_saved_output_replay'] is not True):
            raise ValueError('all four original confirmation settings required')
        identity = {k: r['resources'][k] for k in
            ('slurm_job_id', 'node', 'selected_cpus', 'cpu_topology', 'thread_environment')}
        identity['gpu_uuid'] = r['resources']['gpu']['uuid']
        if allocation is not None and allocation != identity:
            raise ValueError('confirmation allocation changed')
        allocation = identity
        _base.check_workers(r, folder)
        for replica in range(4):
            population = folder / f'population_{replica}'
            worker = r['workers'][str(replica)]
            if worker['command'] != _base.driver_command(d, SEEDS[replica], population):
                raise ValueError('confirmation command differs from fixed recipe')
            if worker['output_sha256'] != {name: digest(population / name) for name in _base.OUTPUTS}:
                raise ValueError('confirmation output changed after completion')
            compare_population(d, sources, root / 'setting_0' / f'population_{replica}', population, replica)
        _base.nonnegative(r['setting_wall_seconds'], 'setting wall seconds')
        if r['setting_wall_seconds'] <= 0:
            raise ValueError('positive setting wall time required')
        costs.append(cost(root / f'setting_{index}_process_cost.json', r['setting_wall_seconds']))
        settings.append(r)
    ratios = [dict(repetition=rep, setting_inclusive_speedup=settings[a]['setting_wall_seconds'] / settings[b]['setting_wall_seconds'],
                   outer_process_speedup=costs[a]['wall_seconds'] / costs[b]['wall_seconds'])
              for rep, (a, b) in enumerate(((0, 1), (3, 2)))]
    passed = all(row[key] >= 1.25 for row in ratios
                 for key in ('setting_inclusive_speedup', 'outer_process_speedup'))
    save(output, dict(status='pending_report_process_cost', phase=PHASE,
        declaration_sha256=digest(declaration_path), all_16_saved_output_replay=True,
        confirmation_passed=passed, selected_workers=4, order=ORDER, seeds=SEEDS,
        ratios=ratios, settings=settings, setting_process_costs=costs,
        source_preparation_seconds=d['source_preparation_seconds'],
        internal_report_seconds=time.monotonic() - start,
        limitations=['Repeated seeds across timing settings are not independent scientific replicates.',
            'Same four development observations; seeds are untouched confirmation seeds.',
            'Saved-output replay does not establish posterior accuracy or catalogue shear calibration.',
            'One V100/eight reserved physical cores; no claim about500 CPUs or multiple GPU scaling.',
            'Original mutation states and terminal RNG are not saved.',
            'Original development/pilot costs and launcher setup are excluded; confirmation preparation/report costs retained.']))


def finalize(report_path, cost_path):
    r = load(report_path)
    if r['status'] != 'pending_report_process_cost':
        raise ValueError('original pending confirmation report required')
    report_cost = cost(cost_path, r['internal_report_seconds'])
    total = {str(level): sum(c['wall_seconds'] for c, setting in zip(r['setting_process_costs'], r['settings'])
                            if setting['concurrency'] == level)
             + (report_cost['wall_seconds'] + r['source_preparation_seconds']) / 2 for level in (1, 4)}
    r.update(status='completed', report_process_cost=report_cost,
        complete_new_work_seconds_by_concurrency=total,
        total_baseline_to_four_worker_ratio=total['1'] / total['4'],
        cost_scope='All four original setting subprocess wall times plus measured confirmation preparation and report processes; shared preparation/report charged half to each level. Source freeze bookkeeping, launcher setup, queueing and final JSON-only finalization excluded.',
        scientific_accuracy_accepted=False)
    save(report_path, r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['run-setting', 'report', 'finalize'])
    p.add_argument('--declaration', type=Path)
    p.add_argument('--setting-index', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--cost', type=Path)
    a = p.parse_args()
    def terminate(signum, frame):
        raise RuntimeError(f'confirmation interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    if a.mode == 'run-setting': run_setting(a.declaration, a.setting_index)
    elif a.mode == 'report': report(a.declaration, a.output)
    else: finalize(a.output, a.cost)


if __name__ == '__main__':
    main()
