"""Independent scheduled replay and complete-cost audit of the frozen n24 run."""
from pathlib import Path
import hashlib
import json
import math
import os
import runpy
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    start = time.monotonic()
    root = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021')
    output = root / 'smc_concurrency_parent_audit_v1_20261022'
    declared = load(output / 'declaration.json')
    run = root / 'smc_concurrency_runs_v1_20261022'
    harness = root / 'smc_concurrency_prepared_v1_20261022'
    original_path = run / 'report.json'
    assert digest(original_path) == declared['original_report_sha256']
    assert os.environ.get('SLURM_JOB_ID')
    functions = runpy.run_path(str(harness / 'benchmark_smc_concurrency.py'))
    replay_path = output / 'replayed_report.json'
    functions['report'](harness / 'declaration.json', replay_path)
    original, replay = load(original_path), load(replay_path)
    # The parent re-executes the same strict source/recipe/worker/stage/array
    # checks. Only this new audit's report execution time is allowed to differ.
    original_comparable, replay_comparable = dict(original), dict(replay)
    del original_comparable['internal_report_seconds']
    del replay_comparable['internal_report_seconds']
    assert original_comparable == replay_comparable
    assert original['status'] == 'completed'
    assert original['all_24_saved_output_replay'] is True
    assert original['setting_order'] == [1, 2, 4, 4, 2, 1]
    costs = []
    pins = {str(original_path): digest(original_path)}
    for i, setting in enumerate(original['settings']):
        path = run / f'setting_{i}_process_cost.json'
        cost = load(path)
        assert type(cost['exit_code']) is int and cost['exit_code'] == 0
        for key in ('wall_seconds', 'user_seconds', 'system_seconds', 'max_rss_kib'):
            assert math.isfinite(cost[key]) and cost[key] >= 0
        assert cost['wall_seconds'] + .011 >= setting['setting_wall_seconds']
        assert setting['setting_wall_seconds'] >= setting['execution_wall_seconds'] > 0
        costs.append(cost)
        pins[str(path)] = digest(path)
    path = run / 'report_process_cost.json'
    report_cost = load(path)
    assert type(report_cost['exit_code']) is int and report_cost['exit_code'] == 0
    for key in ('wall_seconds', 'user_seconds', 'system_seconds', 'max_rss_kib'):
        assert math.isfinite(report_cost[key]) and report_cost[key] >= 0
    assert report_cost['wall_seconds'] + .011 >= original['internal_report_seconds']
    pins[str(path)] = digest(path)
    ratios = []
    for repetition, indices in enumerate(((0, 1, 2), (5, 4, 3))):
        baseline = indices[0]
        for index in indices[1:]:
            settings = original['settings']
            ratios.append(dict(repetition=repetition, concurrency=settings[index]['concurrency'],
                execution_speedup=settings[baseline]['execution_wall_seconds'] / settings[index]['execution_wall_seconds'],
                setting_inclusive_speedup=settings[baseline]['setting_wall_seconds'] / settings[index]['setting_wall_seconds']))
    assert ratios == original['ratios']
    eligible = {k: min(row['setting_inclusive_speedup'] for row in ratios if row['concurrency'] == k)
                for k in (2, 4)}
    eligible = {k: v for k, v in eligible.items() if v >= 1.25}
    assert eligible
    best = max(eligible.values())
    selected = min(k for k, v in eligible.items() if v >= .95 * best)
    assert selected == 4
    # Charge the common new preparation and report equally per setting as a
    # sensitivity check, separately from the predeclared primary ratio.
    shared_per_setting = (original['source_preparation_seconds'] + report_cost['wall_seconds']) / 6
    outer_ratios = []
    for repetition, baseline, candidate in ((0, 0, 2), (1, 5, 3)):
        outer_ratios.append(dict(repetition=repetition,
            process_speedup=costs[baseline]['wall_seconds'] / costs[candidate]['wall_seconds'],
            charged_speedup=(costs[baseline]['wall_seconds'] + shared_per_setting) /
                            (costs[candidate]['wall_seconds'] + shared_per_setting)))
    assert all(item['charged_speedup'] >= 1.25 for item in outer_ratios)
    assert all(digest(path) == checksum for path, checksum in pins.items())
    result = dict(status='passed', job=os.environ['SLURM_JOB_ID'],
        original_report_sha256=declared['original_report_sha256'],
        all_24_saved_output_replayed_again=True,
        strict_replayed_report_equal_except_its_elapsed_time=True,
        process_costs=costs, report_process_cost=report_cost,
        original_process_seconds=sum(item['wall_seconds'] for item in costs) + report_cost['wall_seconds'],
        source_preparation_seconds=original['source_preparation_seconds'],
        ratios=ratios, outer_ratios=outer_ratios, selected_concurrency=selected,
        selected_worst_speedup=eligible[selected],
        pinned_reports_and_costs=pins, elapsed_seconds=time.monotonic()-start,
        cost_scope='Original complete setting/report process timers; shared new preparation/report cost sensitivity split equally. Existing scientific fits, queueing and final serialization excluded.',
        limitations=['Development performance confirmation pending on predeclared fresh seeds.',
                     'No posterior-accuracy acceptance or four-GPU/500-CPU extrapolation.',
                     'This separate parent audit is validation work, not included in original benchmark timings.'])
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
