"""Independent completed-only audit of the four frozen prior-reuse v2 pairs.

Reuses the immutable source/proposal/output validators, never its report or
finalize functions. Recomputes all timing arithmetic from original receipts.
No sampler execution, new settings, acceptance threshold, or report overwrite.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


DECLARATION_SHA256 = 'ce259bd7d64a32086d01ce30b59ffe278599382bb87e5452fe519ad936571093'
SEEDS = [2026103900 + i for i in range(4)]
ARMS = ('baseline', 'optimized')
OUTPUTS = ('result.json', 'protocol.json', 'progress.json', 'particles.npz', 'scenes.npz')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)


def nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid nonnegative finite ' + name)
    return value


def pin(path, pins, expected=None):
    path = str(Path(path).resolve())
    value = digest(path)
    if (expected is not None and value != expected) or (path in pins and pins[path] != value):
        raise ValueError('changed evidence: ' + path)
    pins[path] = value
    return path


def process_cost(path, minimum=0.):
    nonnegative(minimum, 'minimum recorded process time')
    value = load(path)
    if type(value['exit_code']) is not int or value['exit_code'] != 0:
        raise ValueError('successful original process receipt required')
    for key in ('wall_seconds', 'user_seconds', 'system_seconds', 'max_rss_kib'):
        nonnegative(value[key], key)
    if value['wall_seconds'] + .011 < minimum:
        raise ValueError('outer timer is shorter than recorded work')
    return value


def pair_cost(replica, receipt, cost):
    """Independent implementation of the declared equal common-cost allocation."""
    nonnegative(receipt['pair_internal_wall_seconds'], 'pair internal time')
    durations = {arm: nonnegative(receipt['arms'][arm]['wall_seconds'], 'arm time') for arm in ARMS}
    arm_total = sum(durations.values())
    if cost['wall_seconds'] + .011 < max(arm_total, receipt['pair_internal_wall_seconds']):
        raise ValueError('pair process timer undercounts measured work')
    common = max(0., cost['wall_seconds'] - arm_total)
    charged = {arm: durations[arm] + common / 2 for arm in ARMS}
    if any(value <= 0 for value in charged.values()):
        raise ValueError('positive measured arm costs required')
    return dict(replica=replica, seed=SEEDS[replica], order=receipt['order'], arms=receipt['arms'],
        process_cost=cost, common_overhead_seconds=common, charged_seconds=charged,
        baseline_to_optimized_ratio=charged['baseline'] / charged['optimized'])


def cost_summary(pairs, report_cost, preparation):
    if len(pairs) != 4 or [row['replica'] for row in pairs] != list(range(4)):
        raise ValueError('all four ordered original pairs required')
    nonnegative(preparation, 'source preparation time')
    ratios = [row['baseline_to_optimized_ratio'] for row in pairs]
    totals = {arm: sum(row['charged_seconds'][arm] for row in pairs) + report_cost['wall_seconds'] / 2
              + preparation / 2 for arm in ARMS}
    return dict(ratio_range=[min(ratios), max(ratios)], ratio_mean=statistics.mean(ratios),
        ratio_mean_standard_error=statistics.stdev(ratios) / 2,
        new_work_seconds_by_arm=totals,
        total_baseline_to_optimized_ratio=totals['baseline'] / totals['optimized'])


def completed_report(d, declaration_sha, pins):
    """Do not open any population, timing or pair receipt before this barrier."""
    root = Path(d['output'])
    report_path = root / 'report.json'
    if not report_path.is_file():
        raise ValueError('finalized original report required; partial runs are not audited')
    pin(report_path, pins)
    report = load(report_path)
    if (report.get('status') != 'completed' or report.get('declaration_sha256') != declaration_sha or
            report.get('all_four_saved_output_replay') is not True or len(report.get('pairs', [])) != 4):
        raise ValueError('finalized complete four-pair report required')
    required = [root / 'report_process_cost.json', root / 'allocation.txt']
    for replica in range(4):
        folder = root / f'pair_{replica}'
        required += [folder / 'receipt.json', root / f'pair_{replica}_process_cost.json']
        for arm in ARMS:
            required += [folder / arm / name for name in OUTPUTS]
            required += [folder / (arm + '_runtime') / name for name in ('worker_0.json', 'population_0.log')]
    if any(not path.is_file() for path in required) or (root / '.running').exists():
        raise ValueError('all original pair, worker, output and cost receipts required; no running lock')
    for path in required:
        pin(path, pins)
    return report


def frozen_helper(declaration_path, d, pins):
    """Bind every transitive sibling helper before importing any frozen code."""
    root = Path(declaration_path).resolve().parent
    required = ('benchmark_prior_reuse_smc.py', 'benchmark_exchange_full_smc.py',
                'benchmark_smc_concurrency.py', 'prior_reuse_proposal_provenance.py',
                'check_prior_reuse_startup.py')
    for name in required:
        path = str(root / name)
        if path not in d['harness_files']:
            raise ValueError('missing executed frozen helper pin: ' + name)
    for mapping in ('harness_files', 'external_files'):
        for path, checksum in d[mapping].items():
            pin(path, pins, checksum)
    path = root / required[0]
    spec = importlib.util.spec_from_file_location('_completed_prior_reuse_frozen', path)
    helper = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(helper)
    return helper


def validate_pair(helper, d, sources, replica):
    root = Path(d['output'])
    folder = root / f'pair_{replica}'
    receipt = load(folder / 'receipt.json')
    order = ['baseline', 'optimized'] if replica % 2 == 0 else ['optimized', 'baseline']
    if (receipt['status'] != 'completed' or type(receipt['replica']) is not int or receipt['replica'] != replica or
            receipt['seed'] != SEEDS[replica] or receipt['order'] != order or
            receipt['declaration_sha256'] != DECLARATION_SHA256 or receipt['bitwise_saved_output_replay'] is not True):
        raise ValueError('original pair identity, completion or replay differs')
    helper.validate_workers(d, receipt, folder)
    # The original report does not compare this recorded end-of-pair identity.
    # Check it explicitly as evidence that both measured arms shared one GPU.
    for key in ('uuid', 'name', 'cuda_visible_devices', 'memory_total_mib'):
        if receipt['resources']['gpu'][key] != receipt['gpu_after'][key]:
            raise ValueError('GPU identity changed during original pair')
    for arm in ARMS:
        actual = {name: digest(folder / arm / name) for name in OUTPUTS}
        if receipt['arms'][arm]['output_sha256'] != actual:
            raise ValueError('saved output differs from original pair receipt')
        worker = receipt['arms'][arm]['workers']['0']
        start = nonnegative(worker['start_offset_seconds'], 'worker start')
        end = nonnegative(worker['end_offset_seconds'], 'worker end')
        if end < start or end > receipt['arms'][arm]['wall_seconds']:
            raise ValueError('worker interval outside its complete arm timing')
    arrays = helper.compare_pair(d, sources, folder, replica)
    helper.compare_json(arrays, receipt['arrays'], 'protocol')
    cost = process_cost(root / f'pair_{replica}_process_cost.json', receipt['pair_internal_wall_seconds'])
    return receipt, pair_cost(replica, receipt, cost), arrays


def audit(declaration_path, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('complete-source and saved-array audit requires a CPU scheduler allocation')
    started = time.monotonic()
    declaration_path, output = Path(declaration_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('audit output must be new')
    pins = {}
    pin(declaration_path, pins, DECLARATION_SHA256)
    d = load(declaration_path)
    if d['schema_version'] != 2 or d['seeds'] != SEEDS:
        raise ValueError('only the original prior-reuse v2 declaration is eligible')
    if output.is_relative_to(Path(d['output'])) or output.is_relative_to(declaration_path.parent):
        raise ValueError('audit output must be outside original source and run directories')
    report = completed_report(d, DECLARATION_SHA256, pins)
    helper = frozen_helper(declaration_path, d, pins)
    sources = helper.verify(d)
    for arm, item in d['arms'].items():
        root = Path(item['source'])
        entries = list(root.rglob('*'))
        if (any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries) or
                {path.relative_to(root).as_posix() for path in entries if path.is_file()} != set(sources[arm]) | {'SHA256SUMS'}):
            raise ValueError('source tree has missing, unlisted or nonregular files')
        pin(root / 'SHA256SUMS', pins, item['manifest_sha256'])
        for name, checksum in sources[arm].items():
            pin(root / name, pins, checksum)
    template = d['protocol_template']
    pin(template['path'], pins, template['sha256'])
    pairs, replays, allocation = [], [], None
    for replica in range(4):
        receipt, pair, arrays = validate_pair(helper, d, sources, replica)
        identity = helper.allocation_identity(receipt['resources'])
        if allocation is not None and identity != allocation:
            raise ValueError('allocation changed between original pairs')
        allocation = identity
        helper.compare_json(pair, report['pairs'][replica], 'protocol')
        pairs.append(pair)
        replays.append(dict(replica=replica, seed=SEEDS[replica], all_saved_outputs_equal=True,
                            stages_per_arm=64, arrays=arrays))
    report_cost = process_cost(Path(d['output']) / 'report_process_cost.json', report['internal_report_seconds'])
    summary = cost_summary(pairs, report_cost, d['source_preparation_seconds'])
    for key, value in dict(summary, report_process_cost=report_cost,
                           source_preparation_seconds=d['source_preparation_seconds'],
                           previous_failed_startup=d['previous_failed_startup'],
                           failed_startup_included_in_pair_speedups=False).items():
        helper.compare_json(value, report[key], 'protocol')
    helper.verify(d)
    for path, checksum in pins.items():
        if digest(path) != checksum:
            raise ValueError('evidence changed during independent audit: ' + path)
    failed_cost = d['previous_failed_startup']['process_cost']
    successful = sum(pair['process_cost']['wall_seconds'] for pair in pairs) + report_cost['wall_seconds'] + d['source_preparation_seconds']
    result = dict(status='audit_passed', declaration_sha256=DECLARATION_SHA256,
        original_report_sha256=pins[str(Path(d['output']) / 'report.json')],
        all_four_original_pairs_replayed_again=True, all_reported_costs_recomputed=True,
        comparisons=replays, pairs=pairs, summary=summary, report_process_cost=report_cost,
        successful_experiment_wall_seconds=successful,
        preserved_failed_startup_cost=failed_cost,
        experiment_plus_preserved_failed_startup_wall_seconds=successful + failed_cost['wall_seconds'],
        source_preparation_seconds=d['source_preparation_seconds'], allocation=allocation,
        evidence_sha256=pins, checker_sha256=digest(__file__),
        internal_audit_seconds=time.monotonic() - started,
        scientific_accuracy_accepted=False, automatic_adoption=False,
        limitations=['Audit pass verifies recorded replay and descriptive timing arithmetic; no speed threshold was declared.',
            'Exactly the frozen implementation hashes and validated derived-proposal provenance exception are normalized; proposal values and pilot metadata stay exact.',
            'Four development pairs at the declared analytic target law; no numerical mock-target-law or posterior-convergence acceptance.',
            'Recorded scientific JSON and NPZ dtype/shape/bytes are compared; unrecorded mutation states and terminal RNG are unavailable.',
            'Successful costs include pair/report processes and declared preparation (including startup preflight), conditional on existing fits and prior native replay.',
            'Previous failed startup is retained separately and is not added to paired speedups. Audit process cost is separate; queueing, launcher setup and JSON-only finalization remain excluded.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.declaration, args.output)
