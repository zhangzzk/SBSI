"""Mandatory eight-task replay of the generic pool against original serial runs.

All four development and all four confirmation seeds are fixed. This is an
execution replay, not new science or a historical timing acceptance test.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time

_SPEC = importlib.util.spec_from_file_location('_bounded_pool', Path(__file__).with_name('run_sampler_task_pool.py'))
pool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pool)

SOURCE_SHA = 'fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1'
PHASES = {
    'development': dict(prefix='smc_concurrency',
        declaration='34383474bb5c5bdce06be6a711ae9b088b1ee5e2d5b48a14a1545892951ff185',
        report='2d8f7c2f0d106f9da849114b97a9bc16eb0e3f9471988de3bc0ed22b89860c30',
        audit='a933df95479785fbc4fac77ca0e1dc804074bc59f6b22a05363b97879ed058a2',
        helper='benchmark_smc_concurrency.py', seeds=list(range(2026103700, 2026103704)), settings=6),
    'confirmation': dict(prefix='smc_concurrency_confirmation',
        declaration='9a26daf6a7bc1b756485a64f620f43549f8520c1a315051d60d8a1def25b3268',
        report='6f562b2dfd1df339300b193e98828168667eec84410cd45a7666ec808c43989c',
        audit='0b47738bdc3d59e6702c0c4f478c271d6ad09abcffd84ac69a8638a3e4891d48',
        helper='benchmark_smc_concurrency_confirmation.py', seeds=list(range(2026103800, 2026103804)), settings=4),
}
FILES = ('result.json', 'protocol.json', 'progress.json', 'particles.npz', 'scenes.npz')
HARNESS = ('run_sampler_task_pool.py', 'run_sampler_task_pool.sh', 'benchmark_smc_concurrency.py',
           'benchmark_exchange_full_smc.py', 'benchmark_sampler_task_pool_replay.py', 'run_sampler_task_pool_replay.sh')


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def original_evidence(root):
    """Check frozen gates and all eight selected original serial populations."""
    evidence, phases, first_sources = {}, {}, None
    for phase, definition in PHASES.items():
        prefix = definition['prefix']
        paths = dict(declaration=root / f'{prefix}_prepared_v1_20261022/declaration.json',
            report=root / f'{prefix}_runs_v1_20261022/report.json',
            audit=root / f'{prefix}_parent_audit_v1_20261022/result.json')
        for key, path in paths.items():
            if pool.digest(path) != definition[key]:
                raise ValueError('changed original ' + phase + ' ' + key)
            evidence[str(path)] = definition[key]
        d, report, audit = (pool.load(paths[key]) for key in ('declaration', 'report', 'audit'))
        helpers = [p for p in d['harness_files'] if Path(p).name == definition['helper']]
        if len(helpers) != 1 or pool.digest(helpers[0]) != d['harness_files'][helpers[0]]:
            raise ValueError('original phase helper changed')
        helper = imported(helpers[0], '_pool_replay_' + phase)
        sources = helper.verify(d)
        if len(sources) != 320 or d['source']['manifest_sha256'] != SOURCE_SHA or d['seeds'] != definition['seeds']:
            raise ValueError('original source/seed identity differs')
        if first_sources is not None and sources != first_sources:
            raise ValueError('original development and confirmation sources differ')
        first_sources = sources
        if report['status'] != 'completed' or report['declaration_sha256'] != definition['declaration'] or len(report['settings']) != definition['settings']:
            raise ValueError('complete original report required')
        if phase == 'development':
            if report['all_24_saved_output_replay'] is not True or audit['status'] != 'passed' or audit['all_24_saved_output_replayed_again'] is not True:
                raise ValueError('completed original development replay audit required')
        else:
            if (report['all_16_saved_output_replay'] is not True or report['confirmation_passed'] is not True or
                    audit['status'] != 'audit_passed' or audit['all_16_saved_outputs_rechecked'] is not True or audit['confirmation_passed'] is not True):
                raise ValueError('completed independent confirmation gate required')
            if any(row[key] < 1.25 for row in report['ratios'] for key in ('setting_inclusive_speedup', 'outer_process_speedup')):
                raise ValueError('original confirmation speed gate failed')
        for path, checksum in audit['pinned_reports_and_costs'].items():
            if pool.digest(path) != checksum:
                raise ValueError('changed original outer process receipt')
            evidence[path] = checksum
        for mapping in ('external_files', 'harness_files'):
            evidence.update(d[mapping])
        source = Path(d['source']['path'])
        evidence[str(source / 'SHA256SUMS')] = SOURCE_SHA
        evidence.update({str(source / name): checksum for name, checksum in sources.items()})
        evidence[d['protocol_template']['path']] = d['protocol_template']['sha256']
        base = helper if phase == 'development' else helper._base
        folder = Path(d['output']) / 'setting_0'
        receipt = pool.load(folder / 'receipt.json')
        base.compare_json(receipt, report['settings'][0], 'protocol')
        if receipt['status'] != 'completed' or receipt['concurrency'] != 1 or receipt['seeds'] != definition['seeds']:
            raise ValueError('original serial setting zero required')
        base.check_workers(receipt, folder)
        evidence[str(folder / 'receipt.json')] = pool.digest(folder / 'receipt.json')
        for replica, seed in enumerate(definition['seeds']):
            worker = receipt['workers'][str(replica)]
            population = folder / f'population_{replica}'
            if worker['command'] != base.driver_command(d, seed, population):
                raise ValueError('original serial command differs')
            helper.saved_run(d, sources, population, replica)
            for name in FILES:
                path = population / name
                if pool.digest(path) != worker['output_sha256'][name]:
                    raise ValueError('changed original serial population')
                evidence[str(path)] = worker['output_sha256'][name]
            path = folder / f'worker_{replica}.json'
            evidence[str(path)] = pool.digest(path)
        phases[phase] = dict(declaration=d, helper=helper, sources=sources, serial_folder=folder)
    for path, checksum in evidence.items():
        if pool.digest(path) != checksum:
            raise ValueError('changed original evidence: ' + path)
    return phases, evidence


def tasks_for(phases, source, output, python):
    tasks = []
    for phase, definition in PHASES.items():
        d = phases[phase]['declaration']
        template = d['protocol_template']
        for replica, seed in enumerate(definition['seeds']):
            name = f'{phase}_{replica}'
            destination = output / 'results' / name
            # Preserve original driver argument bytes; replace only its declared
            # destination and insert the checked source-authority bootstrap.
            original = pool._cpu.driver_command(d, seed, destination)
            entrypoint = source / 'scripts/check_galsbi_particles.py'
            argv = pool.command(python, str(source), str(entrypoint), original[5:])
            first = argv.index('--observation-indices') + 1
            tasks.append(dict(id=name, source='original', cwd=str(source), entrypoint=str(entrypoint),
                output=str(destination), argv=argv,
                identity=dict(seed=seed, observations=[58165, 0, 1, 8],
                    recipe=dict(path=template['path'], sha256=template['sha256']),
                    target_law=dict(path=template['path'], sha256=template['sha256'])),
                argument_indices=dict(seed=argv.index('--seed') + 1, output=argv.index('--output') + 1,
                    observations=list(range(first, first + 4))), expected_outputs=list(FILES)))
    return tasks


def prepare(root, destination):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('complete saved-output preparation requires a CPU scheduler allocation')
    started = time.monotonic()
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('fresh replay preparation required')
    phases, evidence = original_evidence(root)
    destination.mkdir(parents=True)
    source = destination / 'source'
    source.mkdir()
    old_source = Path(phases['development']['declaration']['source']['path'])
    for name in [*phases['development']['sources'], 'SHA256SUMS']:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_source / name, path)
    harness = {}
    for name in HARNESS:
        path = destination / name
        shutil.copyfile(Path(__file__).with_name(name), path)
        harness[str(path)] = pool.digest(path)
    python = str(Path(sys.executable).resolve())
    output = root / 'sampler_task_pool_replay_runs_v1_20261022'
    inputs = dict(evidence)
    inputs[python] = pool.digest(python)
    manifest = dict(schema=pool.SCHEMA, workers=4, cores_per_worker=2, output_root=str(output), python=python,
        environment=dict(pool.THREADS, PYTHONDONTWRITEBYTECODE='1',
            SBSI_CACHE_DIR='/project/ls-gruen/users/zekang.zhang/sbsi_caches',
            BLENDEMU_RUNS_DIR='/project/ls-gruen/users/zekang.zhang/blendemu_runs'),
        sources={'original': dict(path=str(source), manifest_sha256=SOURCE_SHA)},
        harness_files=harness, input_files=inputs, tasks=tasks_for(phases, source, output, python))
    manifest_path = destination / 'tasks.json'
    pool.save(manifest_path, manifest)
    # Verify the executed frozen helper, not a working-tree surrogate.
    frozen = imported(destination / 'run_sampler_task_pool.py', '_prepared_generic_pool')
    frozen.verify(manifest_path, pool.digest(manifest_path))
    declaration = dict(schema='sbsi-sampler-task-pool-replay-v1', root=str(root),
        manifest=str(manifest_path), manifest_sha256=pool.digest(manifest_path),
        preparation_seconds=time.monotonic() - started,
        pool_process_cost=str(root / 'sampler_task_pool_replay_process_cost_v1_20261022.json'),
        audit_process_cost=str(root / 'sampler_task_pool_replay_audit_cost_v1_20261022.json'),
        report=str(root / 'sampler_task_pool_replay_report_v1_20261022.json'),
        evidence_sha256=evidence,
        limitations=['All eight original tasks mandatory; no retries or replacements.',
            'Exact saved-output replay only; no scientific accuracy or new timing acceptance.',
            'Historical serial timings are descriptive references from separate allocations.',
            'Preparation excludes interpreter startup, final declaration/manifest serialization and permissions.'])
    pool.save(destination / 'replay.json', declaration)
    top = sorted(p for p in destination.iterdir() if p.is_file())
    (destination / 'SHA256SUMS').write_text(''.join(f'{pool.digest(p)}  {p.name}\n' for p in top))
    for path in destination.rglob('*'):
        path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    print('PREPARED_TASK_POOL_REPLAY', destination, pool.digest(destination / 'replay.json'), flush=True)


def verify_replay(declaration_path, fresh=False):
    d = pool.load(declaration_path)
    if d['schema'] != 'sbsi-sampler-task-pool-replay-v1':
        raise ValueError('unknown replay declaration')
    manifest_path = Path(d['manifest'])
    frozen = imported(manifest_path.with_name('run_sampler_task_pool.py'), '_checked_generic_pool')
    manifest, hashes = frozen.verify(manifest_path, d['manifest_sha256'], fresh=fresh)
    phases, evidence = original_evidence(Path(d['root']))
    if evidence != d['evidence_sha256'] or any(manifest['input_files'].get(p) != h for p, h in evidence.items()):
        raise ValueError('original replay evidence missing or changed')
    source = Path(manifest['sources']['original']['path'])
    if manifest['sources']['original']['manifest_sha256'] != SOURCE_SHA:
        raise ValueError('pool replay requires unchanged original source')
    expected = tasks_for(phases, source, Path(manifest['output_root']), manifest['python'])
    if manifest['tasks'] != expected:
        raise ValueError('all eight original tasks and their exact recipe required')
    pool._cpu.nonnegative(d['preparation_seconds'], 'preparation seconds')
    return d, manifest, hashes, phases


def successful_cost(path, minimum=0.):
    cost = pool.load(path)
    if type(cost['exit_code']) is not int or cost['exit_code'] != 0:
        raise ValueError('successful complete process cost required')
    for key in ('wall_seconds', 'user_seconds', 'system_seconds', 'max_rss_kib'):
        pool._cpu.nonnegative(cost[key], key)
    if cost['wall_seconds'] + .011 < minimum:
        raise ValueError('outer process timer is shorter than internal work')
    return cost


def check_intervals(rows):
    events = [(row['start_offset_seconds'], 1, row['cpus']) for row in rows] + [
              (row['end_offset_seconds'], -1, row['cpus']) for row in rows]
    active, maximum, occupied = 0, 0, set()
    for _, delta, cpus in sorted(events):
        active += delta
        maximum = max(maximum, active)
        if delta == 1:
            if occupied.intersection(cpus):
                raise ValueError('concurrent tasks overlap physical CPU slots')
            occupied.update(cpus)
        else:
            if not set(cpus) <= occupied:
                raise ValueError('invalid task process interval')
            occupied.difference_update(cpus)
        if active < 0 or active > 4:
            raise ValueError('invalid or oversubscribed process intervals')
    if active or occupied or maximum != 4:
        raise ValueError('complete four-worker bounded execution required')
    return maximum


def audit(declaration_path, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('saved-output replay audit requires scheduler allocation')
    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise FileExistsError('original replay report cannot be overwritten')
    d, manifest, hashes, phases = verify_replay(declaration_path)
    if pool.absolute(str(output)) != pool.absolute(d['report']):
        raise ValueError('audit report path must match the replay declaration')
    root = Path(manifest['output_root'])
    receipt = pool.load(root / 'receipt.json')
    if (receipt['status'] != 'completed_execution' or receipt['manifest_sha256'] != d['manifest_sha256'] or
            receipt['hashes_before'] != hashes or receipt['hashes_after'] != hashes or receipt['hashes_unchanged'] is not True or
            len(receipt['tasks']) != 8 or pool.load(root / 'manifest.json') != manifest):
        raise ValueError('all eight original tasks and unchanged inputs required')
    selected = receipt['resources_before']['selected_cpus']
    topology = receipt['resources_before']['cpu_topology']
    if len(selected) != 8 or len({tuple(topology[str(cpu)]) for cpu in selected}) != 8:
        raise ValueError('eight physical cores required')
    comparisons = []
    for index, (task, row) in enumerate(zip(manifest['tasks'], receipt['tasks'])):
        if (row['index'] != index or row['task'] != task or row['id'] != task['id'] or
                row['status'] != 'completed_execution' or row['returncode'] != 0 or
                row['cpus'] not in [selected[i:i+2] for i in range(0, 8, 2)]):
            raise ValueError('missing, reordered or failed original task')
        runtime = root / 'tasks' / task['id']
        if row['runtime_sha256'] != pool.tree_hashes(runtime):
            raise ValueError('changed worker runtime evidence')
        worker = pool.load(runtime / 'worker.json')
        if (worker != row['worker'] or worker['command'] != task['argv'] or worker['pid'] != row['pid'] or
                worker['affinity'] != row['cpus'] or worker['cuda_visible_devices'] != receipt['resources_before']['gpu']['cuda_visible_devices']):
            raise ValueError('worker command/affinity/device differs')
        folder = Path(task['output'])
        if row['output_sha256'] != pool.tree_hashes(folder):
            raise ValueError('changed task output')
        phase, replica = task['id'].rsplit('_', 1)
        replica = int(replica)
        original = phases[phase]
        helper, old = original['helper'], original['declaration']
        adapted = dict(old, source=manifest['sources']['original'])
        a = helper.saved_run(old, original['sources'], original['serial_folder'] / f'population_{replica}', replica)
        b = helper.saved_run(adapted, original['sources'], folder, replica)
        for left, right, kind in zip(a, b, ('result', 'protocol', 'progress')):
            helper.compare_json(left, right, kind)
        arrays = {name: helper.compare_npz(original['serial_folder'] / f'population_{replica}' / name, folder / name)
                  for name in ('particles.npz', 'scenes.npz')}
        comparisons.append(dict(id=task['id'], seed=task['identity']['seed'], all_scientific_json_equal=True,
            complete_64_stages=True, arrays=arrays, output_sha256=row['output_sha256']))
    maximum = check_intervals(receipt['tasks'])
    for key in ('uuid', 'name', 'cuda_visible_devices'):
        if receipt['resources_before']['gpu'][key] != receipt['resources_after']['gpu'][key]:
            raise ValueError('pool GPU identity changed')
    verify_replay(declaration_path)
    cost = successful_cost(d['pool_process_cost'], receipt['internal_cost']['wall_seconds'])
    # Pin the pool and complete cost receipt as well as the scientific outputs.
    proof = {str(root / 'receipt.json'): pool.digest(root / 'receipt.json'),
             d['pool_process_cost']: pool.digest(d['pool_process_cost']),
             str(Path(declaration_path).resolve()): pool.digest(declaration_path)}
    pool.save(output, dict(status='pending_audit_process_cost', all_eight_original_saved_outputs_equal=True,
        comparisons=comparisons, maximum_active_tasks=maximum, pool_process_cost=cost,
        preparation_seconds=d['preparation_seconds'], audit_internal_seconds=time.monotonic() - started,
        proof_sha256=proof, replay_declaration=dict(path=str(Path(declaration_path).resolve()),
            sha256=proof[str(Path(declaration_path).resolve())]),
        scientific_accuracy_accepted=False, limitations=d['limitations']))


def finalization_proof(result, output, cost_path):
    """Recheck the small proof chain, without repeating source/array replay."""
    output, cost_path = pool.absolute(str(output)), pool.absolute(str(cost_path))
    item = result.get('replay_declaration')
    pool.fields(item, 'path sha256', 'audited replay declaration')
    declaration_path = pool.absolute(item['path'])
    pool.checksum(item['sha256'])
    proof = result.get('proof_sha256')
    if not isinstance(proof, dict) or proof.get(str(declaration_path)) != item['sha256']:
        raise ValueError('pinned audited replay declaration required')
    if pool.digest(declaration_path) != item['sha256']:
        raise ValueError('audited replay declaration changed')
    d = pool.load(declaration_path)
    if d.get('schema') != 'sbsi-sampler-task-pool-replay-v1':
        raise ValueError('unknown audited replay declaration')
    if pool.absolute(d['report']) != output or pool.absolute(d['audit_process_cost']) != cost_path:
        raise ValueError('report and audit-cost paths must match the audited declaration')
    manifest_path = pool.absolute(d['manifest'])
    if pool.digest(manifest_path) != pool.checksum(d['manifest_sha256']):
        raise ValueError('audited task manifest changed')
    manifest = pool.load(manifest_path)
    receipt_path = pool.absolute(manifest['output_root']) / 'receipt.json'
    pool_cost_path = pool.absolute(d['pool_process_cost'])
    required = {str(declaration_path), str(receipt_path), str(pool_cost_path)}
    if set(proof) != required:
        raise ValueError('exact declaration, pool receipt and pool cost proof required')
    for path, checksum in proof.items():
        if pool.digest(pool.absolute(path)) != pool.checksum(checksum):
            raise ValueError('audited proof changed: ' + path)
    receipt = pool.load(receipt_path)
    if (receipt.get('status') != 'completed_execution' or receipt.get('hashes_unchanged') is not True or
            receipt.get('manifest_sha256') != d['manifest_sha256']):
        raise ValueError('original completed pool receipt required')
    pool_cost = successful_cost(pool_cost_path, receipt['internal_cost']['wall_seconds'])
    if pool_cost != result['pool_process_cost']:
        raise ValueError('reported pool cost differs from its pinned original receipt')
    pool._cpu.nonnegative(d['preparation_seconds'], 'preparation seconds')
    if result['preparation_seconds'] != d['preparation_seconds']:
        raise ValueError('preparation cost differs from the audited declaration')
    return successful_cost(cost_path, result['audit_internal_seconds'])


def finalize(output, cost_path):
    started = time.monotonic()
    result = pool.load(output)
    expected = [(f'{phase}_{i}', seed) for phase, definition in PHASES.items() for i, seed in enumerate(definition['seeds'])]
    if (result['status'] != 'pending_audit_process_cost' or result['all_eight_original_saved_outputs_equal'] is not True or
            [(row['id'], row['seed']) for row in result['comparisons']] != expected or
            any(row['all_scientific_json_equal'] is not True or row['complete_64_stages'] is not True for row in result['comparisons'])):
        raise ValueError('original complete eight-task pending replay report required')
    cost = finalization_proof(result, output, cost_path)
    audit_cost_sha256 = pool.digest(cost_path)
    validation_seconds = time.monotonic() - started
    result.update(status='completed', audit_process_cost=cost, audit_process_cost_sha256=audit_cost_sha256,
        finalization_validation_seconds=validation_seconds,
        complete_new_work_seconds=result['preparation_seconds'] + result['pool_process_cost']['wall_seconds'] + cost['wall_seconds'] + validation_seconds,
        generic_pool_replay_passed=True,
        cost_scope='Measured source/evidence preparation, complete pool and audit Python processes, and measured finalization proof validation; launcher metadata, queueing, finalizer interpreter startup and final report serialization excluded.')
    pool.save(output, result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'audit', 'finalize'])
    p.add_argument('--root', type=Path)
    p.add_argument('--destination', type=Path)
    p.add_argument('--declaration', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--cost', type=Path)
    a = p.parse_args()
    if a.mode == 'prepare': prepare(a.root, a.destination)
    elif a.mode == 'audit': audit(a.declaration, a.output)
    else: finalize(a.output, a.cost)
