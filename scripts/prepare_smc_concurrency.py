"""Freeze a small, counterbalanced CPU-concurrency test of unchanged SMC code."""
import argparse
import json
from pathlib import Path
import shutil
import time

from benchmark_exchange_full_smc import digest, load, manifest, save


SOURCE_MANIFEST = 'fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1'
SEEDS = [2026103700, 2026103701, 2026103702, 2026103703]


def prepare(root, destination, resources_script):
    started = time.monotonic()
    root, destination = Path(root).resolve(), Path(destination).resolve()
    worktree = Path(__file__).resolve().parents[1]
    source = root / 'source_corrected_heldout_v3_20261022'
    if digest(source / 'SHA256SUMS') != SOURCE_MANIFEST:
        raise ValueError('complete frozen sampler source manifest changed')
    entries = manifest(source)
    if len(entries) != 320:
        raise ValueError('complete 320-file source required')
    old_path = root / 'exchange_full_smc_prepared_v2_20261022/declaration.json'
    previous = load(old_path)
    external = dict(previous['external_files'])
    external[str(old_path)] = digest(old_path)
    external[str(source / 'SHA256SUMS')] = SOURCE_MANIFEST
    for name, checksum in external.items():
        if digest(name) != checksum:
            raise ValueError('changed frozen input: ' + name)
    template = previous['protocol_template']
    if digest(template['path']) != template['sha256']:
        raise ValueError('changed audited recipe template')
    protocol = load(template['path'])
    original_source = Path(template['source']).resolve()
    for name, checksum in protocol['implementation_sha256'].items():
        relative = str(Path(name).relative_to(original_source))
        if entries.get(relative) != checksum:
            raise ValueError('active sampler differs from audited template: ' + relative)

    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'source'
    target.mkdir()
    for name in entries:
        output = target / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, output)
    shutil.copyfile(source / 'SHA256SUMS', target / 'SHA256SUMS')
    assert manifest(target) == entries
    harness_files = {}
    names = ['benchmark_smc_concurrency.py', 'benchmark_exchange_full_smc.py',
             'prepare_smc_concurrency.py', 'run_smc_concurrency.sh']
    for name in names:
        output = destination / name
        shutil.copyfile(worktree / 'scripts' / name, output)
        harness_files[str(output)] = digest(output)
    resource_copy = destination / 'detect_resources.py'
    shutil.copyfile(resources_script, resource_copy)
    harness_files[str(resource_copy)] = digest(resource_copy)
    declaration = dict(
        schema_version=1,
        source=dict(path=str(target), manifest_sha256=SOURCE_MANIFEST),
        protocol_template=template,
        external_files=external,
        harness_files=harness_files,
        python='/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python',
        output=str(root / 'smc_concurrency_runs_v1_20261022'),
        reference=previous['reference'],
        physical_geometry=previous['physical_geometry'],
        initial_proposal=previous['initial_proposal'],
        observations=[58165, 0, 1, 8], seeds=SEEDS,
        order=[1, 2, 4, 4, 2, 1], threads_per_worker=2,
        required_allocated_cpus=8,
        settings=previous['settings'],
        expected_runtime=previous['expected_runtime'],
        source_preparation_seconds=time.monotonic() - started,
        selection_rule={
            'eligible': 'All 24 complete populations replay bitwise, no operational failure, both per-repetition speedups versus concurrency one at least 1.25.',
            'ranking': 'Largest minimum speedup across the two repetitions; within five percent prefer fewer workers.',
            'promotion': 'Development only; require separate fixed untouched-seed full-recipe timing/replay before production adoption.',
            'future_test_seeds': [2026103800, 2026103801, 2026103802, 2026103803],
        },
        limitations=[
            'Fixed existing development fits; no new pilot, phase-control fit, or posterior accuracy comparison.',
            'Same four seeds repeated for counterbalanced timing; these repeats are not independent scientific replicas.',
            'One GPU with eight allocated physical CPU cores throughout; not a 500-core capacity benchmark.',
            'Original four-observation groups, density 4096, two worker threads and all sampler settings fixed.',
            'Exactness covers saved outputs and all 64 stage records, not unrecorded mutation states or terminal RNG.',
            'Source preparation timing excludes final declaration serialization and manifest/permission writes.',
        ])
    save(destination / 'declaration.json', declaration)
    top = sorted(p for p in destination.iterdir() if p.is_file())
    (destination / 'SHA256SUMS').write_text(''.join(
        f'{digest(p)}  {p.name}\n' for p in top))
    for path in destination.rglob('*'):
        path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    print(json.dumps(dict(destination=str(destination),
                         declaration_sha256=digest(destination / 'declaration.json'),
                         manifest_sha256=digest(destination / 'SHA256SUMS'),
                         submitted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--resources-script', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.destination, args.resources_script)
