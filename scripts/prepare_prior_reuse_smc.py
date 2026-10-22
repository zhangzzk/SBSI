"""Freeze the original sampler and the three-file prior-reuse candidate."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmark_exchange_full_smc import digest, load, manifest, save

SOURCE_MANIFEST = 'fd96ec78e4b95886a3355082102a25d7a40ffa3dd1a5f4d52443a46458a64df1'
CHANGED = ('sbsi/galsbi_particle_target.py', 'sbsi/galsbi_prior.py',
           'sbsi/galsbi_target_coordinates.py')


def prepare(root, destination):
    start = time.monotonic()
    root, destination = Path(root).resolve(), Path(destination).resolve()
    worktree = Path(__file__).resolve().parents[1]
    source = root / 'source_corrected_heldout_v3_20261022'
    if digest(source / 'SHA256SUMS') != SOURCE_MANIFEST:
        raise ValueError('complete frozen sampler source changed')
    entries = manifest(source)
    if len(entries) != 320:
        raise ValueError('complete 320-file baseline required')
    previous_path = root / 'smc_concurrency_prepared_v1_20261022/declaration.json'
    previous = load(previous_path)
    native_path = root / 'prior_reuse_native_replay_20261022/result.json'
    native = load(native_path)
    if (native['status'] != 'passed' or native['all_exactly_equal'] is not True
            or native['development_rows'] != 73728 or native['probe_rows'] != 15):
        raise ValueError('complete successful installed-photometry replay required')
    for name in CHANGED:
        replay_hashes = {checksum for path, checksum in native['source_sha256'].items()
                         if path.endswith('/' + name) and checksum != entries[name]}
        if replay_hashes != {digest(worktree / name)}:
            raise ValueError('candidate differs from the native-replay implementation: ' + name)
    failed_root = root / 'prior_reuse_smc_runs_v1_20261022'
    failed = dict(job='17446115', optimized_output=str(failed_root / 'pair_0/optimized'),
        process_cost=load(failed_root / 'pair_0_process_cost.json'))
    for key, name in [('receipt', 'pair_0/receipt.json'), ('log', 'pair_0/optimized_runtime/population_0.log'),
                      ('cost', 'pair_0_process_cost.json')]:
        path = failed_root / name
        failed[key] = dict(path=str(path), sha256=digest(path))
    from prior_reuse_proposal_provenance import validate_failed_startup
    validate_failed_startup(failed)
    external = dict(previous['external_files'])
    external.update({failed[key]['path']: failed[key]['sha256'] for key in ('receipt', 'log', 'cost')})
    external.update({str(previous_path): digest(previous_path),
                     str(native_path): digest(native_path)})
    for path, checksum in external.items():
        if digest(path) != checksum:
            raise ValueError('changed frozen input: ' + path)
    destination.mkdir(parents=True, exist_ok=False)
    arms = {}
    for arm in ('baseline', 'optimized'):
        folder = destination / arm
        folder.mkdir()
        arm_entries = {}
        for name, checksum in entries.items():
            original = worktree / name if arm == 'optimized' and name in CHANGED else source / name
            output = folder / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, output)
            arm_entries[name] = digest(output)
            if arm == 'baseline' and arm_entries[name] != checksum:
                raise ValueError('baseline copy changed')
        (folder / 'SHA256SUMS').write_text(''.join(
            f'{checksum}  {name}\n' for name, checksum in sorted(arm_entries.items())))
        arms[arm] = dict(source=str(folder), manifest_sha256=digest(folder / 'SHA256SUMS'))
    from benchmark_prior_reuse_smc import HASHES
    from prior_reuse_proposal_provenance import derived_payload
    copied_sources = {arm: manifest(item['source']) for arm, item in arms.items()}
    derived_path = destination / 'optimized_initial_proposal.json'
    save(derived_path, derived_payload(previous['initial_proposal'], native_path, arms, copied_sources, HASHES))
    external[str(derived_path)] = digest(derived_path)
    files = ['prior_reuse_proposal_provenance.py', 'check_prior_reuse_startup.py', 'benchmark_prior_reuse_smc.py', 'benchmark_smc_concurrency.py',
             'benchmark_exchange_full_smc.py', 'prepare_prior_reuse_smc.py',
             'run_prior_reuse_smc.sh']
    harness = {}
    for name in files:
        output = destination / name
        shutil.copyfile(worktree / 'scripts' / name, output)
        harness[str(output)] = digest(output)
    declaration = dict(schema_version=2, arms=arms, previous_failed_startup=failed,
        external_files=external, harness_files=harness,
        protocol_template=previous['protocol_template'],
        reference=previous['reference'], initial_proposal=previous['initial_proposal'],
        initial_proposals=dict(baseline=previous['initial_proposal'], optimized=str(derived_path)),
        proposal_migration=dict(original=previous['initial_proposal'], derived=str(derived_path),
            native_replay=str(native_path), native_replay_sha256=digest(native_path)),
        physical_geometry=previous['physical_geometry'], settings=previous['settings'],
        expected_runtime=previous['expected_runtime'],
        seeds=[2026103900, 2026103901, 2026103902, 2026103903],
        observations=[58165, 0, 1, 8], python=previous['python'],
        output=str(root / 'prior_reuse_smc_runs_v2_20261022'),
        source_preparation_seconds=time.monotonic() - start,
        cost_scope='Source verification/copy preparation, paired full production processes and diagnostics/reporting, conditional on existing frozen fits; no new pilot/phase fit. Preparation timer excludes final declaration/manifest serialization and permission changes.',
        limitations=['Development efficiency only; no posterior-accuracy acceptance.',
                    'Exactly three runtime files differ; original observations, seeds by pair, model and numerical grouping fixed.',
                    'Native CPU replay and actual CPU production startup preflight are prerequisites, not replacements for full saved GPU replay.',
                    'Original failed v1 process cost retained separately and excluded from successful paired speedup.',
                    'No automatic promotion, retries, replacement seeds or discarded failed pairs.'])
    provisional = destination / 'declaration_preflight.json'
    save(provisional, declaration)
    preflight_root = destination / 'startup_preflight'
    subprocess.run([sys.executable, str(destination / 'check_prior_reuse_startup.py'), 'preflight',
        '--declaration', str(provisional), '--output', str(preflight_root)], check=True)
    preflight = load(preflight_root / 'result.json')
    if preflight['status'] != 'passed':
        raise ValueError('real production startup preflight failed')
    for arm in preflight['arms'].values():
        external.update(arm['evidence_sha256'])
    preflight_path = preflight_root / 'result.json'
    external[str(preflight_path)] = digest(preflight_path)
    declaration['startup_preflight'] = dict(path=str(preflight_path), sha256=digest(preflight_path))
    declaration['source_preparation_seconds'] = time.monotonic() - start
    save(destination / 'declaration.json', declaration)
    top = sorted(p for p in destination.iterdir() if p.is_file())
    (destination / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in top))
    for path in destination.rglob('*'):
        path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    print('PREPARED_PRIOR_REUSE', destination, digest(destination / 'declaration.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.destination)
