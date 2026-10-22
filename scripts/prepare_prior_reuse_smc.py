"""Freeze the original sampler and the three-file prior-reuse candidate."""
import argparse
from pathlib import Path
import shutil
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
    external = dict(previous['external_files'])
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
    files = ['benchmark_prior_reuse_smc.py', 'benchmark_smc_concurrency.py',
             'benchmark_exchange_full_smc.py', 'prepare_prior_reuse_smc.py',
             'run_prior_reuse_smc.sh']
    harness = {}
    for name in files:
        output = destination / name
        shutil.copyfile(worktree / 'scripts' / name, output)
        harness[str(output)] = digest(output)
    declaration = dict(schema_version=1, arms=arms,
        external_files=external, harness_files=harness,
        protocol_template=previous['protocol_template'],
        reference=previous['reference'], initial_proposal=previous['initial_proposal'],
        physical_geometry=previous['physical_geometry'], settings=previous['settings'],
        expected_runtime=previous['expected_runtime'],
        seeds=[2026103900, 2026103901, 2026103902, 2026103903],
        observations=[58165, 0, 1, 8], python=previous['python'],
        output=str(root / 'prior_reuse_smc_runs_v1_20261022'),
        source_preparation_seconds=time.monotonic() - start,
        cost_scope='Source verification/copy preparation, paired full production processes and diagnostics/reporting, conditional on existing frozen fits; no new pilot/phase fit. Preparation timer excludes final declaration/manifest serialization and permission changes.',
        limitations=['Development efficiency only; no posterior-accuracy acceptance.',
                    'Exactly three runtime files differ; original observations, seeds by pair, model and numerical grouping fixed.',
                    'Native CPU replay is a prerequisite, not a replacement for full saved GPU replay.',
                    'No automatic promotion, retries, replacement seeds or discarded failed pairs.'])
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
