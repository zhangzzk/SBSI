"""Freeze the predeclared n24 confirmation after complete development replay."""
import argparse
from pathlib import Path
import shutil
import time

from benchmark_exchange_full_smc import manifest
from benchmark_smc_concurrency_confirmation import (
    audit_development, digest, load, save, PHASE, SEEDS, ORDER, SOURCE_SHA)


def prepare(root, destination):
    started = time.monotonic()
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('confirmation preparation cannot overwrite an earlier attempt')
    declaration = root / 'smc_concurrency_prepared_v1_20261022/declaration.json'
    report = root / 'smc_concurrency_runs_v1_20261022/report.json'
    parent_audit = root / 'smc_concurrency_parent_audit_v1_20261022/result.json'
    selection = audit_development(declaration, report, parent_audit, replay=True)
    old = load(declaration)
    destination.mkdir(parents=True)
    source = destination / 'source'
    original = Path(old['source']['path'])
    entries = manifest(original)
    source.mkdir()
    for name in entries:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, path)
    shutil.copyfile(original / 'SHA256SUMS', source / 'SHA256SUMS')
    if manifest(source) != entries or digest(source / 'SHA256SUMS') != SOURCE_SHA:
        raise ValueError('confirmation source changed')
    selection_path = destination / 'development_selection.json'
    save(selection_path, selection)
    worktree = Path(__file__).resolve().parents[1]
    harness = {}
    for name in ('benchmark_smc_concurrency_confirmation.py', 'benchmark_smc_concurrency.py',
                 'benchmark_exchange_full_smc.py', 'prepare_smc_concurrency_confirmation.py',
                 'run_smc_concurrency_confirmation.sh'):
        path = destination / name
        shutil.copyfile(worktree / 'scripts' / name, path)
        harness[str(path)] = digest(path)
    external = dict(selection['evidence_sha256'])
    external[str(selection_path)] = digest(selection_path)
    d = dict(old, phase=PHASE, selected_workers=4, seeds=SEEDS, order=ORDER,
        source=dict(path=str(source), manifest_sha256=SOURCE_SHA), external_files=external,
        harness_files=harness, output=str(root / 'smc_concurrency_confirmation_runs_v1_20261022'),
        development_selection=dict(path=str(selection_path), sha256=digest(selection_path)),
        source_preparation_seconds=time.monotonic() - started,
        selection_rule=dict(confirmation='All16 replay, both internal-inclusive and outer-process speedups>=1.25; no replacements or retries.',
                            scientific_accuracy_accepted=False),
        limitations=['Original development selection is fixed; no further tuning.',
            'Fresh seeds2026103800..03 on the same four development observations.',
            'Preparation includes development proof replay and source/harness copies; excludes final declaration, top manifest and permission writes.',
            'Existing pilot/fit and original development/search costs excluded from confirmation timing.',
            'No production posterior accuracy or500-core scaling claim.'])
    save(destination / 'declaration.json', d)
    top = sorted(p for p in destination.iterdir() if p.is_file())
    (destination / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in top))
    for path in destination.rglob('*'):
        path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    print('PREPARED_CONFIRMATION', destination, digest(destination / 'declaration.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    prepare(a.root, a.destination)
