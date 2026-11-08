# Repository organization

## Active layout

| Directory | Contents |
| --- | --- |
| `sbsi/` | Model loading, prediction, likelihood, selection and inference library; compatibility modules retain their regression tests |
| `scripts/` | 32 current inference helpers and reusable finite-catalogue preparation, cache and reporting tools |
| `jobs/` | Retained inference/cache/test wrappers; scientific paths and prerequisites remain explicit |
| `tests/` | Core regression, artifact replay, model identity and retained runtime-contract tests |
| `configs/` | Current model/inference identities and historical compatibility configurations |
| `models/` | Both pinned GalSBI flows and companions; small configured compatibility artifacts |
| `doc/` | Current guides, core math/conventions, history and a small pinned evidence set |
| `logs/`, `plots/` | Runtime output locations; previous contents archived |
| `archive/` | Research snapshots, excluded from active imports and default test discovery |

Start with [current inference](CURRENT_INFERENCE.md#entry-points),
[the fiducial model](LIKE_1_0_GALSBI.md#decision-and-identity), and
[history](HISTORY.md#how-to-read-this-history).
The package and its compatibility interfaces have not been pruned by filename:
reusable code and its tests remain even when a historical study is archived.

## November 8 cleanup

The owner requested an active tree centered on `infer-1.0-galsbi`,
`like-1.0-galsbi`, the retained multistage comparison, core math and conventions.
The [archive manifest](../archive/research-2026-11-08/manifest.json) records
original paths, archived paths, byte counts, SHA256 hashes, tracking status
and move/snapshot actions. Every previously tracked archived file was also
checked against its original Git blob. Full old work logs are snapshots;
[HISTORY.md](HISTORY.md#galsbi-response-and-photometry-november-2026) provides the
short findings index. Both model checkpoints and all library code retain
their bytes.

Originally ignored logs, plots and evidence outputs are preserved locally
inside the archive, with hashes in the manifest. Their bytes remain outside
Git, as before; a Git clone alone does not reproduce that local output archive.
Tracked research source and reports are included in Git. Source worktrees,
external datasets/caches and scheduler state were not changed.

The earlier consolidation is recorded in the
[archived receipt](../archive/research-2026-11-08/doc/evidence/dev_consolidation_20261108/validation.json).
Its current production-source receipt and model-artifact receipt remain in
`doc/evidence/dev_consolidation_20261108/`. The recovery ref
`refs/codex-snapshots/pre-galsbi-integration-20261108` preserves the earlier dev tree.

## Research archive

- [November 8, 2026](../archive/research-2026-11-08/README.md#contents-and-restoration): historical documents, experiments, launchers, their tests, logs and plots.
- [September 25, 2026](../archive/research-2026-09-25/manifest.json): completed F8/C8 investigation and pre-cleanup snapshots.
- [September 20, 2026](../archive/research-2026-09-20/manifest.json): closed joint/ConstGold work and historical inference diagnostics.
- `archive/research-2026-09-14/`: earlier research and mathematical records.

Historical documents retain original bytes and may contain links to their old
layout. Resolve those references through the manifest; active guides link to
the retained or archived locations directly.

## Restoring an experiment

1. Use an isolated checkout at the archive's baseline commit, or restore the
   manifest's original paths and their full dependency closure.
2. Verify source, model and data hashes. Originally untracked artifacts need
   the local archive or their original external storage, not just Git.
3. Review scheduler paths and resource limits, then run the restored tests and
   smoke checks before submitting scientific work.

Do not import directly from `archive/` or treat an archived launcher as an
active interface. Closed investigations and paused evaluations need a new
owner request before resumption. Restoration does not transfer historical
validation to new model identities or scientific settings.
