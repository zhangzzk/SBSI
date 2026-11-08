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

## Package layout

| Location | Responsibility |
| --- | --- |
| `sbsi/` | Public model loading/prediction, catalogue utilities, coordinates and shared transforms |
| `sbsi/flows/` | Factorized, disk and sign-symmetric flow architectures behind `load_measurement_model` |
| `sbsi/galsbi/` | Continuous population prior, scenes, feature cache, likelihood summaries, coordinates and MH helpers |
| `sbsi/disk/` | Output-conditioned emulator, transport, radial precision and disk likelihood |
| `sbsi/finite/` | Finite-catalogue priors, cache infrastructure, sampling, selection normalization and estimators |

The public `sbsi` exports, `sbsi.models`, `sbsi.measurement_model` and
`sbsi.selection_model` remain in place. Thirty internal modules moved to these
subpackages; their original-to-current names are in the
[migration manifest](../archive/package-2026-11-08/manifest.json).
Repository callers and tests use the new names. External notebooks importing
those internal modules must update their imports using that map.

The GalSBI disk likelihood still shares cache/likelihood infrastructure with
`finite/`; this is an actual dependency. `galsbi/features.py` retains the
historical `V39CatalogueCache` class name and feature contract. Its use by
current code does not change a model identity. Flow checkpoints keep their
original formats, state names and weights.

## November 8 package review

The retained entry points and public exports require 46 of the original
65 modules. Archive the other 19 modules and 15 associated tests: experimental
particle/SMC, exchange/relocation, phase-control, gradient and initialization
studies. Their original source, dependency context and tests remain in the
[package archive](../archive/package-2026-11-08/README.md#contents-and-restoration).
The retained package has 16 root modules and four subpackages, each with a
lightweight initializer. Package discovery includes all four in builds.

The current inference source snapshot previously referenced an archived
launcher and assumed the checkout was the working directory. Source capture
now resolves the checkout explicitly and includes subpackages recursively.
The disk driver also hashes subpackages. This changes source/cache provenance;
old prepared caches must pass existing identity checks or be rebuilt.
It does not change proposals, random streams, evaluation order or model weights.

## November 8 cleanup

The owner requested an active tree centered on `infer-1.0-galsbi`,
`like-1.0-galsbi`, the retained multistage comparison, core math and conventions.
The [archive manifest](../archive/research-2026-11-08/manifest.json) records
original paths, archived paths, byte counts, SHA256 hashes, tracking status
and move/snapshot actions. Every previously tracked archived file was also
checked against its original Git blob. Full old work logs are snapshots;
[HISTORY.md](HISTORY.md#galsbi-response-and-photometry-november-2026) provides the
short findings index. That first cleanup preserved model and library bytes;
the subsequent package reorganization is recorded separately above.

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

- [November 8 package review](../archive/package-2026-11-08/README.md#contents-and-restoration): unused modules/tests, original retained source and module migration map.

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
