# Repository organization

## Active layout

| Directory | Supported contents |
| --- | --- |
| `sbsi/` | Model loading/prediction; finite-prior likelihood, selection, sampling and inference |
| `scripts/` | Reusable scene/prior preparation, caches, inference, combination, audits, atom maps and the bright-prior extension experiment |
| `jobs/` | Reusable inference/cache launchers, `job_tests.sh`, atom maps and bright-prior experiment launchers |
| `tests/` | Core regression and artifact/feature/transport contract tests |
| `configs/` | Separate numerical, likelihood and frozen model identities |
| `doc/` | Maintained guides, concise work log and pinned V3.9 evidence |
| `archive/` | Dated research source and historical documentation; excluded from tests/imports |

Start with [V3.9-like](V39_MODEL.md#decision-and-identity) for the retained
model and [the inference workflow](INFERENCE.md#active-workflow) for commands.
The reusable V3.6 disk driver remains explicitly V3.6-specific; naming V3.9
does not change its feature or prepared-cache contracts.

## October 25 sampler simplification

`scripts/check_galsbi_local_sampler.py` now runs only the production recipe: persistent scene, one
environment, exact type sum, per-chain scales, SED jumps, joint morphology, cached invariant work,
likelihood-drawn starts from a fresh bank, and co-moving readings with no saved chain. Tempering and
narrow steps remain options. The chains are bit-identical to snapshot d776abb on this recipe; see
[Simplified sampler](TEMPERED_MH.md#simplified-sampler--october-25-2026).

The launchers that drove the removed modes were deleted from the active tree. They remain in git history
at commit `821e65f` (`git show 821e65f:<path>`):

- `scripts/run_galsbi_population_shard.py`, `scripts/run_galsbi_throughput.py`,
  `scripts/check_galsbi_cache_replay.py`
- `jobs/job_galsbi_initialization.sh`, `jobs/job_galsbi_overnight.sh`, `jobs/job_galsbi_local_sampler.sh`,
  `jobs/job_galsbi_throughput.sh`

`jobs/job_galsbi_population.sh` keeps its prepare/report stages. `scripts/run_galsbi_bright.py` keeps its
prepare/selection stages, and its old MH+profile stage is gone. Studies run from frozen source snapshots,
so existing results are unaffected.

## September 25 cleanup

The owner closed development and named F8+C8+disk V3.9-like on 2026-09-25.
The cleanup removes from the active tree 37 experiment scripts, 38 fixed-run
launchers, 15 experiment tests and five detailed investigation documents.
All 95 files are preserved byte-for-byte under `archive/research-2026-09-25/`.
Another 66 snapshots capture the dirty working-tree versions before editing,
including core code, configurations, documentation and the full work log.
The [manifest](../archive/research-2026-09-25/manifest.json) records every original
path, archived path, size, SHA-256 and move/snapshot action.

The model registry shares one disk-preset loader for V3.6 and V3.9. Obsolete
circularized-feature training helpers and a one-off historical cache-identity
exception were removed; the predictor, transformations, physical transport,
selection and inference regression tests remain. Stale prepared caches now
require rebuilding. Unique scientific results, datasets, checkpoints, external
research checkouts and figures were preserved. No git reset or clean was used.

Detailed histories:

- [Feature simplification](../archive/research-2026-09-25/doc/V36_FEATURE_SIMPLIFICATION.md#owner-retains-c8-provisionally-emulator-inputs)
- [Model comparison](../archive/research-2026-09-25/doc/V36_MODEL_COMPARISON.md)
- [Domain recovery](../archive/research-2026-09-25/doc/V36_DOMAIN_RECOVERY.md)
- [Inference review](../archive/research-2026-09-25/doc/V36_INFERENCE_REVIEW.md)
- [Full prior work log](../archive/research-2026-09-25/snapshots/doc/WORKLOG.md)

## Research archive

The 2026-09-20 cleanup preserves 453 retired files under
`archive/research-2026-09-20/`, mirroring original relative paths:

- 150 research scripts and 199 scheduler launchers;
- 90 experiment-specific tests and seven plotting programs;
- four unused experimental package modules;
- three detailed ConstGold investigation documents.

Nineteen additional snapshots preserve files retained or revised during
cleanup, including the full work log, joint investigation, and model registry.
The machine-readable [manifest](../archive/research-2026-09-20/manifest.json)
records original path, archive path, SHA-256, byte count, and move/snapshot
action for every file. Earlier research remains under
`archive/research-2026-09-14/`.

The archive preserves working-tree edits present at cleanup, including files
that had never been committed. No model checkpoints, datasets, or scientific
results were deleted. Unique research code was moved out of the active tree
rather than erased. Original source bytes remain available for provenance.

## Restoring an experiment

Archived source is historical and is not a supported execution tree. Its
imports, scheduler paths, and source pins refer to the original repository
layout. Do not add `archive/` to active imports or simply launch an archived
job. To restore an experiment deliberately:

1. Create an isolated checkout and use the manifest to restore original paths.
2. Restore the required dependency closure, including any earlier archive
   dependencies, and verify all source/model/data hashes.
3. Review original absolute paths and Slurm resources, then run its tests and
   smoke checks before a new submission.

The stopped independent validation requires a new owner request to resume.
Its original frozen protocols and partial data remain in external storage.

## Historical documents

- [Joint investigation](../archive/research-2026-09-20/doc/JOINT_CALIBRATION.md#goal-and-retained-population)
- [Full work log before cleanup](../archive/research-2026-09-20/doc/WORKLOG.md)
- [500k inference review](../archive/research-2026-09-20/doc/INFERENCE_CONSTGOLD_500K_REVIEW.md)
- [20k numerical diagnostics](../archive/research-2026-09-20/doc/INFERENCE_CONSTGOLD_20K_DIAGNOSTICS.md)
- [ConstGold prior-domain rerun](../archive/research-2026-09-20/doc/INFERENCE_CONSTGOLD_RE037_SIZE050.md#authorized-setup)

Historical documents retain their original contents and relative links.
Resolve old repository-relative references via the archive manifest when a
link no longer matches the active layout.
