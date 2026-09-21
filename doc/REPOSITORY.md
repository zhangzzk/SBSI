# Repository organization

## Active layout

`sbsi/` contains model prediction/loading and the finite-prior likelihood,
normalization, sampling, and inference implementation. V3.6-like runtime
building blocks include `crowding.py`, `disk_response_transport.py`,
`output_conditioned_response.py`, and the joint-flow loader.

`scripts/` retains the documented scene/prior preparation, cache building,
partition combination, inference, uncertainty audit, and summary commands.
See [the command sequence](INFERENCE.md#active-workflow). `jobs/` contains
the two inference wrappers and the two blend-response cache wrappers.

`tests/` retains regression tests for these supported paths and the model
feature/transport contracts. Archived experiments are excluded from default
pytest discovery. `configs/` separates likelihood and estimator configurations
from the V3.6-like model manifest. Documentation lives in `doc/`.

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
