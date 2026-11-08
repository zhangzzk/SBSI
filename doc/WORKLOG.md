# Work log

## 2026-11-08 — Clean active tree around the GalSBI identities

Owner requested a smaller doc/scripts/jobs/logs/plots tree. Retain core math,
conventions, runtime contracts, infer-1.0-galsbi, like-1.0-galsbi and the
multistage comparison. Add [research history](HISTORY.md#how-to-read-this-history),
rewrite the active navigation and distinguish current continuous inference
from finite-catalogue compatibility. Archive 1421 documentation/evidence
files, 81 study scripts, 26 launchers, 20 study-specific tests, 1022 logs
and 89 plots. Preserve originals of revised documents/configurations in
snapshots. Models, all sbsi modules and all retained scripts/tests are unchanged.

The [manifest](../archive/research-2026-11-08/manifest.json) records 2674
move/snapshot entries. All 1331 previously tracked archived files match their
original Git blobs. Previously ignored output bytes remain in the local
archive, outside Git, with hashes recorded. Relative source symlinks are
preserved. Historical V3.9 evidence references now resolve to the archive;
all evidence and artifact hashes remain unchanged.

Validation: clean-export CPU job17834966 completed0:0; 649 tests passed,
4 skipped (one existing NumPy deprecation warning), 75.68s. Both bundled
model validators, isolated10 integration tests (1 skip), and direct/module
inference CLI help passed. All2674 archive checksums, 25 artifact/evidence
pins, 89 active documentation links/anchors and active script dependencies
passed. Every retained runtime/model/config file still matches the tested tree.
[Validation receipt](evidence/cleanup_20261108/validation.json). No scientific runs, source-worktree
cleanup, held-job changes or automated follow-ups. Next: use the current
pipeline/model guides; restore historical studies deliberately from the archive.

## 2026-11-08 — Name and consolidate current GalSBI references

Named infer-1.0-galsbi for the continuous GalSBI-prior inference core, separate
from model identities. Retain control_nine as like-1.0-galsbi and the weighted
multistage flow as a comparison. Both checkpoints and companions are bundled;
the standard loader replays original physical densities and draws exactly.
The current inference entry preserves the executed cache-off numerical sampler
and rejects model/reference mismatches. No end-to-end inference result exists
for these new model sets; multistage still needs an expanded summary adapter.

Consolidation clean-export job17834811 passed 758 tests, 4 skips, both model
validators and both CLI forms. Detailed changes, failed validation attempts,
scientific findings and receipts are in the
[complete earlier log](../archive/research-2026-11-08/snapshots/doc/WORKLOG.md).
