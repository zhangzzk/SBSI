# Work log

## 2026-11-08 — GalSBI model scene inputs and image-reference builder for the pilot

Owner request: test infer-1.0-galsbi with like-1.0-galsbi on measured GalSBI
constant-shear image catalogues, starting with small subsets (up to 6 GPUs and
500 CPUs).

Found before any run: the GalSBI models were trained on LSST-i magnitudes and on
the rendered Sersic index (clipped to 0.3-6.2, 122-point log grid). The sampler
gave them the prior's HSC-r `m_r` and raw `n`. On scene files LSST-i is
0.33 ±0.25 mag brighter. This affected flow/C8 features, crowding and emulator
pairs. Fixes:
- The manifests of `like-1.0-galsbi` and `galsbi-multistage` declare
  `scene_inputs`. `ModelPaths` carries `scene_inputs` and `observing_conditions`.
- `sbsi/galsbi/model_inputs.py`: `SceneMarks` hooks the model band into
  `build_prior`'s photometry. Drawn and decoded galaxies then carry `m_model`,
  computed once from the photometry's own coefficients. `scene_model_inputs` and
  `response_pairs` take `marks=`. The default `None` is unchanged for V3.x.
- `prior.sample` keeps any extra photometry output.
- `run_galsbi_inference.py` and the `run_galsbi_bright.py` selection use the
  named model's marks. Both refuse a reference whose conditions differ from the
  model's pinned training conditions. Selection takes `--reference` and records
  the model and scene inputs.
- New `scripts/prepare_galsbi_image_reference.py` builds paired plus/minus
  `galsbi_population_v1` references. One shared identity draw serves both legs,
  each leg applies its own normalization cuts, unmatched rows are counted, and
  the unsheared inputs and declared shear are checked.
- New `scripts/check_galsbi_scene_marks.py` replays the scene inputs on written
  scene files.

Validation: Slurm job 17835256 replays scenes 1000 and 1077. Magnitude max
difference is 1.4e-14 and is batch-independent; other photometry is unchanged;
Sersic differs by at most 8.9e-16 (one rounding step) and radii are exact. 256
drawn targets and 5620 neighbours carry the exact hooked magnitude. Full suite:
387 passed, 1 skipped. A one-case real build gives 76/75 galaxies at fraction
5e-4. About 1560 detections per leg have no crossmatch; they are dropped and
counted.

Limitations: no inference has been calibrated by this. Pilot launched in
`galsbi_image_pilot_20261108` (rule.md); results to follow.

## 2026-11-08 — Review and reorganize the Python package

Trace the dependency closure from all retained entry points and public exports.
Archive19 unused experimental modules and15 associated tests, preserving their
bytes and the full original retained source in the
[package archive](../archive/package-2026-11-08/README.md#contents-and-restoration).
Move30 retained modules into flows/galsbi/disk/finite subpackages; retain16
root modules and four lightweight initializers. Update package discovery,
callers, tests, model-loader source paths and documentation. Public loading
and prediction paths remain; direct internal imports use the migration map.
Format the retained factorized-flow implementation without changing its AST.

Review finds a startup source-copy failure: the current inference entry still
required the launcher archived in the previous cleanup. Replace its partial,
cwd-dependent source list with recursive package capture from the checkout.
Add source-snapshot regression coverage outside the checkout and require the
disk driver's source identity to include nested flow/transport implementations.
The disk driver now hashes subpackages. Existing cache identity checks remain;
old caches are not relabelled. Source provenance changes, science does not.

Validation: clean-tree CPU job17835049 passes378 tests/3 skips in59.06s,
focused12 tests/1 skip, both model validators and direct/module GalSBI plus
finite CLI help. Wheel build includes all50 Python sources and excludes archive
code. Its isolated import then fails because the model registry requires the
checkout manifests, consistent with the existing editable-only resource
contract; this is not a supported standalone-wheel deployment. Follow-up
job17835061 completes0:0 after an isolated editable installation: all48 modules
and10 root exports import from outside the checkout, both bundled model
validators and the direct inference CLI pass. Full tests were not repeated.

All46 retained package ASTs equal the originals after declared import/path
mapping; sampling/evaluation body bytes are unchanged. All184 archive hashes,
305 package imports and99 documentation links pass. Model weights and pinned
report/source bytes are unchanged. Final runtime/config/package metadata match
the tested tree. [Verification receipt](evidence/package_cleanup_20261108/validation.json).
No scientific jobs or closed investigations were restarted.

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
