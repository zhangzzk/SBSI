# Work log

## 2026-11-08 — Model band as the sampler magnitude coordinate (owner-approved)

Change (aaa48bc, doc 978ce77): `GalSBITargetCoordinates` uses `m_model` as its
magnitude coordinate when a model band is hooked into the prior photometry
(SceneMarks, i.e. models declaring `scene_inputs`). Otherwise it keeps `m_r`.
`run_galsbi_inference.py` writes the coordinate names to `protocol.json` and
refuses to run if the constructed coordinates differ. The native V3.x behaviour
is unchanged.

Validation:
- `tests/test_galsbi_target_coordinates.py`: a new test uses a stand-in band
  with its own z/SED colour. It checks the round trip and the finite-difference
  Jacobian (1e-8), and that a z/SED move keeps `m_model` while `m_r` changes.
- `scripts/check_galsbi_scene_marks.py` now also checks the real LSST-i band
  on scenes 1000/1077 (about 885k rows each). The band shifts one-for-one with
  M_B (max error 1.7e-14), and the coordinate round trip holds to 3.2e-14 (M_B)
  and 2.8e-14 (band). all_pass. Job 17846892 ran this and the full suite:
  389 passed, 1 skipped.
- A/B in `$DATA_DIR/sbsi/catalogue_prior/galsbi_magcoord_ab_20261108`
  (job 17846891, 1 GPU, 38 min): 12 pilot c1 units, the first 3 bright and
  3 faint per leg (1080 galaxies), same reference and recipe.

| R-hat >= 1.1 | m_r coordinate (pilot) | m_model coordinate |
|---|---|---|
| log z, all | 69.8% ±1.4% | 8.0% ±0.8% |
| log z, mag<21 (48) | 94% | 42% |
| log z, 23-24 (323) | 95% | 10% |
| shear ratio g1, all | 5.6% | 1.5% |

Median log-z ESS rose from 9-32 to 77-510. The old-model 100k run had 9.5%.
Mixing of the remaining parameters (m, e, Re, crowding) is unchanged within
noise; bright m/Re/e mixing stays poor, as in all earlier runs. The ± values
are binomial.

Limitations: no shear estimate has been made with the new coordinate yet. The
pilot result (m1 -3.2% ±5.2%) was made with the old coordinate.
Next: rerun the pilot or a larger subset with the new coordinate, using larger
units.

## 2026-11-08 — GalSBI image pilot result (infer-1.0-galsbi, like-1.0-galsbi)

Run: `$DATA_DIR/sbsi/catalogue_prior/galsbi_image_pilot_20261108`
(`rule.md`, `paired_estimate.json`). Measured catalogues of constant_g002
cases 1000-1139 (±0.02 legs), 5e-4 identity draw: 9603 plus / 9638 minus
selected (9422 on both legs, 181 plus-only, 216 minus-only). Sampler recipe
unchanged. All 219 c1 and 42 c2 units completed with no failures or retries.
Selection: 4 replicas x 50 x 8192.

| leg | g1 | g2 |
|---|---|---|
| plus (truth +0.02) | +0.02296 ±0.00356 | +0.00133 ±0.00357 |
| minus (truth -0.02) | -0.01576 ±0.00354 | +0.00204 ±0.00361 |

Paired: m1 = -3.20% ±5.19% (-0.6σ), c1 = +0.00360 ±0.00336 (+1.1σ),
c2 = +0.00168 ±0.00347 (+0.5σ), g1-to-g2 response -0.018 ±0.052. The quoted
errors are the largest of the case bootstrap and the galaxy influence. The
chain floor is ±1.8% on m1.
This is an end-to-end check at 10k galaxies per leg, not an acceptance test.

Finding: redshift mixing collapsed. 69% of c1 galaxies have R-hat(log z)
>= 1.1, versus 9.5% in galsbi_constgold100k_20261025 (old model). Median log-z
ESS is 10-30, versus 120-200. Magnitude, shape and size mixing are similar to
the old run. The shear ratio (R-hat >= 1.1) is flagged in 22% of mag<21
galaxies in c1 (old: 5%) and 0% after c2; at fainter magnitudes it is 0-11%
(old: 1-7%). Likely cause: the sampler's magnitude coordinate is HSC-r `m_r`,
while the model now uses LSST-i. At fixed `m_r`, a redshift move changes the
model magnitude, so adaptation shrinks redshift steps. Proposed (owner
decision, not applied): use the model band as the magnitude coordinate. At
fixed z and SED the two differ by a shift, so the Jacobian is 1.

Also cancelled extra c1 worker 17835427 (all units already claimed).
Small units are inefficient: a 40-galaxy bright unit takes about 1.7 h.
Next: decide on the coordinate change; then a larger subset with larger units.

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

Pilot preflight then found a second gap: every nonzero-shear likelihood
evaluation failed for like-1.0-galsbi. `FactorizedCondition` lacked the
`TabularPreprocessor` interface that `CatalogueLikelihood._shear_standardized_context`
uses: the device was keyword-only, and there were no `means`/`scales`/`log_features`.
The fix delegates these to the base preprocessor, which owns e1/e2. The regression
test `tests/test_factorized_shear_context.py` on the real checkpoint matches
standardizing a sheared frame to 2e-6. Suite (job 17835377): 386 passed, 3 skipped.

Pilot `$DATA_DIR/sbsi/catalogue_prior/galsbi_image_pilot_20261108` (rule.md):
- Reference: 9603 plus and 9638 minus galaxies (fraction 5e-4 over cases
  1000-1139), 9422 selected on both legs, 219 c1 units.
- Preflight 17835392 passed. The bright tempered, faint nonzero-shear and selection
  smokes are all finite; the selected fraction is 0.154 at all five nodes.
- Running from snapshot e81e8de: selection array 17835393 (4 x 50 chunks), c1
  workers 17835394-97, plus 17835426-27 after selection (6 GPUs, at most
  192 CPUs). The c1 estimate 17835398 then submits c2 and the paired estimate.
- Rate: a 40-galaxy bright unit takes about 1.7 h (fixed per-step overhead
  dominates small units). Use larger units when scaling up.

Limitations: no inference has been calibrated by this; results to follow.

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
