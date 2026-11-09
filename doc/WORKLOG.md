# Work log

## 2026-11-09 (evening) — Where a sampler step's time goes (profile, no code change)

Owner: "look deeply into the pipeline and see if we can improve the computation efficiency and speed by a lot".

Profile (`$DATA_DIR/sbsi/speed_profile_20261109/`, job 17856944, cip A40-16Q slice, one process alone on the
GPU, cProfile, 200k-run source snapshot, 640 steps / 256 warm-up / stride 16, plus-leg galaxies):

- c1 bright, 160 galaxies (5120 rows, 8 levels): sampling 76 s = 119 ms/step. Share of sampling time: neighbour
  refresh 31% (of which ~85% is the truncated betaprime Sersic cdf/ppf of new neighbours), prior density
  `log_mixture_prior` 23% (photometry tables called 4x per row), flow+classifier likelihood ~24%, `fill_b`
  9%, co-moving ratios 5%, scene copies ~7%.
- c1 faint, 1022 galaxies (4088 rows): sampling 72 s = 113 ms/step. Refresh 27%, prior 23%, likelihood 17%,
  `fill_b` 20%, co-moving ratios 13%. Setup 68 s, diagnostics 25 s (at full length ~2-3% of a unit).
- c2 bright, 20 galaxies at (0.02071, 0.00140): 49 ms/step alone vs ~135 ms/step in production.
- The GPU part (flow density) is a small share; every step is CPU-bound in one Python process.
- xgboost on GPU: building the DMatrix on one thread costs more than the prediction (faint: 17 s vs 14 s).
- Production units at full length (200k c1, 80 units each): bright 40-galaxy 149 ms/step, faint 140-galaxy
  136 ms/step; sampling is 99% / 97% of a unit.
- Worker layout: `-c 32` gives 16 physical cores plus their hyperthreads (c2 worker 17853252 on kng-cl-nv01:
  CPUs 2-9,11-18 and siblings 58-65,67-74), so 30 slots share 16 cores. GPU busy 28-37% (batch test).

Candidate changes (not implemented; estimated from the shares above, one profiled run each, so roughly
+-20% relative; cProfile inflates many-small-call Python code):

1. Exact: draw neighbour Sersic uniforms as now but invert only when b needs them (filled in place on the
   persistent scene). At zero shear b is needed only for stored states. ~-25% bright, ~-23% faint.
2. Exact: `fill_b` only at profile steps (and before the invariant replay). ~-8% bright, ~-17% faint.
3. Exact if the photometry tables are row-independent (to check bitwise): reuse decode's photometry in
   `log_intensity` instead of recomputing per type. ~-7%.
4. To check bitwise: xgboost `inplace_predict` instead of a new DMatrix per call. ~-2 to -5%.
   Together 1-4: ~1.7x bright, ~2x faint per slot. Next tier (exact, more work): pandas-free likelihood
   adapter, block-aware prior caching, one summary pass for the four co-moving nodes (~1.3x more).
5. Layout, no code: more CPUs per GPU worker (MPS allows 48 clients on V100/A40); e.g. `-c 50` -> 25 physical
   cores, 48 slots, ~1.5x per GPU if the GPU keeps up; 6 x 50 = 300 CPUs, inside the 500 cap; depends on
   free cores on the GPU nodes.
Not proposed: CUDA graphs / other density batches (rejected 2026-10-17, shift log K 3-5e-4; GPU share is small
anyway), fewer levels or shorter chains (science settings; w8l5 failed Test 1).

## 2026-11-09 — Batch-size test; bigger sampler batches for future GalSBI image runs

Owner: "run the batch-size test during c2", then "let's do batch-size test first
and use cip", then "use bigger batches for future runs, update the scripts".

Why: the 200k run's c1 took about 12 h on 6 GPUs. Per galaxy per slot, bright
units ran at 77-89 s and faint units at 20 s, against 24 s and 5.1 s in the
27 Oct 100k run, which used 609-/1022-galaxy units instead of 40/140.

Test (`$DATA_DIR/sbsi/catalogue_prior/galsbi_image_200k_20261109/batch_test/`,
job 17853361, 56 min, 0 of 50 sampler runs failed):
- Same code, model, reference and recipe as c1; chains 1/8 length (2560 steps,
  1024 warm-up); 10 slots on a cip A40-16Q slice with MPS (inter GPUs were full).
- Speed at full chain length (± = standard error over the 10 slots):

  | galaxies per process | s per galaxy per slot | galaxies per GPU-hour |
  |---|---|---|
  | bright 40 (current) | 42.2 ± 0.2 | 853 ± 3 |
  | bright 160 | 22.4 ± 0.2 | 1606 ± 12 |
  | bright 640 | 16.6 ± 0.2 | 2163 ± 20 |
  | faint 140 (current) | 11.4 ± 0.1 | 3156 ± 25 |
  | faint 1022 | 3.1 ± 0.0 | 11669 ± 56 |

- A sampler step costs about 60-80 ms whatever the batch size: 40 → 640 bright
  galaxies (16x) makes a step only 6.3x slower (82 → 520 ms). GPU busy 28-37%,
  GPU memory under 4 GB for 10 processes.
- Bigger units are 2.5x faster for bright and 3.7x for faint, which explains
  most of the slowdown against 27 Oct.
- Step-size adaptation is per galaxy and chain (`log_scales` rows in
  `run_galsbi_inference.py`). Batch size changes only which random numbers each
  galaxy gets and the batched-evaluation rounding.
- Limitation: 10 slots on a GPU slice, not 30 on a full A40, so the gain in
  production may differ.

Template: new `scripts/galsbi_image_run/` (study.py, worker.sh, pipeline.sh,
launch.sh), from the 200k run's scripts.
- Unit size = min(cap, ceil(galaxies / (slots - 2))), with slots = 30 x stage
  workers, so each slot gets about one unit. The caps live in `rule.json`
  `unit_size_caps` (bright 640, faint 1022, bright21 640).
- Seeds, sample fraction, cases, worker counts and job-name prefix are read
  from `rule.json`. The run directory is set once with `sed @RUN_DIR@`, and
  `study.py` uses its own location.
- The estimator is unchanged: its bootstrap resamples simulation cases, not
  sampler units.

Validation:
- `bash -n` on the shell scripts and `py_compile` on `study.py`.
- Dry run on the 200k reference (read-only):
  - step-1 bright: 255 per unit (178 units); faint: 879 per unit (179 units);
    step 2: 46 per unit (176 units), for 180 slots;
  - all 201,828 step-1 and 8,021 step-2 galaxies in exactly one unit.
- A fake `sbatch` shows `launch.sh` would submit the same jobs, seeds,
  fraction, case range and dependencies as the 200k launch.
- Not yet run end-to-end with real jobs.

Limitation: with one unit per slot, a stage takes two rounds when fewer
workers start than planned. Today 4 of 5 c2 workers started and the fifth was
queued until 18:50.

200k run operations, same day:
- estimate-c1 17848354 was cancelled and resubmitted as 17849980 (5 c2
  workers plus the batch test).
- The batch test moved from inter to cip (17853258 cancelled, 17853361 run).
- c2 workers 17853252-56 were held, then released by the owner at about 14:36.
- The 200k run itself keeps its recorded unit sizes (40/140/20).

## 2026-11-09 — GalSBI image run at 100k galaxies per leg (launched)

Owner: "go ahead with the 200k. shall we do one leg only?" Answer: two paired
legs of 100k each (200k in total, the same cost as one 200k leg).
- Scaled from the pilot, two paired legs give m1 ±1.6%. One 200k leg would
  give m1 ±3.9%, with m1 not separable from c1.

Run: `$DATA_DIR/sbsi/catalogue_prior/galsbi_image_200k_20261109`
(`rule.md`, `rule.json`, `launch.sh`, `jobs.txt`).
- Code snapshot 75cc2e5 is identical to the magnitude-coordinate rerun, with
  the m_model coordinate. The recipe, cuts, unit sizes and estimator are
  unchanged.
- New seeds: identity draw Bernoulli(5.2e-3) with seed 20261109 over cases
  1000-1139, c1 2029100001/2034100001, c2 2029200001/2034200001, selection
  base 2027408001, bootstrap 20261109.
- New selection normalization with 8 replicas (the pilot had 4). Its receipts
  are tied to the new reference.
- No GPU preflight, because the code is the same as the completed rerun.
- Workers have a 24 h limit. Step 2 uses 6 workers.

Reference (job 17848346, 4 min):
- Selected 100,888 plus and 100,940 minus galaxies: 98,870 on both legs,
  2,018 plus-only and 2,070 minus-only.
- 2253 c1 units.
- Unmatched rows dropped across all 140 cases (`reference/selection_report.json`):
  - plus: 193,917 shapes without a crossmatch, 5,255 unusable;
  - minus: 197,120 shapes without a crossmatch, 5,376 unusable;
  - no crossmatch rows without shapes and no matches without truth.
- The `cluster` queue had about 7,800 jobs pending. The reference job was
  allowed onto `inter` as well, where it ran on a CPU node (met-cl-vis02).

Jobs:
- selection array 17848347 (8 tasks, 2 at a time, inter A40);
- c1 workers 17848348-51, with 17848352-53 after the selection;
- estimate-c1 17848354, which submits 6 c2 workers and the finish job.
At 01:30, 3 workers had 90 units sampling with no errors.

Expectation from earlier evidence:
- Forward response of like-1.0-galsbi on 4 of these scenes: +0.39% ±0.40%.
- Mock inference (V3.9, older recipe): -0.5% ±0.8%.
- Together: m1 ≈ -0.1% ±0.9%, plus ±1.6% statistical error.
- Untested: the mirror symmetry of like-1.0-galsbi in c2, and the effect of
  poor bright-galaxy (< 21) mixing.

Expected finish: late on 2026-11-09 (c1 ~9 h, c2 ~3 h).

## 2026-11-09 — GalSBI image pilot rerun with the model-band coordinate

Run: `$DATA_DIR/sbsi/catalogue_prior/galsbi_image_magcoord_20261108`
(`rule.md`, `paired_estimate.json`; source snapshot 75cc2e5 = code of
aaa48bc). It reuses the pilot's galaxies, cuts, unit list and selection
normalization through links, with fresh sampler seeds. `study.py` refuses
chains run with another magnitude coordinate. Five c1 GPU workers ran from
21:51 to 00:44 (2 h 53 min). The sixth worker was cancelled while still
queued because it had no units left to claim. All 219 c1 and 42 c2 units
completed with no failures, retries or tracebacks. Matched rows are the same
as in the pilot: 9422 on both legs, 181 plus-only, 216 minus-only.

| | pilot (m_r coordinate) | rerun (m_model coordinate) |
|---|---|---|
| plus g1 (truth +0.02) | +0.02296 ±0.00356 | +0.02289 ±0.00353 |
| plus g2 | +0.00133 ±0.00357 | +0.00014 ±0.00360 |
| minus g1 (truth -0.02) | -0.01576 ±0.00354 | -0.01569 ±0.00355 |
| minus g2 | +0.00204 ±0.00361 | +0.00217 ±0.00363 |
| m1 | -3.20% ±5.19% (-0.6σ) | -3.54% ±5.09% (-0.7σ) |
| c1 | +0.00360 ±0.00336 (+1.1σ) | +0.00360 ±0.00336 (+1.1σ) |
| c2 | +0.00168 ±0.00347 (+0.5σ) | +0.00116 ±0.00347 (+0.3σ) |
| g1-to-g2 response | -0.018 ±0.052 | -0.051 ±0.051 (-1.0σ) |
| chain floor on m1 | ±1.76% | ±1.52% |
| log z R-hat >= 1.1 (c1) | 68.5% ±0.3% | 7.9% ±0.2% |

The two runs share the galaxies and the noise, so most of the ± cancels in
their difference. The difference is set mainly by the chains: m1 moved
-0.34% ±2.3% and c2 -0.00053 ±0.00070, where the ± combines the two chain
floors in quadrature. Nothing moved beyond chain noise.

Stuck fraction by magnitude, log z R-hat >= 1.1 (c1, pilot to rerun):

| Magnitude | Galaxies | Pilot | Rerun |
|---|---|---|---|
| < 21 | 820 | 91% | 42% |
| 21–23 | 3603 | 60% | 5% |
| 23–24 | 5245 | 92% | 10% |
| 24–25 | 9573 | 57% | 5% |

Median log-z ESS rose from 10–30 to 62–502. The shear-ratio g1 flags fell from
22% to 13% (< 21) and from 11% to 4% (23–24). Bright (< 21) mixing in shape,
size and crowding is unchanged, with e1/Re/crowding flags of 69%/77%/63% in
both runs. Step 2 (mag < 21) still has 46% of chains stuck in log z.

Limitations:
- The pilot is too small for this test: the m1 ± (5%) is shape noise at
  10k galaxies per leg.
- This is development data with known shear, not an acceptance test.
- The c1 offset of +1.1σ is the same in both runs, so it comes from the
  shared galaxies and noise, not from the sampler.

Next: scale up (for example 100k per leg) to bring the m1 ± near 1.5%.
Bright-galaxy shape and size mixing is still the weakest part of the
sampler.

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
