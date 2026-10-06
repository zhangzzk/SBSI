## 2026-09-20 — Diagnose independent-scene slowdown and prepare unchanged-worker recovery

Read-only preflight16602263 completes69s, verifies frozen pins and all33
currently completed measurement receipts/output hashes, with empty stderr.
Recovery16602287 is queued afterany of full generation16593294. Preparation
16594999 now depends on successful recovery16602287; binding/evaluation/
report dependencies remain intact. `jobs.json` records the change and both
snapshot handles. No incomplete scene is omitted from the validation.

The old BlendEMU log `fs2_lsst_r_sim.14387750.th-cl-naples04.out` records
100 cases at three shear settings rendered in11731s (May29). Log
`fs2_g002_full.14814317.out` records200 one-shear cases rendered and measured
in18732s (June26; rendering6774s, shape measurement11882s). Both use100 MPI
ranks. The current frozen multi-axis generator instead renders each scene's
legs sequentially with `--threads 1`, at array throttle8; eight CPUs per task
are used for the later fitting stage. Thus its rendering concurrency is at
most eight, not64, and the larger per-scene leg count alone does not explain
the slower wall time. These logs are useful operational comparisons, not a
controlled benchmark with identical measurement conventions.

At10:25CEST,32 of40 independent scenes have completed and eight are running.
Case22000 on th-cl-rome01n1 has run13h11m, with nine-image fitting running
about7h43m. Worker samples show roughly14% CPU utilization and repeated
Lustre/page waits; one worker has read about2.77TB against an11.9GB image
set. Node NUMA0 has very little file cache despite substantial free memory
on NUMA1. This supports an I/O bottleneck; its exact filesystem/NUMA cause
is not established. The other nine-leg cases completed in about3h42m–4h04m.
A requested extension beyond the16-hour allocation was denied by Slurm.

Cache-local `warm_scene22000_cache.py` reads/hashes all nine images in48s
and checks unchanged size/mtime/inode. A read-only mmap/mlock cache keeper
reports successful setup but its actual mappings show zero resident/locked
image pages. Neither attempt improves measured worker throughput; the
keeper step16593295.9 is terminated. Receipts and before/after samples are
under `independent_joint_validation_20260919_v1`. A read-only GDB inspection
of the parent stalled loading libraries and was terminated; it detached,
then the parent was explicitly resumed with SIGCONT. A subsequent process
check confirms sleeping/futex state and TracerPid0, with workers still
running. No scientific outputs or fitting state were edited.

Added BlendEMU `scripts/resume_independent_measurements.py` and
`jobs/job_resume_independent_measurements.sh`, with separate frozen
`recovery_protocol.json`. They verify existing source/model pins, wait for
the entire original generation array to be terminal, retain completed
measurements, and invoke the exact original eight-worker measurement module
only for incomplete scenes with completed matching render audits. Partial
output files cause failure for explicit review; nothing is deleted or
regenerated. The fallback excludes the pathological node and records every
retained/retried case. The existing downstream receipt/data checks remain
unchanged. Python syntax, shell syntax and Ruff pass; scheduled read-only
preflight16602263 checks frozen pins and completed measurement hashes.

Updated `jobs/job_check_independent_joint.sh` to include cache diagnostic
logs; shell syntax passes. New one-shot snapshot16602111 is scheduled about
12:16CEST. It records status but cannot wake the task. Pending numerical
validation still requires all40 predeclared scenes; no independent bias
result or acceptance claim is available. Inspect recovery and each dependent
preparation/evaluation stage when it starts. A future rendering launcher
should batch independent fixed-seed scenes/legs efficiently, with explicit
seed/byte-equivalence validation, rather than carrying this serial pilot
execution layout into a larger production run.

## 2026-09-19 — First new scene passes the preparation receipt contract

CPU16595031 completes27s with empty stderr. It runs the exact pinned
preparer's `require_declared_scene` and `validate_receipts` on completed
case22004: all five declared shear legs, scene/noise seeds and linked
generation/render/measurement receipt hashes pass. The check retains the31
caught measurement KeyErrors in its receipt. It reads no catalogue values or
calibration outcomes and makes no scene-selection decision.

Cache-local `check_first_scene_receipts.py` and
`first_scene_receipt_check.json` preserve the check and source/receipt hashes
under `independent_joint_validation_20260919_v1`. This verifies compatibility
with one actual newly generated scene; it does not substitute for the queued
full40 preparation and data binding. Scientific workers, frozen protocols,
candidate models, endpoints and downstream job dependencies remain unchanged.

## 2026-09-19 — Queue the complete independent validation chain

Preflight16594998 completes10s with all79 pins verified and both historical
g1/g2 result schemas accepted. This is a compatibility check only. Earlier
preflight16594995 failed on invoking a cache-local script outside the package
module path;16594997 then passed the two compatible schemas but attempted an
unrelated historical `parent_v2` layout. The corrected cache-local
`preflight_v2.py` runs through `runpy` from the repository and limits this
smoke to the two matching layouts. No scientific worker, frozen protocol or
result was changed to pass the smoke.

Submitted CPU preparation16594999 after all generation16593294 and passing
preflight16594998; CPU binding16595000 follows preparation; GPU array16595003
follows binding at throttle2; CPU report16595004 follows the entire response
array. Slurm had already removed completed emulator report16592966 from its
dependency registry, so an attempted GPU submission with that dependency was
rejected before creating the array. Its COMPLETED state was reverified in
accounting, with no other active/pending SBSI GPU jobs, before submitting the
new array with the still-live binding dependency. Thus the global two-GPU
limit remains satisfied. `jobs.json` records the exact handles and reason.

Added `jobs/job_check_independent_joint.sh`; shell syntax passes. One-shot
snapshot16595007 is scheduled in two hours and captures every generation and
downstream stage. It records logs but cannot wake the Codex task; no callable
wake-up tool is available. Generation is progressing normally, including a
completed five-leg measurement receipt for case22004. Its31 caught per-object
KeyErrors remain in the measurement receipt with invalid shape sentinels and
are not silently recast as successful fits. Full-data preparation, binding,
response and density reporting remain pending; inspect each dependent stage
when it starts. No independent calibration result is claimed yet.

## 2026-09-19 — Complete emulator repeat and freeze independent evaluation

Emulator response array16592825, density array16592853 and combined report
16592966 all complete successfully. The repeated emulator gives primary joint
m=-0.166180 +/-0.263913% on main80 and +0.437508 +/-0.188822% on the existing
40-scene endpoint. Paired changes from the retained original are +0.033799
+/-0.001339 and +0.035034 +/-0.001790 percentage points. Four-scene g1/g2
changes are +0.030995 and +0.030838pp. All eight primary normalized density
score changes have magnitude below0.000056nat/object. These are one-factor
repeats, not an estimate of the training-seed distribution or a model choice.
The completed five-variant report retains every original endpoint and arm.

Added `scripts/validate_independent_joint.py` and
`scripts/report_independent_joint.py` with bind, GPU evaluation and CPU report
launchers. The binder checks complete generation/measurement receipts, exact
scene and noise seeds, all declared shear legs, and parent/pair/geometry
identity before hashing the prepared products. Evaluation executes the same
unchanged source bytes as the development checks, with64 shared antithetic
draws, seed7301 and every diagnostic arm/cut retained. The original single
flow/classifier/emulator remain fixed. The report includes both primary axes,
both auxiliary endpoints and absolute predictive discrepancies without
automatic acceptance or outcome-dependent choices.

Frozen `independent_joint_validation_20260919_v1/preparation_protocol.json`
and `joint_template.json` contain67 and79 source/model/protocol pins. Targeted
CPU job16594969 passes5 design/parent tests in17.53s; Ruff and launcher shell
syntax pass. Read-only preflight16594995 is submitted to verify all pins and
the new result checker against historical result schemas. This schema check
does not relabel historical results as independent evidence. Full preparation,
binding and independent numerical evaluation still await the complete scene
set. Generation16593294 has completed its first case,22004, and proceeds to
case22008 under the unchanged CPU throttle. See the
[independent validation design](JOINT_CALIBRATION.md#prospective-independent-two-axis-validation).

## 2026-09-19 — Independent two-axis scenes launched; cross terms remain secondary

Frozen generation protocol
`/project/ls-gruen/users/zekang.zhang/blendemu_runs/joint_independent_validation_20260919_v1/protocol.json`
declares cases22000–22039, both shear axes at +/-0.05, and an auxiliary
+/-0.02 check on the first four cases. It pins the original single flow,
single smooth classifier and disk emulator before generation, retaining every
declared case without outcome-dependent selection. The primary population
uses measured FLUX_RADIUS>0.6 arcsec and MAG_AUTO<25.8, with predicted
detection/usability. Cross-response components remain secondary to m1/m2.

BlendEMU launcher `jobs/job_joint_independent_validation.sh` submitted CPU
array16593294, throttle8. A status check finds tasks0–7 running and8–39 waiting
on the array limit. Existing emulator repeat response tasks16592825_2/_3
occupy the two GPUs; density16592853 and report16592966 remain dependent.
Existing simulations continue to supply the reproducibility checks. The new
scenes address untouched validation and the limited four-scene g2 evidence;
the bounded archive inventory above found no unused compatible sample.

Draft SBSI files `scripts/prepare_independent_joint_validation.py`,
`tests/test_independent_joint_validation.py`, and
`jobs/job_prepare_independent_joint_validation.sh` prepare the declared scene
set and check its design contract. Preparation is not submitted or validated
end to end; downstream binding/evaluation remains unfinished. The current
goal state is paused. This checkpoint records existing work and submits no
additional jobs. Validation here is read-only protocol and Slurm inspection,
not a new calibration result.

## 2026-09-19 — Emulator repeat trains and passes fixed-recipe binding

CPU16592759 completes14m44s with218 trees, all128,809,729 fit rows and
unchanged intercept0.00811731707072. The repeated model SHA256 is
`1a20393ef749e90d483ee0f7743b14e7e2611551c5a68a79fc45e3f9bd531baf`,
distinct from the original. Final training physical-moment RMSE is0.997070
versus original0.997075; this is a training diagnostic, not joint calibration
or a reason to select the repeat. Source checks pass before and after fitting.

Binder16592811 completes59s, including1 passing recipe test in19.84s.
Every fixed metadata/parameter field agrees, the prescribed seed is the only
recipe change, and model hashes match completed metadata. All error logs
are empty. The frozen129-pin joint protocol releases fresh_g1/fresh_g2
tasks16592825_0/_1; main80/independent40 wait under throttle2. Joint response
and physical-density evidence remain pending. See
[trained emulator repeat](JOINT_CALIBRATION.md#emulator-repeat-trained-and-frozen).

## 2026-09-19 — Queue the complete five-variant reproducibility report

Added `scripts/report_emulator_training_repeat.py` and its CPU launcher.
Job16592966 waits for the entire emulator-density array16592853. It extends
the completed flow-repeat evidence, retaining every previous branch/paired
change/density score, and appends the emulator repeat at all four endpoints.
It verifies the frozen model/protocol/receipt identities and all42 density
arm/cut/leg combinations before writing the primary tables and complete JSON.
The source explicitly notes one-factor-at-a-time repeats, unchanged original
candidate, reused development scenes and untested full flow pretraining.

Ruff and shell syntax pass; report runtime remains pending. The new table's
column counts are checked during source review. Emulator CPU16592759 reaches
tree40 with128,809,729 fitting rows, fixed intercept0.00811731707072 and no
logged errors. No additional model fitting or GPU job is introduced by the
report. See [emulator report](JOINT_CALIBRATION.md#complete-emulator-repeat-evidence-report).

## 2026-09-19 — Freeze one emulator-training seed repeat

Completion review finds an untested stochastic component: the disk-response
emulator's fixed trial9 XGBoost recipe uses row subsampling0.763454 and
feature subsampling0.965194. Added BlendEMU
`scripts/train_disk_response_moment_repeat.py`, a source-checked copy changing
only seed20260816 to20260817, its protocol checker and32-CPU launcher.
CPU16592759 reuses all existing161,012,399 prepared pairs, the same fit/dev
split, physical-moment objective and218 trees. Nine source/data/model pins
and exact one-replacement source equality pass before training. No development
or joint response chooses a tree count or a preferred seed.

Added SBSI `scripts/validate_emulator_training_repeat.py`, fixed-recipe test,
freeze/evaluation launchers and127-pin joint template. Binder16592811 waits
for training; two-GPU array16592825 waits for the binder and retains all four
existing endpoints, all classifier arms and all cuts, with original flow and
classifier fixed. It checks exact selected-mass equality and unchanged model,
data and evaluator inputs before paired response comparison. CPU16592812
passes the recipe test in13.25s with empty stderr.

Added `scripts/audit_emulator_repeat_density.py`, cancellation-contract test
and density launcher. It evaluates each emulator's own inverse-transport
physical density using the same flow; radius/flux cuts and classifier-weighted
selection masses must agree exactly. Density array16592853 waits for the whole
response array, throttle2. CPU16592854 tests its contract and existing score
normalization:4 tests pass6.89s with empty stderr. Ruff and shell syntax pass;
full-data validation is pending. At5m45s training has53 CPU-minutes and
4.39GiB peak RSS during data preparation. It subsequently constructs all
128,809,729 fit rows, reproduces the fixed training intercept0.00811731707072
and logs tree0 physical-moment RMSE0.99986, with empty stderr. The completed
model remains pending. New one-shot snapshot16592894 is scheduled in two
hours and includes training, binder and every response/density task log;
it captures status but does not wake the Codex task.
The original model remains retained, not averaged with or replaced by this
repeat. See [emulator seed repeat](JOINT_CALIBRATION.md#fixed-recipe-emulator-training-repeat).

## 2026-09-19 — Flow-refinement response and density repeat complete

All flow-repeat response tasks16591210 complete, including main80 in1h35m20s
and independent40 in49m34s. Primary smooth predicted-U m is-0.198490776
+/-0.263669529% main and+0.404368251 +/-0.188782903% independent.
Paired changes from the retained original are+0.001488611 +/-0.000818pp
and+0.001894245 +/-0.000431pp. The four-scene g1/g2 changes remain
+0.005007011 and+0.001106905pp. No result selects between flows.

Density tasks16591245 all complete (44s,44s,4m08s,2m28s), with empty error
logs. Properly normalized primary predicted-U NLL changes are
main+0.00000604/+0.00000787nat and independent+0.00000453/+0.00000960nat
for plus/minus legs; all eight endpoint/leg changes have magnitude<1e-5nat.
Each density worker runs its four contract/normalization tests successfully.
Combined report16591355 completes3m38s, verifies all provenance and all
declared arms/cuts/endpoints, and writes `flow_refinement_repeat_report`.

These results establish stability to the tested flow-refinement streams,
conditional on the original flow initialization and emulator. Together with
the classifier and latent repeats, they strengthen the approximate calibration
evidence without a required model ensemble. Emulator retraining and an
untouched acceptance sample are not tested by these repeats. No production
configuration or original candidate changes. See
[completed flow repeat](JOINT_CALIBRATION.md#completed-flow-refinement-response-and-density-repeat).

## 2026-09-19 — Recheck available constant-shear scene coverage

Read-only directory inventory while the flow repeat runs confirms the
existing constant-shear store spans140 cases0–139, the g005 store100 cases
140–239, REPRO five cases0–4, and the indomain store50 cases40–89.
The current domain manifest assigns every case0–199 to flow fitting or
validation. Cases200–239 are already the independent40 development endpoint;
the four newly generated scenes20010–20013 have also been used in this
investigation. This corroborates the earlier
[acceptance inventory](JOINT_CALIBRATION.md#fresh-acceptance-data-inventory).

This bounded metadata review finds no newly unused compatible constant-shear
sample among those archives. It is not a claim to have exhausted every
simulation family or to have verified independent image/noise identity from
case numbers alone. No shape rows, response values, fitting, new images or
GPU jobs are introduced. Existing results remain development/reproducibility
evidence; no subset is relabeled as untouched acceptance. Main80 and
independent40 flow-repeat validations remain live.

## 2026-09-19 — Queue combined flow-repeat evidence report

Added `scripts/report_flow_refinement_repeat.py` and
`jobs/job_report_flow_refinement_repeat.sh`. CPU16591355 is chained after
successful completion of every density task16591245. It combines the original,
latent-repeat, classifier-training-repeat and flow-refinement-repeat diagonal
results without averaging, selecting a favorable run or changing the candidate.
It retains all response branches and all42 density arm/cut/leg combinations
per endpoint, checks protocol/receipt/source hashes and scene/shear identities,
and includes the original absolute-density evidence. Primary tables show
paired shifts and properly normalized predicted-U density changes.

Ruff and shell syntax checks pass. End-to-end report validation awaits the
remaining scientific outputs; no report success or new calibration result is
claimed yet. Main80 and independent40 validation tasks16591210_2/_3 remain
live; no extra GPU work, images or fitting are introduced. See
[combined evidence report](JOINT_CALIBRATION.md#combined-flow-refinement-evidence-report).

## 2026-09-19 — First flow-repeat diagonal results and absolute density review

Flow-repeat four-scene g1/g2 tasks16591210_0/_1 complete successfully.
Smooth predicted-U primary m are+0.352175 and-0.442323%, changing only
+0.005007 +/-0.002370pp and+0.001107 +/-0.001917pp from the original.
Original candidate remains fixed; main80 and independent40 tasks start with
first-scene progress and empty stderr. These are conditional refinement
repeats, not independent flow pretraining or final calibration acceptance.

Added `scripts/report_smooth_density_diagnostics.py` and
`jobs/job_report_smooth_density_diagnostics.sh`, submitted CPU16591324.
They use existing original-candidate summaries across every endpoint/leg,
verify result hashes/replay, and retain all-cut moment and support diagnostics.
The figure reports primary mean/variance effect sizes; the table explicitly
labels histogram TV as uncut. Ruff and shell syntax pass. CPU16591324
completes25s with empty stderr; all four receipt/hash checks and histogram
normalizations pass, and PNG is visually inspected with no clipping.
Primary mean residuals are below0.6% of observed SD; e1/e2 variances remain
underpredicted1.32–1.73% and2.41–2.56% across every endpoint/leg.
Uncut coarse histogram TV is3.54–3.92%, including sampling noise.
No new GPU work or simulation; cross-component
investigation remains paused. See
[density review](JOINT_CALIBRATION.md#first-flow-repeat-diagonal-checks-and-absolute-density-review).

## 2026-09-19 — Flow repeat selects a distinct step3 checkpoint within all gates

GPU16591200 completes13m38s. Selected flow SHA256 is
`33fc71c0636b5f8e030fc152319022736d904e9ec4e20e617d50b583e9fbd5db`,
step3 versus original step2. Full validation NLL cost is0.000868040256nat/object
(initial-4.768710489910869, selected-4.767842449654797), within the unchanged
0.001 budget. Soft/hard primary probability residuals are0.0010542616 and
0.0012273332; guard loss falls0.4706957998 to0.3243362633 in the repeat's
integration stream. All fourth-step trials fail and training stops under the
prespecified rule. No step count, budget or checkpoint choice is tuned from
joint results, and the original flow is retained.

Binder16591209 completes30s after successful training and passes all identity
checks. Response tasks16591210_0/_1 start with the exact frozen evaluator
hash and first1024-parent progress, both stderr empty. Normalized density
16591245 remains chained after response validation at two GPUs maximum. Added
[training outcome](JOINT_CALIBRATION.md#flow-refinement-repeat-trained-and-frozen-for-validation).
Training gates are not joint calibration or absolute-density acceptance;
scientific validation of the repeated flow remains pending.

## 2026-09-19 — Full suite passes after repeat and density instrumentation

CPU16591268 completes1m07s with **589 passed,4 skipped,1 existing warning
in57.57s** using the documented py31 `python -m pytest tests -q` with one
thread. The warning is the existing NumPy small-sample variance warning;
regression/snapshot failures are zero. This covers the new response-vector,
spatial-contribution and fixed-flow-density contract tests alongside the
existing library. No test expectation or scientific candidate changes.

Flow repeat16591200 remains live: it accepts a third refinement step under
the fixed constraints, while all fourth-step proposals are rejected. The
original run selected step2. Different stochastic streams therefore produce
a different optimization path; this is not yet a claim about joint bias.
The dependent binder, all-endpoint response validation and selected-density
comparison remain queued. Cross-component investigations remain paused under
the owner's priority guidance.

## 2026-09-19 — Prepare normalized predicted-U density for the flow repeat

Added `scripts/audit_flow_repeat_density.py`, a cancellation-contract test,
and GPU launcher. Array16591245 waits for all flow-repeat response tasks
16591210 and is throttled at2, preserving the GPU limit. All four endpoints,
three cuts, six predicted-U classifiers and actual-U oracle are retained.
The physical inverse-transport density uses the existing Jacobian and CUDA
emulator convention. Fixed classifier factors cancel only in the observed
raw ratio; each model retains its own classifier-weighted selection mass.
Thus changes in pass probability cannot masquerade as selected-density gains.

The six-pin density protocol also pins the122-pin joint template. Completed
joint results/comparison receipts, unchanged non-flow inputs, and measured
counts/shape sums are verified. Nonfinite densities fail rather than dropping
rows. CPU16591247 completes12s;4 contract/normalization tests pass7.30s,
stderr empty. Ruff and shell syntax pass. A mistaken test filename was caught
and corrected before submission; no job ran against that filename.
Full-data density results remain pending. Training16591200 is verified live,
with accepted refinement steps and empty stderr. No original candidate or
canonical catalogue changes. Replaced pending snapshot16591214 with an
updated two-hour snapshot including the density stage.
See [flow-repeat density](JOINT_CALIBRATION.md#predicted-u-density-check-for-the-flow-repeat).

## 2026-09-19 — Completed classifier/latent repeats; freeze flow-refinement repeat

All eight latent/classifier validation endpoints complete successfully.
Main80 original/latent-repeat/classifier-repeat primary m are
-0.199979/-0.209397/-0.134885%; independent40 values are
+0.402474/+0.391591/+0.467089%. Corresponding case SEs remain about0.264
and0.188pp. Large-sample latent shifts are about-0.01pp; classifier training
shifts are+0.065095 +/-0.001101pp and+0.064615 +/-0.001095pp. All endpoints,
cuts and arms remain in `reproducibility_report/evidence.json` and `report.md`.
Replica selected-density improvements remain about-0.0332 to-0.0335nat/object
on main/independent legs, without scene exclusions or seed selection.
Comparison/influence16587400 completes72s, density16587403 completes51s,
and consolidated report16587896 completes27s; all error logs are empty.

The remaining flow-side randomness question motivates one bounded refinement
repeat. Added `refine_joint_guard_probability_flow_repeat.py`, checker and
launcher. Source is identical to the original after adding101 to eight
random-seed constants;29 input/code pins verify this. Same initial checkpoint,
reference integrals, target banks, six-step recipe, learning rates and full-NLL/
probability gates remain. This is conditional refinement reproducibility,
not independent pretraining or emulator retraining. Neither flow is selected
by joint bias or averaged into deployment. Original candidate stays frozen.

CPU smoke16591198 completes40s,8 tests pass12.72s; the new stream selects
smoke step1 and replays. Full GPU16591200 starts, passes identity checks,
and loads full populations with empty stderr. A CPU-partition GPU submission
was rejected before starting; accepted job uses inter/cip and one A40.
Added binder/validator `validate_flow_refinement_repeat.py` and launchers;
122-pin response template16591209->array16591210 runs all four endpoints,
original smooth classifier and all arms/cuts after successful training.
Observed sufficient statistics must match exactly via the existing paired
comparator. Array throttle2 and dependencies preserve the two-GPU limit.
Ruff, shell syntax and source/pin checks pass; downstream response and density
assessment remain pending. A one-shot two-hour scheduler snapshot is queued.
See [completed repeats](JOINT_CALIBRATION.md#completed-sampling-and-classifier-training-reproducibility)
and [flow repeat](JOINT_CALIBRATION.md#fixed-recipe-flow-refinement-repeat).

## 2026-09-19 — First classifier-retraining endpoint and consolidated report

Added `scripts/report_smooth_reproducibility.py` and CPU launcher16587896,
dependent on16587387,16587400 and16587403. The report verifies original
candidate fingerprints, repeat comparison receipts, unchanged retained-arm
statistics and density/influence result identities. It retains every endpoint,
cut and arm in JSON, and displays the fixed-primary-cut original, latent-repeat
and classifier-training-repeat m values separately in Markdown. It does not
pool different amplitudes, select a seed, average deployment models or
assert calibration. Ruff and shell syntax pass; full-data execution remains
pending with the expected dependencies. Output: scene-classifier cache
`reproducibility_report/`.

Replica16587398_0 completes5m22s: fresh-g1 primary smooth
m=+0.410181 +/-0.810691%, versus original+0.347168 +/-0.817220%.
Retained five-arm sufficient-statistic replay is exactly0.0. The point shift
is+0.063013pp; paired uncertainty remains for the queued comparison job.
No favorable seed selection follows this small four-scene result. Replica
16587398_1 starts; main80 latent repeat remains live. Added
[consolidated evidence reporting](JOINT_CALIBRATION.md#consolidated-reproducibility-evidence-report).

## 2026-09-19 — Restore replica throughput automatically after the latent repeat

Revalidated main80 repeat16587387_2 and replica16587398_0 as live with
meaningful object progress. Added `jobs/job_restore_smooth_replica_throttle.sh`
and submitted CPU16587872 with afterok dependency on the full latent array
16587387. It restores replica16587398 throttle2 only after checking that no
other running/configuring GPU job owned by the user remains. If another GPU
job is present, it fails without changing the throttle. Shell syntax passes;
current squeue formatting correctly identifies the two live GPU array IDs.
The helper remains pending on the expected dependency.

This scheduling-only change prioritizes diagonal/density reproducibility and
avoids leaving the second GPU idle after the remaining repeat completes.
No scientific source, checkpoint, data, cut or acceptance criterion changes.
Next inspect replica endpoint receipts as they finish; the scheduled snapshot
16587409 remains queued. Cross-component work remains lower priority under
the owner's latest guidance.

## 2026-09-19 — Prioritize diagonal reproducibility; start replica on idle GPU

Owner clarifies that cross-components have less priority. Updated the goal
section and paused further cross-component development. Diagonal joint m,
density and sampling/training reproducibility are the immediate priorities;
existing cross diagnostics remain caveats, with no correction or exclusions.

Validated binder16587397 completed successfully and replica protocol has no
pins on unfinished latent-repeat outputs. Changed array16587398 throttle
from2 to1, then cleared its resource-ordering dependency so one replica can
overlap the remaining main80 latent repeat16587387_2. Reattaching the already
completed binder was rejected by Slurm as a dependency problem, so its success
was verified via sacct before clearing the dependency. Scientific protocols
are unchanged. Restore throttle2 only after the other GPU repeat completes.

Immediately preceding this clarification, CPU16587840 completed60s with
empty stderr. `scripts/audit_cross_tail_objects.py` verifies receipts and
replays saved shapes, contribution means and largest-object IDs. In the
largest1% cross-derivative group,81–86% have opposing endpoint shape vectors;
median centroid shifts are only0.023–0.029 pixels and all fit flags are zero.
Large centroid jumps do not dominate these tails; optimizer failure is not
proved. Ruff and shell syntax pass. Saved `cross_tail_objects.json` and
[tail-object interpretation](JOINT_CALIBRATION.md#tail-object-inspection-and-owner-priority).

A bounded CPU same-image optimizer probe16587861 had already started before
the clarification. New BlendEMU script/job `probe_cross_tail_optimizer.py`
reuses the frozen worker with three declared common optimizer seeds, requires
original-stream replay, retains ranked tail objects and40 randomly selected
parent controls, and changes no canonical catalogue. Its input/code protocol
is `cross_optimizer_protocol.json`. No additional cross investigation is
scheduled; the main replica validation is the current priority. Replica16587398_0
starts with the frozen evaluator hash and meaningful first-case progress,
stderr empty. Probe16587861 subsequently writes `cross_optimizer_probe.json`
with all80 objects and original-stream replay passing; no follow-up analysis
is promoted ahead of the diagonal/density validation.

## 2026-09-19 — Spatial audit shows noisy, heavy-tailed cross derivatives

Added `scripts/audit_measured_cross_contributions.py`, a weighted-block
jackknife test, and CPU launcher. Job16587813 completes1m09s with one test
passing20.43s and empty stderr; Ruff and shell syntax pass. Input hashes and
canonical per-leg observed sufficient statistics replay before diagnostics.
Saved `scene_classifier_geometry_20260918_v1/measured_cross_contributions.json`,
retaining all scenes/cuts/axes, fixed8x8 spatial cells and ranked-tail
contributions to the full mean. No objects, cells or amplitudes are excluded.

On the same78,404 case20010 all-eight-leg selected objects, the R21 amplitude
change is +0.008024837 +/-0.00779344 under a paired spatial-block jackknife;
R12 changes -0.002595862 +/-0.00707766. These within-scene conditional-mask
errors do not replace the original scene errors or define acceptance. They
weaken the interpretation of amplitude dependence as a demonstrated systematic
effect. Individual cross derivatives reach roughly -47 to +49. The largest1%
by absolute derivative supply substantial signed contributions in cases20012
and20013; no single spatial block explains their negative R21. All extreme
measurements remain in the estimand. Source inspection confirms per-object
optimizer RNG reset in every leg, but does not identify the origin of the
large shape changes. Next inspect paired centroids, shapes and fit flags.

Added [spatial/object interpretation](JOINT_CALIBRATION.md#spatial-and-object-contributions-to-measured-cross-response).
Main80 latent-repeat16587387_2 remains live; replica validation dependencies
and two-GPU limit remain intact. The cross discrepancy is not declared solved,
and no model or calibration claim changes.

## 2026-09-19 — Existing amplitudes show unstable measured cross response

Added `scripts/audit_existing_cross_amplitude.py` and CPU launcher. Job16587781
completes22s with empty stderr; Ruff and shell syntax pass. The audit verifies
parent/crossmatch/measurement hashes and replays h=0.02 observed sufficient
statistics for both axes, all cuts, actual-U and matched-U. These runtime
checks cover the canonical data path; no simulation, measurement refit,
model change or favorable amplitude selection occurs.

On the same78,404 case20010 objects selected in all eight existing nonzero
legs, measured R21 changes from-0.007809566 at h=0.02 to+0.000215271 at
h=0.04, whereas R11 changes by only+0.000046961. Uncut common-usable R21
also changes from-0.005093295 to+0.001729898. All masks/cuts/axes are saved
in `scene_classifier_geometry_20260918_v1/existing_cross_amplitude.json`.
Bare input cross Jacobians are small and separately labeled; they are not
measured responsivity. One scene cannot separate finite-step, noisy/nonsmooth
measurement and missing-model effects, but fitting a fixed cross correction
from the h=0.02 point would be unsupported. Next inspect spatial/object
contributions and measurement/noise provenance on these existing images.
See [amplitude audit](JOINT_CALIBRATION.md#existing-image-cross-response-amplitude-audit).

Independent40 latent repeat16587387_3 completes48m23s: smooth primary
m=+0.391591 +/-0.187383%, seed change-0.010883 +/-0.005890pp; actual-U
m=+0.469058 +/-0.182518%. Density collector16587401 completes18m13s,
stderr empty, all four endpoints retained. Main80 repeat16587387_2 remains
live; replica GPU and final density stages retain their dependencies and
the two-GPU limit. No candidate promotion or calibration claim follows.

## 2026-09-19 — Cross-component discrepancy survives scene and latent checks

Added `scripts/audit_smooth_cross_response.py` and its CPU launcher, reusing
the tested response-vector helper. CPU16587535 completes40s, stderr empty;
Ruff and shell syntax pass. It verifies all original receipts and both
completed seed7302 receipts, exact observed sufficient statistics and frozen
latent-repeat provenance. All scenes/cuts/arms remain in the report.

Primary smooth leave-one-scene-out cross leakage ranges are
[-1.636382,-1.054891]% for R21 and [-1.041605,-0.569279]% for R12.
Seed7302 changes the full-sample values by only +0.000153 +/-0.011507pp
and +0.009916 +/-0.019688pp. Uncut and matched-usable branches also retain
the discrepancy: uncut matched R21=-1.322374%, R12=-0.560332%.
This excludes neither a measurement issue nor missing model physics, but
neither one scene, these two latent streams, photometric cuts nor usability
turnover alone explains the finding. No favorable stream or scene is selected.

Saved `scene_classifier_geometry_20260918_v1/smooth_cross_response_audit.json`
and extended [response-vector validation](JOINT_CALIBRATION.md#smooth-classifier-response-vector-validation).
The four image manifests use distinct scene/noise seeds. Their identical
noise-metadata hashes are settings hashes, not proof of shared pixel noise.
Existing case20010 includes h=0.04 on both axes; next inspect measured
amplitude and matched-object responses against h=0.02, without new images.
Large-sample latent repeats and classifier replica density remain live;
classifier replica joint evaluation still depends on their GPU predecessors.

## 2026-09-19 — Full response vectors expose a remaining four-scene discrepancy

Previous turn was a verified wait; the repeated live-handle check alone did
not change the candidate evidence. Reviewed acceptance scope and added
`scripts/summarize_smooth_response_vectors.py`, its paired unequal-leg-mass
unit test, and `jobs/job_smooth_response_vectors.sh`. CPU16587517 completes35s,
one test passes13.37s, stderr empty; Ruff and shell syntax pass. The reporter
verifies result receipts, reproduces diagonal m and bootstrap SE, and retains
all existing cohorts/cuts/arms with both response components. No pinned
running evaluator, model, simulation, or selection rule changes.

Primary smooth cross residuals, normalized by the corresponding model
diagonal, are +0.049824 +/-0.294751% main80 and -0.199118 +/-0.184632%
independent40 (R21). On the four additional scenes they are
-1.387767 +/-0.346398% (R21) and -0.788142 +/-0.262163% (R12).
Observed-U oracle values remain -1.488643 +/-0.304932% and
-0.839496 +/-0.298054%; the discrepancy is not removed by predictive usability.
These are cross leakages, not multiplicative m. The small scene sample limits
bootstrap precision but does not license dropping this adverse evidence.

Saved `scene_classifier_geometry_20260918_v1/smooth_response_vectors.json`;
added [response-vector validation](JOINT_CALIBRATION.md#smooth-classifier-response-vector-validation)
and linked it from the headline. Next inspect case influence and existing
second-latent-stream cross responses, then localize measurement/selection,
scene anisotropy and flow/emulator contributions. The frozen training and
latent repeats continue; the full calibration goal remains unproven.

## 2026-09-19 — Independent classifier training and both freeze stages complete

CPU16587392 completes19m06s with all100 declared epochs. Seed20260914 selects
epoch98 by minimum tune mean-case BCE0.11873550994501654, versus original
seed20260913 epoch99/BCE0.11856610059211203. The5377-parameter replica hash is
`c917266eb507eb5abd00a7fbdc536d59425b9a995bc2e7d9e636ce54e8b932eb`.
The selected model replays its tuning loss; `response_read=false` and
`production_accepted=false` remain recorded. Similar probability loss is
not yet evidence of equal joint response, and neither seed is selected by bias.

Joint binder16587397 completes35s with two contract tests13.25s and freezes
`smooth_replica_joint_protocol.json`; density binder16587399 completes19s
with two tests6.71s and freezes `smooth_replica_density_protocol.json`.
All hashes, declared seed,100-epoch history and minimum-loss selection checks
pass. Density collector16587401 starts and passes three tests6.88s with empty
stderr. It reuses existing observations and will report all six arms. Replica
GPU array16587398 remains correctly dependent on completion of the live
latent-repeat array16587387, preserving two GPUs total. The original frozen
candidate and its manifest are unchanged; no new model is promoted.

## 2026-09-19 — Save an all-arm four-endpoint comparison figure

Added `scripts/plot_smooth_joint_comparison.py` and
`jobs/job_plot_smooth_joint.sh`. CPU16587455 completes41s with empty stderr,
writing PNG, vector PDF and a source/value/hash manifest under
`scene_classifier_geometry_20260918_v1/comparison_figure/`. The plot reads
verified complete result/comparison receipts and displays all six classifier
arms plus the observed-U oracle, all four endpoints, the fixed measured
0.6-arcsec primary cut and±1 case-bootstrap SE. The±0.3% band is explicitly
a reference, not a confidence-test acceptance threshold. Shared four-scene
axes and ongoing repeat limitations are labeled. Ruff and shell syntax pass;
visual inspection confirms readable, unclipped labels/error bars. This is a
reporting-only artifact; no scientific values or model choices change.
Training replica16587392 remains verified live at15m36s; GPU repeats and
dependent validation jobs are unchanged.

## 2026-09-19 — Consolidate the mechanism diagnosis and historical scope

Added [mechanism summary and V3 scope](JOINT_CALIBRATION.md#mechanism-summary-and-v3-scope)
near the start of `doc/JOINT_CALIBRATION.md`. It separates the established
new-pipeline findings—population/response definitions, shared-row guard
variance, disk support and physical-response training, and the dominant
predicted-U blend-weighting discrepancy—from the historical V3 headline.
The current classifier changes hold one flow/emulator fixed and give about
+0.94pp on both large samples, with improved probability scores; this does
not isolate the causal contribution of V3's ensemble versus narrow support.
No retrospective numerical attribution or same-population density comparison
is invented. Existing section anchors and recorded numbers substantiate the
summary; documentation-only change requires no new test run. Revalidated
live latent-array16587387 and training16587392 handles; their model/data and
queued validations are unchanged. This continuation adds a scientific
synthesis, while acceptance still waits for the frozen repeats.

## 2026-09-19 — Full test suite passes with smooth and reproducibility code

Previous continuation was a verified wait on the live latent-repeat and
classifier-retraining jobs; the next status-only continuation added no new
evidence. Revalidated their live Slurm handles before taking the independent
software-validation step. CPU16587440 runs the documented py31 command
`python -m pytest tests -q` with one thread and completes53s: **586 passed,
4 skipped, 1 warning in46.07s**, stderr empty. The warning is the existing
small-sample NumPy variance warning in the corner-extrapolation test. No
scientific checkpoint, sampling stream, test expectation or acceptance
criterion changes. This verifies the new smooth features, six-arm validation,
density/influence and training-replica code alongside the existing library;
it does not itself establish calibration. The scientific repeat chain and
scheduled snapshot16587409 remain intact.

## 2026-09-19 — Two-axis latent repeat completes on the four additional scenes

Array16587387 tasks0/1 complete both previously generated four-scene endpoints.
The strict existing latent-repeat comparator verifies identical model/data/
evaluator provenance and exactly matched observed sufficient statistics.
At the primary measured>0.6-arcsec cut, smooth predicted-U seed7302 gives
g1m=+0.289840±0.824666% and g2m=-0.425409±0.777288%. Changes versus fixed
seed7301 are-0.057327±0.014104pp and+0.018021±0.021073pp respectively.
Both seeds and both axes are retained; no seed averaging or favorable-stream
selection is performed. These changes are smaller than0.3pp but do not
bound finite-draw bias or replace the larger-sample repeats. Main/independent
tasks2/3 have now started; classifier training replica16587392 remains live.

## 2026-09-19 — Frozen single-candidate manifest and repeat startup

Created `scene_classifier_geometry_20260918_v1/smooth_candidate_manifest.json`
with the single flow, single required smooth classifier, disk emulator metadata,
physical measured>0.6-arcsec/MAG_AUTO<25.8 cut, parent population, model hashes,
all four endpoint results/receipts and density/influence source hashes. Every
joint receipt is checked against its result hash and exact0.0 replay. This is
a reproducible development artifact, explicitly not a production-default
configuration or completed calibration claim. It records that the17-input
classifier needs the state-major feature helper. No model weights or results
are altered; reporting validation is receipt/hash verification.

Latent repeat16587387 starts both fresh axes and passes complete scenes with
empty stderr. Classifier replica16587392 passes two contract tests11.15s and
progresses normally (through epoch14 at the startup check). All queued replica
GPU work remains dependent on completion of the latent array and uses throttle2.
Updated `jobs/job_check_joint_development.sh` with repeat/replica/influence
prefixes and replica artifact paths; shell syntax passes. One-shot snapshot
16587409 is scheduled about two hours after submission to capture both chains;
it is a log snapshot, not an agent wakeup. No new image simulation is launched.

## 2026-09-19 — Smooth classifier reaches the practical response scale; freeze repeats

Training16586130 completes19m29s, selecting epoch99 by tune BCE0.11856610059,
5377 parameters, SHA256ad975edf7020aacce00ee3efcc30353a87c5bc1eec030b2017f67369c3c78f3c.
Binder16586148 completes38s. Joint16586160 completes fresh tasks5m53s each,
main1h35m11s and independent48m20s; all five previous arms replay exactly0.0.
Smooth primary m is-0.199979±0.263649%(main),+0.402474±0.188664%(independent),
+0.347168±0.817220%(fresh g1),-0.443430±0.794924%(fresh g2). Paired shifts
versus the previous corrected classifier are+0.408929±0.004146,
+0.415675±0.004709,+0.392458±0.016440,+0.418826±0.018523pp respectively.
Predicted-U versus actual-U response differences shrink to+0.00044627,
+0.00051353,+0.00037981,-0.00018311. No response correction is fitted.

Density binder16586193 completes20s, collector16586194 completes18m09s,
summary16586197 completes43s with three tests16.03s and no errors. Primary
selected NLL gains versus historical are0.03306–0.03363nat/object across
every endpoint/leg, improving over the previous classifier by0.00972–0.00983.
Full influence16586211 completes43s with two tests16.07s: main LOO range
[-0.260896,-0.106446]%, independent[+0.327589,+0.454937]%; every omission
retains a positive gain versus historical. Fresh four-scene ranges remain
wide. Coarse histogram/mean-shape density discrepancies remain explicit.
Updated current summary and detailed results in `doc/JOINT_CALIBRATION.md`
[smooth results](JOINT_CALIBRATION.md#completed-smooth-classifier-and-reproducibility-repeats).

Added `scripts/validate_smooth_latent_repeat.py` and
`jobs/job_smooth_latent_repeat.sh`. Frozen121-file protocol changes only
latent seed7301 to7302 and output names. Array16587387 repeats all four
endpoints with throttle2 and the same six-arm evaluator. The existing
`compare_truth_parent_response --independent-latent-repeat` requires identical
model/data/evaluator provenance and exactly matched observed statistics;
all branches/cuts are retained. Both fresh tasks start and complete their
first full scene, with empty stderr. Ruff and shell syntax pass.

Added classifier training replica seed20260914 via
`scripts/train_smooth_classifier_replica.py`,
`scripts/freeze_smooth_classifier_replica.py`, their training/freezing jobs,
and `tests/test_smooth_replica_contract.py`. CPU16587392 passes two tests11.15s
and progresses through its first three epochs. Tests verify the trainer
changes only its random seed and the receipt enforces the declared seed and
minimum probability-loss choice. No joint bias selects either model.
Joint binder16587397 follows training; four-endpoint array16587398 follows
both binder AND completion of16587387, preserving two GPUs combined.
CPU comparison/influence16587400 follows the replica array, using
`jobs/job_compare_smooth_replica.sh`. Replica density binder16587399 and
collector16587401 run after training; full summary16587403 also waits for
the replica array. Separate `replica_results/` retains every outcome.
This checks classifier initialization and Monte Carlo variation, not full
flow/emulator retraining. No new images, ensemble deployment, or model
promotion occurs; target reproducibility remains under validation.

## 2026-09-19 — Retain quantitative density discrepancies before smooth validation

Read the completed corrected-classifier predictive summaries and verified
their four result hashes against comparison receipts. Saved
`scene_classifier_geometry_20260918_v1/primary_density_baseline_summary.json`
with all five classifier arms, both legs, every endpoint, primary moment
residuals/SEs, diagonal relative variance residuals, selection-fraction
residuals, uncut log-radius variance and photometry failure fractions.
This is light reduction of existing summaries, not a new simulation or
acceptance gate; no test is needed for a reporting-only artifact.

Corrected predicted-U primary shape variances are low by1.52–1.64%(e1)
and2.45–2.49%(e2) on main, and1.45–1.71%/2.41–2.54% on independent.
Primary mean-shape residuals are roughly(-0.0008,+0.0013–0.0018), while
MAG_AUTO residuals are+0.0038–0.0040(main) and+0.0046–0.0047(independent).
Primary selection fractions are overpredicted by0.00100–0.00106(main)
and0.00072–0.00079(independent), substantially smaller than earlier actual-U
diagnostics but not identically zero. These modest persistent discrepancies
must remain visible alongside relative selected-log-score improvements;
"good density" is not redefined as exact equality of every moment.

Uncut log-radius variance excess is0.0126–0.0129(main),0.0212–0.0238
(independent), and0.0808–0.0837(four fresh axis/leg combinations). This
moment is sensitive to very small positive radii; the existing diagnostics
separately retain zero/invalid photometry. No floor, tail exclusion or
empirical moment correction is applied. A new classifier cannot by itself
certify the fixed conditional flow density. Smooth training16586130 is
verified live through epoch30 with best tune BCE0.1203348 and empty stderr;
joint/density/influence stages retain their frozen dependencies.

## 2026-09-19 — Predeclare six-arm scene-influence summary

Previous continuation was a verified wait on live training16586130. It remains
healthy, now through epoch20 (best tune BCE0.1210858), with empty stderr; no
joint response is available yet. Added
`scripts/summarize_smooth_classifier_stability.py`,
`tests/test_smooth_classifier_stability.py`, and
`jobs/job_smooth_classifier_influence.sh`. The summary reuses the existing
pooled-response and result-receipt helpers, includes all six classifier arms,
all four endpoints/three cuts, and every single-scene omission. It verifies
the unaltered full-population point estimate against each evaluator result.
No scene is excluded or chosen by bias, and omission ranges are not confidence
intervals. Tests use an analytic pooled-versus-individual-ratio example and
reject disagreement with the stored full point estimate. Ruff and shell syntax
pass. Unit job16586212 completes13s: two tests pass8.53s, stderr empty.
Summary16586211 waits for
successful completion of the unchanged four-endpoint joint array16586160.
No additional training, simulations or GPU allocations are introduced.

## 2026-09-19 — Smooth classifier training starts; full joint and density chain frozen

Preparation smoke16586119 completes56s (four tests23.53s); training smoke
16586128 completes29s with exported checkpoint replay and two declared
epochs. Full preparation16586129 completes5m28s on all80 cases. Full CPU
training16586130 passes four tests6.10s and progresses through epoch5,
with minimum tune mean-case BCE0.1263820 versus the old corrected model's
0.1380879. This is early probability-loss progress, not a shear-bias result
or an alternative checkpoint-selection rule.

Added `scripts/smooth_classifier_joint.py`, the frozen six-arm evaluator
`scripts/evaluate_smooth_classifier_joint.py`, validator/freezer
`scripts/validate_smooth_classifier.py`/`scripts/freeze_smooth_classifier.py`,
and corresponding job launchers. Tests in `tests/test_smooth_classifier_joint.py`
check unchanged old-arm predictions and the limited evaluator wiring change.
Operational full-scene smoke `scripts/smoke_smooth_classifier_joint.py` and
`jobs/job_smoke_smooth_classifier.sh` complete as16586157 in27s, with three
tests5.92s, valid six-arm probabilities and old-arm Brier replay within the
existing2e-7-per-row tolerance. It uses smoke weights only and makes no
calibration claim. Joint binder16586148 waits for full training and this
smoke. GPU array16586160 follows, all four endpoints with throttle2 and
unchanged64 draws/seed7301/flow/emulator/cuts. No GPU is currently consumed
by the CPU training. Its105-file template is `smooth_joint_template.json`.

Added `scripts/audit_smooth_classifier_density.py`, its three tests and
`jobs/job_smooth_classifier_density.sh` to retain the selected-density
comparison for all six arms. The first synthetic fixture mistakenly supplied
five probabilities for six arm names; test16586187 exposed this before any
scientific density evaluation. Cancelled pending original density jobs
16586184/16586185/16586186, preserved the original fixture in cache, and
corrected the fixture to include all six. The first retry16586192 still
reported the old missing-arm failure; an isolated-bytecode-cache retry
16586199 completes34s with all three tests passing19.39s and empty stderr.
No scientific score, model, or threshold changed.
Replacement binder16586193 requires successful joint binder and test retry;
collector16586194 follows, and full summary16586197 additionally waits for
all joint endpoints. Revised template `smooth_density_template_v2.json`
retains the original as provenance.

Updated `doc/JOINT_CALIBRATION.md` with completed large-sample, influence and
predictive-density evidence and the smooth experiment contract. Archive
metadata checks find main h0.02 cases through139 and existing h0.05 starting
at140, so these roots do not supply an immediate same-scene amplitude pair.
No additional images are generated. Snapshot16586189 was replaced with
16586207 to include the corrected density dependency chain, about two hours
after submission; it captures logs, not an agent wakeup. Full training is
healthy through epoch10 with best tune BCE0.1232037 and empty stderr.
Goal remains unachieved pending joint/density results and reproducibility.

## 2026-09-19 — Full corrected validation and predictive selected density complete

Full16580812 completes all endpoints (fresh6m00s/5m50s, main1h30m10s,
independent55m22s); all previous sufficient statistics replay exactly0.0.
At the measured>0.6-arcsec primary cut, consistently trained coordinate
classifier m is-0.608908±0.261944% main80 and-0.013201±0.187994%
independent40. The fresh-axis results exactly retain the earlier outcomes.
The correction improves both large samples by about0.53pp versus historical
predicted U, but does not establish reproducible0.3% calibration. Main and
independent use different cases and h=0.02/0.05; their difference cannot be
assigned to amplitude alone. Full influence16581504 completes39s: main LOO
range[-0.670045,-0.515687]%, independent[-0.088174,+0.038848]%. No scene is
removed; the main residual is not attributable to one influential case.

Selected-density collector16581685 completes7m50s; early/full summaries
16581688/16581689 complete29s/47s. Three tests pass at every stage (full25.08s),
all provenance/replay checks pass, stderr empty. Corrected-minus-historical
normalized primary selected NLL is-0.02378085/-0.02379564nat per object
main plus/minus and-0.02353859/-0.02353554 independent. Respective case SEs
are0.00006650/0.00006814 and0.00009164/0.00008021. Fresh axes/legs retain
improvements0.02334–0.02353nat. These are selected finite-parent/output
probability laws with predicted U; common flow/emulator density cancels.
They establish a relative probability-law improvement, not absolute density
adequacy. All arms/cuts/legs are in `classifier_selected_density_full.json`.

Updated the opening current-state summary of `doc/JOINT_CALIBRATION.md` to
remove a stale checkpoint description. Added the classifier density/influence/
continuous audit prefixes to `jobs/job_check_joint_development.sh`; syntax
passes. One-shot snapshot16581697 successfully completed4s, supplementing
the earlier snapshot; these capture logs rather than waking an agent.

Next controlled experiment adds all four predeclared Gaussian crowding
scales to the existing13 state-major-coordinate inputs. New files:
`scripts/smooth_scene_classifier_features.py`, `scripts/prepare_smooth_classifier.py`,
`scripts/train_smooth_classifier.py`, `tests/test_smooth_classifier_features.py`,
`jobs/job_smooth_classifier.sh`. Original sources/models remain immutable.
Preparation verifies both legs' existing annuli, preserves labels/identities,
and derives smooth features from the same transported truth-only distances.
The trainer preserves seed20260913, splits40–99/100–119,64-wide two-layer
network, optimizer and minimum mean-case tune-BCE selection; the four new
columns add256 weights and start at zero alongside the other added inputs.
No measured response or joint bias chooses a checkpoint. Frozen protocol
`smooth_classifier_protocol.json` pins12 files and commits to all four joint
endpoints/three cuts/both fresh axes for later validation. Ruff and shell
syntax pass; CPU preparation smoke16586119 is submitted, followed by a
dependent training smoke. Full preparation/training follow a successful smoke.

## 2026-09-19 — Main reference replay completes; predicted-U selected density audit

Main reference16580808_2 completes1h45m21s with exact replay (maximum
sufficient-statistic difference0.0). Primary measured-radius>0.6-arcsec
m is-1.140485±0.261272% historical,-1.122124±0.261323% control,
-0.936300±0.261356% static geometry, and-0.135051±0.265483% actual U.
Binder16580811 completes21s and freezes the full corrected protocol;
16580812 tasks0/1 start on the two authorized GPUs and pass their first
complete scenes with empty stderr. Larger main/independent corrected
comparisons remain queued behind these declared fresh-axis confirmations.

Added `scripts/audit_classifier_selected_density.py`,
`tests/test_classifier_selected_density.py`, and
`jobs/job_classifier_selected_density.sh`. This closes a scope gap in the
preceding actual-U selected-density comparison: it compares predicted-U
selected parent/output laws across all five classifiers, holding the single
flow and disk emulator fixed. Their common output log density cancels;
candidate-minus-historical selected NLL is the observed selected mean
log(p_historical/p_candidate) plus log(M_candidate/M_historical), where
M sums classifier-weighted cut probabilities across the finite parent.
No normalization is an empirical response correction. This score conditions
on selection across the supplied parent population, not separately on each
parent; it does not establish absolute output-density quality.

Protocol `scene_classifier_geometry_20260918_v1/classifier_selected_density_protocol.json`
pins62 files, all80 main/40 independent/four additional scenes on both axes,
all three cuts and five arms, bootstrap2000/seed20260919, and both existing
CUDA comparison protocols. The CPU collector checks measured labels and
classifier probabilities against the completed coherent audit. Summaries
verify model/parent/measurement provenance and result receipts, then reuse
the64-draw selected masses from the joint evaluator. No new simulations or
optimizer steps are involved. Tests cover the strict measured3-pixel cut,
unusable-row exclusion, selected-law invariance to uniform usability scaling,
and rejection of incompatible model/population inputs. Ruff and shell syntax
pass. CPU collector16581685 is submitted; early summary16581688 depends on
it, and full summary16581689 also depends on16580812. Monitor startup/tests
and keep all outcomes; density and0.3% calibration acceptance remain unproven.

## 2026-09-19 — Continuous crowding audit reuses existing simulations

Added `scripts/audit_continuous_crowding_residuals.py`, three analytic tests
in `tests/test_continuous_crowding_residuals.py`, and
`jobs/job_audit_continuous_crowding.sh`. Frozen protocol and output are
`scene_classifier_geometry_20260918_v1/continuous_crowding_protocol.json`
and `continuous_crowding.json` under the cache root. CPU16581578 completes
3m46s; all three targeted tests pass15.57s, with empty stderr. Input, model,
measurement and probability replay checks pass.

The fixed diagnostic uses all40 independent scenes and the existing four
additional scenes on both axes, all five classifier arms, and Gaussian
radial scales0.5,1,2,4. It measures shear-even predicted-minus-observed
usability residual associations with continuous crowding, after linear
projection on the thirteen existing static classifier inputs plus intercept.
There are2000 paired case-bootstrap replicates, seed20260919. No model is
trained, no measured-output cut is applied, and no new images are simulated.

For the consistently trained coordinate classifier, independent partial
correlations at the four declared scales are0.010924,0.091685,0.138767,
0.078285 (case-bootstrap SE0.000434,0.000573,0.000714,0.000689).
Fresh g1 values are0.013329,0.095971,0.136963,0.075976; fresh g2 values
are0.013416,0.095852,0.137277,0.075312. All arms/scales are retained in the
artifact. This supports investigating smooth crowding features, but linear
partial associations do not establish missing conditional information beyond
a nonlinear use of the existing inputs, nor explain a particular shear bias.
Training uncertainty and absolute calibration are not assessed here.

Owner clarification: current validation uses the existing80 main scenes,
40 independent scenes and four previously generated additional scenes.
"Fresh" is the historical name of that last cache, not a new simulation
request. Main reference16580808_2 is through case119; the existing binder
and two-GPU-throttled validation chain remain queued. Next: finish strict
reference replay and inspect the corrected classifier on the larger samples
before selecting another model experiment.

## 2026-09-19 — Both corrected fresh axes and case-influence audit complete

Early16581487 tasks0/1 finish5m04s/5m56s, with exact replay of all reference
sufficient statistics (max difference0.0). At the measured0.6-arcsec primary
cut, the consistently trained single classifier gives g1
-0.045290±0.819529% and g2-0.862256±0.784785%. Static-to-corrected paired
changes are+0.355147±0.011685pp and+0.439124±0.021469pp. The second axis is
retained despite its less favorable point estimate; this is not0.3% acceptance.
Fixed density-histogram TV remains ~0.0382–0.0392 across axes/legs.

CPU16581503 completes32s, two tests pass14.71s, stderr empty. Its all-arm/cut
leave-one-out audit reproduces pooled point estimates and keeps all cases in
the reported result. Corrected primary LOO m ranges are[-0.836247,+0.648088]%
g1 and[-1.277733,+0.037561]% g2. Case20012 has the largest influence on both;
it is not excluded. Paired gains versus historical remain positive for every
omission:0.542911–0.570011pp g1 and0.501398–0.547313pp g2. The improvement is
robust within these scenes, while absolute calibration is poorly constrained.
See [corrected fresh results](JOINT_CALIBRATION.md#corrected-classifier-on-both-fresh-axes-and-case-influence).

Corrected predicted-U minus actual-U total response remains+0.00307194(g1)
and+0.00272766(g2), dominated by blend differences+0.00298312/+0.00306550.
This is a diagnostic of remaining probability/output/response coupling, not
an offset to subtract. Source inspection confirms the flow already uses
three neighbour-flux summaries but lacks the added five annular geometry
features. A future conditional-density/continuous-crowding residual audit
would need to control for those existing inputs; no new model is fitted on
this hypothesis before the larger joint comparisons.

Main reference16580808_2 is live through case109; binder16580811 retains
successful-reference and early-termination dependencies, then full five-arm
16580812 and influence16581504 follow. At most two GPUs remain authorized.
Snapshot16580814 remains00:42 CEST. All completed outputs/receipts are retained,
and the goal remains unproven.

## 2026-09-19 — Corrected fresh g1 completes; retain the second-axis decision

Early16581487_0 completes the four fixed fresh g1 scenes with exact replay of
all reference branches (max sufficient difference0.0). The consistently
trained single classifier gives m=-0.045290±0.819529%, compared with
-0.400438±0.819196% static geometry and-0.603985±0.808292% historical.
Its paired change versus static is+0.355147±0.011685pp; the original geometry
checkpoint with transported coordinates gives-0.039801±0.819371%.
The trained-versus-coordinate-only paired difference is-0.005490±0.000769pp.
No checkpoint is selected from these response numbers.

The improvement comes primarily from the flow contribution changing
0.52874710(static) to0.52628830(consistent), while the blend term is nearly
unchanged at0.16205. Fixed joint-histogram TV is also essentially unchanged
(~0.03819/0.03915 across legs). These four scenes have large absolute
uncertainty; the near-zero g1 point estimate is not acceptance. Early g2 is
scheduled next under throttle1. Main80/independent40 full corrected results
remain necessary, as do integration/robustness checks.

Case-influence unit job16581505 completes26s with two tests passing17.04s and
empty stderr. The early/full summaries retain all scenes and model arms;
they do not introduce an exclusion rule. Existing source/model hashes and
the two-GPU dependency chain remain intact.

## 2026-09-19 — Independent reference completes; use the free GPU for both corrected fresh axes

The preceding goal turn completed a proper selected-density normalization and
analytic tests. Current Slurm state confirms independent reference16580808_3
completed56m49s with exact sufficient-statistic replay (max difference0.0),
while main task2 remains live. At the primary measured0.6-arcsec cut,
independent m is-0.542806±0.185983% historical,-0.524321±0.186046% matched
single control, and-0.340327±0.187548% static geometry. Static-minus-historical
paired change is+0.202479±0.005984pp. These are not corrected-coordinate results.

Static geometry changes the model flow response from0.52773579(control) to
0.52973671, but lowers the blend contribution0.16513577 to0.16185565, giving
net model response0.69159236. Thus the better total hides opposing changes;
the corrected-coordinate comparison remains essential. Fixed histogram TV is
0.0364–0.0370 across these classifier arms and legs, with negligible change.

With only the main reference occupying a GPU, created an early-execution
copy of the already declared two fresh endpoints, retaining every model,
argument, cut, scene, draw and strict reference check. Outputs gain `_early`;
full four-endpoint confirmation remains queued. Binder16581486 checks the
completed training and both fresh reference receipts. Array16581487 (0–1%1,
30min/task) follows and uses only the spare GPU. Full binder16580811 is updated
to require original reference success AND early-array termination; dependent
full array16580812 still cannot exceed the two-GPU combined limit. No source
or checkpoint is modified. Early templates/protocols are separate immutable
artifacts, not replacements for the pending full protocol.

Binder16581486 completes24s, verifying selected epoch94 and both completed
reference receipts. Early g1 starts and processes its first two full scenes
without errors. Added `scripts/summarize_classifier_joint_stability.py`, two
analytic tests and `jobs/job_classifier_joint_stability.sh`. It reports every
case omission for all five arms/three cuts, confirms the full pooled m against
the evaluator, and retains every scene in the point estimate. Influence
ranges are not confidence intervals or exclusion rules. CPU summaries16581503
and16581504 follow successful early/full arrays respectively; targeted tests
are submitted separately to catch implementation errors before those stages.
Ruff and launcher syntax pass. Next: both early-axis strict replays and their
influence diagnostics, with main/independent full validation unchanged.

## 2026-09-18 — Normalized primary selected density improves slightly

CPU16580912 completes32s; all three normalization tests pass14.37s and stderr
is empty. Frozen-input/model/measurement/count checks pass for all80 main
scenes, three cuts and both legs. For the primary measured radius>0.6 arcsec /
MAG_AUTO<25.8, common-descent minus initial normalized selected NLL is
-0.00050716±0.00002575 (+0.02) and-0.00053071±0.00002367 (-0.02) nat/object.
The raw restricted NLL increases by about0.00314, but log selection-mass ratios
of about-0.00365 reverse that comparison under the explicitly defined pooled
parent-and-output selected probability law. Uncut NLL still costs about
+0.00076nat/object; no contrary result or secondary cut is discarded.

See [normalized selected density](JOINT_CALIBRATION.md#normalized-selected-density-comparison-of-the-two-single-flows)
for the derivation, every-cut results and limitations. This uses actual U,
finite64-draw normalization and paired case bootstrap; it is neither predicted
detection validation nor absolute density/calibration acceptance. No live
scientific model or evaluator is changed. Joint response jobs continue under
the existing two-GPU dependency chain; the corrected-classifier comparison
remains the next decision-bearing result.

## 2026-09-18 — Proper normalization for the cached selected-density comparison

The previous turn was a verified wait on the live joint-validation array.
Independent density review confirms the composed disk-transport physical NLL
is evaluated, but selected-cut summaries are explicitly unnormalized. Added
`scripts/normalize_selected_density_comparison.py`, three analytic tests and
`jobs/job_normalize_selected_density.sh` to use existing immutable initial/
common-descent response masses and transported observed-row likelihoods.

For the pooled finite-parent-and-output distribution conditional on actual U
and a measured cut, the candidate-minus-reference selected NLL is the raw
observed-row NLL change plus log(total candidate selected mass / total
reference selected mass). This is normalization of a specified probability
law, not an empirical calibration correction. It is not the likelihood
conditional separately on each parent, and uses oracle U rather than testing
predicted detection. The tests exercise equal selected laws despite doubled
pass probabilities, pooled versus per-case conditioning, paired case weights,
and invalid masses.

The code checks common population, h, cuts, draws/seed, checkpoint and emulator
identities, exact observed sufficient statistics, per-leg measurement receipts,
and observed density counts before computing all three cuts/both legs. It
preserves the original density bootstrap seed20260918/2000replicates and
retains raw and normalization terms separately. Finite64-draw normalization
uncertainty and absolute density adequacy remain limitations. Ruff/shell
syntax pass; a short CPU job is submitted for tests and aggregation. No live
joint-evaluator source or numerical setting changes.

## 2026-09-18 — Both fresh GPU reference comparisons complete with exact replay

Reference16580808 tasks0/1 complete6m22s/5m31s on g1/g2. Both receipts verify
against the new frozen protocol and result hashes; every original sufficient
statistic replays exactly (maximum difference0.0). Thus restoring the original
CUDA response device resolves the numerical execution contract as well as the
CPU bottleneck. No tolerance or scientific setting was changed.

On all four fresh scenes and the primary measured0.6-arcsec/MAG_AUTO<25.8 cut,
historical/single-control/static-geometry m values are
-0.60399/-0.58459/-0.40044% (g1) and-1.39571/-1.38208/-1.30138% (g2).
Case SE is roughly0.81%/0.77%; these few-scene estimates do not certify0.3%.
Fixed joint-histogram TV remains about0.038–0.039 for all classifier arms,
with no material improvement from static geometry despite better probability
scores. This is a coarse descriptive density measure, not an acceptance test.
Both axes/all models remain recorded in `fresh_cuda_reference_summary.json`.

Main task2 and independent task3 have started automatically; main has reached
51200 parents with empty stderr. Their full outputs and the subsequent
five-arm corrected-classifier comparison remain pending. See
[fresh GPU reference comparison](JOINT_CALIBRATION.md#fresh-gpu-classifier-reference-comparison).
Snapshot16580814 remains scheduled00:42 CEST. The two-GPU cap is preserved.

## 2026-09-18 — Restore reference GPU prediction and avoid a CPU replay mismatch

The preceding goal turn was a verified wait on live16580144 tasks. Independent
runtime investigation now finds that **every original joint reference used
CUDA emulator prediction**, while the later classifier protocols accidentally
specified CPU. This is both the dominant slowdown and a numerical replay risk;
extending the time limit alone did not address the device mismatch.

Added `scripts/benchmark_disk_prediction_devices.py`, a bounded diagnostic on
three fixed prepared development parts and the exact disk checkpoint. It runs
as an overlapping step within an existing GPU allocation16580751, adding no
third GPU. Each part has41957–42902 rows; CPU prediction takes0.302–0.313s,
warm CUDA including DMatrix construction0.01694–0.01697s. Predictions differ
on roughly3600 rows/part by at most1.1920928955e-7 in raw tree output. Thus
CPU and CUDA are not bitwise interchangeable, and strict sufficient-statistic
replay must not be weakened. Result: scene-classifier root
`disk_device_benchmark.json`. Ruff passes. This finite device benchmark is
not a joint-calibration or full-equivalence claim.

Cancelled CPU-device array16580144 and downstream binder16580568/array16580569
before any completed scientific output. Original protocols/logs remain.
Created new `joint_full_cuda_protocol.json` and
`transported_joint_full_cuda_template.json`, preserving all models, sources,
cases, draws64/seed7301, batch1024, cuts, h/axes and density diagnostics. The
execution-device change restores the reference; output paths gain `_cuda`.
Fresh g1/g2 run first solely to exercise complete strict replay earlier; no
endpoint is dropped or selected by result. Old protocols and the benchmark
are hash-pinned in the replacements.

New reference array16580808 (0–3%2,12h/task) depends on old-array termination;
binder16580811 depends on its success, then five-arm array16580812 (0–3%2)
depends on binding. Both fresh reference tasks have started, with two GPUs
total. The completed training receipt and all original numerical tolerances
remain mandatory. Static execution audit passes: old file pins and all arguments are identical
except response device/output paths, with the same endpoint set (receipt
`cuda_execution_revision_audit.json`). Both fresh jobs reach25600 parents in
19.2–21.5s, versus178.1s with CPU. Snapshot16580814 replaces16580649 and is
scheduled2026-09-19 00:42 CEST; it is a log capture, not an agent wakeup.
Next: verify fresh-axis full replay, then main/independent results and the
new classifier stages.

## 2026-09-18 — Signed probes confirm the coordinate mechanism and expose neighbour residual

CPU16580775 completes1m11s, five tests pass18.06s, stderr empty. The new
signed audit exactly replays all old model fields and signed probability
swaps; maximum coherent-moment difference from the independent audit is6.25e-9.
The consistently trained model reduces self-U-response error from
+0.01117220/+0.01052455 to+0.00080171/-0.00013598 on g1/g2. Its g2 coherent
residual-0.00194726 is largely the missing neighbour-shape contribution
(-0.00181127), which all present classifier inputs omit by construction.
All five arms, both axes and magnitude bins remain in the external result.
See [signed coordinate validation](JOINT_CALIBRATION.md#signed-probe-validation-of-the-coordinate-correction).

These are one-scene intrinsic-shape-weighted event diagnostics, not measured
shear m or population acceptance. No additional component-level calibration
requirement, empirical correction or checkpoint selection is introduced.
The frozen total-response/density comparisons determine practical acceptance.
Fresh joint16580144_2/3 each reaches25600 parents in178.1s with no errors;
remaining success dependencies and the two-GPU cap are intact. The scheduled
snapshot16580649 remains00:34 CEST. Next: inspect completed joint results and
case/axis stability before deciding whether neighbour-shape modeling is needed.

## 2026-09-18 — Signed-probe check of the consistent-coordinate classifier

The preceding goal turn made progress: completed probability-response and
latent-repeat evidence changed the scientific diagnosis, and corrected the
joint-validation walltime without changing numerical inputs. The live Slurm
state in this turn confirms original fresh-axis tasks16580144_2/3 running;
remaining tasks and five-arm validation retain success dependencies.

Added `scripts/audit_signed_transported_classifier.py` and its CPU launcher,
a deliberate extension of the frozen original signed audit. It retains all
five arms on the same scene20011, two source-sign probes and both axes, applies
the existing state-coordinate helper, replays each coherent moment against
the completed five-arm audit, and replays every old signed-model field against
the original signed result. No checkpoint fitting, re-selection, flow or
emulator evaluation enters this diagnostic. The physical question is whether
the coordinate correction repairs the previously located self-selection
response failure. The model still lacks neighbour-shape dependence by design.

Ruff and shell syntax checks pass. CPU16580775 is submitted with the original
three signed-response tests and two coordinate-helper tests before the audit;
all source/model/reference hashes are frozen in
`scene_classifier_geometry_20260918_v1/signed_transported_protocol.json`.
The new output is `signed_transported.json`; one scene/two probes provide
conditional diagnostics, not population calibration or measured shear m.
No live or queued evaluator source is changed. Next: inspect test/replay and
both-axis self/neighbour results when this short job completes.

## 2026-09-18 — Consistent-coordinate response audit and two latent repeats complete

CPU16580567 completes all four declared endpoints and five classifier arms.
The new single classifier retains improved probability scores and reduces the
static geometry shape-weighted U-response error from about+0.009 to
-0.00044032±0.00013123 on independent40; this diagnostic is not joint m.
Both fresh axes and all residual bins remain reported. Added a hash-linked
compact external summary and documented the full result in
[coordinate response results](JOINT_CALIBRATION.md#consistent-coordinate-classifier-event-response-results).
Training selection remains epoch94 by tune BCE only.

Matched-model16577323 completes both latent seeds with identical scene/probe
provenance. Maximum exact endpoint-m seed difference is0.066595pp; one-scene
g1 self/neighbour cancellation persists. Added both model/seed/axis results
and uncertainty limits in
[matched-image repeat](JOINT_CALIBRATION.md#matched-image-latent-repeat-completed).
These actual-U comparisons do not establish predictive detection calibration.

Original joint16580144 progresses but its CPU emulator prediction is too slow
for four hours. Read-only process/stack inspection locates XGBoost prediction;
GPU snapshot is idle. No evaluator code is changed. After Slurm rejects a
running time-limit extension, tasks0/1 are requeued under the same identities
with36-hour limits and preserved startup logs; tasks2/3 and dependent16580569
also receive36 hours. Success dependencies and the two-GPU throttle remain.
See [runtime correction](JOINT_CALIBRATION.md#joint-validator-runtime-correction).
Source tests remain565 passed/4 skipped; no code change warrants rerunning.
Fresh-axis tasks2/3 restart successfully and each logs its first1024 parents
in9.5s with empty stderr. Audit16580567 final elapsed is14m19s, exit0.
One-shot snapshot16580649 is moved to2026-09-19 00:34 CEST (about two hours);
it captures logs and is not an agent wakeup. Next: joint bias/density/reference
replay, including the dependent main/independent and new-classifier stages.

## 2026-09-18 — Consistent-coordinate classifier completes; automatic response audit starts

CPU training16580498 completes22m24s and selects epoch94 by the unchanged
minimum-mean-case-tune-BCE rule. Tune BCE is0.13808794354 and Brier0.03843268363;
selected SHA is `e3077bf30426925e6ea3bdaff0a438b6770b4a2df8eb293e842b222752d24f76`.
It remains a single classifier with response_read=False and
production_accepted=False. The scores are close to the original static
geometry model; the scientific question is whether consistent coordinates
improve response and joint calibration, not whether this small loss difference
alone is meaningful.

Dependent CPU audit16580567 starts, passes5 tests in7.36s and verifies the
completed minimum-loss selection receipt before freezing its execution
protocol. It has begun the fixed main80 cases with the new classifier and
all reference arms. No source, checkpoint, case or threshold is changed at
the hand-off. The preceding goal turn completed the evaluator/provenance
pipeline and tests; this turn now verifies the completed model and its
dependent scientific audit. The full goal remains active and unproven.

## 2026-09-18 — Coordinate-aware validation passes full suite and receipt integrity checks

CPU16580623 completes32s with5 receipt/stopping/reference-integrity tests
passing20.74s. Added `tests/test_transported_reference_receipts.py` to verify
that future original joint results can only enter validation through the
predeclared upstream protocol and matching result receipt; corrupted result
or protocol hashes are rejected before creating an execution protocol.

Full-suite16580631 completes44s:565 tests pass,4 skip and the existing
corner-extrapolation degrees-of-freedom warning remains (36.40s). No further
source change or broader test rerun is needed on this evidence. The frozen
training/audit/reference/joint dependency chain remains unchanged; the goal
still requires scientific bias/density results, not merely passing tests.

Updated one-shot checker covers all new classifier validation stages and
artifacts; bash syntax passes. Verified snapshot16580649 at23:05 CEST replaces
cancelled16580504. Training16580498 is healthy through epoch86; the probability
audit, reference binding and four-endpoint joint array remain dependency queued.
This is a log snapshot, not an agent wakeup. Yield with the goal active after
completing the independent evaluator, provenance and test work.

## 2026-09-18 — Exact coordinate-aware evaluator replay; full validation stages queued

Added a coordinate-aware classifier helper, extended joint evaluator, paired
comparison runner and operational smoke. Five classifier arms retain all old
comparisons and add the old checkpoint with transported inputs plus the new
consistently trained checkpoint. All arms share one flow/emulator draw stream,
with all cuts and classifier-weighted density diagnostics retained. See
[coordinate-aware validation](JOINT_CALIBRATION.md#coordinate-aware-joint-evaluator-and-predeclared-validation-stages).

CPU16580542 completes2m01s:3 tests pass26.08s, stderr is empty, and every old
branch replays exactly (max sufficient difference0.0). Its candidate is the
two-epoch operational model, not a scientific acceptance result. Full-model
loading requires explicit coordinate metadata and the full fixed training
split. Ruff and bash checks pass; frozen original sources are unchanged.

Added a five-arm probability/response audit and completed-training binder.
The binder checks the minimum-tune-BCE choice against training history and
pins future completed reference stages via their receipts. Receipt/stopping
and deferred-reference tests are submitted in CPU16580623. CPU audit16580567
depends on training16580498; joint binding16580568 depends on that training
and original joint16580144; full array16580569 (0–3%2) depends on binding.
This maintains two GPUs total and does not choose checkpoints from response
or joint bias. The case sets remain development tests, not untouched final
acceptance data. Training16580498 is progressing normally through epoch52;
the preceding goal turn made progress through the coordinate mechanism and
consistent training, and this turn creates its concrete downstream tests.

## 2026-09-18 — Consistent-coordinate preparation and training smoke complete

Full preparation16580483 completes2m36s:8,876,218 parents and all labels/base
inputs preserved, zero-leg geometry unchanged everywhere,713595 g05 geometry
rows changed. Four targeted tests pass and stderr is empty. Training
smoke16580484 completes38s with checkpoint/tune-BCE replay verified. Full CPU
training16580498 is submitted under the same seed/architecture/optimizer and
minimum-tune-BCE selection as the original geometry model, with explicit new
coordinate metadata and pinned successful receipts. See
[consistent-coordinate training](JOINT_CALIBRATION.md#consistent-coordinate-classifier-preparation-and-training).

CPU execution keeps the two-GPU limit intact and leaves running/queued
comparisons unchanged. A new coordinate-aware evaluation is required after
training; no model is promoted and no joint calibration claim is made.

Training16580498 passes protocol verification and its first two full epochs:
tune BCE0.15431464 then0.14887911, no clipped gradients or stderr. Epoch2 takes
12.79s. Updated checker syntax passes; verified snapshot16580504 at23:05 CEST
replaces cancelled16580378 and includes all coordinate/preparation/training
jobs and artifacts. It remains a Slurm log capture, not an agent wakeup.
The job is progressing; keep the goal active and evaluate the resulting
checkpoint with consistent coordinates before any acceptance decision.

## 2026-09-18 — Coordinate audit removes most geometry response error; prepare consistent training

Added major-axis coordinate transport, three analytic tests, a frozen
counterfactual audit and CPU launcher. Full16580457 completes2m55s with3 tests
passing0.13s, empty stderr and exact original moment replay. Independent
U-response error changes+0.00879871 +/-0.00011913 to-0.00059140 +/-0.00013096;
both fresh axes also improve. This is an intrinsic-shape event diagnostic,
not m; the remaining residual and static-training mismatch prohibit a model
acceptance claim. See [coordinate audit](JOINT_CALIBRATION.md#major-axis-coordinate-transport-explains-most-of-the-geometry-response-failure).

Failed smokes16580393/16580397 are preserved: the first uses a raw replay
tolerance inappropriate for tiny CPU numerical variation; the second executes
the superseded check despite revised source. The launcher now runs hash-verified
source bytes directly. Revised smoke16580451 completes33s. Counts/U/D stay exact;
prediction replay uses the existing2e-7 per-parent tolerance and records actual
differences. No bias threshold or scientific population is adjusted.

Added versioned consistent-coordinate preparation, a label/base-input
preservation test, and an explicitly tagged trainer preserving the prior fixed
recipe. Preparation smoke16580469 completes18s (4 tests,6.10s); zero-leg features
replay exactly. Full preparation16580483 and two-epoch CPU training smoke16580484
are running. Ruff/bash checks pass. See
[consistent-coordinate training](JOINT_CALIBRATION.md#consistent-coordinate-classifier-preparation-and-training).
The preceding turn made progress by locating the training/self-response
failure; this turn supplies a coordinate mechanism and a concrete consistent
dataset/recipe. Existing two GPU jobs and the frozen joint comparison remain
unchanged; no model is promoted.

## 2026-09-18 — Training-pair audit confirms response failure before coherent transfer

CPU16580358 completes1m53s with3 existing analytic tests passing22.74s and
empty stderr. All historical sufficient statistics replay exactly. Geometry
predicts positive intrinsic-shape-weighted U response on both train60 and
tune20 where the images and matched control give negative response. Its
training errors are(+0.00896767,+0.00891706), SE(0.00019447,0.00022373);
tuning errors are(+0.00929212,+0.00911601), SE(0.00037216,0.00049654).
See [training response audit](JOINT_CALIBRATION.md#geometry-response-failure-is-already-present-on-the-training-pairs).

This changes the next development direction: probability score improvement
alone misses response information in the fitted data; coherent transfer is
not the sole cause. Also recorded the source-verified distinction between
major-axis-normalized neighbour distances and circularized size in the base
context. A physically consistent alternative geometry or response-aware
training needs a declared new experiment, not an offset or a retroactive edit
to frozen models. The signed and training audits are read-only; no promotion
or calibration claim is made. Live matched and queued joint jobs remain the
next source of combined bias/density evidence.

Updated the one-shot checker with both response audits; bash syntax and Ruff
checks pass. Verified snapshot16580378 at23:05 CEST replaces cancelled16580330.
Both matched GPU tasks remain RUNNING,36m17s at the final scheduler check;
joint16580144 remains pending their completion. The snapshot is a log capture,
not an agent wakeup. Yield while these long comparisons progress, with the
goal active and the full acceptance requirements unchanged.

## 2026-09-18 — Signed images localize the geometry classifier's self-response error

Added `scripts/audit_signed_classifier_response.py`, three analytic tests and
its CPU launcher. It reuses frozen scene20011 and both existing sign probes,
retaining all classifiers, axes and magnitude bins. CPU16580345 completes55s
with3 tests passing18.91s; all coherent and signed-swap probability replays are
exact. Ruff/bash checks pass and stderr is empty. See
[signed classifier audit](JOINT_CALIBRATION.md#signed-image-usability-response-localizes-the-geometry-failure).

Image self U moments are-0.00834745/-0.00743210 on g1/g2; geometry predicts
+0.00282475/+0.00309244. Its self errors+0.01117220/+0.01052455 are much larger
than the image neighbour terms+0.00096837/+0.00181127. This disfavors missing
neighbour response as the sole explanation. These are intrinsic-shape event
moments, not m; two-probe conditional SE and finite-step limitations remain.
No checkpoint is selected, updated, or promoted from these results.

Added `scripts/audit_classifier_training_response.py` and a CPU launcher to
compare fixed train60/tune20 random-shear moments with the old classifier's
cached audit, using corrected labels and all three frozen arms. Syntax/Ruff
checks pass; CPU16580358 is submitted with existing analytic moment tests and
source/checkpoint/label pins. This tests whether the response failure already
exists on the fitted probability objective's own data. The long GPU joint
comparisons remain unchanged, with two GPUs total. The preceding goal turn
made progress through completed evidence and documentation; this turn adds a
new diagnostic that changes the next causal test.

## 2026-09-18 — Geometry improves held-out probability scores but fails coherent U response

Added `scripts/audit_scene_classifier_calibration.py`, three analytic tests and
`jobs/job_audit_scene_classifier_calibration.sh`. All frozen classifier arms
are evaluated on main80, independent40 and both fresh axes, retaining BCE,
Brier, reliability bins, geometry residuals and intrinsic-shape-weighted U
response. Ten-thousand paired case bootstraps quantify score changes; no
checkpoint, flow, emulator or empirical correction is updated. See
[full calibration audit](JOINT_CALIBRATION.md#held-out-geometry-calibration-improves-while-coherent-response-worsens).

Smoke16580204 completes1m02s with3 tests passing25.11s. Full16580213 completes
4m54s with3 tests passing16.94s and empty stderr. Historical label counts
replay exactly; per-row score differences are at most2.11e-10. Ruff and bash
syntax checks pass. Independent geometry-minus-control BCE improves by
-0.06012486 +/-0.00015810; neighbour-correlated residuals shrink substantially.
However, independent coherent U-response error worsens to+0.00879871
+/-0.00011913, versus control-0.00015848 +/-0.00011317. The same failure appears
on main and both fresh axes. These are event-response moments, not m, and all
bins/endpoints remain visible. Better probability scores do not demonstrate
correct shear response; possible missing scene dependence is not yet isolated.

Keep frozen joint array16580144, queued behind healthy matched-model16577323,
to measure both effects together with the unchanged flow/emulator and density
diagnostics. No model is promoted. Updated checker includes calibration logs
and artifacts; verified snapshot16580330 at23:05 CEST replaces cancelled
16580149. This snapshot captures logs rather than waking an agent. Two GPU
tasks are running; no additional GPU work is launched. The goal remains active
while these long comparisons progress; the next decisions require their joint
results, not probability scores alone.

## 2026-09-18 — Exact classifier evaluator replay passes; full fixed comparison queued

Revised CPU smoke16580123 completes1m39s. Five targeted identity/replay/paired
comparison tests pass in7.93s. On the correct operational case20000, the new
classifier-arm evaluator reproduces all old branches' sufficient statistics
exactly (maximum difference0.0), plus the fixed response-component tolerance.
The small-fixture bias is not used to select or accept a model. See
[shared-draw validation](JOINT_CALIBRATION.md#completed-geometry-training-and-shared-draw-validation).

Full frozen array16580144, tasks0–3%2, is queued afterany:16577323. It evaluates
main80, independent40 and both fresh axes using identical64-draw CRN streams
for reference/control/geometry classifiers, with one unchanged flow and disk
emulator. It retains all cuts, original branches, classifier-weighted density
diagnostics and paired differences. Sources, checkpoints, geometry manifests,
reference results and successful smoke receipts are pinned. No promotion or
0.3% claim is made; probability-score improvement still needs joint validation.

Updated one-shot checker for all new job prefixes/artifacts; syntax/Ruff checks
pass. Snapshot16580149 at23:05 CEST replaces cancelled16579893. Both running
matched-model tasks are past20000/111741 parents for the initial flow, with
verified inputs and empty stderr. They are long, healthy runs; yield rather
than poll them to completion. The next independent work is held-out probability
calibration for the new classifier arms, preserving the declared case sets.
The goal turn made concrete progress through prepared data, trained models,
exact evaluator replay and queued full validation; the goal remains active.

## 2026-09-18 — Complete geometry training and signed images; build shared-draw comparison

Classifier training16580012 completes8m54s. Control/geometry tune BCE are
0.19845420/0.13817770 (epochs95/94), Brier0.05783104/0.03845942. All preparation
identities match:8,876,218 training/tuning parents,4,437,748 independent and
444,404 fresh. The probability improvement is substantial but does not prove
joint shear calibration. See
[completed training](JOINT_CALIBRATION.md#completed-geometry-training-and-shared-draw-validation).

Added a classifier-arm joint evaluator, identity-checked geometry helper,
paired-comparison runner and tests. Reference/control/geometry share all flow
and emulator draws; classifier-weighted density diagnostics are retained too.
Full suites pass542 tests (job16580019,81.17s) and545 tests (job16580095,45.72s),
both with4 skips and the existing warning. Smoke16580095 then fails before
model evaluation because the older small fixture is case20000, not20010.
The empty partial directory and failed receipt are preserved; revised
smoke16580123 corrects only this fixture identity and runs targeted replay/
comparison tests. Ruff and launcher syntax pass.

Signed-image16574297 completes3h51m12s: zero-shear fitted shapes/selection replay
exactly,111741 parents across9 new legs,59 caught fitter KeyErrors retained as
unusable. Image-only self/neighbour estimates have two-probe conditional errors,
not scene uncertainty. Matched-model16577323_0/1 starts successfully, verifies
all13 legs and progresses on both latent seeds. No model is promoted; no flow
or emulator is updated from these diagnostics. The next step is a successful
reference replay followed by the frozen full classifier comparison.

## 2026-09-18 — Prepare truth geometry and start matched classifier experiment

Added truth-only five-annulus geometry features, corrected-label preparation,
matched control/geometry classifier training, tests and job launchers. Inputs
retain the strict true-r<26 full parent, exact identity/provenance checks and
no measured selection. Pair support is capped at19 neighbours; its scaled
distance unit is inherited explicitly. Learned pair responses are excluded.
See [geometry experiment](JOINT_CALIBRATION.md#geometry-preparation-and-matched-classifier-training).

Preparation smoke16579927 completes41s with7 tests passing in0.44s. Full80-case
preparation16579955 completes1m52s. Training smoke16579954 completes1m31s,
including8 tests in17.18s and exported-checkpoint score replay for both arms.
Ruff and launcher syntax checks pass. Validation geometry16580007 is progressing
through independent40/fresh4 with the same function and no label/model reads.

Full training16580012 starts on one GPU: fixed seed20260913, matched initial
core/data order, eight-input control versus13-input geometry, checkpoint choice
only by tune BCE. The frozen protocol pins both smokes and prepared manifest.
Matched-model16577323 is held briefly while dependencies are updated, then
released with afterok:16574297 and afterany:16580012. This preserves the two-GPU
limit without making the matched physics diagnostic depend on classifier
success. Image measurement16574297 is still live. The prior goal turn made
concrete diagnostic progress; this turn now tests the resulting hypothesis.

## 2026-09-18 — Coherent classifier audit completes; define controlled geometry comparison

Full CPU16579820 completes every declared endpoint, with empty stderr and
3 tests passing in18.61s. Main/independent all-parent intrinsic-shape-weighted
U-response errors are -0.00055125 ±0.00019267 and -0.00042568 ±0.00011345;
fixed bright/faint bins have opposite-sign errors. These are event-response
moments, not m. All bins/axes remain reported. See
[coherent results](JOINT_CALIBRATION.md#coherent-scene-classifier-response-diagnostic).

Together with the much larger blending-weighting discrepancy, this motivates
a controlled current-input versus truth-geometry classifier experiment,
selected on held-out probability loss, with independent/fresh joint outcomes
kept downstream. Recorded a bounded first geometry feature contract and its
unit checks; existing `distance_scaled` is separation divided by primary
PSF-convolved size, not PSF FWHM. No classifier training, flow update or model
promotion is performed this turn. See
[next controlled classifier experiment](JOINT_CALIBRATION.md#next-controlled-classifier-experiment).

Updated the one-shot checker to include the coupling analysis; bash syntax and
Ruff checks pass. Snapshot16579893 at21:50 CEST replaces cancelled16579825.
Image measurement16574297 remains live; matched-model16577323 is pending its
completion, while prerequisite gradient audit16579583 has completed. Preserve
the combined two-GPU limit if classifier training is added before that array.
This turn made concrete progress through completed coherent/coupling/gradient
evidence; the full goal remains active and unproven.

## 2026-09-18 — Locate modeled-U discrepancy in emulator weighting; gradient audit completes

Added `scripts/analyze_usability_blend_coupling.py`, two analytic tests and a CPU
job launcher. It replays all cached response components and uses paired case
bootstraps, retaining all declared cuts/endpoints. Exact order-symmetric
numerator/denominator attribution is diagnostic only. CPU16579853 completes
in46s, with2 tests passing in0.40s; Ruff and bash syntax pass.

Modeled-U minus actual-U primary blending response is +0.00696902 ±0.00005598
on main80 and +0.00715813 ±0.00008008 on independent40. Flow changes are
+0.00000734 ±0.00010642 and -0.00012007 ±0.00008918. Fresh g1/g2 blending
changes agree in sign and scale. Normalization explains very little of the
main/independent shift. See
[usability/blending coupling](JOINT_CALIBRATION.md#usability-discrepancy-is-concentrated-in-the-blending-contribution).
The next model experiment should test scene information that predicts
usability conditional on blending severity, selected by held-out probability
scores; no empirical response correction is justified.

Own-normalized GPU audit16579583 completes in11m01s. Independent-half direction
cosine0.99941660 passes the fixed0.99 stability gate. A response-only step would
worsen primary probability; a common descent direction exists. This permits
future numerical refinement but does not justify an immediate flow update:
matched physical decomposition and classifier/emulator coupling still matter.
See [full own-normalized result](JOINT_CALIBRATION.md#own-normalized-gradient-audit).
Full coherent classifier16579820 is progressing through its fixed case list;
image measurement16574297 remains live. No model is changed or promoted.

## 2026-09-18 — Add frozen coherent-scene classifier response audit

Added `scripts/audit_coherent_classifier_response.py`, three analytic tests,
and `jobs/job_coherent_classifier_response.sh`. The CPU-only diagnostic
compares frozen predicted U, actual U and raw detection D under coherent
antithetic shear on all four declared validation endpoints. It replays parent
and measurement provenance, reports fixed truth-magnitude bins and paired
case uncertainty, and never fits a correction. See
[coherent classifier diagnostic](JOINT_CALIBRATION.md#coherent-scene-classifier-response-diagnostic).

Smoke16579795 completes in2m50s, including3 tests in63.40s and successful
measurement-report replay on all four fixed single-case endpoints. Full CPU
audit16579820 is submitted with the successful smoke receipt pinned. Ruff and bash syntax pass. The preceding turn made concrete
progress: completed200-case label-transfer and fresh-flow evidence motivate
this coherent diagnostic rather than blind classifier retraining. The active
goal remains unproven. Existing GPU audit16579583 is live and progressing;
full signed-image16574297 is still measuring, with matched-model16577323
waiting on the established dependencies. No additional GPU is requested.

## 2026-09-18 — Selected flow fresh validation completes; next GPU audit starts

Common-descent fresh array16575483 completes both axes with all comparisons.
At measured radius>0.6 arcsec/MAG_AUTO<25.8 and actual per-leg U, g1/g2 m are
+0.402785 ±0.805590% and -0.469881 ±0.733269%. Paired changes from initial are
+0.024185 ±0.013944pp and -0.079965 ±0.011723pp. Modeled-U results remain
-0.603985 ±0.808292% and -1.395712 ±0.770364%. Four scenes are insufficient
for absolute acceptance; no axis/branch/model is promoted. See
[fresh two-axis evidence](JOINT_CALIBRATION.md#fresh-continuation-comparisons-on-both-shear-axes).

Dependent own-normalized gradient audit16579583 starts on one GPU, verifies
its frozen protocol, passes3 analytic tests in11.26s, loads all160 training
cases and reaches reference integration (2048/17,755,103 parents), with empty
stderr. Matched-model16577323 still waits for this
audit and full signed-image measurement16574297. Measurement has no per-row
logging; scheduler CPU usage is increasing and stderr has only module loads.
Snapshot16579723 remains scheduled for21:50 CEST. Continue from these actual
job states; do not repeat completed classifier or fresh evaluations.

## 2026-09-18 — Corrected-label classifier audit completes without explaining joint shift

Full read-only CPU16579715 completes in3m52s, including3 passing tests and empty
stderr. All80 old-cache cases have identical inputs/predictions after identity
matching; zero new-parent misses and zero detection-label changes. U changes
689/737 of8,876,218 parent rows in g0/g05. On120 classifier-unseen cases,
truth-shape-weighted U-response diagonal errors are +0.00005299 ±0.00018124 and
-0.00004218 ±0.00017938, consistent with zero for this random-shear diagnostic.
Mean probability gaps are small but detectable, +0.00034730/+0.00032091.
See [full transfer interpretation](JOINT_CALIBRATION.md#classifier-transfer-to-corrected-measurement-labels).

This disfavors broad label drift as the explanation for the approximately1pp
modeled-U joint-bias shift. No classifier retraining or correction is justified
from these results. The next targeted diagnosis is coherent scene-shear
usability and its coupling to the selected conditional measurement density;
random-shear moment agreement alone is insufficient.

Updated `jobs/job_check_joint_development.sh` to include classifier diagnostics;
bash syntax passes. Snapshot16579723 at21:50 CEST replaces cancelled16579587.
It is a one-shot Slurm log capture, not an automatic agent wakeup. Model and
measurement jobs continue under the existing two-GPU dependency chain.

## 2026-09-18 — Audit classifier label transfer and retain fresh two-axis failures

Added `scripts/audit_classifier_measurement_transfer.py`, three tests and
`jobs/job_classifier_measurement_transfer.sh`. The read-only CPU diagnostic
checks frozen classifier calibration on corrected labels, original classifier
train/tune/unseen splits, identity-matched old/new transitions, and forward
truth-shape-weighted U moments. It records missing matches, source/model hashes
and detection-with-shape-row limitations. No model is trained or corrected.
See [classifier transfer](JOINT_CALIBRATION.md#classifier-transfer-to-corrected-measurement-labels).

Smoke 16579707: 3 tests pass in 12.93s; cases0/40/100 complete, empty stderr.
Old/new features and predictions are exactly equal on the overlap for cases40
and100; respectively 7/12 and10/11 U labels change across g0/g05. Full200-case
CPU audit16579715 is queued behind a verified smoke receipt and frozen inputs.
Ruff and bash syntax pass. Full results remain necessary before attributing the
modeled-U bias discrepancy to classifier calibration rather than label drift.

Fresh initial/control/augmented array16567110 and paired comparison16567262
complete; old rejected per-leg candidate16573420 explicitly skips in 1s.
The control/augmented continuations worsen both-axis actual-U response,
particularly g2: paired shifts +2.617366 ±0.021522pp and +3.623599 ±0.041717pp
from initial. All results are retained; see
[fresh comparisons](JOINT_CALIBRATION.md#fresh-continuation-comparisons-on-both-shear-axes).
Selected common-descent validation16575483 starts successfully on both GPUs.
Full signed-image16574297 passes all9 render audits and starts measurement of
111741 parents on9 legs. Later GPU stages retain the combined two-GPU limit.

## 2026-09-18 — Clarify practical acceptance and detection scope with owner

The owner accepts performance at roughly the target level provided it is not
lucky or cherry-picked. Treat 0.3% as a practical reproducible scale, rather
than forcing every noisy estimate below a hard boundary; retain all declared
axes, seeds, populations and comparisons. No model or estimator is changed.

Checked the current candidate's completed raw results against
[usability definitions](CONVENTIONS.md#5-usability-detection-and-selection).
The primary actual-U branch includes observed per-leg detection/measurement
losses, including missing crossmatches. It does not validate predicted U.
The modeled-U branch gives m=-1.140485 +/-0.261272% on main80 and
-0.542806 +/-0.185983% on independent40. This is the combined detection and
usable-measurement probability model, not a separately identified detection
error. See [acceptance and detection scope](JOINT_CALIBRATION.md#practical-acceptance-and-detection-scope).

The near-target actual-U results therefore cannot be called fully predictive
selection calibration. Continue the fixed fresh and signed-probe comparisons
without checkpoint reselection; report both actual-U and modeled-U branches
and investigate their difference before any end-to-end claim.

## 2026-09-18 — Schedule read-only gradient audit during remaining image rendering

Slurm confirms fresh control tasks 16567110_2/3 and full image job 16574297
are still running; both later GPU stages are pending. Reordered dependencies
so own-normalized audit 16579583 follows fresh candidate array 16575483, while
matched-model array 16577323 additionally waits for this audit. This uses the
likely GPU-idle interval before the signed images finish. At most one audit GPU
or two matched-model GPUs run after the existing two-task validation chain.

The first matched-model update rejects the already completed preparation ID
16567098, which is no longer an active controller handle. `sacct` reconfirms
its successful completion; retry omits that fulfilled dependency and succeeds.
Both final dependency chains are verified with `scontrol show job`. Scientific
protocols, seeds, models and sources are unchanged, and the audit remains
read-only. The initial schedule receipt preserves the planned DAG, and
`own_normalized_guard_schedule_confirmed.json` records the failed first update
and verified active dependencies; snapshot 16579587 remains set for 21:50 CEST.
See [gradient scheduling](JOINT_CALIBRATION.md#own-normalized-gradient-audit).

## 2026-09-18 — Verify own-normalized response gradients and queue read-only full audit

Added `scripts/own_normalized_guard.py`,
`scripts/audit_own_normalized_guard_gradient.py`, three analytic tests and a job
launcher. Nine full-parent moments define the response of the model's own
selected mean. Gradient replay includes both the baseline mean and mass
derivatives, retaining each leg's own usability and independent row sampling.
There is no optimizer or prediction correction. See
[own-normalized gradient audit](JOINT_CALIBRATION.md#own-normalized-gradient-audit).

Analytic CPU job 16579484: 3 tests pass in 12.70s, including invariance to a
common probability scaling and agreement with full autograd under entering/
leaving usability. Smoke 16579565 completes in 54s, including 3 tests in
21.39s, and verifies unchanged model weights. Its independent-half direction
cosine is only 0.66109, so the small sample does not justify an update. Full
suite 16579569: 526 passed, 4 skipped, one existing warning in 44.94s. Ruff and
bash syntax pass.

Full read-only GPU audit 16579583 is queued after matched-model array 16577323,
preserving the combined two-GPU limit. It uses 160 training cases, four full
moment integration groups, and four independent sampled-gradient replicas.
Frozen inputs and execution receipts record the experiment. Updated snapshot
16579587 at 21:50 CEST supersedes cancelled 16578601.

Fresh initial-flow tasks complete: four-scene primary m is +0.378600 +/-0.818046%
along g1 and -0.389917 +/-0.723574% along g2. One-case h=0.04 results retain
undefined case uncertainty and cannot identify an amplitude correction. Control
tasks 16567110_2/3 pass startup and progress normally on both axes. These new
scenes are useful development evidence but too few to establish 0.3% calibration.
No checkpoint is promoted or updated.

## 2026-09-18 — Complete common-descent joint validation; isolate normalization gain

Main/independent array 16575454 completes in 1h57m23s / 52m41s, including the
main transported-density audit. At measured radius >0.6 arcsec with actual
per-leg U, selected step2 gives m=-0.135051 +/-0.265483% on main80 and
+0.477637 +/-0.183980% on independent40. Paired changes from initial are
+0.033644 +/-0.003382pp and +0.031135 +/-0.002190pp. The goal is not established.
See [completed common-descent evidence](JOINT_CALIBRATION.md#completed-common-descent-joint-and-normalization-evidence).

Self-response audit 16576138 completes in 3m42s. Soft primary own-normalized
R11 changes only -0.0003343 and R22 increases +0.0000894; the measured-reference
changes are -0.0022334 and -0.0018130. Most primary guard improvement therefore
comes from the shrinking normalization discrepancy. Hard-cut behavior agrees.
Independent selection-fraction residual improves 0.00393413 to 0.00092028,
while uncut transported NLL worsens by about 0.00076 nat/object. These mixed
results do not justify promotion. A hash-linked external evidence receipt
records exact results and the normalization decomposition; no model is changed.

Reviewed the earlier emulator amplitude evidence: the compatible four-scene
h=0.02 versus h=0.2 pilot has mean difference +0.010033 +/-0.008710 and does
not identify a finite-amplitude correction. No fitted offset or scaling is
introduced. Future flow objectives must distinguish model-normalized response
from baseline probability; matched signed-image evidence is still needed to
attribute the remaining joint error to flow and emulator components.

All full-scene tasks 16564237 complete. Dependent preparation 16567098 passes
3 tests in 16.21s and completes in 1m35s: 444,404 parents across four scenes,
zero unmatched-parent drops. Fresh GPU tasks 16567110_0/1 pass identity checks
and begin both-axis evaluation with empty stderr and meaningful progress.
Matched image job 16574297 remains live after seven rendered legs. Next
one-shot snapshot 16578601 is scheduled for 21:50 CEST. All later GPU stages
retain their dependency chain and the two-GPU total.

## 2026-09-18 — Complete matched model/image pilot and queue full-area comparison

Added `scripts/evaluate_matched_model_probe.py`, four tests, and small/full job
launchers. The evaluator verifies image-control receipts and exactly replays
the observed response aggregation before evaluating both fixed flows on the
same 13 legs. It preserves actual per-leg U, joint-output cuts, CRN draws,
selection centering, both normalization conventions and paired probe errors.
See [integrated comparison](JOINT_CALIBRATION.md#integrated-matched-modelimage-comparison).

CPU job 16577296 passes the full suite: 522 passed, 4 skipped, one existing
warning in 39.78 seconds; then it fails before model draws on the historical
small-pilot receipt's missing scope field. Added explicit compatibility for
that known small format while requiring the original-prior receipt in full
mode. Retry 16577302 completes in 2m32s, including 12 targeted tests in 7.12s.
A stale compiled test omitted the new compatibility test; removed that cache
and job 16577308 explicitly passes the test in 14.43s. Ruff and bash syntax pass.

Both flows complete the 1,075-parent, two-probe pilot. Initial own-normalized
self-response errors are -0.0277561 (g1), +0.0710388 (g2); paired conditional
probe SEs are 0.0277113 and 0.00613188. These are response differences, not m.
The small scene has the operational py31 prior and cannot support a full-prior
calibration claim or checkpoint selection. The measured-side replay succeeds.

Queued full-area array 16577323 with fixed latent seeds 7301/7302, each comparing
the initial and selected single-flow checkpoints. It depends on full images
16574297, preparation 16567098 and the existing fresh candidate array 16575483;
the two-task throttle and dependency chain preserve the two-GPU total. Frozen
source/model/result hashes and a separate execution receipt record this run.
Updated checker and resubmitted snapshot 16577325 at 17:56 CEST, cancelling
16576327. Full-area decomposition and joint calibration remain pending.

## 2026-09-18 — Add identity-matched model shear-field primitives

Added `scripts/matched_model_probe.py` and five tests for the model side of
the random-sign experiment. Intrinsic flow contexts use each primary's sign;
disk-response pair coefficients use each secondary's sign before summation
and radial-tanh transport. The helper reuses exact output-cell pooling and
rejects missing source identities and self pairs. Existing live evaluators
and simulation sources are unchanged. See [model shear-field contract](JOINT_CALIBRATION.md#model-side-matched-shear-field-contract).

Ruff passes. CPU job 16577170 completes in 14 seconds: 18 targeted tests pass
in 9.59 seconds across the new helper, matched influence estimator, existing
output-conditioned response and disk transport. Tests cover exact all-positive
coherent replay, mixed secondary signs including a nonparent source, row-order
invariance, isolated primaries, signed primary contexts and identity failures.

The main/independent validation jobs remain live. Fresh scene task 16564237_3
has started and passed 32 startup tests in 10.69 seconds. This turn adds tested
evaluation primitives, not a completed model/image decomposition or calibration
result. Next integrate them with the measured signed legs, actual per-leg U,
shared CRN draws and explicit own/reference baseline normalization. Scheduled
snapshot 16576327 remains pending for 17:56 CEST; the goal remains active.

## 2026-09-18 — Verify full-area zero pixels and queue normalization audit

CPU job 16575858 passes in 9 seconds: scene 20011's replayed 18181-square
raw image and 48-square PSF match the original pixels exactly. The remaining
random-sign legs and fitted-catalogue replay are pending. This establishes the
zero-image control, not a response or calibration result.

Added `jobs/job_audit_common_descent_self_response.sh`, reusing the frozen
40-case flow auditor for the initial and selected step2 checkpoints with fixed
identity checks. `bash -n` passes. Job 16576138 waits for independent validation
task 16575454_1; fresh array 16567110 now also waits for this audit, preserving
the two-GPU total. The audit separates own-model and measured-reference
normalization because reduced guard loss alone does not establish reduced
self-response error. See [normalization follow-up](JOINT_CALIBRATION.md#common-descent-normalization-audit-and-full-area-zero-control).

Main and independent evaluations have progressed through cases 50 and 211.
Replacement one-shot snapshot 16576327 at 17:56 CEST includes the new audit;
superseded snapshot 16575587 is cancelled. A separate execution supplement
records the dependency change and zero-control receipt. Joint bias, matched
response decomposition and fresh-scene acceptance remain pending.

## 2026-09-18 — Select eligible common-descent step2 and start fixed joint validation

Full training16574987 completes10m35s. Step2 reduces validation response loss
0.4569601 to0.3180714 and soft/hard primary probability residuals to
0.00113818/0.00128465. NLL costs0.000676600nat/row, within the unchanged0.001
budget. Third-step trials fail training NLL; no constraint is relaxed. Final
checkpoint SHA5434cfaf... is verified against its completed receipt. See
[selected checkpoint](JOINT_CALIBRATION.md#selected-common-descent-checkpoint-and-fixed-joint-validation).

Added `scripts/validate_joint_probability_candidate.py` and its job launcher,
reusing the tested frozen endpoint implementation. Ruff/bash syntax pass.
Queued array16575454 for main/independent joint+density diagnostics; both
real two-case smokes complete and full evaluations start with empty stderr.
This checks the new wrapper's routing and identity gates on the actual selected
checkpoint. No scientific calibration claim follows from these startup checks.

Rewired original fresh16567110 after this array and CPU prep; new candidate
fresh16575483 follows the old step-zero skip array16573420. This serializes all
two-task GPU arrays within the total limit. Frozen downstream protocol and a
separate execution/checkpoint receipt record the experiment. Snapshot16575587
at17:56 supersedes16575356. Full matched probe16574297 completes its first
g0 replay render and continues; it uses no GPU. The0.3% goal remains unproven.

## 2026-09-18 — Train along stable common descent; launch matched full-area probe

Full gradient16573829 completes4m35s: response/probability gradient cosine
about-0.1823, independent common-direction halves agree0.99943071. Response-only
updates increase training probability error; common descent has both slopes
negative. Added `scripts/refine_joint_guard_probability_flow.py` and its launcher.
It uses CRN model-reference integration, joint response/probability backtracking,
and the unchanged full-NLL plus soft/hard validation gates. See
[full-scene probe and refinement](JOINT_CALIBRATION.md#full-scene-matched-probe-and-common-descent-refinement).

Smoke16574698 completes41s with8 tests passed; a training step is accepted but
validation rejects it and step0 replay passes. Full suite16574895:514 passed,
4 skipped,existing warning in61.67s. Ruff/bash syntax pass. Full GPU16574987
passes startup and is loading full training data. Fresh16567110 now depends on
this and CPU preparation16567098, preserving the combined two-GPU limit.

Extended the BlendEMU random-sign pilot through explicit full-scene opt-in,
original-prior receipt checks, historical simulation-runtime checks and recorded
runtime. Added a full-area launcher and scope test; adapted SBSI analysis to
label full and small scopes explicitly. Preserved the earlier successful pilot
sources with matching hashes. Full probe16574297 uses the first completed
fresh scene20011,699568 unchanged objects, two independent probes along both
axes, and exact g0 replay. Three tests pass6.32s; nine legs are prepared in
sims1 Python3.9.20/NumPy1.26.4 and rendering has started. This is8CPU/48GiB,
no GPU. It cannot establish calibration from one scene/two probes.

Updated/resubmitted the snapshot as16575356 at17:56, including the full probe,
and cancelled16573832. External execution protocols record sources and DAG.
No model is promoted. Next: inspect finite-update constraint outcomes; compare
matched probe/model quantities only under identical masks and normalization.

## 2026-09-18 — Complete density and response evidence; audit a joint training direction

Both density tasks16568039 complete5m10s: uncut transported NLL improves by
about0.0061--0.0066nat/object, despite both continuations worsening joint m.
Control independent16564328 completes44m12s at+1.101258 +/-0.185056%; paired
comparison16564445 gives changes against initial+0.654755 +/-0.002898pp
(control) and+1.301656 +/-0.003964pp (augmented). See
[completed evidence](JOINT_CALIBRATION.md#completed-continuation-density-independent-response-and-per-leg-rejection).

Full refinement16573416 completes11m15s with step0 retained. First two updates
fail probability gates; all third-step trials fail train NLL. Existing downstream
array16573419 emits two explicit skipped receipts after1s. No model is promoted.

Random-sign image pilot16573694 completes6m42s. Nine legs pass rendered-truth
checks; g0 image/PSF pixels, fitted shapes and primary selection reproduce
exactly. Two fit exceptions remain unusable. The two-probe,1075-parent response
estimates have much larger uncertainty than required for calibration. See
[pilot completion and gradient audit](JOINT_CALIBRATION.md#completed-image-replay-and-joint-gradient-feasibility-audit).

Added `scripts/selection_probability_gradient.py`,
`scripts/audit_joint_guard_probability_gradient.py`, its job launcher and three
meaningful gradient/direction tests. Failed smoke setup16573787/16573793 exposed
a missing synthetic cut name and stale compiled fixture; repaired before running
physical integration. Smoke16573818 completes45s,3 tests pass13.89s, weights are
unchanged. Shared-mass direction agreement0.99620 is stronger than independent-
half agreement0.84746, so full sampling is required before any update.

Queued full read-only GPU audit16573829 after successful smoke; fresh16567110
now waits for this and CPU preparation16567098. Other prerequisite GPU chains
are confirmed complete. Source hashes and dependencies are recorded in external
`joint_gradient_execution_protocol.json`. Updated one-shot checker for the new
audit and resubmitted16573832 at17:56, cancelling16573422. Ruff/bash syntax pass.
The0.3% joint goal remains unproven; next inspect full gradient feasibility and
then test any proposed update under unchanged NLL and probability constraints.

## 2026-09-18 — Launch matched random-sign image pilot; probability gate rejects early updates

Added BlendEMU `scripts/prepare_random_sign_probe_pilot.py`, its CPU launcher and
two tests; added SBSI `scripts/analyze_random_sign_probe_pilot.py`. The bounded
pilot reuses operational scene20000 with6996 objects, independent signs on every
source, both axes at+/-0.02, and an exact g0 replay. Image/PSF pixels, fitted
shapes and primary0.6arcsec selection must reproduce before interpreting probe
responses. Own-leg selection and explicit finite-step/probe-variance limitations
are retained. See [image pilot](JOINT_CALIBRATION.md#bounded-random-sign-image-pilot-with-zero-shear-replay).

Startup16573641 fails before images on a historical outer-driver hash difference;
unchanged core fitting modules are verified. Retry16573694 explicitly records
that difference and requires exact replay, with immutable v2 source pins. Two
setup tests pass in4.79s; Ruff and bash syntax pass. It uses4CPUs/16GiB and no GPU.
Image rendering, audit, fitting and diagnostic completion remain pending.

Full refinement16573416 has completed two training updates. Both reduce the
validation response loss within the full-NLL budget, but both increase soft and
hard primary selection-mass errors. Both are rejected at checkpoint selection;
step0 remains selected provisionally. No gate, target, source or model output is
altered to accommodate these results. Current two-GPU schedule remains intact.

## 2026-09-18 — Repair decorated-helper provenance; independent augmentation fails target

Full refinement16572851 failed startup before training because its smoke receipt
hashed PyTorch's `no_grad` wrapper instead of the selection-mass helper. Fixed
`scripts/refine_per_leg_guard_flow.py` to hash the helper module file. No change
to scientific helper, numerical recipe or selection gates. Preserved old
receipts; new smoke16573410 passes39s with identical baseline NLL/guard and
correct source identities. Full suite16573411:511 passed,4 skipped,existing
warning in58.09s. Ruff/bash syntax pass. See
[repair](JOINT_CALIBRATION.md#per-leg-startup-provenance-repair-and-independent-augmentation-result).

Replacement full16573416 has passed startup and is loading training parents.
Repaired density dependencies and resubmitted the two per-leg validation arrays
as16573419/16573420 with explicit immutable v2 protocols. Cancelled stale
arrays16573305/16573308 and snapshot16573309; new snapshot16573422 at17:56.
Existing fresh-array16567110 dependencies updated. Combined two-GPU limit
verified with control16564328 and refinement16573416 active.

Augmented independent16564329 completes48m43s: primary actual-U joint
m=+1.748159% +/-0.186377pp,95%CI[1.393908,2.126961]%. Together with main
+1.126205%, this rejects the NLL-only augmented continuation as a0.3% solution.
Control and formal paired independent comparisons are pending.

Added `scripts/matched_response_probe.py` and four estimator-oracle tests for
same-scene self/neighbour response separation, including own-leg selection
normalization and rectangular source/target populations. CPU16573403:4 passed
in0.11s. The exhaustive linear oracle recovers known derivatives to1e-14;
[derivation and limits](JOINT_CALIBRATION.md#matched-random-sign-response-decomposition-estimator-oracle).
This is an experimentally unvalidated diagnostic helper, not evidence of
calibration. A bounded image pilot is needed before any full matched experiment.

## 2026-09-18 — Queue frozen downstream per-leg validation and audit fresh fit failures

Added `scripts/validate_per_leg_candidate.py`, its two-task job launcher, and
nine targeted tests. The runner rejects invalid full-refinement receipts and
failed density/response/probability gates, explicitly skips step zero, and
runs unchanged joint evaluators plus paired response and density diagnostics.
See [downstream validation](JOINT_CALIBRATION.md#downstream-validation-of-the-per-leg-refinement).
CPU test16573294:9 passed in0.10s; Ruff and bash syntax pass. No live scientific
training/evaluator sources were edited.

Queued existing-endpoint array16573305 after16568039; added16573305 to the
original fresh-array16567110 prerequisites; queued per-leg fresh-axis
array16573308 after16567110. Two GPUs maximum throughout. Source/input hashes
and the schedule are recorded separately in the external downstream protocol
and execution receipt. Updated one-shot checker to include new logs/statuses,
resubmitted as16573309 at17:56CEST and cancelled superseded16572869, because
Slurm captures batch scripts on submission.

Fresh case20011 completed rendering and measurement. Read-only exception replay
16573300 completes1m10s:32/32 recorded KeyErrors are ngmix flag1 results without
`g`, with nonfinite stored shapes. No valid shapes were dropped by this audit,
no catalogue/source was changed, and those fits remain unusable per leg.
Other scenes and refinement/joint results remain pending. The0.3% goal is not
established; no model is promoted.

## 2026-09-18 — Constrain selected probability and queue full per-leg refinement

Full reference-gradient audit16568798 completes13m52s with bank-direction
cosine0.99999875. Conditional gradient RMS noise falls from39.435 to4.681/4.663.
Training primary reference R11/R22 is0.5321742/0.5332933 versus measured
0.5207415/0.5210140. Independent reference stability supports a controlled
update; it does not prove scientific calibration.

Added `scripts/guard_selection_mass.py` and extended per-leg checkpoint selection
to require non-increasing absolute errors in both soft and exact hard primary
baseline selected mass. This prevents an apparent response improvement from
being selected through a worse probability prediction. All-parent normalization
and own-leg U are explicit; exact0.6arcsec radius equality is excluded. Full
hard-label sources match the40-case soft bank. Preserved the old smoke's source
with its original SHA; the new smoke writes separate artifacts. See
[selection constraints](JOINT_CALIBRATION.md#selection-probability-constraints-and-first-full-per-leg-refinement).

CPU smoke16572361 completes40s;16 targeted tests pass in13.59s. It retains
step0 after all trial NLL gates fail. Full suite16572827:500 passed,2 skipped,
existing warning in48.08s. Ruff and bash syntax pass. New full launcher pins
the revised smoke and helpers.

Queued six-step full refinement16572851 after augmented independent16564329
and the passing suite. Added it to density-array16568039 dependencies so that
the existing control chain and new refinement together remain at two GPUs.
Fresh-array dependencies remain valid transitively. Frozen external protocol
records all settings and hashes; next snapshot16572869 is at17:56. Full
training and joint-validation outcomes remain pending; no model is promoted.

## 2026-09-18 — Quantify self-response errors and selection-normalization effects

Full model audit16566470 completes6m12s; CPU decomposition16568526 completes20s,
both with empty stderr. On the40-case corrected validation population, primary
soft per-leg initial own-normalized residuals are+0.0101294/+0.0100386 in
R11/R22 (case SE about0.00158; two-group integration SE below0.000067).
Control gives+0.0059105/-0.0075356; augmentation+0.0008532/-0.0135664.
Hard-cut results confirm the initial positive discrepancy. Neither the old
aggregate guard nor the improved NLL establishes both-axis calibration.
See [completed model audits](JOINT_CALIBRATION.md#completed-per-leg-model-and-normalization-audits).

Initial selected-mass error is+0.0037614, or0.518409% relatively. It explains
almost all of the0.0027448/0.0027513 reference-versus-own response difference;
centering contributes only about1e-6. Exact decomposition reconstruction error
is below4.08e-16 across the bootstrap. No normalization correction is applied.

Read-only inspection confirms the emulator's forward h0.2 physical-moment
training requires usable g0 and both response-label shapes; native validation
also uses common-usable anchors. The initial small coherent joint bias and
separate positive self-response residual motivate an error-cancellation
hypothesis, but different experiments prevent identifying an emulator offset.
No emulator source or weights were modified.

GPU reference audit16568798 has passed full target-bank loading and is
integrating17,755,103 training parents. Independent augmentation evaluation
16564329 is also progressing, preserving two GPUs total. Next: inspect gradient
stability before the controlled per-leg refinement and require subsequent joint
validation. The0.3% joint-calibration goal remains unproven.

## 2026-09-18 — Complete continuation response evidence and repair comparison scheduling

Main80 control16564323 and augmentation16564324 complete1h42m46s/1h42m14s.
At measured radius>0.6arcsec, actual-U joint m is+0.480712+/-0.266823% and
+1.126205+/-0.268783%, versus initial-0.168695+/-0.265339%. Paired changes
are+0.649407+/-0.003353pp and+1.294900+/-0.004612pp. The augmented absolute
95% case interval[0.588554,1.635113]% excludes the target band. Control is also
not an established0.3% result. Training NLL/old guard improvements do not
establish joint calibration; neither continuation is promoted. See
[completed continuation evidence](JOINT_CALIBRATION.md#completed-80-scene-continuation-response-and-density-diagnostics).

CPU16564444 completes29s with paired control/augmentation response and density
diagnostics. Both retain selected-fraction excesses of about0.0042 and even
MAG_AUTO mean offsets of about0.0042mag under the existing diagnostic definitions.
Added `jobs/job_compare_initial_main_response.sh`: Slurm's submitted-script copy
did not include edits made after16564444 was queued. Separate CPU16569482
completes34s and strict comparability checks pass for both initial-flow
comparisons. A dependency submission using purged completed handles failed;
resubmission after verifying completed inputs succeeds. Bash syntax passes.

Model audit16566470 is running; augmented independent smoke16564327 completes
3m00s and full16564329 starts with meaningful progress. Changed pending gradient
audit16568798 to physical A40 nodes on cip/inter, preserving its dependency
chain and one-GPU request. Resubmitted the updated one-shot checker as16569483
at15:56 and cancelled16568103, so Slurm captures the newly added audit coverage.
No scientific settings or model artifacts changed; two GPUs remain active.

Next: complete the population/normalization audit and gradient-reference study
before choosing the full per-leg refinement. The goal remains active.

## 2026-09-18 — Add paired continuation-versus-initial main-sample comparisons

Added `scripts/project_truth_parent_audit.py`, two focused tests and CPU
preparation launcher. It preserves all80 original cases and the initial
corrected flow's nine truth-parent response branches, removing only other
reporting cohorts. Original statistics/summaries remain unchanged and the
derived artifact records source/projection hashes. Added both main-sample
continuation-versus-initial comparisons to `job_compare_shape_continuation.sh`.
See [initial-flow comparisons](JOINT_CALIBRATION.md#paired-main-sample-comparisons-against-the-initial-corrected-flow).

CPU16569419 completes39s:9 targeted tests pass in13.81s, and the full real-data
self-comparison yields exactly zero paired changes/errors/intervals. Ruff and
bash syntax pass. Job16564444 now also depends on successful preparation; GPU
scheduling is unchanged. No predictive-density moments are invented for the
old initial audit, which did not record them. Next: inspect complete paired
results rather than differences with unpaired error bars.

At15:35, full-area case20011 has completed all five renders and its render
audit; measurement has started for111,741 parents across five legs. The batch
job is active. Verified699,568 rendered objects and111,741 generated parents
in every leg, with no unrendered identities. During measurement startup its
aggregate CPU time advances by over10CPU-minutes and reported memory stays
below the48GiB allocation (peak about6.9GiB). No startup traceback is present;
fit outcomes remain pending. This is pipeline progress, not a calibration result.

## 2026-09-18 — Validate density-constrained per-leg refinement

Added `scripts/refine_per_leg_guard_flow.py`, its isolated CPU smoke launcher,
and two checkpoint-selection tests. The runner reuses independent-row replay
with the new per-leg population and compatible frozen reference banks. Full
validation NLL is checked before any candidate is selected; step0 competes and
the selected artifact is reloaded and replayed. No ConstGold result is read.
The existing six-step backtracking and0.001nat relative density budget remain
explicit experimental settings, not calibration criteria. See
[refinement runner](JOINT_CALIBRATION.md#density-constrained-per-leg-refinement-runner).

CPU smoke16568936 completes27s;13 targeted tests pass in6.63s. Every trial
lowers its training response score but exceeds the density budget (smallest
NLL increase+0.0010833nat), so step0 is restored exactly and marked ineligible
for joint validation. This verifies rejection behavior on the small smoke
subsets; it predicts neither the full-population update nor calibration.

Full CPU suite16569008:495 passed,2 skipped, existing warning in42.64s. Ruff
and bash syntax pass. No full optimization was submitted and the existing GPU
chains remain unchanged. Next: inspect the full model/normalization and
reference-gradient audits before selecting the scientific refinement recipe.

## 2026-09-18 — Prepare population-consistent reference integrals for training

Added `scripts/audit_per_leg_guard_gradient.py` and its CPU/GPU launcher.
The frozen initial corrected flow is integrated over the complete training
parent with own-leg usability; full targets must reproduce the existing bank.
Four independent reference integrals and repeated gradients from two separate
banks measure update-direction stability without fitting a model. The old
both-usable reference bank is not reused. See
[per-leg gradient audit](JOINT_CALIBRATION.md#per-leg-frozen-reference-integration-and-gradient-stability).

CPU smoke16568742 completes29s, with finite gradients, unchanged flow hash,
and complete reference/gradient artifacts for1,024 random training parents.
Ruff and launcher bash syntax pass. This is an operational check, not evidence
of full-population gradient stability or any calibration gain.

Queued GPU16568798 after full model audit16566470 and the successful smoke;
rerouted control independent smoke16564326 after16568798. Squeue confirms
both active GPUs remain the original main evaluations, with the new audit
serialized in the control chain. Updated one-shot check16568103 coverage and
saved execution_update_20260918_1511.json. No optimizer step or model selection
was performed. Next: inspect model/normalization and gradient evidence before
choosing a density-constrained refinement. The full0.3% goal remains unproven.

## 2026-09-18 — Separate selected mass and centering in the response guard

Added `scripts/decompose_guard_normalization.py`, six focused tests, and a CPU
launcher. The diagnostic exactly decomposes measured-reference minus
model-normalized response into selected-mass and mean-centering terms using
the frozen audit's additive moments and paired case bootstrap. It changes no
model or evaluator and applies no fitted correction. See
[normalization identity](JOINT_CALIBRATION.md#exact-decomposition-of-guard-normalization-effects).

Ruff and bash syntax pass. First CPU smoke16568456 rejects harmless reduction
roundoff in the common shear design; fixed with a relative1e-12 matrix-norm
comparison while retaining exact parent counts and testing rejection of real
design changes. Retry16568502 completes28s:6 tests pass in14.05s, all twelve
operational model/population/cut decompositions reconstruct within2.23e-16,
empty stderr. Full CPU16568526 is queued after model audit16566470 and the
successful smoke; the two-GPU schedule is unchanged.

Next: inspect full population terms together with per-leg model residuals
before specifying further training. Smoke magnitude estimates are not used.
Measured radius>0.6arcsec remains primary and joint0.3% calibration is unproven.

## 2026-09-18 — Validate continuation density on the sheared endpoint

Extended `scripts/audit_disk_candidate_density.py` with explicit device and
existing-leg options; legacy defaults remain unchanged. Added a frozen
continuation density launcher that compares both single flows to the initial
corrected flow on coherent+/-0.02 observations, preserving actual-U support
and excluding historical g0 training labels. Restricted scores remain
unnormalized and are paired with selection-fraction diagnostics.

CPU smokes16567996 finish2m42s/3m10s. Source measurements and row counts match
the old reference exactly; CPU/GPU reference NLL differs by at most4.386e-7nat.
Both arms improve smoke uncut NLL by about0.0061--0.0065nat despite their
upward response-bias shifts. This is preliminary evidence of the density/
response tradeoff, not an acceptance result. Full suite16568033:487 passed,
2 skipped, existing warning in44.92s; Ruff/bash syntax pass.

Queued two-task full80 density array16568039 after both existing GPU chains.
Fresh array16567110 additionally depends on this array; squeue verifies the
two-GPU scheduling limit. New external transported_density_protocol.json
records identities/settings. Snapshot16568103 supersedes16567266 at15:56.
See [coherent density comparison](JOINT_CALIBRATION.md#coherent-endpoint-physical-density-comparison-for-the-continuations).
Next: inspect full response, population-consistent model audit and density
results before further training. Joint0.3% calibration remains unproven.

## 2026-09-18 — Complete target banks and verify real-flow per-leg gradients

Training-bank CPU16567533 completes4m20s, retaining17,755,103 parents across160
cases. Primary soft per-leg R11/R22 is0.520741494/0.521013953, versus paired
0.525975189/0.526499356. Verified disjoint160/40 training/validation cases,
matching source/domain identities and cut definitions; froze bank checksums
in `per_leg_guard_target_banks.json`. No model was fitted to these targets.

Added `scripts/smoke_per_leg_guard_flow.py`. CPU16567735 completes22s:
actual frozen physical-flow gradients from independent-row replay match direct
autograd at relative L2 error9.52e-8 (<1e-6 declared mixed-precision tolerance).
Scores agree and participating gradients are finite. Input checkpoint hash is
unchanged; zero optimizer steps and no new checkpoint. Ruff passes. The
small operational subset makes no population-response or calibration claim.
See [banks and gradients](JOINT_CALIBRATION.md#complete-per-leg-target-banks-and-physical-flow-gradient-check).

Next: inspect the pending full model-side audit and joint density/response
comparisons before selecting a new optimization recipe. The approximately0.3%
joint-calibration goal remains active and unproven.

## 2026-09-18 — Implement and validate the population-consistent response guard

Added experimental `scripts/per_leg_guard_response.py`, four focused tests and
`jobs/job_validate_per_leg_guard.sh`. The adapter retains full truth-parent
contexts and applies each leg's own U to both measured and generated influence
contributions. It reuses CRN sampling and independent-row gradient replay,
without modifying legacy training, live evaluators or any checkpoint.

CPU16567454 completes2m33s. Full suite:487 passed,2 skipped, existing warning
in64.06s. Tests cover the exact oracle, analytic loss gradient, direct versus
replayed gradients, all-usable legacy equivalence and invalid/missing targets.
The full40-case real-data bank reproduces legacy paired and audited per-leg
reference targets within2e-12. Ruff/bash syntax pass. Training-bank CPU16567533
starts normally; its160-case preparation fits no model.

Prioritized model audit16566470 after control main16564323, with control
independent16564326/16564328 following it. Fresh array16567110 now explicitly
waits for both GPU-chain endpoints. Initial dependency updates rejected completed
job IDs; retry using outstanding handles succeeds and squeue verifies the
acyclic graph and two-GPU limit. Completed smoke artifact checks remain.
External execution_update_20260918_1437.json records scheduling-only changes.
See [per-leg adapter](JOINT_CALIBRATION.md#experimental-per-leg-guard-adapter-and-exact-reference-validation).

Next: complete target preparation and inspect the frozen model audit before
choosing a new optimization run. Joint calibration and density acceptance
remain unproven; no model is promoted.

## 2026-09-18 — Verify a structural inconsistency in paired-only guard supervision

Added an exact oracle-distribution regression test to
`tests/test_flow_per_leg_response.py`. The model matches the entire usable
per-leg output distribution, yet conditioning measurements on both-leg
usability induces a false guard residual: true/model response0.5I versus
paired measured0.9I. Per-leg weighted measured/model responses agree exactly.
This proves that the paired guard requires a cross-leg conditional-independence
assumption absent from a marginal flow; it does not estimate the real-data bias.

Ruff passes. CPU16567387:4 targeted tests passed in15.83s, job25s, empty stderr.
No model, live evaluator, audit source or scientific catalogue was modified.
Documented the expectation identities and a consistent per-leg influence
statistic in [matched-usability issue](JOINT_CALIBRATION.md#why-matched-usability-can-contradict-an-exact-per-leg-density).

Next: use queued full model audit16566470 to measure the real residual change
before choosing the next training experiment. Retain the measured0.6-arcsec
selection and joint response/density requirements; calibration remains unproven.

## 2026-09-18 — Make paired response comparisons use the correct shear axis

Updated `scripts/compare_truth_parent_response.py` to use R22 for g2 bootstrap
bias, preserve legacy g1 defaults, reject mixed axes/vector orders and reject
single-case operational confidence intervals. Added three regression tests;
no live evaluator or model was changed. Saved the previous helper externally.

CPU validation16567253 completes65s. Full suite:482 passed,2 skipped, one
existing warning in44.18s. The complete40-case independent g1 comparison
replays with every numerical comparison field exactly unchanged. New
`jobs/job_compare_fresh_joint_development.sh` queues response/density comparisons
for all frozen model pairs on each fresh axis, plus one-scene amplitude point
differences with explicit null uncertainty. Bash and embedded-Python syntax
pass. CPU16567262 waits for fresh array16567110 and validation;16567261 was
replaced before execution to tighten emulator-provenance checks.

Monitor16567266 supersedes16567122 at15:56 and includes these logs.
See [axis comparison](JOINT_CALIBRATION.md#explicit-axis-paired-comparisons-and-amplitude-reporting).
Next: await full response/density evidence and fresh measurements. The
calibration goal remains open; these changes ensure correct interpretation
of the queued g2 results and do not establish target calibration.

## 2026-09-18 — Connect fresh full-area scenes to validated two-axis analysis

Added `scripts/prepare_fresh_joint_development.py`, three tests, preparation
and evaluation launchers. Preparation verifies completed render/measurement
provenance, full-area replay scope, both axes and independent seeds; it retains
the generated truth population, explicitly drops/reports unrendered parents
across legs and links source catalogues without changing measurements.

CPU preparation smoke16566975 completes39s:1,075 parent identities/contexts
exactly match the previous small-pilot arrays, with16,977 neighbour pairs.
Two-axis evaluation smokes16567097 finish46/59s with empty errors; every
measured/model sufficient statistic exactly matches the original staged runs.
Full test16567100:479 passed,2 skipped, one existing warning in46.55s.
Ruff/bash syntax pass. Full g0 image rendering also completes (~17min);
additional fresh legs and measurements remain live.

Queued full preparation16567098 after all fresh scenes complete, then six
single-flow/axis GPU tasks16567110 with throttle2. Dependencies on both old
GPU chains preserve the total limit; each task records h0.02/four cases plus
h0.04/one-case amplitude diagnostics without claiming single-case uncertainty.
Source hashes and dependencies are frozen in the external fresh-analysis
protocol. Snapshot16567122 replaces16566473 at15:56 and covers new stages.
See [fresh analysis chain](JOINT_CALIBRATION.md#fresh-full-scene-analysis-chain-and-exact-pilot-replay).

Next: inspect dependent stage startup and full response/density evidence when
available. These four fresh scenes are development data; the original joint
bias/density/single-flow objective remains unproven and active.

## 2026-09-18 — Add the matching per-leg model response audit

Added `scripts/audit_flow_per_leg_response.py`, three focused tests and
`jobs/job_audit_flow_per_leg.sh`. The audit evaluates the three frozen single
flows under the same measured per-leg U, cuts and forward extraction as the
completed label audit. Common random numbers span models/legs/cuts. It reports
model versus measured baseline normalization separately, selected-mass errors,
full response tensors, scene-bootstrap errors and independent latent groups.
No missing measurement or numerical zero-flux tail is silently discarded.

CPU smoke16566462 completes35s with3 tests passing and empty stderr. Two512-row
prefixes validate operation only; no scientific conclusion is drawn from them.
Full40-case audit16566470 waits on this smoke and control-chain16564328, retaining
at most two GPUs. Sources/models/domain are hash checked; the new sources are
frozen pending execution. Full-suite16566471 completes:476 passed,2 skipped, one existing warning
in37.36s. Ruff and bash syntax pass. Recorded protocol externally and updated
[model audit](JOINT_CALIBRATION.md#frozen-model-audit-with-per-leg-usability-and-normalization-controls).
Snapshot16566473 replaces16564479 at15:56 and covers the new stage.

Next: follow full audits when their dependencies finish, determine whether the
training guard residual survives matching support/selection/normalization,
and use fresh multi-axis scenes to separate finite-step effects. Joint0.3%,
density quality and a validated single-flow replacement remain unproven.

## 2026-09-18 — Quantify guard population and cut mismatch on corrected labels

Added `scripts/audit_harmonized_usability_response.py`, three tests and CPU
launcher `jobs/job_audit_harmonized_usability.sh`. The label-only audit retains
all corrected validation truth parents, independently weights each usable leg,
and compares paired/per-leg populations under soft/exact measured0.6-arcsec
cuts. No missing shape is imputed; full-parent identities, label hashes and
source provenance are checked. Bootstrap uses40 independent scene units.

Audit16565769 completes55s, including3 passing targeted tests. The paired-soft
primary target reproduces the saved training guard to3e-13. Exact-cut per-leg
minus paired R11/R22 is-0.00498238 +/-0.00038123 /-0.00463552 +/-0.00044361.
Per-leg hard-minus-soft is-0.00213676 +/-0.00046590 /-0.00144057 +/-0.00051628.
These are consequential estimand differences, not fitted corrections; model
predictions must be evaluated on the same per-leg population before replacing
a training target. The aggregate guard also hides component tradeoffs.
See [label audit](JOINT_CALIBRATION.md#corrected-label-per-leg-usability-and-exact-cut-response-audit).

Ruff/bash syntax pass. Full CPU suite16565880:473 passed,2 skipped, one
existing warning in45.00s. Main single-flow evaluations16564323/16564324 and
fresh image array16564237 remain live; no extra GPU was used or live source
changed. Next: inspect full joint comparisons, perform a consistently defined
model-side per-leg response audit, and follow fresh multi-axis measurements.
The calibration/density/single-flow goal remains unproven and active.

## 2026-09-18 — Restore exact scene prior and validate completed single flows

Historical-runtime replay16564191 exactly recovers all699,568 original scene
objects; current-runtime sampling differs despite identical loader/source.
BlendEMU full-scene launcher uses sims1 generation/rendering and corrected
py31 measurement. Replacement CPU array16564237 passes29 tests and exact
replay in both initial tasks, then starts image rendering; two cases remain
pending. Failed outputs are preserved. Matched-prior symmetry audit16564238
completes65s over69,956,780 rows and preserves the approximate-symmetry
conclusion. Runtime versions are now recorded by the standalone audit.
Updated [environment contract](ENVIRONMENT.md#historical-scene-prior-runtime-compatibility)
and [runtime evidence](JOINT_CALIBRATION.md#historical-runtime-replay-and-full-scene-restart).

Control/augmentation training16563497/16563498 completes7m03s/6m52s.
Training-only selection chooses epoch1/epoch2; full validation NLL is
-4.774971864/-4.775263274 vs initial-4.768710490, with guard
0.116155758/0.183126290 vs0.356833213. Both are eligible for joint evaluation,
not accepted models. New `jobs/job_evaluate_shape_continuation.sh` pins
checkpoint/protocol/source identities, verifies successful smoke before full
runs, and retains primary measured0.6 arcsec, actual-U and fixed joint-density
diagnostics. Bash syntax passes. Submitted two dependency chains16564321/23/26/28
and16564322/24/27/29 for main80(h.02) then existing independent40(h.05).
Both smokes begin with successful frozen-artifact validation; two GPUs total.
No scientific evaluator or frozen BlendEMU source changed in this step.
Smokes finish3m10s/3m01s and both full evaluations begin normally with empty
error logs. Preliminary two-scene m shifts upward to+1.7922%/+2.4497%
from initial+1.1578%, despite improved training gates; full endpoints remain
pending and there is no calibration claim. Added syntax-checked CPU comparison
launcher (jobs16564444/16564445) and extended the one-shot monitor with explicit
validation-job IDs. Snapshot16564479 replaces16564263 at about+2h. Frozen
external `joint_validation_protocol.json` records models, sources and job chains.

Next: follow smoke completion/full startup and fresh rendering stages, compare
both frozen single-flow arms without choosing favorable endpoints, and inspect
fresh full-scene multi-axis response. Goal remains open. See
[continuation validation](JOINT_CALIBRATION.md#completed-continuation-and-frozen-joint-validation).

## 2026-09-18 — Independent comparison completes; fresh prior replay catches mismatch

Independent baseline16563591 and candidate16563245 complete51m32s/52m14s.
At measured0.6 arcsec, actual-U m is-0.0110521 +/-0.1827114% baseline and
+0.4465029 +/-0.1835285% candidate. Paired shift+0.4575550 +/-0.0011960pp
repeats the refinement effect but does not establish target calibration.
This endpoint differs from development in both scenes and h(.05 vs.02).
Density analysis16563973 finds persistent candidate primary mean offsets
(-0.000814,+0.001353,-0.000709,+0.006345) in e1,e2,log radius,MAG_AUTO,
and selection-fraction excess about0.0039--0.0040. Baseline is similar;
matching response alone is insufficient. Updated the goal headline and
[independent evidence](JOINT_CALIBRATION.md#completed-independent-response-and-density-evidence).

Both preregistered training arms16563497/16563498 start, reproduce their
frozen baseline NLL/guard and pass first-epoch progress with empty error logs.
They are the only two GPU jobs. Their selection gates remain training-only.

Full-scene array16564117 fails all four tasks before rendering because exact
prior replay fails, despite29 tests passing. Preserved preparation outputs.
Audit16564158 rules out the historical5-arcsec radius variant; diagnostic
16564165 bounds the original sampling-pool size to69,956,594--69,956,915,
versus current69,967,938. Matching physical galaxies have shifted pool indices
and some position angles differ at float32 precision. This is a real pipeline/
runtime compatibility issue, not a reason to bypass the guard. Runtime replay
16564191 is submitted in the historical sims1 environment. No new full images
or calibration results were produced by the failed array.

Next: resolve prior replay before restarting fresh generation; retain current
training and inspect its completion. Calibration goal remains open.

## 2026-09-18 — Plan precision and start full-area fresh development pilot

Applied the statistical-power skill to prospective same-scene-variance
precision, using the80-case SE0.2653391pp: about251 scenes for0.15pp SE or564
for0.10pp. Recorded sensitivity to stipulated biases/variance, with explicit
limits: no measured g2 variance, no area/shear-step extrapolation, and no
post-hoc power calculation or new owner acceptance requirement.

Extended BlendEMU's pilot generator with explicit full-scene development
scope, original-prior replay, and matching `one_tile` geometry. Preserved the
completed small pilot's source before editing; its default remains capped
at0.01 square degree. Added tests for scope and changed-prior rejection.
The current loader's implied count differs from training (699,680 vs699,568),
so full scenes must replay the old scene exactly and retain its count before
rendering. No empirical response scaling or prior-angle adjustment is used.

Frozen four-case protocol20010--20013 has g0 and both axes at+/-0.02;
case20010 also has+/-0.04. Source hashes, scene/noise seeds and measurement
conventions are recorded externally. Initial submission used array indices
beyond the scheduler limit; corrected to indices0--3 mapped to cases20010--20013.
CPU16564117 now starts two tasks with throttle2,8CPU/48G each;29 BlendEMU tests
pass and source catalogue loading begins. Ruff and bash syntax checks pass.
No additional GPU is used. These four scenes are development/variance data,
not a calibration acceptance sample.

See [fresh precision planning](JOINT_CALIBRATION.md#fresh-full-area-development-and-precision-planning).
Next: inspect exact-prior replay and first rendering progress, then schedule
long-run checks; follow independent validation and augmentation startup.

## 2026-09-18 — Audit finite-prior symmetry and preserve density offsets

Added `audit_scene_prior_shape_symmetry.py`, two tests and CPU launcher.
Audit16563921 completes54s over the exact69,967,938-row processed FS2 prior.
Number-weighted mean intrinsic e=(-7.99064e-5,-3.85251e-6), with36-bin angular
TV0.000199838 from the four-sign average. Flux-weighted mean is
(-0.000455437,-0.000325882), TV0.0158859; rare bright strata are less symmetric.
This is exact finite-prior description, not a calibration bound or proof of
conditional scene symmetry. Keep the augmentation experiment's untransformed
density/response gates and do not alter angles or subtract empirical offsets.

Added `compare_joint_predictive_results.py`, two tests and CPU launcher. It
separates shear-even moment residuals from shear-odd derivatives, compares
fixed histogram marginals and bootstraps candidate changes by common scenes.
This prevents a small response discrepancy from hiding shared shape offsets.
Observed statistics and diagnostic definitions must match exactly. Smoke
16563974 passes targeted tests and completes19s; full analysis16563973 depends
on both independent40-case evaluations. Full suite16563977 is submitted.
Ruff and shell syntax checks pass; existing evaluator/trainer sources stay frozen.

See [prior and density diagnostics](JOINT_CALIBRATION.md#finite-scene-prior-symmetry-and-residual-decomposition).
Next: use full independent density/response results and inspect dependent
augmentation startup. Neither diagnostic alone certifies the requested model.

Full validation16563977 completes:470 passed,2 skipped and the existing warning
in35.33s. The two-case predictive smoke retains primary shear-even shape
residuals about(-0.00178,+0.00180) for the corrected candidate; these two scenes
are insufficient to quantify generalization. Full40-case analysis remains
pending and will distinguish persistent density offsets from response errors.

## 2026-09-18 — Stage and validate explicit-axis joint response evaluation

Added `scripts/joint_response_axes.py`, truth-parent pilot preparation and
CPU launchers, plus eight helper/full-main test cases. An external evaluator
copy gains explicit shear axis, CPU support, and a clearly marked single-case
operational mode. Axis selection reaches the intrinsic-shape map, disk
transport and m diagonal while response vectors stay in sky e1,e2 order.
G1 contexts/summaries remain bitwise identical. Full-main synthetic tests
recover a known Mobius-composed response on g1 and g2, including flow/blend
decomposition. Existing running evaluator/trainer sources are untouched.

Tests16563831 found NaN JSON serialization in single-case uncertainty output;
fixed recursively in the new helper. Full suite16563850 passes466 tests,
2 skips and the existing warning in35.25s. Ruff and bash syntax checks pass.
Truth preparation16563852 completes1,075 parents and16,977 neighbour pairs;
no measured cut defines the parent. CPU array16563859 is submitted for
g1/g2 h=0.02/0.04 operational smoke with the frozen corrected checkpoint.
No additional GPU is requested and no pilot bias is an acceptance result.

See [explicit-axis staging](JOINT_CALIBRATION.md#explicit-axis-joint-evaluator-staging).
Next: follow smoke startup/completion; integrate only after independent runs
finish, then complete fresh-scene and joint-density validation.

All four real-data CPU smokes complete34--48s. Verified frozen source/model
identities, finite sky-ordered response vectors, flow+blend decomposition and
null single-case uncertainties; receipt is `smoke_validation.json` under the
staging root. Primary m on this small scene varies+0.795/+0.038% (g1 h=.02/.04)
and-12.539/-2.059% (g2). Preserve all values without choosing a favorable axis
or step: these are operational results without case-based precision, not an
acceptance result. No configured likelihood, checkpoint or science cut changes.

## 2026-09-18 — Prepare a fresh whole-scene multi-axis rendering pilot

Added BlendEMU `prepare_joint_multiaxis_pilot.py`,
`audit_joint_multiaxis_pilot.py`, ten targeted test cases and two CPU launchers.
Current catalogue generation's `constant` mode actually shears half the scene;
the pilot instead copies one generated scene and assigns exact whole-scene
g1/g2 legs. Frozen pilot settings: case20000, scene seed20123, noise seed2026291801,
0.01 square degree without a galaxy-count cap, g0 and +/-0.02,+/-0.04 on both
axes. Existing simulation source/config/catalogue hashes are recorded. Matching
noise seeds are explicit, simulator return codes checked, and render audit
reports missing generated/parent identities and checks shears/crossmatches.

CPU16563738 passed25 tests, then failed48s before rendering due to a NumPy
float32 manifest field. Preserved its output, converted that scalar explicitly,
and submitted retry16563748 with dependent audit16563763. The retry passes all
26 BlendEMU tests in13.25s. Ruff and bash syntax checks pass. SBSI evaluator and
queued training sources are unchanged; both GPU chains retain the two-GPU cap.

This small-area rendering pilot is not acceptance data or a bias result.
Corrected subpixel/common-optimizer measurements and fresh-scene independence
audits remain before any scientific use; primary measured cut remains0.6 arcsec.
See [pilot details](JOINT_CALIBRATION.md#fresh-multi-axis-rendering-pilot).
Next: verify rendering/audit completion and add matching multi-axis measurements.

Added the matching `measure_joint_multiaxis_pilot.py` and CPU launcher16563789,
dependent on audit16563763. It reuses the corrected target-fit worker, explicitly
joins detection identities, and preserves per-leg non-detections and fit flags.
An additional test covers shuffled, noncontiguous detection IDs and missing
parents. Rendering has passed six of nine legs with6,996 generated objects;
Ruff and launcher syntax checks still pass. Measurement/test completion is pending.

Completion: rendering16563748 finishes5m01s, audit16563763 finishes9s, and
measurement16563789 finishes80s after27 tests pass10.32s. Each of nine legs
contains6,996 rendered objects and1,075 generated target parents, none lost
at rendering. Missing detections114--118 and usable fits957--961 vary by leg;
no fit exceptions/nonzero flags occur. This closes the pipeline pilot only.
Recorded completion receipts externally. Fresh full-scene model freeze and
two-component joint evaluation remain; no calibration result or model
promotion follows from this pilot.

## 2026-09-18 — Reach -0.169% primary development bias; repair independent diagnostics

Corrected-flow16561643 completes83m41s: primary measured0.6-arcsec actual-U
m=-0.1686949 +/-0.2653391%, versus-0.6230785 +/-0.2641523% before refinement.
Paired improvement is+0.4543836 +/-0.0021160 percentage points. This uses one
flow and the unchanged disk-response model. Uncut and0.75-arcsec diagnostics
remain-1.55784% and+1.04538%; no cut is substituted. The primary95% interval
[-0.6992,+0.3349]% and reused scenes prevent a completion claim. See
[completed joint response](JOINT_CALIBRATION.md#completed-corrected-flow-joint-response-and-latent-repeat).

Baseline latent repeat16561899 completes101m31s with identical source/model/data
provenance and measured sufficient statistics. Seed7302 gives-0.6293408% at the
primary cut, a-0.0062623 +/-0.0055841pp change from7301. The observed integration
scatter is small relative to the current case uncertainty; no favorable seed
selection or finite-draw-bias bound is claimed.

Independent smoke16563242 then failed29s in the new diagnostic, after encountering
finite nonpositive generated photometry. Updated `joint_predictive_diagnostics.py`
to retain this mass in a separate histogram bin and explicit failure fractions,
while reporting log moments conditional on positive photometry and their
undefined mass. No values are floored and no response estimate is altered.
Preserved failed diagnostic source/manifest, then integrated the tested
truth-only/predictive evaluator into the active script after original runs
completed. Added two underflow tests and two synthetic integration cases.

Full CPU16563585 passes458 tests with2 skips45.65s and the existing warning;
Ruff and diff checks pass. Baseline retry16563587 and candidate smoke16563243
pass early draw progress. Slurm denied releasing16563244 from its administrative
hold after failed dependency; replaced it with16563591, redirected comparison
16563246 and control training16563497, and cancelled the obsolete job.
Both GPU chains remain limited to two. See
[diagnostic repair](JOINT_CALIBRATION.md#independent-diagnostic-underflow-repair-and-retry).
Next: inspect smoke completion and full-stage startup, then independent response
and density evidence. No acceptance or configured-likelihood promotion is made.

Both revised smokes then complete176/184s. Invalid-photometry mass is below
0.9ppm, with no undefined moments at the primary measured cut; shape boundary
mass is4.8--7.0ppm and is reported explicitly. Full baseline16563591 and
candidate16563245 start and pass meaningful draw progress with empty error
logs. Snapshot16563602 replaces16563501 about two hours after startup. The
dependent augmentation jobs remain queued and the combined GPU count is two.

## 2026-09-18 — Queue controlled single-flow shape-augmentation experiment

Added `refine_flow_shape_augmentation.py`, three tests and a CPU-smoke/GPU-full
launcher. Both arms start from the same corrected checkpoint and use identical
row streams and fresh AdamW settings. Augmentation applies exact physical
shape sign actions through the unchanged affine standardizers; it introduces
no calibration offset or inference ensemble. Full runs use four epochs of
4million rows, lr1e-5, batch8192 and clip_norm5.

Checkpoint selection requires full untransformed validation NLL improvement
without an increase in the initial validation guard loss. Epoch zero competes;
selected density and response are recomputed after reloading. The guard uses
the same corrected targets and CRN integration settings as the preceding
refinement. No ConstGold or independent-scene response is read. See
[controlled augmentation](JOINT_CALIBRATION.md#controlled-single-flow-augmentation-continuation).

CPU smokes16563490/16563491 pass76/72s, with identical starting scores and both
correctly retaining epoch zero. They are explicitly ineligible for scientific
joint validation. Full tests16563492 pass454 with2 skips59.05s and the existing
NumPy warning. Targeted tests, Ruff, shell syntax and diff checks pass.

Queued control16563497 after independent baseline16563244, and augmented
16563498 after independent candidate16563245. This extends the existing two
GPU chains without increasing concurrency. The preregistered external
experiment protocol records all seeds, inputs and source hashes. Snapshot
16563501 replaces16563249 at the unchanged13:21 Berlin time. Freeze trainer
source until both runs finish and inspect dependent stages at startup. No
full augmentation result or0.3%-calibration claim exists yet.

## 2026-09-18 — Check native ngmix stamp symmetries and optimizer variability

Added BlendEMU `scripts/audit_measurement_shape_symmetry.py`, two geometry/noise
tests and its CPU launcher. Existing stamps, PSFs and subpixel centres transform
together through four pixel actions; measured shapes return to the original
frame. The two optimizer streams remain separate, and identity stream0 must
replay the existing harmonized fit artifact. No source catalogue is overwritten.

CPU array16563404 completes cases4/96 in55/49s. Of1024 generated truth parents
sampled per scene,112/122 lack detection coordinates; these remain recorded.
Identity replay passes1e-12 for both cases. One transformed fit fails with
KeyError and is recorded, without common-success selection.

Added SBSI `audit_measurement_symmetry_pilot.py` plus two tests. Combined CPU
test/summary16563411 completes65s. Primary-cut maximum absolute moment z-scores
are2.371 and1.505 across the transformations and streams, versus1.841/1.347 for
identity optimizer repeats. Mean-shape paired-object errors are0.005--0.008;
this is not a tight symmetry bound or calibration certificate. It supports a
controlled training-augmentation test while preserving untransformed validation
and the frozen independent comparison. See
[native symmetry pilot](JOINT_CALIBRATION.md#native-measurement-symmetry-pilot).

Validation:16 BlendEMU tests pass11.28s;451 SBSI tests pass with2 skips50.99s
and the existing NumPy warning. Targeted tests, Ruff, shell syntax and diff
checks pass. No augmented model or new GPU allocation is introduced. Both
science evaluation chains remain frozen;0.3% remains unverified.

## 2026-09-18 — Identify a held-out shape-symmetry density deficit

Added `audit_flow_shape_symmetry.py`, its CPU launcher and four tests. The
audit transforms intrinsic and measured shape together through the four spin-2
sign actions, preserving photometry and scalar conditions. It verifies circular
PSF metadata, held-out case membership, exact g0 and source hashes. Probabilities
are averaged using stable log-sum-exp; the diagnostic does not fit a model or
select a ConstGold result. See [symmetry audit](JOINT_CALIBRATION.md#discrete-shape-symmetry-density-audit).

CPU16563273 completed98s on4096 rows per each of40 held-out g0 scenes. The
normalized symmetry-orbit density lowers baseline NLL by0.01374021 +/-0.00056932
nat and corrected-flow NLL by0.01373573 +/-0.00056815, improving on all40 cases
for both flows. Absolute directional log-density differences are0.12--0.17nat.
The corrected-label refinement barely changes this defect. This supports
testing symmetry augmentation during single-flow training, after measurement
symmetry checks; it does not establish a joint-bias gain or explain V3 by itself.
No ensemble or correction is introduced into the frozen validation candidates.

Validation: four focused tests pass in5.57s; full CPU16563274 passes449 tests
with2 skips in42.11s and the existing NumPy warning. Ruff, shell syntax and
diff checks pass. The two existing GPU evaluations and queued independent
dependency chains are unchanged. Goal remains unverified.

## 2026-09-18 — Queue independent validation through two GPU dependency chains

Removed an unnecessary workflow dependency: the tested staged evaluator can
run directly from its frozen external copy while the original evaluator stays
unchanged for the current source-hashed jobs. The independent launcher now
accepts an explicit evaluator path and verifies the same staged source/helper
hashes. Verified its actual runpy entry point with `--help`, shell syntax and
source hashes; the earlier synthetic full-main and445-test validations apply.

Submitted baseline smoke16563242 after latent repeat16561899, then baseline
full16563244; candidate smoke16563243 after current evaluation16561643, then
candidate full16563245. CPU paired comparison16563246 depends on both full
runs. All four new GPU jobs are pending dependencies; the two existing GPU
jobs are the only running allocations. Full arms require matching successful
smokes. These are fixed-model comparisons, not a flow ensemble or calibration
selection. See [queued independent comparison](JOINT_CALIBRATION.md#independent-comparison-queued-from-frozen-evaluator-source).

Updated execution manifests and the snapshot launcher for new artifacts.
Snapshot16563249 replaces16562630 at the unchanged13:21 Berlin time. Freeze
the standalone evaluator and cohort/predictive helpers through both full runs.
Next: inspect each stage at startup, repair operational failures if any, and
evaluate complete response and density evidence. No independent result or
0.3%-calibration claim exists yet.

## 2026-09-18 — Prepare absolute joint-density diagnostics for independent evaluation

Review found that the existing density gates constrain relative NLL rather
than absolute joint-output agreement. Added `joint_predictive_diagnostics.py`
and10 tests, with a fixed four-dimensional output histogram and actual-U
selection fractions, means, covariance and disk-violation mass. Generated
outputs are measured after transport using the same integration draws, with
per-draw selection and no observed-photometry conditioning. Paired case errors
cover moments; empirical histogram TV is descriptive, not an acceptance gate.

Extended the external staged evaluator with an optional diagnostics flag and
helper provenance. The independent launcher enables it and requires matching
helper hashes and a smoke that contains diagnostics. Existing GPU evaluator
source is still untouched. See
[predictive diagnostics](JOINT_CALIBRATION.md#prospective-absolute-joint-distribution-diagnostics).

Validation:10 focused tests pass in6.53s. One synthetic CPU integration test
passes in6.69s, exercising missing-g0 parents through the staged main and
checking exact equality of response results with diagnostics on/off. Ruff,
shell syntax and diff checks pass. CPU suite16563124 passes445 tests with2 skips
in46.79s, with the existing NumPy degrees-of-freedom warning.
No new GPU job or model change is introduced; independent validation remains
pending completion of the two current evaluations and application of the
staged integration. The0.3% goal is not established.

## 2026-09-18 — Stage explicit no-g0 evaluation without altering running source

Added `scripts/truth_parent_cohorts.py` and11 test cases. Missing g0 data now
has an explicit prospective evaluator contract: request `truth_parent` only;
never invent observed masks. Available masks must be aligned booleans and
fixed-cut membership must lie within usable-g0 when both are requested.
The new helper is not imported by either running GPU evaluator.

Prepared the evaluator patch and its original/staged/helper hashes under the
external cache `harmonized_secondary_20260917_v1/evaluator_staging/`. Its27
default accumulator branches match the current evaluator exactly; restricting
to truth_parent yields9. The original evaluator is unchanged, preserving the
completion-time source hashes for16561643 and16561899. Apply only after both
finish, then validate integration before submitting any independent run.

Added `jobs/job_evaluate_independent_joint.sh` for frozen baseline/candidate
smokes and40-case runs with common seed7301,64 draws and g1+/-0.05. It checks
the staged source/helper hashes, frozen flow identity, full training density
gate, pair preparation and matching per-arm smoke. Both frozen arms remain
planned regardless of the main ConstGold result. No new GPU job is submitted.

Validation:15 targeted tests pass in6.91s; staged default-statistics equivalence,
Ruff, shell syntax and diff checks pass. Full CPU suite16562981 completes:
435 passed,2 skipped in45.20s, with the existing NumPy degrees-of-freedom warning.
See [staged evaluator](JOINT_CALIBRATION.md#staged-truth-parent-only-evaluation).
The calibration objective remains unverified.

## 2026-09-18 — Complete independent-scene neighbour-pair preparation

CPU array16562810 completed all four blocks in98--125 seconds each, with zero
exit codes and empty error logs. All40 case artifacts exist and their metadata
hashes match successful builder logs:4,437,748 parents and71,357,269 neighbour
pairs. Saved `independent_pair_preparation.json` and updated the external
candidate execution manifest. No measured response or model fit was evaluated.
See [completed independent pilots](JOINT_CALIBRATION.md#completed-independent-parent-and-measurement-compatibility-pilots).

The corrected-flow joint evaluation16561643 and latent repeat16561899 remain
healthy, with empty error logs and progress through cases63 and76 respectively.
Their shared evaluator source remains frozen. Scheduled one-shot16562630 at
13:21 Europe/Berlin captures the running evaluations and preparation logs.
Next: inspect full joint/latent-repeat results, then add explicit truth-parent-only
evaluation support and evaluate the frozen candidates on cases200--239.
The primary measured0.6-arcsec goal is not established.

## 2026-09-18 — Verify independent scene range and prepare parents without g0 anchors

Inventory16562638 confirms by generated positions/IDs that g005 cases140--199
are the same scenes as training, while200--239 are distinct. The initial
submission's completed-job dependency was rejected after Slurm purged that
handle; accounting confirmed the predecessor completed successfully before
resubmission. No measured response was opened.

Added `prepare_independent_truth_parent.py`, two tests, and pilot/full launchers.
The builder selects generated secondary role and strict original-precision r<26,
reports unrendered IDs, constructs intrinsic contexts from raw truth, and checks
crowding against an independent historical cache. It writes no g0 measurements
or masks. Pilot16562654 completed34s:110,640/111,263 parents and crowding agreement
within2.4e-7. Full40-case preparation16562670 is submitted. A truth-parent-only
evaluator mode remains to be implemented after the active source-hashed runs
finish; no incompatible independent evaluation is submitted now.

Added `audit_independent_measurement_pilot.py` and dependent CPU job16562659.
It checks exact non-shape columns, hashes and output/fit alignment, comparing
old/new per-leg moments using independent usability and the declared0.6-arcsec
cut. Reuses the tested paired-marginal influence calculation; no calibration
response offset or matched-success-only population is introduced. See
[independent parent preparation](JOINT_CALIBRATION.md#independent-scene-overlap-and-anchor-free-parent-preparation).

Validation: **424 passed,2 skipped** in30.74s (existing NumPy warning), targeted
Ruff, shell syntax and diff checks pass. Both frozen GPU evaluations and the
two CPU measurement pilots continue normally. Next: inspect pilot marginal
compatibility and complete full-parent preparation, then prepare the disjoint
endpoint with explicit missing-g0 semantics once candidate evidence is ready.
The calibration goal remains active.

Independent-parent16562670 then completed all40 cases:4,437,748 rows, one
unrendered ID dropped, maximum crowding-cache difference2.3842e-7. Both
remeasurement pilots completed15m22s/13m34s; dependent audit16562659 passed36s.
At0.6 arcsec all five moment discrepancies are within1.589 paired-object errors;
the maximum over all declared cuts is2.086. No centering mismatch is detected.
Keep the original existing catalogue for the independent endpoint, without
launching a full40-case remeasurement. This is compatibility evidence, not a
distribution-equivalence or calibration claim. CPU neighbour-pair array16562810
is submitted with two concurrent four-CPU tasks; no GPU or measured-response
evaluation is added. See
[completed pilots](JOINT_CALIBRATION.md#completed-independent-parent-and-measurement-compatibility-pilots).

## 2026-09-18 — Freeze corrected step2 flow; validate density; recover independent g1 scenes

Refinement16561508 completed with step2 selected. Full held-out density loss
increases0.000929921 nat, within the preregistered0.001 budget; held-out guard
loss improves0.498035248 to0.356833213. Further updates fail the density gate.
The selected checkpoint SHA256 is
`fade8aebfb4825344b8e9e36dfc07fce5357649340995670f1bb284df7068bd4`;
80/99 state tensors change while architecture stays fixed. Density/joint smokes
16561633/16561635 pass. Full density16561637 at the primary0.6-arcsec cut shows
small resolved NLL increases0.00012--0.00015 nat acrossg0/+/-0.02. Full joint
16561643 has started. These do not establish the calibration goal. See
[completed refinement](JOINT_CALIBRATION.md#completed-corrected-label-refinement-and-density-validation).

Extended the inventory beyond the main ConstGold root using the archived
cont.633 provenance: a complete separate g1+/-0.05 root covers cases140--239.
New `audit_existing_g005_truth.py` and its CPU launcher verify that cases200--239
are distinct from all200 current training scenes and each other, with exactly
paired truth and expected shear. Job16562378 passes all40 scenes; parent size
4,437,749, one unrendered ID reported. No measured response values were opened.
Added three fingerprint tests. These are an existing historical generalization
endpoint, not fresh multi-axis acceptance; the prior inventory is clarified
as covering the main +/-0.02 store only.

BlendEMU measurement driver now accepts an explicit constant-shear family,
requiring a nonzero antithetic pair and matching generated truth, while keeping
half-shear defaults. Added five layout tests. Frozen package/driver snapshot
and new launcher submit CPU pilot16562406 for cases200/201, reusing existing
images with the training-compatible subpixel/common-optimizer recipe. Both
jobs pass preparation; the old shape timestamps alone cannot certify centering.
No full independent rebuild, image production or new model training is launched.
See [independent endpoint](JOINT_CALIBRATION.md#existing-independent-g1-endpoint-recovered-from-the-archive).

Validation: SBSI **422 passed,2 skipped** in31.98s; BlendEMU **14 passed**
in14.44s; targeted Ruff, shell syntax and diff checks pass. Next: follow full
joint and latent-repeat results, verify pilot measurement compatibility, then
prepare an explicitly truth-parent-only independent endpoint without inventing
a g0 anchor. Two GPU jobs remain the combined maximum. Goal stays active.
Both pilot jobs subsequently passed6,144 rows; one caught optimizer KeyError is
retained as an invalid fit, not a job failure. Full joint16561643 passed its
second case with empty stderr. One-shot capture16562630 replaces16561988 and
includes the independent CPU audit/pilot plus both GPU runs; it records logs,
not a task wakeup. No long job is being held open for continuous polling.

## 2026-09-18 — Separate fresh acceptance from inspected development scenes

The first corrected-label update16561508 passes held-out selection: guard loss
0.498035248 to0.396906241 and subset NLL -4.772255463 to-4.771760186
(+0.000495277 nat). Step1 is provisionally selected; optimization and the final
full-density gate remain in progress. This is a response/density validation
result, not the final joint bias.

While the two GPU jobs run, inventoried available raw-data directories: ConstGold
has cases0--139, entirely inside the main flow-development scene range0--199.
Recorded the fitting/validation overlap in
[fresh data inventory](JOINT_CALIBRATION.md#fresh-acceptance-data-inventory).
Existing scenes therefore remain development checks, not fresh multi-axis
acceptance. Reviewed the BlendEMU simulation entry point; no simulation was
submitted without an explicitly frozen scene and measurement recipe.

`doc/INFERENCE.md` now labels its0.75-arcsec acceptance population as the
historical V3.5-like release criterion and links the separate active owner goal
at measured >0.60 arcsec. No configured likelihood cut changes. Validation is
metadata inspection and clean `git diff --check`; no code or model recipe was
changed. Next action remains completion of the preregistered refinement and
frozen evaluations, followed by fresh acceptance if those results support it.

## 2026-09-18 — Resolve the corrected self-response target shift

Reference16556888 completed all four integral groups on15,630,719 training
pairs and is now testing gradient replicas. At the soft3-pixel/25.8 guard,
corrected measured R11/R22 are0.525975189/0.526499356, versus original
0.536498040/0.538172486. Frozen predictions barely change:
0.537989240/0.538940206 on the corrected population. Four-group integration
errors are0.000041864/0.000018369, excluding measurement/case uncertainty.
Wrote `target_comparison.json` with all16 cuts and both source hashes. This
establishes a materially different training target after harmonizing centering;
it is not a response offset or a prediction of the hard-cut joint m. See
[training target](JOINT_CALIBRATION.md#corrected-training-response-target-and-frozen-reference).

The preceding turn launched an independent integration repeat and passed419
tests. Both GPU jobs remain live and healthy; repeat16561899 has reached its
second case. No model update, new calibration claim or target change is made
in this entry. Gradient-bank agreement and held-out response/density selection
remain prerequisites for the queued refinement. Documentation diff checks pass.

Reference16556888 subsequently completed10m25s. Independent reference-gradient
cosine is0.999999523, comfortably passing the0.99 gate. RMS gradient noise
falls48.7901 to4.2286 (11.54x); relative noise falls25.22% to2.15%.
Refinement16561508 has started successfully and is loading its training data.
This establishes a stable update direction, not generalization or calibration.
Refinement then passed initial full-population validation with finite baseline
subset NLL=-4.772255463 and guard loss0.498035248, empty stderr, and began
updates. This is first meaningful progress for the dependent long stage;
one-shot capture16561988 remains scheduled. The repeat evaluator hash matches
the original baseline, and its8,712 block seeds are unique and disjoint from
all8,712 original block seeds; recorded in its execution manifest.

## 2026-09-18 — Launch independent integration repeat of the frozen disk baseline

The preceding continuation was a verified wait on healthy reference16556888.
The existing joint result uses one64-draw latent stream, leaving numerical
repeatability unresolved at the requested0.3% scale. Added
`job_audit_joint_latent_repeat.sh`: same frozen flow/emulator/classifier,
80 scenes,64 antithetic draws,1024-object batches and+/-0.02 CRN extraction,
with seed7302 instead of7301. Job16561899 has passed25,600 rows of its first
case with empty stderr. This is repeated integration of one model, not a flow
ensemble; neither stream will be selected for a favorable calibration number.

`compare_truth_parent_response.py` now has explicit independent-latent-repeat
mode. It requires different seeds, otherwise identical model/evaluator/data
provenance, unchanged response composition and pair provenance, and exact
measured sufficient-statistic equality. Ordinary model comparison retains its
same-stream requirement. The repeat reports delta m with paired-case errors;
two streams do not bound finite-draw bias or training-seed uncertainty. Added
regression checks reject same-seed repeats and changed flow/response artifacts.
See [precision protocol](JOINT_CALIBRATION.md#independent-latent-stream-precision-check).

Reference16556888 and repeat16561899 occupy the two permitted GPUs. Refinement
and candidate diagnostics form one serialized chain behind the reference; they
cannot add a third GPU alongside the repeat. One-shot capture16561988 replaces
16561663 and includes both chains. Measurement and model selection remain
unchanged; calibration goal is active and no new m is reported yet.

Validation: **419 passed,2 skipped** in29.05s, one existing NumPy warning;
Ruff, launcher syntax and diff checks pass. Next: inspect the independent
reference direction, monitor refinement startup, and assess repeat differences
before interpreting any candidate near0.3% on the measured >0.60-arcsec cut.

## 2026-09-18 — Add frozen corrected-flow density and joint evaluation chain

Extended `scripts/audit_disk_candidate_density.py` with explicit candidate and
reference flow arguments. Both transported models now score identical observed
outputs with their own preprocessing and physical Jacobians; paired differences
isolate a flow change. The default remains transport versus its own self-flow.
Restricted cut scores are explicitly unnormalized because changing the flow can
change selected mass. This does not replace selected-density normalization or
joint-distribution acceptance.

New `jobs/job_evaluate_harmonized_candidate.sh` validates the selected checkpoint
hash and full-validation density gate, and skips redundant evaluation if step
zero wins. Identity job16561604 completed: every paired density difference and
its error is exactly zero for the same flow on both sides across two cases,
three shear legs and three cuts. Its self-flow/transported NLL values also match
the historical density smoke exactly (maximum difference0). Candidate jobs
16561633(density smoke),16561635(joint smoke),16561637(full density), and
16561643(full joint) run serially after refinement16561508 and the identity
check. Frozen joint comparison uses the existing disk model as baseline and
reports the measured >0.60-arcsec actual-U population as primary. See
[frozen evaluation protocol](JOINT_CALIBRATION.md#frozen-corrected-flow-evaluation-protocol).

The preceding goal turn made concrete progress by repairing preparation and
submitting the preregistered corrected-label refinement. Current authoritative
checks confirm preparation16561334 is still running normally, past case170,
with no stderr. Dependent density/reference/refinement jobs remain queued;
there is no new candidate or calibration improvement to claim yet. The
identity check and serialized science chain require at most two GPUs together.

Validation: **417 passed,2 skipped** in27.88s, one existing NumPy warning;
Ruff, launcher syntax and diff checks pass. Next: finish full preparation,
inspect density/reference results and each dependent stage at startup, then
assess the frozen candidate if validation selects a change. Goal remains active.

Preparation16561334 subsequently completed13m05s with empty stderr. It reports
six unrendered parent identities dropped; 22,192,875 parent rows per leg;
19,601,281/19,597,477 valid g0/g05 rows. Compared with old validity,869/884
new rows are retained and893/907 old-only rows dropped; all shared-row checks
and the160/40 split pass. Density16556887 has passed its first60 cases without
stderr. Replacement one-shot capture16561663 supersedes16561512 and includes
the entire candidate evaluation chain. No unchanged status wakeup is requested.

Full density16556887 subsequently completed1m52s. Held-out paired new-minus-old
NLL is +0.0004301 +/- 0.0004487 nat atg0 and -0.0142782 +/- 0.0005131 atg05
(40 cases; shared-row weighting,10,000 case resamples, seed20260918). Training
changes agree with this pattern. `full_density_summary.json` records the source
hash and bootstrap intervals. This establishes compatibility of the corrected
g0 measurements and the g05 centering effect, not a learned-density gain.
Reference16556888 passed its first2,050,048 of15,630,719 training pairs with
empty stderr; refinement and frozen evaluation remain dependent. The long audit
is now progressing normally, so leave it to run with one-shot check16561663.

## 2026-09-18 — Fix primary target at measured 0.6 arcsec; finish joint and amplitude checks

The owner specifies measured FLUX_RADIUS >0.6 arcsec (3 pixels), retaining
MAG_AUTO<25.8, as the primary calibration population. Updated
[goal and population](JOINT_CALIBRATION.md#goal-and-retained-population).
Full disk-transport joint job16556994 completed all80 cases: actual-U
m=-0.623079 +/- 0.264152% on this population, versus original -1.127809%.
Modeled-U m=-1.619293% is a separate diagnostic. Physical-density NLL changes
at this cut are +0.00004479 +/- 0.00002016 and +0.00016627 +/- 0.00002559 nat
for +/-0.02. Neither density improvement nor the 0.3% goal is established.

Primary measurement16557008 and marginal audit16557012 completed. Added
BlendEMU `prepare_harmonized_response_case.py` and optional bounded workers /
secondary identities in `blendemu.response.retrieve_response` (defaults preserved).
Fresh amplitude preparation16557030/16557033 uses each new leg's validity,
not old accepted response rows. New SBSI `audit_harmonized_response_pairs.py`
checks exact pair/source alignment and raw photometric cuts, clustering repeated
neighbour projections by primary. Audit16557060 completed all four pilot cases:
h=0.02 minus h=0.2 response at0.6 arcsec averages +0.010033, conditional
primary-cluster SE0.012660; no clear amplitude shift. Small-amplitude labels
remain about20 times noisier. No full primary rebuild or pilot fit is submitted.
Two launcher submissions16557053/16557054 were cancelled after detecting missing
shell `esac`; corrected smoke16557059 and full16557060 passed. See
[fresh amplitude comparison](JOINT_CALIBRATION.md#completed-fresh-primary-amplitude-comparison).

All200 secondary remeasurement cases completed. Full preparation16556795
failed on role membership because rendering can omit rows, invalidating the
half-rendered-row-count boundary. `prepare_harmonized_flow_domain.py` now
reconstructs membership from the hash-verified generated catalogue, requires
exact agreement with the measured parent, then joins rendered IDs and reports
missing identities. A regression test covers omitted rendered rows and rejects
wrong measured parents. Preserved the empty failed output; retry16561334 has
passed its first70 cases with empty stderr. Density16556887 depends on retry,
then independent-reference16556888 follows. See
[boundary repair](JOINT_CALIBRATION.md#full-domain-generated-role-boundary-repair).

Preregistered the unchanged six-step density-gated refinement on corrected
labels/reference means, without ConstGold checkpoint selection. New launcher
`job_refine_harmonized_guard_flow.sh` submitted16561508 after reference success;
its existing code requires gradient-bank cosine>=0.99 and verifies provenance.
The entire GPU chain is serialized and stays below the two-GPU cap. No refined
candidate exists yet. Next: inspect full domain/density and reference checks,
monitor dependent refinement startup, and evaluate a changed frozen checkpoint
on the primary population only after validation gates. One-shot Slurm status/log
capture16561512 is scheduled in two hours; it records results, not a task wakeup.

Validation: SBSI **417 passed, 2 skipped** in40.37s with one existing NumPy
warning; BlendEMU **9 passed** in1.80s. Targeted Ruff, launcher syntax, and
`git diff --check` pass. Execution manifests and monitoring result discovery
were updated. No configured likelihood or model was promoted.

## 2026-09-18 — Complete native transport validation and start full joint/density checks

The new disk-transport fit and all500 native cases completed. At>=0.75arcsec,
predicted R11=0.170402653 versus measured0.180160094; paired deficit
0.009757441 +/- 0.002299357 remains. Its within-selection component is
0.008297211 +/- 0.002138802. Finite versus linear transport differs by only
0.000020498. A valid shape density therefore does not by itself remove the
response mismatch. See
[completed validation](JOINT_CALIBRATION.md#completed-physical-moment-transport-validation).

Joint smoke16556840 passed in3m51s: all108 checked baseline flow/denominator
arrays are identical, no shape exceeds the disk by more than1e-12, and the
largest squared-norm excess is1.11e-15. Boundary rounding remains reported,
not clipped. The two-case actual-U m=+0.69803 +/- 0.49201% at>=0.75arcsec
is not acceptance.

Density smoke16556926 exposed a nonexistent separate ConstGold g0 directory.
The audit now uses original g0 domain targets with the exact hash recorded
by the truth parent, checks membership and records that reuse. Retry16556983
still ran old source, consistent with filesystem cache delay; immutable
hash-named source retry16556987 passed. Exact zero-shear density identity
holds; nonzero paired NLL changes at>=0.75arcsec are about+0.00025/+0.00030
nat per row on the two smoke cases. No nonfinite score is silently dropped.

Full density16556993 and joint16556994 are submitted for80 ConstGold cases.
Updated harmonized density16556887 to require both preparation16556795
success and completion of16556993; reference16556888 follows it. This makes
the combined two-GPU bound explicit across the chains. Full native
array16556956 and summary16556957 finished successfully.

Validation: **414 passed,2 skipped** in29.08s (one existing NumPy warning),
Ruff and diff checks pass; audit source-path fix was subsequently exercised
by the successful scheduled smoke. The calibration goal remains active.
Full density16556993 subsequently completed4m09s, with exact g=0 identity
on all80 cases. At>=0.75arcsec NLL worsens by0.00009112 +/- 0.00002588
nat at+0.02 and0.00022934 +/- 0.00002885 at-0.02. These small but resolved
regressions are below the earlier0.001nat budget, not a density improvement.
One-shot scheduler/log snapshot16556997 replaces pending16556798, scheduled
two hours after this update and covering the full measurement/preparation,
harmonized audits, and full disk density/joint jobs. Both full evaluation
jobs passed startup and meaningful progress with empty stderr.

The surviving native deficit motivates a clean primary amplitude test:
BlendEMU array16557008 remeasures cases1/40/41 at0/0.02/0.2 with subpixel
centering and common optimizer seeds, using the frozen secondary-rebuild
package/driver. It adds three eight-CPU jobs, no GPU. SBSI audit16557012
will compare original and same-seed geometric measurements, per-leg
marginals, successes and label variance. Source catalogues stay unchanged,
and no full primary rebuild or pilot-trained model is launched. New
launchers pass shell checks. See
[primary amplitude pilot](JOINT_CALIBRATION.md#matched-centering-primary-amplitude-pilot-after-transport-validation).
Final one-shot snapshot16557018 replaces16556997 to include the primary
measurement/audit chain as well as the full joint and harmonized flow jobs.
It is scheduled two hours after the final update.

## 2026-09-18 — Fit a physical-moment response transport with explicit artifact units

Implemented BlendEMU preparation and training of a disk-transport coefficient
on the existing161,012,399 primary-role response pairs. The positive physical
gain enters a conditional moment objective, with analytic gradient/Hessian;
the coefficient is not an old additive response or an average of inverted
labels. Case split, trial9 settings and218 trees are frozen. Preparation
16556819 completed5m16s with all pairs retained; CPU training16556820 is live.
Three BlendEMU tests pass, including a fitted-tree physical-moment check.

SBSI `output_conditioned_response.py` now has a distinct `DiskResponseMoment`
loader that validates artifact format and units. The experimental ConstGold
evaluator supports explicit Mobius composition, preserving baseline flow
draws/denominators and reporting numerical support failures. The independent
native audit compares finite transport and linear physical derivative against
R11/R21. Dependent smoke jobs16556840(GPU) and16556841(CPU) precede full
evaluation. No configured likelihood changes. Details and limitations:
[physical-moment experiment](JOINT_CALIBRATION.md#physical-moment-transport-fit-and-independent-checks).

Validation: full SBSI suite **412 passed,2 skipped** in29.30s, one existing
NumPy warning. Subsequently added physical log-density evaluation through
the inverse transport; observed-disk quadrature verifies normalization and
the unchanged auxiliary marginal to1e-12, and invalid outputs receive -inf.
All six transport tests pass (two added), with Ruff and diff checks clean.
The harmonized
secondary measurement array16556786 continues on CPUs with dependent
preparation16556795. The0.3% objective remains active and unachieved.

Queued full-domain frozen-density audit16556887 after preparation, then
reference/gradient audit16556888 after density. The density script now
accepts explicit cases/domain/output while retaining its pilot defaults.
The new launcher serializes these one-GPU stages; old-domain response
integrals are not reused for harmonized measurements. These stages perform
no optimization and preserve the combined two-GPU cap.

Added `audit_disk_candidate_density.py` and a launcher for observed g=0 and
+/-0.02 physical density, comparing the transported model to the same
self-response flow. The audit hashes sources, verifies exact zero-shear
identity, reports per-cut paired changes with case bootstrap errors, and
never drops nonfinite scored densities silently. Density smoke16556926
follows joint smoke16556840, sharing its GPU allocation chain.

Training16556820 subsequently completed14m44s, MaxRSS11.14GiB; fixed218-tree
artifact SHA256 `194c1d11b67d6fd52039060270663e8e1cd610b478b9e9a66e34271701761fe7`.
Development pair-mean prediction/label at>=0.75arcsec is0.01140667/0.01147197;
these correlated pair diagnostics are not a catalogue bias estimate.
Native smoke16556841 passed; finite-minus-linear R11 differs by only
about-2.2e-5 at>=0.75arcsec. Full native array16556956 and summary16556957
are submitted, using at most eight four-CPU tasks concurrently.

## 2026-09-18 — Start the harmonized full training-data rebuild and test a physical response transport

The three-case secondary pilot and all dependent stages completed:
measurement16556705/16556736, audit16556737(30s), preparation16556746(45s),
and frozen-flow density16556752(16s). g=0 first/second moments agree with
originals within1.85 paired object errors. g=0.05 uncut e2 shifts by
-0.00328 to -0.00398 (5.5--6.8 errors), confirming inconsistent centering in
the current flow's training legs. Shared optimizer seeds reduce label variance
by63--70%. g=0 paired NLL differences remain consistent with zero; g=0.05
NLL decreases by0.0109--0.0161nat per row under the same frozen flow. This
is a measurement compatibility check, not a trained-model improvement or
calibration claim. Preparation retains33 new-only and drops15 old-only valid
leg rows. See [completed pilot](JOINT_CALIBRATION.md#completed-secondary-pilot-and-full-rebuild).

Full measurement array16556786 now rebuilds197 remaining cases, reusing
validated cases0/40/41 by links. It runs eight eight-CPU jobs concurrently,
no GPU, against a frozen copy of the BlendEMU package and measurement driver
under `blendemu_runs/harmonized_secondary_20260917_v1/code_snapshot/`.
Source hashes match the pilot, package versions are recorded, and original
images/catalogues remain untouched. All eight initial jobs passed6,144--8,192
targets with empty stderr; counted fit exceptions remain invalid rows.
Dependent preparation16556795 will produce the full200-case flow domain
with the original grouped split and independent per-leg validity.

Added experimental `sbsi/disk_response_transport.py`: a disk-preserving
Möbius map with bounded shift, exact inverse and density Jacobian, plus its
local physical-response matrix. Four tests cover inversion, shape support,
zero identity, derivatives and the full four-output Jacobian when the shift
depends on auxiliary outputs. Existing physical response coefficients are
not substituted into this map. A conditional physical-moment fitting
objective is derived in the documentation; fitting and joint evaluation
remain to be done. See
[transport derivation](JOINT_CALIBRATION.md#experimental-disk-preserving-response-transport).

Validation: full SBSI suite **412 passed,2 skipped** in30.13s (one existing
NumPy warning); Ruff, shell checks and `git diff --check` pass. The conditional
joint comparison completed with actual-U m=+0.71947% at0.75arcsec, not a0.3%
acceptance. No GPU job remains live after the density audit. Updated one-shot
snapshot16556798 replaces16556726 and runs at2026-09-18 02:02:07 local,
covering the full rebuild, preparation and completed measurement count.
The goal remains active and unachieved; no model is promoted.

## 2026-09-17 — Prepare a full-parent path from harmonized measurements to flow training

Conditional joint arm16556472_1 subsequently completed1h04m01. Actual-U
full-parent m improves to -1.96343 +/- 0.26988% uncut,
-0.62183 +/- 0.26426% at >0.60arcsec and +0.71947 +/- 0.21885% at >=0.75arcsec.
The paired changes are +1.48732 +/- 0.00557, +0.50598 +/- 0.00332 and
-0.37852 +/- 0.00316 percentage points. The checkpoint hash and all162
inspected flow-numerator/denominator arrays match baseline exactly.
Modeled-U m at >=0.75arcsec is -0.27191%, but actual-U and physical-support
checks still fail the objective; no model is promoted. See
[joint candidate result](JOINT_CALIBRATION.md#completed-output-conditioned-joint-comparison).

The previous turn made progress: it established mixed measurement centering
and launched actual remeasurement. Live scheduler checks confirm the paired
flow pilot and conditional joint evaluation continue. Checking the authoritative
flow split caught an important distinction: cases0/40 are both training cases
under seed501, although case0 is development for the emulator. Added held-out
flow case41 as job16556736_2, now beyond20,480 targets with empty stderr.
Cancelled pending audit16556719 and replaced it by16556737, dependent on all
three measurement cases. See
[secondary pilot](JOINT_CALIBRATION.md#matched-centering-secondary-training-pilot).

New `scripts/prepare_harmonized_flow_domain.py` builds conditions from raw truth
and the existing truth-only crowd catalogue, determines new measurement
validity independently in each leg, and preserves the original global case
split. It does not merely overwrite the old valid-row cache. It checks parent
identity, invariant truth, shear amplitude, measurement source hashes, unchanged
photometry and compatible conditions on old/new common rows; it reports old-only
and newly successful rows. Full-parent usability arrays are retained separately
for the declared true-r<26 population. Two tests verify distinct per-leg success
sets and physical validity without a science cut. Pilot preparation16556746
waits for the measurement audit.

New `scripts/audit_harmonized_flow_density.py` and dependent GPU job16556752
will evaluate the same frozen flow on original and new targets, reporting
independent-population NLL and paired common-row changes by case and leg.
This directly checks compatibility with the existing density; it does not
fit the pilot or claim a model improvement. This chain adds at most one GPU
alongside the existing conditional joint run, preserving the two-GPU total.

Full-data measurement and preparation launchers are prepared but not submitted.
They preserve the same0/0.05 measurement contract and case split; the CPU array
would use eight jobs of eight workers, with validated pilot cases reused and
measurement source frozen. The filesystem has7.3TiB available. Full-scale
execution still awaits pilot evidence.

Validation: full SBSI suite **408 passed,2 skipped** in33.45s with one existing
NumPy warning; Ruff, shell syntax and `git diff --check` pass. No configured
model, population cut or response scale changed. The0.3% objective remains
active and unachieved.

## 2026-09-17 — Confirm mixed centering and launch a harmonized self-response pilot

Subpixel case0 pilot16556618 and audit16556648 completed20m26/17s. Original
g=0/0.2 first/second moments agree within1.86 paired object errors across all
declared cuts. The same recipe changes g=0.02 uncut e2 by -0.002431
(4.03 errors) and its second moments by -0.001146/-0.001387 (5.45/6.51 errors).
Together with geometric-pilot results and the archived July remeasurement
record, this confirms mixed centering in the random-neighbour source family.
The original-leg amplitude differences cannot be interpreted as pure shear
step effects. See [centering investigation](JOINT_CALIBRATION.md#measurement-centering-investigation).

All seven sampled fingerprint jobs16556668 and summary16556685 completed.
Neither the full samples nor fixed bright strata discriminate centres by
pointwise squared error: differences are within1.32 paired object errors.
Optimizer variability dominates this small-sample metric. This negative
result does not identify the native or secondary catalogues' centering;
the larger marginal audit and historical provenance carry that evidence.
Native case400's shape product is dated August12. No blanket claim that all
native catalogues use the current geometric default is justified.

New BlendEMU `scripts/remeasure_common_optimizer_targets.py` generalizes the
validated driver to explicit structural role and amplitude lists. Pilot
16556705_0/1 remeasures complete true-r<26 secondary populations in cases0/40
at0/0.05 with subpixel centering and shared optimizer seeds. Both jobs passed
startup and4,096 targets with empty stderr. Source catalogues remain untouched;
outputs and source snapshots live in
`blendemu_runs/common_optimizer_secondary_pilot_20260917_v1/`.
New SBSI `scripts/audit_secondary_measurement_pilot.py`, dependent16556719,
will check full-parent marginals, exact photometry, parent membership, paired
2x2 self-response labels and variance. Full training-set replacement and
retraining await this diagnostic; see
[secondary pilot](JOINT_CALIBRATION.md#matched-centering-secondary-training-pilot).

Validation: existing full SBSI suite406passed/2skipped; subsequent provenance
job16556676 matched391,435 current cached targets to raw shapes exactly,
and new scripts/launchers pass Ruff and shell checks. The four-case marginal
audit tests remain passing. One GPU is active for conditional joint arm16556472_1;
the two new measurement jobs use CPU only. Updated one-shot snapshot16556726
replaces16556505 at the same 2026-09-18 00:50:37 local time, now including all
new job IDs and result availability. No model is promoted and the0.3% goal
remains unachieved.

## 2026-09-17 — Find a measurement-provenance mismatch and nonphysical response tails

Completed geometric-centre common-RNG pilot16556489_0--3 and audit16556497.
Exact 512-object replay passes for cases0/1, and h=0.02 pair-label variance
falls by about 86--94%. However, per-leg marginal distributions shift in
g=0/0.2 and largely agree at g=0.02. This fails the measurement-replacement
gate; the earlier original-leg amplitude comparison is now explicitly
qualified as potentially confounded by measurement provenance. See
[centering investigation](JOINT_CALIBRATION.md#measurement-centering-investigation).

Further archive inspection found explicit July15 records of subpixel
remeasurement for ConstGold and the main g=0 primary/secondary catalogues,
while nonzero secondary legs remained older. New SBSI provenance script and
CPU job16556676 completed22s: all391,435 targets in current flow-domain
cases0/1 at0/0.05 match those raw secondary measurements exactly, with no
unmatched rows. The diagnostic is therefore relevant to the current model.
The archive's old response-rescaling proposals are not adopted.

Historical BlendEMU commit46107b8 records an explicit subpixel Jacobian centre,
whereas the current code used a geometric stamp centre. Restored optional
`gal_cen` in `blendemu/shape.py` and added a `--centering` choice to
`scripts/remeasure_common_optimizer_primaries.py`; geometric remains the
default. Separate case0 subpixel pilot16556618 has passed 69,632 rows with
empty stderr and counted fit exceptions. Dependent SBSI audit16556648 verifies
the identical image/position/seed sources and compares old, geometric and
subpixel marginal distributions. Both pilots retain their source snapshots.
`doc/CONVENTIONS.md` now requires centering and optimizer-stream provenance
for every measurement comparison.

New BlendEMU `scripts/audit_measurement_centering.py` and launcher16556668
sample 4,096 previously usable true-r<26 detections per catalogue, with two
optimizer replicas shared between geometric/subpixel fits. Seven CPU tasks
(two concurrent) cover secondary g=0/0.05 in cases0/1, ConstGold case40 +0.02,
and native case400 +/-0.02. They report paired pointwise discrepancies,
optimizer variability and all dropped/failed rows. This is a provenance
fingerprint, not a full-parent marginal gate or calibration acceptance.
The first two tasks completed; their full-sample pointwise errors are dominated
by optimizer variability. Dependent summary16556685 uses fixed true-r<24 and
measured-mag<24 diagnostic strata to check more stable bright fits, alongside
the complete sampled population. These cuts do not select a calibration
population or model.

Physical-support audit16556544 failed only when writing its final report:
the shared SHA helper expected a Path but received a string. The helper now
accepts both types, with a binary-content regression test. Retry16556609
completed1m55. About 0.10--0.11% of selected >=0.75arcsec draws leave the disk
after either blend translation. Their conditional-flow R11 contribution is
0.039095 for the candidate and 0.038118 for the original scalar model. These
two-case diagnostics exclude classifier weights and cannot be subtracted as
a correction. A support-preserving density composition and matched training
target are required before promotion. See
[support results](JOINT_CALIBRATION.md#density-identity-and-physical-support-check).

Validation: full SBSI suite **406 passed,2 skipped** (30.87s, one existing
NumPy degrees-of-freedom warning), relevant Ruff checks, shell syntax and
`git diff --check` pass. No configured model, source catalogue, population cut
or response scale changed. Conditional joint arm16556472_1 remains in progress;
the one-shot scheduler/log snapshot16556505 remains scheduled for
2026-09-18 00:50:37 local. The 0.3% goal is unachieved.

## 2026-09-17 — Add marginal-distribution and physical-support gates for conditional response work

The previous turn was substantive progress; current Slurm state confirms both
joint arms and all four remeasurement jobs remain live. The joint truth-only
arm has passed case110; the common-RNG pilot has passed 65,536 primary rows
in case0. These are scheduled computations in progress, not completed goal
evidence.

Extended `scripts/audit_common_optimizer_pilot.py` to check each leg's original
and remeasured marginal distributions on the complete declared truth parent,
using each arm's own fit success. It reports selected usable mass, one-sided
success disagreements, first/second shape moments and paired object influence
errors. It verifies that original and new radius/magnitude columns agree
exactly. Common-success restrictions remain confined to the separate label
variance calculation. Two tests check different success populations and
exact paired cancellation; the expanded audit will be loaded by pending
job16556497. Small-probe mean-shape differences are not clearly inconsistent
with optimizer noise, but do not establish marginal equivalence. See
[marginal gate](JOINT_CALIBRATION.md#per-leg-marginal-gate-before-training-label-replacement).

Documented the exact inverse and volume preservation of the output-conditioned
shape translation. Its g=0 density is the original single-flow density; fixed
radius/flux selection mass is unchanged. Normalization over translated support
does not prove preservation of the physical ellipticity disk or calibration.
New `scripts/audit_conditional_response_support.py` and its launcher compare
flow-only, old scalar and conditional shifts on the same two-scene,64-draw
sample and all three cuts. Job16556544 waits for truth-only job16556472_0,
preserving the total two-GPU limit. It records existing/newly invalid shape
mass with no clipping or renormalization. This is a conditional-flow sampling
diagnostic, not a classifier-weighted population bound. See
[density identity](JOINT_CALIBRATION.md#density-identity-and-physical-support-check).

Validation: full SBSI suite **405 passed,2 skipped**, with the same existing
NumPy warning; Ruff and shell syntax pass. The existing one-shot check16556505
remains scheduled for 2026-09-18 00:50:37 local and includes the common log root.
The goal remains active and unachieved; no configured model or measurement
catalogue has been replaced.

Joint truth-only control16556472_0 subsequently completed21m23 with its paired
comparison. Actual-U full-parent m is -2.69186 +/- 0.26761% uncut,
-0.59811 +/- 0.26415% at >0.60arcsec and +1.56819 +/- 0.22039% at >=0.75arcsec.
The corresponding paired changes are +0.75889 +/- 0.00340,
+0.52970 +/- 0.00270 and +0.47020 +/- 0.00249 percentage points. The full
flow-numerator and denominator arrays match baseline exactly (maximum
difference0); the checkpoint hash also matches. The >0.75 selection therefore
worsens even as other cuts improve. See
[joint control](JOINT_CALIBRATION.md#completed-truth-only-joint-control).
Dependent support audit16556544 started successfully, passed its first25,600
case40 rows with empty stderr, and is the second GPU alongside the remaining
conditional joint arm. The four CPU measurement jobs continue normally.

## 2026-09-17 — Shared measurement seeds remove most optimizer noise in the small-shear probe

The completed 200-case amplitude audit passed exact intrinsic-scene and shear
checks, but its 0.02-minus-0.2 response differences were noisy and inconsistent
between fitting and development cases. At the >=0.75 arcsec cut they are
+0.015916 +/- 0.007217 and -0.010758 +/- 0.012612, respectively. This does not
justify replacing labels by an amplitude correction. Details:
[amplitude results](JOINT_CALIBRATION.md#completed-amplitude-audit).

Inspection found that `blendemu/shape.py::ngmix_psf_correct` creates an
independent optimizer RNG on every call. Added an optional explicit RNG while
preserving the historical default. New BlendEMU script
`scripts/audit_measurement_seed_response.py` remeasured 512 objects in each of
cases 0 and 1 on existing 0/0.02/0.2 images, with two independent optimizer
replicas coupled across shear legs. Jobs16556468_0/1 completed in 2m12/1m53;
all 1,024 sampled objects remained usable, no fit exception occurred, and
identical-seed replay passed. At h=0.02, estimated common/independent optimizer
variance ratios are 0.0056--0.0447 across the two shape components and cases,
or about 95--99% reduction. This is a noise diagnosis, not a response correction
or a calibration result.

The new BlendEMU script `scripts/remeasure_common_optimizer_primaries.py`
and its launcher expand to complete true-r<26 primary populations in cases
0,1,40,41 at 0/0.02/0.2. Job array16556489 uses four jobs with eight CPU workers each,
no GPU. It retains original images and per-leg positions, never gates work
on an old fitted shape, and writes separate catalogues and fit-status records
under `blendemu_runs/common_optimizer_primary_pilot_20260917_v1/`. Source
catalogues remain untouched. All four jobs passed startup and initial fitting
progress; fit exceptions are recorded as invalid measurements. Dependent
SBSI audit16556497 (`scripts/audit_common_optimizer_pilot.py`) checks exact
replay of the earlier probe, common-row label variance and paired amplitude
differences. A changed both-leg-usable subset must be accounted for before
new labels are used in training. See
[probe and pilot](JOINT_CALIBRATION.md#completed-optimizer-probe-and-expanded-pilot).

Added optional CUDA prediction to the experimental response loader and joint
diagnostic. `scripts/benchmark_output_response_prediction.py` verified the
frozen trees on 171,608 feature rows: CPU/GPU maximum standardized difference
2.98e-8. Ordinary DMatrix prediction works on virtual A40 hardware. The
benchmark ran inside the existing allocation; queued standalone16556442 was
cancelled. GPU joint smoke16556459 completed both cases40/41 in 2m08 with empty
stderr. Redundant CPU smoke16556391 was cancelled after its first case took
about fourteen minutes. No model, draw convention or response scaling changed.

Full joint A/B16556472_0/1 is progressing normally on the same80 truth-parent
cases and64 common draws, followed by paired comparisons with the original
baseline. Both retain the original single flow. Together these are the only
two active GPUs. The truth-only arm is a population/training control; no arm
is selected on ConstGold. The approximately0.3% joint acceptance goal remains
unachieved. Next: assess full joint responses and the independently specified
measurement pilot before deciding whether to rebuild small-shear labels.

Validation: full SBSI suite after CUDA integration **403 passed,2 skipped**,
with the existing NumPy warning. The new measurement APIs were checked by
the scheduled real-image reproducibility probe. Ruff and launcher syntax
checks pass; final checks after adding the dependent pilot audit are recorded
below. CPU default prediction and default ngmix RNG behavior remain unchanged.

Final validation after adding the pilot audit: **403 passed, 2 skipped**, with
the same existing NumPy warning; Ruff, shell syntax and diff-whitespace checks
pass. All four measurement jobs have passed 4,096 fitted parent rows, and
both joint arms have passed multiple cases with empty stderr. One-shot
status/log capture16556505 replaces16556226 and is eligible at
**2026-09-18 00:50:37 local time**. It includes the active joint and measurement
jobs and their dependent comparisons. The pilot's returned fit exceptions
are explicitly counted as invalid outputs, not silently accepted.

## 2026-09-17 — Conditional emulator improves native response partially; joint and amplitude tests follow

The six-step density-gated CV flow refinement16556200 completed16m03 and
selected step0. Selected full-validation NLL is unchanged at-4.76270675245;
dependent16556221 correctly skipped a duplicate joint evaluation. Full-training
case audit16556249 completed4m38 on15,630,658 pairs. Training/validation
self-response residual differences on the displayed cuts are only about
0.35--2.11 componentwise case standard errors; the training improvement does
not establish a correction that generalizes. Added `--split` and optional
reference subset to `scripts/audit_guard_population_score.py`. Details and
fixed-nuisance uncertainty limitations:
[completed refinement](JOINT_CALIBRATION.md#completed-refinement-and-case-uncertainty).

CPU emulator training16556189 completed both218-tree arms. The native500-case
array16556222 and summary16556225 completed. At the>=0.75arcsec cut, conditional
prediction0.17026532 improves on the original0.16578059 and truth-only
control0.16375420, but measured total is0.18016009. The remaining total
residual is+0.00989477+/-0.00229968; after separating cut motion, the residual
is+0.00843455+/-0.00213917. Neither is a successful joint calibration claim.
See [native comparison](JOINT_CALIBRATION.md#completed-independent-native-validation).

`scripts/audit_response_label_tensor.py`, a scheduler launcher and synthetic
matrix-recovery test examine information lost by scalar labels. Job16556325
completed the frozen161m-pair data. Selected R11-minus-scalar is only
+0.00033616+/-0.00085898 fitting and+0.00045716+/-0.00145315 development.
The scalar/tensor difference is not supported as the explanation for the
native deficit. Source outputs and estimands:
[directional audit](JOINT_CALIBRATION.md#completed-directional-label-audit).

`sbsi/output_conditioned_response.py` now computes each primary's pair sum
for each joint flow draw, pooling only exactly equal tree cells. Tests compare
against exhaustive predictions, exercise both output features and isolated
primaries, and check exact one-sided tree limits when float32 softplus
underflows to zero. No physical clipping, interpolation or response scaling
is introduced. The optional candidate path in
`scripts/evaluate_constgold_truth_parent.py` integrates selection and response
on the same draw; a test rejects the product-of-means shortcut. Pair artifacts
from the extended `scripts/build_constgold_fixed_g0_blend_lookup.py` must match
parent/model provenance and reproduce the old response lookup before use.
Preparation array16556348 covers all80 cases. Joint smoke16556391 uses cases40,
41 and64 draws. The full A/B retains the original single flow and declared
population/cuts; it is developmental and leaves configured likelihoods alone.

Found existing 0.02/0.05/0.2 primary measurements for all200 training-family
cases. New `scripts/audit_response_label_amplitude.py` and its launcher compare
all three forward amplitudes on common-usable primaries with identical pairs
and baseline output cuts. It verifies exact non-shear truth fields, scaled
sky shear vectors and reconstruction of original0.2 labels. Smoke16556399
completed; array16556401 and summary16556406 are running/dependent. No new
simulation or training-data replacement was needed. See
[amplitude protocol](JOINT_CALIBRATION.md#experiment-8-matched-forward-amplitude-audit).

Validation so far: full SBSI suite before the final candidate integration
**402 passed,2 skipped**, with the existing NumPy warning; subsequent targeted
candidate/correlation/identity tests pass. Ruff and launcher syntax checks
pass. Full-suite rerun follows the final edits. All substantial computations
are scheduled; the joint smoke is the only GPU job. Joint0.3% acceptance is
still unachieved. Next: finish smoke, run frozen joint A/B, and use the matched
amplitude evidence to decide whether small-shear labels need replacement.

## 2026-09-17 — Density gate holds on first response step; held-out response has not improved

Refinement16556200 reproduced baseline validation NLL-4.759908605 and
guard score0.044834131 with its registered streams/subset. Its first1e-5
gradient step reduced training guard score to0.02497784 while spending only
0.00032080nat/row of the0.001 density budget. Validation NLL changed by
0.000284656nat/row, but guard score worsened to0.060760717. Step0 therefore
remains selected. Reduced gradient noise and controlled density loss are
established; held-out response improvement is not. The six-step protocol
continues without modifying its targets or selection rule.

CPU emulator training16556189 completed the truth-only control's218 trees,
saved its frozen model/development diagnostics, and started the conditional
arm. No tree count or hyperparameter changes were made. Downstream launchers
were resubmitted so their stored Slurm scripts use `models_cpu/`: smoke16556220,
array16556222 and summary16556225. Joint flow diagnostic16556221 replaces
16556204 and additionally runs the new
`scripts/compare_truth_parent_response.py`. That comparison checks exact
population/input/stream agreement and identical measured arrays before paired
case resampling; two tests check common-case cancellation and rejection of
different latent streams. It does not select a checkpoint using joint bias.

One-shot job16556226 supersedes16556208 and captures the current pipelines
after two hours, including emulator and refinement logs/results. The prior
wrappers are cancelled; training jobs continue and only one GPU is active.
All progress remains developmental, with no promotion or joint calibration
claim. The goal remains unchanged.

Startup follow-through: step 2 also satisfies the training density gate
(+0.00056312 nat/row), but validation guard loss rises to 0.070529131 and
step 0 remains selected. The conditional emulator has reached its first tree
on all 128,809,729 fitting pairs with empty stderr. Final validation after
adding the paired comparator: **401 passed, 2 skipped**, with the same existing
NumPy warning; Ruff, shell syntax and diff-whitespace checks pass. The current
scheduled snapshot is eligible at **23:58:04 local time**.

## 2026-09-17 — Reference integration reduces gradient noise; density-gated refinement starts

Audit16556078 completed20m07s,exit0,empty stderr. Ordinary independent-row
gradient RMS noise was46.4149 across eight draws, versus1.9670/1.8988 with two
independently integrated reference banks. Relative noise fell from98.60% to
5.75%/5.58%; bank mean directions have cosine0.9999523. Their mean-gradient
distance is0.37246 at norms34.1946/34.0249. The maximum component integration
SE of the four-group reference mean is4.5644e-5. This establishes substantial
variance reduction at the original checkpoint, not a calibration improvement.
Source: `guard_control_variate_20260917_v1/audit_v2/result.json` and `reference.json`.

Added `scripts/refine_guard_control_variate_flow.py`, a launcher and a density
gate test. Six gradient-descent steps use independent row/latent replicas,
the frozen reference integral, and backtracking from1e-5 through six rates.
Training score must improve while NLL stays within0.001nat/row of baseline;
validation checkpoint selection has the same budget and retains step0.
The selected checkpoint additionally faces full-validation NLL. Job16556200
has passed training/validation data loading. Dependent joint diagnostic16556204
uses the unchanged emulator/classifier on the full80-case truth parent, or
explicitly skips if step0 remains selected. No ConstGold data enter training
or model selection. Details:
[density-gated refinement](JOINT_CALIBRATION.md#completed-variance-audit-and-density-gated-refinement).

The initial output-conditioned emulator job16556129 failed at its first tree:
XGBoost3.0.5 requested CUDA virtual-memory allocation unavailable on the
A40-16Q device. Its partial `models/` directory is preserved. Both A/B arms
now run on CPU in job16556189 (32cores,48G,cluster), with identical full data
and frozen218-tree recipe, writing `models_cpu/`. It has reached its first
training loss with empty stderr. The native smoke dependency16556158 was
repaired to the replacement job; array16556159 and summary16556163 remain
downstream. The CV refinement is the only active GPU job.

Validation: SBSI full suite **399 passed,2 skipped**, with the same existing
NumPy warning; the five new BlendEMU tests pass. Ruff, launcher syntax and
`git diff --check` pass. No model has met the approximately0.3% joint acceptance
goal, and no model is promoted.

## 2026-09-17 — Full-domain conditional-response emulator A/B prepared and launched

Started a controlled emulator experiment to address the independently measured
selected-response deficit without changing flow density or fitting ConstGold.
BlendEMU now contains `scripts/prepare_output_conditioned_response.py` and
`scripts/train_output_conditioned_response.py`, two scheduler launchers and
five focused tests. Preparation16556121 completed the200-case source catalogue:
248,515,320 pair rows,87,487,614 excluded by strict true-r<26,15,307 invalid
response-label rows dropped,161,012,399 retained, and zero unavailable baseline
joins after label validity. Every retained label's g0 shape agrees with its
exact primary-role raw measurement. The160 fitting cases40--199 contain
128,809,729 pairs; cases0--39 remain disjoint development data. The common
label mean/std are0.00811749324/0.99585391949; these are target standardization,
not a response correction. All3,992 prepared parts have hashes.

Training16556129 runs two arms sequentially on one GPU: the original seven
truth/pair features and the same features plus log baseline radius and baseline
magnitude. Both use exactly the same population, ordinary squared error,
frozen trial9 parameters,218 trees and seed20260816. Development data select no
parameters or tree count. Chunked QuantileDMatrix construction limits memory;
the deployed flow is untouched. Both models are explicit experimental
artifacts under `blendemu_runs/output_conditioned_response_20260917_v1/`.

Added SBSI's `sbsi/output_conditioned_response.py` with hash/feature/units
validation and three tests. Extended the native amplitude audit to compare
the two candidates on identical scene objects and cuts. The conditional inputs
use mean plus/minus physical photometry because this instrument lacks g0;
that finite-h proxy remains an acceptance limitation. Smoke16556158 depends
on training, then CPU array16556159 and summary16556163 evaluate500 cases.
The existing reference-flow variance audit16556078 remains the other GPU;
the combined limit is two.

Baseline diagnostic16555762_0 completed100 cases in10m10s with empty stderr.
On the valid-g0 true-r<26 cohort, uncut actual-U R11 bias remains
-3.00552 +/- 0.23160%, matched-U -2.51059 +/- 0.22806%, and modeled-U
-2.21485 +/- 0.23215% (64 antithetic draws,seed7301;10,000 case bootstrap
replicates,seed20260914). This population differs from the full truth-parent
diagnostic. Neither guard continuation changes these predictions because
both selected checkpoints are exactly identical to baseline.

Validation: SBSI full suite **398 passed,2 skipped**, with the existing NumPy
warning; BlendEMU's new focused suite5 passed; Ruff and shell syntax passed.
Protocol, population and remaining limitations are in
[Experiment6](JOINT_CALIBRATION.md#experiment-6-full-domain-output-conditioned-response-emulator).
No approximately0.3% calibration claim or model promotion is made.

## 2026-09-17 — A six-standard-error selected-population emulator deficit

Amplitude array16556028 completed all twenty blocks; summary16556033 completed
in17s, all exit0 with empty stderr. On cases400--899, pooling inverse-sampling
strata within cases and averaging cases equally, 3,441,882 common-usable anchor
rows give native R11 measured-minus-emulator differences:
`+0.0009740 +/- 0.0029603` uncut;
`+0.0032861 +/- 0.0030763` for MAG_AUTO<25.8/radius>0.60arcsec;
`+0.0143795 +/- 0.0022995` for MAG_AUTO<25.8/radius>=0.75arcsec.
Errors are paired case-bootstrap standard deviations, 10,000 resamples,
seed20260917, neighbour shear+/-0.02 with unsheared primaries. Definitions:
[native amplitude audit](JOINT_CALIBRATION.md#experiment-5-native-emulator-amplitude-and-composition).

At0.75arcsec the measured response is0.1801601 and native emulator prediction
0.1657806. The deficit is6.25 standard errors from zero; after removing the
exact cut-boundary contribution, the within-pair deficit remains
0.0129193 +/- 0.0021387 (6.04 standard errors). Both other cuts have confidence
intervals containing zero. No predicted pair contains an unsheared secondary
anchor. The primary-ellipticity composition proxy changes the0.75 prediction
by only-0.0000327, insufficient to explain the discrepancy.

This redirects priority toward selected-population emulator amplitude and
response/selection correlation. It does not authorize a numerical correction:
the V2 anchor parent differs from full ConstGold and the fixed-g0 emulator's
training selection differs from this reselected test. No model is promoted.
Result: `neighbour_amplitude_audit_20260917_v1/result.json`. The reference-flow
variance audit16556078 has passed data loading and integrated its first two
million training pairs, with empty stderr. Baseline joint diagnostic16555762_0
is progressing; combined GPU count remains two. The next scheduled snapshot
is16556085 at23:32 local time.

## 2026-09-17 — Both guard continuations reject every update; independent emulator and variance audits

Both arms of array16555698 completed twelve epochs and retained epoch0. Exact
checkpoint comparison confirms identical state tensors, condition preprocessors,
target transforms and architecture to the original wide flow; only metadata
differs. The full validation NLL is -4.76270675245 for both. Identity proof is
`population_guard_ab_20260917_v1/checkpoint_identity.json`. No density or
calibration improvement resulted. Cancelled redundant evaluation tasks1--5 of
16555762 and their six-arm summary16555768 after verifying identity; the
already-progressing baseline uncut task0 remains. No duplicate prediction
files are manufactured as independent evaluations.

Added optional frozen-reference integration to
`sbsi/flow_guard_response.py::sampled_guard_response_backward`. Each of the
two independent row/latent groups computes current-minus-reference with common
draws and adds its own independently integrated reference mean. Replay
differentiates only the current flow, and shared parameter storage is rejected
by parameter identity. This changes training-estimator variance, not target
responses or deployed predictions. Two exact tests check expected loss and
gradient at/away from the reference, and reject reference parameter sharing.
Defaults remain unchanged. `scripts/audit_guard_control_variate.py` and its
launcher integrate four full-training-population reference groups and compare
eight gradient replicas for the ordinary estimator and two reference banks.
Record finite integration uncertainty and bank agreement before training.
Audit16556045 was cancelled during data loading to separate reference and
gradient seed ranges; replacement16556078 writes `guard_control_variate_20260917_v1/audit_v2`.
Its GPU allocation and the surviving baseline diagnostic total two GPUs.
The bounded two-hour scheduler/log capture was updated to include both new
audits: job16556085 is eligible at23:32:16 local time. It replaces the older
capture16555773; it records results/status and does not send external messages.

Added `scripts/audit_neighbour_emulator_amplitude.py`, launcher and three tests.
This tests the frozen emulator on the same common-usable anchor identities,
stratum weights and three cuts as the preceding selection decomposition.
It preserves the full finite-k neighbour list, zeros unsheared-secondary
contributions afterwards, and compares native per-unit-shear response with the
primary-shape proxy used by the likelihood. Paired total/within-response
residuals and uncertainties are measured directly, without a flow subtraction
or correction. Initial smoke16556019 failed because BlendEMU is not packaged
in py31; the launcher now specifies the same explicit companion source path
as existing cache jobs. Smoke16556025 completed two cases in19s with empty
stderr. CPU array16556028 and dependent summary16556033 cover cases400--899.
Products: `neighbour_amplitude_audit_20260917_v1/`.

Validation: full CPU suite **395 passed, 2 skipped** (one pre-existing NumPy
degrees-of-freedom warning); targeted reference tests7 passed and amplitude
tests3 passed; Ruff and launcher syntax passed. Protocol and limitations are
recorded in [the investigation](JOINT_CALIBRATION.md#next-optimization-decision)
and [native emulator audit](JOINT_CALIBRATION.md#experiment-5-native-emulator-amplitude-and-composition).
These are development diagnostics; the approximately0.3% joint goal is not
established, and no model is promoted.

## 2026-09-17 — Neighbour-cut response is not established as the dominant missing effect

CPU array16555940 completed all ten 50-case blocks with exit0 and empty stderr;
summary16555957 completed in 12 seconds. On the primary-unsheared, neighbour
+/-0.02 scenes, the exact selected-mean R11 decomposition gives the following
boundary terms: +0.0000114 +/- 0.0002993 for MAG_AUTO<25.8 and radius>0.60
arcsec; +0.0014602 +/- 0.0008586 for the same magnitude cut and radius>=0.75
arcsec. The uncut term is identically zero. Errors are 10,000-resample case
bootstrap standard deviations, seed20260917, pooling the two anchor strata
with their inverse-sampling weights within each of cases400--899 and then
averaging cases equally. Both selected-cut 95% intervals include zero; the
0.75-arcsec effect is about 1.7 standard errors from zero.

The total measured blending response on that scene parent is 0.1375230 uncut,
0.1416114 at 0.60 arcsec, and 0.1801601 at 0.75 arcsec. Every per-case result
satisfies total = paired-change + boundary to the asserted numerical tolerance.
This does not identify output-cut motion from neighbour shear as the dominant
source of the remaining joint error. A smaller 0.75-arcsec contribution remains
possible, but the scene parent differs from ConstGold and no numerical transfer
or fitted correction is made. Result:
`neighbour_selection_audit_20260917_v1/result.json`.

The investigation protocol now records this completed test and the next
optimization issue: removing the shared-row bias was insufficient because the
response steps remain noisy, clipped and harmful to density. A frozen-reference
control variate is a proposed integration-variance reduction to evaluate next;
it is not yet implemented, and no new calibration success is claimed. The
scientific priority is still joint flow-plus-emulator accuracy, not the favorable
classifier-weighted headline alone.

## 2026-09-17 — True-parent selected result approaches 0.3% only through classifier weighting; neighbour-cut audit launched

Full truth-parent GPU evaluation16555872 completed eighty cases in 9m15s with
exit0 and empty stderr. All 27 summaries are finite; zero-shape sentinel counts
are zero across both legs in every case. On the strict true-r<26 target-role
parent, without any observed-g0 requirement, joint `m=R_meas/R_model-1` is
-3.451 +/- 0.266% uncut, -1.128 +/- 0.263% with per-leg MAG_AUTO<25.8 and
radius>0.60 arcsec, and +1.098 +/- 0.220% with radius>=0.75 arcsec, using actual
per-leg U flags. Classifier weighting gives -4.272 +/- 0.263%, -1.885 +/-
0.260%, and +0.329 +/- 0.221%, respectively. These are R11 at antithetic
+/-0.02, cases40--119, 64 common flow draws (seed7301 plus case/batch offsets),
and 10,000 common case-bootstrap resamples (seed20260914). No |e| science cut
is applied. The uncut usable-g0 subset is -3.006 +/- 0.263%, confirming that
requiring a g0 measurement changes the diagnostic even without science cuts.

The near-0.3% classifier-weighted 0.75-arcsec value is not accepted. The flow
response barely changes between actual-U and classifier branches (0.6961067
versus 0.6961231); their blending contributions are 0.1872262 versus 0.1939847,
against a shared measured total0.8930318. Thus the favorable total chiefly
comes from blending-population reweighting while the actual-U joint error
remains 1.10%. Other declared samples fail, the emulator is extrapolated,
and these are reused development scenes. Output:
`constgold_truth_parent_20260917_v1/baseline_response.json`.

Added `scripts/audit_neighbour_selection_response.py` and two launchers to
measure an independent composition limitation: coherent neighbour shear can
move measured radius/flux across cuts, whereas the current external R_blend
shift changes only ellipticity. Existing primary-unsheared neighbour +/-0.02
scenes permit an exact finite-stencil decomposition into average paired shape
change and changing selection of the pair-average shape. The script checks
the sum identity, preserves original/complement stratum weights, reports
missing/common-usable counts, and excludes both flow and emulator fits.
CPU array16555940 scans cases400--899 in ten blocks throttled to four workers;
dependent CPU summary16555957 requires complete exact case coverage. All
started blocks show finite progress with empty stderr. The measured selection
term will not be transferred numerically across the differing scene/ConstGold
populations without a further justified comparison.

Two focused tests recover the exact decomposition, its uncut zero boundary
term, and inverse-stratum-weight invariance. Final full suite: 390 passed,
2 skipped, one pre-existing NumPy warning. Ruff, shell syntax and whitespace
checks pass. [Population results and the neighbour-cut protocol](JOINT_CALIBRATION.md#completed-truth-parent-results)
record estimands and remaining limitations. The shared-row training control
and its dependent A/B evaluations remain scheduled within the two-GPU limit.

## 2026-09-17 — Independent-row continuation rejected; truth-parent smoke passes

Independent-row training task16555698_1 completed twelve epochs and retained
epoch zero: none of the continuation checkpoints beat the initial declared
NLL-plus-full-population-guard score. The selected model therefore remains
the original wide-guard baseline, with full-validation NLL -4.76270675 and
fixed-stream guard score 0.04437478. All guard steps clipped their gradients;
in the first three epochs the mean pre-clip norms were about 108--131 against
a clip of 5. The NLL training passes remained near -4.76, but post-guard
validation NLL degraded to -4.65--4.69. This is evidence against the chosen
alternating continuation recipe, not a calibration gain from the corrected
objective. The slower shared-row control remains active and is not restarted.

Truth-parent CPU lookup array16555864 completed all eight blocks with empty
stderr. Its manifests exactly cover 8,876,218 parent keys across cases40--119;
every per-case row count agrees with the prepared parent. GPU smoke16555865
completed cases40--41 in 23 seconds, producing all 27 cohort/cut/usability
summaries with empty stderr. Both cases have zero exact-zero usable shape
rows. This smoke establishes execution and population accounting, not an
accuracy claim from two cases.

Reordered pending GPU dependencies to use the GPU released by the independent
arm: full truth-parent evaluation16555872 now follows its successful smoke
while the control continues. A/B evaluation16555762 follows both the training
array and the full truth-parent evaluation; summary16555768 remains dependent
on the A/B evaluation. The A/B array was briefly held during the dependency
update and then released. The combined allocation remains at most two GPUs.
The full parent evaluation has begun normal case progress with empty stderr.

## 2026-09-17 — Truth-parent joint-response audit prepared without a g0 detection requirement

While the controlled guard A/B trains, addressed a distinct diagnostic
population issue: even the latest "uncut" ConstGold panels require an observed
usable g0 measurement. Added `scripts/prepare_constgold_truth_parent.py` to
recover the full target-role truth parent from the existing classifier data,
with the declared strict r<26 cut taken from original-precision ConstGold truth.
Observed g0 usability and fixed-cut membership are stored only as diagnostic
masks; no measurements are fabricated for undetected objects.

CPU preparation16555828 stopped at case105 because a g0 row with input ID349783
is outside the target-role parent. Scheduled read-only check16555842 verified
that this key is absent from the full-parent classifier input, rather than
being lost through a magnitude cut. Revised preparation drops and reports such
unmatched g0 keys, and added a separate regression test for truth-cut precision.
The failed partial `parent/` remains preserved; replacement16555863 completed
all eighty cases under `parent_v2/` with empty stderr. Counts: 8,876,218 truth
parents, 7,841,110 usable-g0 objects, 6,472,963 fixed-cut-g0 objects, and
1,035,108 newly retained g0-unusable objects (11.662% of the parent). Three g0
keys outside the target-role parent were dropped, in cases105,106,118.

Added `scripts/evaluate_constgold_truth_parent.py` and tests. It evaluates the
same frozen wide-guard flow plus existing emulator on the truth parent and
both observed-g0 subcohorts. For each it reports uncut and independently
reselected per-leg magnitude/radius samples at the existing 0.60/0.75-arcsec
boundaries. Generated selection acts on the full joint outputs before shape
averaging; the external blend shift is averaged over that same selected
population. The output includes actual/matched/classifier usability branches,
model flow/blend contributions, case-bootstrap errors, per-case statistics,
and missing/invalid/zero-shape counts. Exact zero ellipticity follows the
current full-domain flow validity contract and is counted explicitly because
the classifier preparation treats exact-zero shapes as sentinels.

The existing blend lookup builder now accepts explicit population provenance;
its historical behavior is unchanged. Added CPU and GPU launchers. Failed
preparation's dependent lookup16555830 was cancelled; replacement lookup
array16555864 follows successful preparation16555863 and has started on four
CPU workers. GPU smoke16555865 depends on those lookups and completion of the
independent training task16555698_1, so it can use the released GPU while the
control continues. The dependency chain was subsequently reordered as recorded
in the newer entry above, preserving the combined two-GPU cap.

Validation: six new tests cover parent retention, dropped keys, original
truth-cut precision, joint-output selection, strict/inclusive cut boundaries,
blend averaging and raw measurement alignment. Full suite: 388 passed,
2 skipped, one pre-existing NumPy warning. Ruff, launcher syntax and whitespace
checks pass. This is a diagnostic on reused cases40--119, not acceptance; the
emulator is extrapolated beyond its original fixed-g0 training population.
See [the population-test protocol](JOINT_CALIBRATION.md#experiment-3-remove-observed-g0-conditioning-from-the-joint-diagnostic).

## 2026-09-17 — Population-loss defect measured; controlled single-flow A/B launched

The full-validation audits completed successfully: sharp task16555643_1 in
1m47s and repaired wide task16555687_0 in 3m40s, both with empty stderr. The
historical shared-row loss has an estimated extra variance penalty of 0.02711
for the wide recipe versus its intended population-mean loss of 0.01326; for
the sharp recipe these are 0.04743 versus 0.05363. Thus this is substantial
on the scale of the optimization loss, though a calibration benefit from
removing it is not yet established.

Using each recipe's per-leg soft guard on all 3,908,555 validation pairs,
`m = measured/model - 1` for the half-trace self response is +0.070 +/- 0.423%
globally, +0.896 +/- 0.394% for radius, and +0.709 +/- 0.398% for the joint cut
in the wide model. Sharp gives +0.189 +/- 0.423%, +1.022 +/- 0.406%, and
+0.737 +/- 0.411%. Errors are paired case-bootstrap standard deviations;
finite two-replica latent integration uncertainty is not included. These are
matched-usable forward half-shear responses at |g|=0.05, with per-leg selection,
not fixed-g0 carried-cohort or combined emulator results. The old validation
subsets shift the wide/sharp joint model responses by +0.001495/+0.000595.

Extended `scripts/refine_full_domain_guard_flow.py` with an explicit
population-objective mode, full-validation guard coverage, an epoch-zero
baseline, and a shared-row control. Historical defaults remain available.
Added `jobs/job_refine_population_guard_ab.sh`; two-task array16555698 starts
from the same wide checkpoint with identical NLL, guard widths, seeds, optimizer
settings and validation rule, changing only independent versus shared row
replicas. It uses at most two 16GB A40 allocations, 38GB host memory each.
Both tasks loaded 31,358,178 NLL rows and 15,630,658 response pairs with empty
stderr. The run is a twelve-epoch experiment, not an accepted model.

Added a frozen six-task, two-GPU-throttled ConstGold evaluation launcher for
baseline/control/corrected checkpoints on both uncut valid-g0 and fixed-cut
cohorts, plus `scripts/summarize_population_guard_ab.py` for paired-bootstrap
comparisons. Both populations retain strict true r<26. The summarizer rejects
changed seeds, populations, binning, anchors, classifiers, or emulator lookups;
two tests recover an exact multiplicative difference and reject an unpaired
seed. Final full suite: 382 passed, 2 skipped, with the pre-existing NumPy
warning. Ruff, shell syntax and whitespace checks pass.

Both training tasks have reached finite optimizer/validation progress with
empty stderr. Their epoch-zero metrics match exactly: NLL -4.76031004,
guard loss 0.04437478, combined score -4.71593526. First-epoch density and
response scores worsened in both arms, so the initial checkpoint remains
selected; no improvement is claimed. The slower control was checked inside
its allocation and is actively using its GPU (6.5GB resident), not stalled.
Dependent evaluation array16555762 has six tasks throttled to two GPUs and
starts only after both training arms succeed; CPU summary16555768 depends on
all evaluations. One-shot status/log capture16555773 is scheduled about two
hours later. All artifacts and logs remain under
`population_guard_ab_20260917_v1/`.

See [the A/B protocol](JOINT_CALIBRATION.md#experiment-2-controlled-population-objective-continuation).
Next: verify first training progress, finish the controlled experiment, then
evaluate the frozen joint models. The uncut branch extrapolates the existing
fixed-g0-trained emulator beyond its fit cohort; all ConstGold cases are reused
development scenes and cannot certify a fresh 0.3% acceptance claim.

## 2026-09-17 — Autonomous joint-calibration goal: audit the guard population objective

The owner requested autonomous work toward approximately 0.3% joint
flow-plus-emulator bias, retaining good density and removing the need for a
flow ensemble. Added [the investigation protocol](JOINT_CALIBRATION.md#goal-and-retained-population),
including the distinction between the historical narrow-domain V3 milestone
and the current true-mag-limited full domain, and between per-leg reselected
guard responses and fixed-g0 carried-cohort diagnostics.

Found a mathematical defect in the interpretation of the current aggregate
guard loss: independent latent replicas on the same sampled pairs remove
latent variance but retain `Var_i(E_z response_i)/(2 N scale^2)` in expectation.
This parameter-dependent term can shrink real response variation even at the
correct population mean. Added opt-in independent row replicas to
`sbsi/flow_guard_response.py`, preserving the historical default for existing
recipes and keeping common random numbers between shear legs. An exact
two-object enumeration in `tests/test_flow_guard_response.py` verifies the
expected loss and gradient, including the replay implementation.

Added `scripts/audit_guard_population_score.py` and its array launcher to
measure that penalty and compare full-validation response with the historical
fixed-subset selector. Each checkpoint is evaluated on all 3,908,555 validation
pairs with two independent latent groups of eight antithetic draws; outputs
include full 2x2 response matrices, case-bootstrap errors, and per-case
statistics. This audit is self-response only and is not an acceptance claim.

Validation: focused suite 5 passed; full suite 380 passed, 2 skipped, with the
pre-existing NumPy degrees-of-freedom warning. System Ruff, Bash syntax, and
whitespace checks pass. The py31 environment has no Ruff module, so the
installed `/usr/bin/ruff` was used. No heavy work ran on the login node.

Initial full-A40 array16555630 was cancelled while pending because its estimated
start was overnight. Replacement array16555643 uses two available 16GB A40
allocations, 4 CPUs and 32GB host memory each. Sharp-checkpoint task1 started
normally with empty stderr. Wide-checkpoint task0 failed before evaluation
because its checkpoint lives in `flow/guard/selected.pt`, not `flow/selected.pt`;
the launcher was corrected and task0 resubmitted as array16555687. This keeps
at most two GPU tasks active/pending for the experiment. Results are under
`$DATA_DIR/sbsi_caches/guard_population_audit_20260917_v1/`. Next: inspect both
completed audits before choosing the next controlled training experiment.

## 2026-09-17 — Underlying responses exposed for uncut size panels

Added a focused response-rendering mode to
`plots/constgold_property_bias.py` and remade the former B (true size) and D
(fixed-g0 measured size) panels of the no-measured-cuts diagnostic as
`R_meas` and `R_model`, rather than their unstable ratio `m`.  The new mode
reconstructs both responses and their case-bootstrap standard errors from the
saved per-case sufficient statistics, so it requires no new flow evaluation.
Branch identity uses the existing accessible colors and markers; filled solid
curves denote measured response and open dashed curves denote model response.
Because the classifier branch intentionally shares the actual-per-leg measured
comparator, that exactly duplicated measured curve is drawn once rather than
obscuring itself.  The original B and D panel letters and uncut anchor
histograms are preserved.

The focused figure, vector companion, response table, and histogram table are
`plots/figures/constgold_response_true_size_g0_radius_full_domain_guard_refine_uncut_true_mag26_c40_139.{png,pdf,csv}`
and the matching `_histograms.csv`.  It directly shows that the first two
measured-radius bins have measured responses about -0.26 and -0.034, versus
model responses about -0.09 and +0.10; the huge ratio-defined `m` was therefore
caused by a misplaced response zero crossing.  The PNG was visually inspected.
Validation: 12 focused tests and the full suite pass (379 passed, 2 skipped;
one pre-existing NumPy degrees-of-freedom warning).  Ruff, Python compilation,
and whitespace checks pass.

## 2026-09-17 — Sharp-guard property panels remade without measured cuts

Remade the 100-case 2x3 ConstGold property figure on the full valid-measurement
fixed-g0 population of the sharp-guard model.  This version applies no measured
`MAG_AUTO` or `FLUX_RADIUS` selection.  It retains the model's declared strict
truth boundary `r_input_p < 26` and requires a valid g0 measurement because
otherwise there is no anchor measurement to bin; the ConstGold plus/minus legs
are not recut.  The same uncut anchor identities define the quantile bins,
population histograms, measured responses, and model predictions.

The old `R_blend` lookup was keyed only to the post-cut cohort and could not be
silently reused.  Added `jobs/job_build_constgold_full_domain_rblend.sh` to
construct exact lookups for all full-domain anchor keys in ten CPU blocks.
Array job16544314 completed all blocks with exit 0, empty stderr, and exact
primary coverage.  Added
`jobs/job_constgold_property_bias_full_domain_uncut.sh`; dependent one-A40
job16544336 completed in 10m35s with exit 0 and empty stderr.  The final cohort
contains 9,802,054 anchors, 1,711,992 more than the corresponding true-mag-
limited post-cut plot.

With 64 common antithetic draws and 10,000 common case-bootstrap resamples, the
uncut collapsed `m = R_measured / R_model - 1` values are -2.4241 +/- 0.2282%
for matched usability, -2.9192 +/- 0.2317% for actual per-leg flags, and
-2.1115 +/- 0.2323% for classifier-weighted usability.  The corresponding
post-cut values were +1.5020%, +1.1302%, and +0.9316%, so the fixed-g0
magnitude/radius selection changes both the sign and magnitude of the global
diagnostic.

The uncut radius bins now span 0.206--32.234 arcsec.  The first two radius bins
have negative measured response, while the model response changes sign at a
different boundary; for example, the 0.5918--0.6486-arcsec bin has measured
response about -0.0337 and model response about +0.106 for the actual-flag
branch.  Ratio-defined `m` is consequently -131.8% there.  A similar
near-zero/sign-changing response occurs in the second true-size bin.  These
large plotted ratios mark response-crossing regimes, not ordinary global
multiplicative calibration offsets.

Final artifacts are
`plots/figures/constgold_property_bias_full_domain_guard_refine_uncut_true_mag26_c40_139.{png,pdf,csv}`
plus `_histograms.csv`; the source result and ten lookup blocks are under
`full_domain_true_mag26_guard_refine_v1/constgold_property_bias_uncut_true_mag26_c40_139_v1/`.
The PNG was visually inspected at full resolution; it preserves the accessible
three-branch encodings, case-bootstrap errors, target band, and uncut anchor
histograms, with a vector PDF companion.  The JSON has 144 finite summary rows,
its aggregate anchor count matches all lookup-block manifests, both launchers
pass `bash -n`, whitespace checks pass, and no SBSI GPU job remains active.
Cases40--139 remain reused development scenes, so this is a diagnostic rather
than fresh acceptance evidence.

## 2026-09-17 — Sharp-guard ConstGold property panels remade on the true-mag-limited domain

Remade the 2x3 ConstGold property-binned `m` figure with histogram overlays for
the sharp-guard full-domain checkpoint.  Because the existing fixed-g0 anchor
extends to true `r` about 29 while this checkpoint's declared training
population is strictly `r_input_p < 26`, `plots/constgold_property_bias.py` now
accepts `--true-magnitude-max`.  The leg-invariant truth cut is applied before
quantile construction and before either ConstGold shear leg is read, and the
same retained population supplies both response curves and histogram overlays.
The result records pre-cut and retained row counts separately and shows the
declared population boundary in the figure title.

Final one-A40 job16543392 completed all 100 cases in 8m13s with exit 0 and empty
stderr.  It retained 8,090,062 of 9,077,958 fixed-g0 anchors (89.1176%).  With
64 common antithetic draws and 10,000 common case-bootstrap resamples, the
collapsed `m = R_measured / R_model - 1` values are +1.5020 +/- 0.2336% for
matched usability, +1.1302 +/- 0.2337% for actual per-leg flags, and +0.9316
+/- 0.2324% for classifier-weighted usability.  The smallest measured-radius
bin, 0.6002--0.6524 arcsec, remains a response-collapse regime: its three `m`
values are -107.7%, -108.6%, and -109.5%, respectively, because the measured
response is near zero while the model response is about 0.12--0.13.

The production artifacts are
`plots/figures/constgold_property_bias_full_domain_guard_refine_true_mag26_c40_139_v2.{png,pdf,csv}`
plus `_histograms.csv`; the source result is under
`full_domain_true_mag26_guard_refine_v1/constgold_property_bias_true_mag26_c40_139_v2/`.
The PNG was inspected at full resolution and preserves the colour-blind-safe
three-branch encodings, uncertainties, target band, and grey population
histograms; the PDF remains vector output.

Two launch-time/provenance issues were caught rather than hidden.  Job16543118
failed before processing any case because the file-style entry point did not
expose the installed package on the compute node; module invocation fixed it.
Job16543153 then completed the scientifically correct calculation and figure,
but an audit found `anchor_rows_by_case` still copied the pre-cut rather than
retained counts.  The final versioned v2 run fixes that metadata; its science
rows are unchanged.  The superseded v1 files are not final products.

Validation: 12 focused tests and the full suite pass (379 passed, 2 skipped;
one pre-existing NumPy degrees-of-freedom warning).  Ruff, Python compilation,
launcher syntax, and whitespace checks pass.  The final JSON contains 144
finite summary rows, its true-magnitude upper edge is below 26, retained counts
agree between aggregate and per-case records, and all four figure/table files
are nonempty.  No SBSI GPU jobs remain active.  As before, cases40--139 reuse
finite development scenes, so this is a diagnostic rather than fresh
acceptance evidence.

## 2026-09-17 — Sharp guard refinement completed without hard-cut improvement

One-A40 job16537164 completed in 25m32s with exit 0 and empty stderr.  The
0.005-pixel/0.01-mag sharp-guard refinement selected epoch4, with full
validation NLL -4.76383982 and fixed-subset guard loss 0.05555938.  On that
400,000-pair fixed subset its scalar response residual is -1.45% at
`FLUX_RADIUS > 3` and -1.03% at the joint `FLUX_RADIUS > 3`,
`MAG_AUTO < 25.8` guard.  The raw loss is not directly comparable with the
earlier wide-guard loss because the response scale and guard widths changed.

Added `jobs/job_validate_refined_guard_hard_cuts.sh` and ran job16542077 on all
40 held-out cases with 64 coherent draws and sampling seed57301, exactly
matching the previous checkpoint comparison.  It completed in 4m04s with exit
0 and empty stderr.  The previous wide-guard checkpoint had radius-only bias
-1.627% +/- 0.384% and correctly anchored joint bias -1.506% +/- 0.396%; the
sharp refinement gives -1.641% +/- 0.384% and -1.527% +/- 0.396%, respectively.
Thus this refinement is effectively unchanged and marginally worse, despite a
slightly better NLL, and is not promoted.  The result rules out simply
sharpening the same aggregate guard recipe as the next route to sub-percent
hard-boundary robustness; the next experiment should change the response
objective or its variance/coverage rather than continue this checkpoint.

Validation: the new launcher passes `bash -n`; its output is
`full_domain_true_mag26_guard_refine_v1/hard_cut_validation_c40_v1/refined_heldout.json`.
No other SBSI GPU job was active when it was submitted, so the two-GPU limit
was respected.

## 2026-09-17 — Full-domain guard recovered most hard-cut response bias; sharp refinement launched

Closed the first direction-2 training/evaluation loop.  The true-mag-limited
full-domain job16536097 had completed all 140 NLL epochs but failed in guard
phase2 because an ultra-faint softplus draw can round to exactly zero.  The
forward magnitude guard was then finite and zero, but autograd formed a
nonfinite `0 * infinity` through `log10(flux)`.  `torch_guard_weights` now
clamps only this float underflow tail to the smallest representable positive
value, where the guard already has its limiting zero weight and zero gradient.
The focused reproducer changed from a NaN flux gradient to a finite zero
gradient.  This is numerical stabilization, not a catalogue cut, response
offset, or change to the likelihood domain.

Resumed one-A40 job16536920 from the completed NLL and guard-phase1 state.  It
completed in 18m51s with exit0 and empty stderr, selected guard phase12, and
preserved full-validation NLL at -4.76270675.  On its fixed soft-guard subset,
the scalar half-trace response residual is +0.14% globally, -0.71% at
`FLUX_RADIUS > 3`, and -0.49% at the joint `FLUX_RADIUS > 3`, `MAG_AUTO < 25.8`
guard.  These are checkpoint-selected soft-guard diagnostics, not hard-cut or
end-to-end acceptance numbers.

Completed the original no-radius-cut control with job16534684 and evaluated it
with job16536921 on 40 training plus all 40 held-out cases.  Removing the
training radius cut improves the held-out first `[0.6000,0.6154)` arcsec bin
from the old roughly +0.22 response residual to +0.04196 +/- 0.01317 when the
model is read on its generated radius.  It still leaves a hard post-cut
response ratio bias of -3.96% +/- 0.44% held out (-3.16% +/- 0.31% on training
cases).  Thus broadening the domain helps the edge coupling but ordinary NLL
plus the old global paired-response loss does not make test-time cuts robust.

Added post-cut aggregation and a catalogue-anchor-matched reference to
`scripts/diagnose_flow_generated_radius_coupling.py`.  The latter correction is
essential for a full-domain likelihood: generated joint radius/magnitude cuts
must be compared with measured rows passing both g0 cuts, not with a radius-only
reference.  Jobs16537006 and 16537080 then compared the full-domain NLL-only and
guard-selected checkpoints on all 40 held-out cases with 64 common antithetic
draws.  For the hard radius-only cut, NLL-only is -3.94% +/- 0.38% and guarded
is -1.62% +/- 0.39%.  For the correctly matched joint radius/magnitude cut,
NLL-only is -3.82% +/- 0.40% and guarded is -1.51% +/- 0.40%.  The guard removes
about 60% of the domain-shift bias but does not yet meet the sub-percent goal.
The coherent-draw joint result changes to -4.30% when only the sheared draw
order is permuted, reiterating that the per-leg likelihood does not identify
the cross-leg latent copula; all quoted primary results use the declared common
latent extraction convention on both measured/model sides.

Added `scripts/evaluate_full_domain_guard_widths.py`.  Job16537126 evaluated
400,000 validation pairs excluded from checkpoint selection with common random
numbers.  The current checkpoint's radius residual moves from -0.947% at the
trained 0.05-pixel softness to -1.061% at 0.01 and -1.057% at 0.005; the joint
residual remains between -0.78% and -0.84%.  Sharpening therefore exposes a
real but numerically stable hard-edge gap.  Added
`scripts/refine_full_domain_guard_flow.py` and its launcher to refine the
successful checkpoint at 0.005-pixel/0.01-mag softness, 0.01 response scale,
65,536 pairs and eight draws per guard step while retaining full-domain NLL.
One-A40 job16537164 started with no other GPU job active, loaded the expected
31,358,178 training rows and 15,630,658 response pairs, and improved its finite
validation objective from -4.64327 at epoch1 to -4.65357 at epoch2 with empty
stderr.  It is a new experiment, not an accepted model.

Files added or changed in this loop: `sbsi/flow_guard_response.py`,
`scripts/diagnose_flow_generated_radius_coupling.py`,
`scripts/evaluate_full_domain_guard_widths.py`,
`scripts/refine_full_domain_guard_flow.py`, their focused tests, and immutable
launchers for the no-radius evaluation, hard-cut comparisons, width sweep, and
sharp refinement.  Shell syntax, Python compilation, Ruff, whitespace checks,
and the full suite pass: 377 passed and 2 skipped (one pre-existing NumPy
degrees-of-freedom warning).  Next: let job16537164 reach normal completion,
then repeat the corrected 40-case hard radius/joint evaluation on its selected
checkpoint before any ConstGold or end-to-end claim.

## 2026-09-17 — True-magnitude-limited full-domain likelihood with cut-guarded response launched

Implemented a direction-2 training experiment that keeps the likelihood on
the complete valid measurement domain while explicitly protecting its induced
shape response near prospective test-time cuts.  New
`sbsi/flow_guard_response.py` defines smooth lower-radius/upper-magnitude guard
functions and the first-order selected-mean influence
`w(y) [e - mean_0(e|w)] / P_0(w)`.  Regressing its per-leg change on the actual
two-component applied shear gives a four-component response target without
conditioning the flow on any measurement or introducing a paired-output
likelihood.  Independent generated-draw replicas give an integration-noise-
subtracted cross-quadratic, and gradient replay keeps the large aggregate
response batches memory bounded.

Added `scripts/prepare_full_domain_flow.py` and
`scripts/train_full_domain_guard_flow.py`.  Preparation retains every valid
`U=1` row independently in each leg after the owner's strict declared parent
boundary `r_input_p < 26`; it applies no measured `MAG_AUTO` or `FLUX_RADIUS`
science cut.  Training first performs the ordinary joint
four-output NLL and then constrains the global 2x2 shape response plus radius
guards at 2.8/3.0/3.2 pixels, magnitude guards at 25.6/25.8/26.0, and all nine
joint combinations.  The global target uses a 0.01 response scale and guard
targets use 0.02; these are optimization scales, not claimed calibration
uncertainties or empirical response corrections.  The guards have 0.05-pixel
and 0.05-magnitude soft widths so the generated density receives gradients on
both sides of each prospective boundary.

Added focused tests and immutable scheduler entry points.  The focused suites
report 9 passed; the full suite reports 374 passed and 2 skipped.  Python
compilation, Ruff, shell syntax, and whitespace checks pass.  A real-catalogue
case-level algebra check recovered finite global and joint response matrices.
The initial no-truth-cut preparation job 16535966 completed successfully with
31,411,766/31,398,334 valid g0/g05 rows, but its dependent training job
16535967 was cancelled during data loading, before the first optimization
epoch, when the owner added the true-magnitude boundary.  That immutable
unused preparation remains under `full_domain_guard_v1`.  Replacement CPU
preparation job 16536096 completed in 2m18s with exit 0 and empty stderr,
retaining 19,601,305/19,597,500 g0/g05 rows.  Dependent one-A40 training job
16536097 loaded 31,358,178 training rows, 7,840,627 validation rows, and
15,630,658/3,908,555 matched train/validation response pairs; by NLL epoch3
its validation objective had improved from -3.228844 to -3.643598 with empty
stderr.  Existing one-A40 no-radius job 16534684 is the only other SBSI GPU
job, preserving the owner's two-GPU limit.  This is a training experiment,
not an accepted likelihood: hard-cut cumulative response curves and held-out
end-to-end SBSI closure remain required after completion.

## 2026-09-16 — Root cause of the selected small-radius training residual

Audited why the fixed-g0 staged flow fails in the first measured-radius bin
even on cases that supplied its response labels.  The result separates two
effects: the response target used by the loss is a truth-conditioned mean,
whereas the diagnostic bin is defined by a realised measurement; and the
implemented global stochastic objective has extremely low sensitivity to the
small localized signed error.

GPU job 16534876 evaluated the NLL-only, staged-response, and from-scratch
direct-response checkpoints on the same 20 explicit training cases with 64
common antithetic draws.  In `[0.6000,0.6154)` arcsec (56,241 objects), the
measured `R_self` is -0.257598, while the three predictions are respectively
-0.017806, -0.031250, and +0.022115.  Thus staged response supervision moves
the NLL result in the right direction by only -0.013444, recovering 5.6% of
the -0.239792 required change; direct response training makes the bin worse.
The job completed in 5m46s with exit 0 and empty stderr.  Outputs are under
`fixed_g0_m258_r060_v2/root_cause_radius_v1/`.

The recorded training history explains why the regularizer has little power
there.  The first bin is 3.112% of these training rows, so eliminating its
entire mean parallel residual would reduce the globally averaged shape score
by only `p * residual^2 / 4 = 0.0004474`: 0.0055% of the roughly 8.108 shape
loss, below its approximately 0.001 validation Monte Carlo error.  The
selected checkpoint's shape loss changes only from 8.108024 at NLL epoch140
to 8.105279, while the radius component is about 17.2.  The largest 1%/0.1%
of radius-response targets contribute 88.25%/75.22% of the radius loss.  All
35 staged epochs report a clipped-step fraction of 1.0; mean pre-clip gradient
norms range from 57 to about 3,970 and maxima reach 1.9e6 against a clip of 5.
The total validation objective and selected epoch15 are consequently governed
by heavy-tailed radius and integration fluctuations, not this boundary shape
mean.  Uniform response sampling also means only about 54.1% of the 14.45M
pairs are expected to have appeared at least once by phase epoch15, although
the boundary still has ample aggregate samples; omission alone is not the
explanation.

Added `scripts/diagnose_truth_only_boundary_predictability.py`, its focused
tests, and a CPU launcher for a direct predictability control.  Job16534912
fit flexible LightGBM models on 2.4M sampled training pairs, then evaluated
800k disjoint objects from training cases and 800k objects from all40 held-out
cases.  The truth-only model uses the two declared eight-feature flow contexts
plus shear, but no measurement.  In the first bin it predicts +0.008773 for
unseen training objects against -0.248607 measured, and +0.007773 on held-out
cases against -0.259352.  An oracle given the realised g=0 radius predicts
-0.265174 and -0.267252, leaving residuals consistent with zero at the
case level.  Truth features predict first-bin membership well (AUC 0.912 on
both splits), but do not predict the response conditional on the realised
within-bin radius.  The oracle result is diagnostic only and is not a proposed
measurement-conditioned model or correction.

Together with the existing leg decomposition--the measured first-bin zero-leg
projected term is -0.000415 while the sheared-leg term is -0.243598--this shows
that the missing signal is not merely a g=0 within-leg shape/radius trend.  It
is response information associated with the realised g=0 radius after the
current truth and coarse neighbour-flux summaries are held fixed.  This does
not prove that a truth-conditioned density is impossible: a cut-free density,
richer scene/blending conditions, and a correct post-selection integral may
still recover it.  It does show that simply placing the same rows in an
individual-pair mean-response loss does not force the present flow to match an
arbitrary measurement-defined subgroup.

Validation: the new focused suite reports 2 passed; Python compilation, Ruff,
and both launcher syntax checks pass.  Job16534912 completed in 2m53s with
exit 0 and 1.1GiB peak RSS; stderr contains only six harmless LightGBM feature-
name warnings.  The no-radius-cut domain preparation separately completed as
job16534683 in 32m08s with 20,208,269 g=0 anchors and 20,112,754 usable carried
g=0.05 rows; dependent existing-flow training job16534684 started normally on
one A40.

## 2026-09-16 — Existing-flow no-radius-cut control launched

Prepared a controlled test of whether the current truth-conditioned per-leg
flow can learn through the 3-pixel boundary without changing to a paired-flow
model.  `fixed_g0_anchor_mask` and `scripts/prepare_fixed_g0_domain.py` now
support an explicit no-`FLUX_RADIUS`-science-cut mode.  It retains the same
g=0 detection and `MAG_AUTO < 25.8` anchor, finite measured radius, no truth
cut, exact key carry to g=0.05, and the existing per-leg measurement-validity
requirements; only `FLUX_RADIUS(g=0) > 3` is omitted.  The original strict-cut
default is unchanged.  Added a focused mask test and separate immutable Slurm
entry points under `fixed_g0_m258_no_radius_cut_v1`.

This run deliberately keeps the existing four-output flow architecture,
truth/blending-summary conditions, NLL, paired mean-response regularizer,
case split, and training recipe fixed, so it isolates the population change.
The test-time radius cut must be audited with signed post-cut response and fine
boundary bins rather than only the full-domain scalar objective.  Opposite
signed residuals do not literally cancel inside the current per-pair squared
response score, but the small boundary population and large irreducible
pair scatter can still make a severe local residual cheap in the global loss.

CPU preparation job 16534683 and dependent one-A40 training job 16534684 were
submitted with no other SBSI jobs active, preserving the two-GPU limit.  The
preparation job reached all 480 g=0 batches and began the response-anchor pass
with empty stderr.  Validation before submission: the focused domain suite
reports 8 passed; Python compilation and both job-script syntax checks pass.
Next: verify preparation counts/manifest, monitor the training startup, then
evaluate both uncut and `FLUX_RADIUS(g=0)>3` validation/test residuals without
using a full-domain aggregate as the acceptance criterion.

## 2026-09-16 — Correction: the fixed-g0 flow does not identify the cross-leg radius/response coupling

The preceding generated-radius interpretation was too strong.  Added an
invariance control to `scripts/diagnose_flow_generated_radius_coupling.py`:
within every object, retain exactly the same g0 and g05 draws but randomly
permute the g05 draw order before pairing it with g0.  This preserves each
leg's empirical distribution and per-object mean exactly.  Consequently both
the per-leg NLL and the paired response loss used in training are unchanged;
only the cross-leg latent copula changes.

Job 16531223 completed cases 0--19 in 1m57s with exit 0, empty stderr, and
794,620 KiB peak RSS.  In `[0.6000,0.6154)` arcsec:

| extraction | `R_self` | model - measured +/- paired case SEM |
|---|---:|---:|
| measured catalogue | -0.243182 | -- |
| generated radius + anchor, same latent order | -0.246314 | -0.003131 +/- 0.024073 |
| generated radius + anchor, permuted g05 order | -0.066280 | +0.176903 +/- 0.023704 |

The 0.1800 response shift occurs under a transformation to which the complete
training objective is blind.  The same-latent agreement therefore does not
show that the flow learned the physical paired distribution; it is an
architecture/transport-dependent coupling selected by the common-random-number
convention.  This also explains why the catalogue-radius residual is nearly
identical on training and held-out cases: per-leg likelihood plus a difference
of latent-averaged means targets `E[delta y|truth,S0]`, not the distribution of
`delta y` jointly with the realised g0 measurement.  The hard g0 selection and
unrecut g05 leg make that missing dependence largest at the 3-pixel boundary.

The correct truth-only formulation is a paired generative model of
`p(y0,yg|x0,xg,S0)` (or an equivalent joint output such as `y0` plus a valid
paired change), trained with a paired joint likelihood.  Feeding the observed
measurement as an external condition is unnecessary.  The g0 radius/magnitude
support must also be represented exactly while the carried g05 leg remains
untruncated.  The new result supersedes the claim below that the current flow
had learned the cross-leg coupling.  Result:
`$DATA_DIR/sbsi_caches/fixed_g0_m258_r060_v2/generated_radius_coupling_c0_19_v2/result.json`.
Four focused tests and the full suite (367 passed, 2 skipped) pass; Python
compilation, Ruff, Bash syntax, and whitespace checks are clean.

## 2026-09-16 — Original small-radius table recomputed on 40 training cases

Recomputed the original four `FLUX_RADIUS` boundary intervals from the existing
64-draw staged-flow cache restricted to 40 explicit training-split cases
(cases 20--71 after split filtering; 3,615,648 matched objects).  The first
four measured/model `R_self` rows are -0.269620/-0.030218,
-0.081934/+0.081550, +0.118022/+0.198766, and +0.262673/+0.300831, with
111,998, 113,781, 113,701, and 113,634 objects.  Equal-weight measured-response
case SEMs are 0.013612, 0.017085, 0.020260, and 0.019829.  Scheduler job
16530869 completed in 19 seconds with exit 0 and empty stderr; the result is
`$DATA_DIR/sbsi_caches/fixed_g0_m258_r060_v2/flow_self_response_v1/training_radius_original_edges.json`.
No model sampling, fitting, or source-code change was made.

## 2026-09-16 — Initial generated-radius check (interpretation superseded)

**Superseded interpretation.**  The numerical same-latent extraction below is
correct, but the newer latent-permutation control above proves that its
cross-leg coupling is not identified by the training objective.  It must not
be cited as evidence that the current flow learned the physical paired
response/radius distribution.

Added `scripts/diagnose_flow_generated_radius_coupling.py`, its focused tests,
and `jobs/job_diagnose_flow_generated_radius_coupling.sh` to distinguish two
different response-versus-radius calculations.  The earlier diagnostic first
averaged every flow draw for an object and then binned that unconditional mean
on the object's realised catalogue g=0 `FLUX_RADIUS`.  The new generative
calculation instead bins each common-random-number g=0/g05 flow draw on that
draw's own generated g=0 radius.  Both calculations keep the flow conditioned
only on the eight declared truth/context features; neither supplies a measured
quantity as a condition or fits a correction.

GPU job 16527537 completed cases 0--19 in 1m54s with exit 0 and empty stderr,
using 64 antithetic draws (115,670,464 total paired draws).  In the boundary
bin `[0.6000,0.6154)` arcsec the comparison is:

| extraction | `R_self` | model - measured +/- paired case SEM |
|---|---:|---:|
| measured catalogue | -0.243182 | -- |
| unconditional object mean, catalogue radius bin | -0.030177 | +0.213005 +/- 0.024133 |
| flow draws, generated radius bin | -0.232147 | +0.011035 +/- 0.024108 |
| flow draws, generated radius bin and generated full anchor | -0.246314 | -0.003131 +/- 0.024073 |

Thus the apparently catastrophic first-bin flow error vanishes when the same
random variable defines the model response and its radius bin.  The first-bin
leg decomposition is also diagnostic: the measured zero/sheared projected
terms are -0.000415/-0.243598, while the generated-and-anchored flow gives
-0.002848/-0.249162 under the same-latent convention.  The sharp negative
response is a correlation between generated g=0 radius and paired sheared-leg
shape; the newer control above shows that the current loss does not identify
that correlation.  The old operations nonetheless do not commute:
`E[R|catalogue radius bin]` cannot be estimated by binning `E_flow[R|truth]`
on a different realised radius draw.

One real boundary limitation remains in the marginal radius density.  The
catalogue places 3.1027% of selected rows in the first bin, compared with
2.0509% of generated radius-valid draws and 2.0236% after the generated full
anchor.  The flow also puts 1.0584% of all draws below the strict 3-pixel
boundary; that leaked mass is almost exactly the missing first-bin mass.  An
unconstrained smooth density has blurred the hard truncation.  This is a
support/normalization issue to address with an explicit truncated-support
construction, not by conditioning the model on measured `FLUX_RADIUS`.

The immutable result is
`$DATA_DIR/sbsi_caches/fixed_g0_m258_r060_v2/generated_radius_coupling_c0_19_v1/result.json`.
Validation: 9 focused tests and the full suite (367 passed, 2 skipped) pass;
Python compilation, Ruff, Bash syntax, and `git diff --check` are clean;
scheduler peak RSS was 762,448 KiB.  Limitations: the comparison is the
flow-only half-shear family at `|g|=0.05`, cases 0--19 include 17 training and
three validation cases, and the finite-draw error is folded into the reported
case scatter.  This result corrects the interpretation of the catalogue-radius
residual plot but does not by itself revise the ConstGold total `m`, whose
blending and classifier terms remain separate.

## 2026-09-16 — Staged-flow training residual resolved at the fixed-g0 radius boundary

At the owner's request, added
`plots/flow_training_residual_vs_flux_radius.py` and the reproducible CPU entry
point `jobs/job_plot_flow_training_residual_radius.sh`.  The diagnostic uses
the existing 64-draw staged-flow cache for 40 cases explicitly contained in
the flow's training split (3,615,648 matched objects).  The model remains
conditioned only on truth/context: measured g=0 `FLUX_RADIUS` is used only
after prediction as the diagnostic bin coordinate.  The residual sign is
`R_self(measured) - R_self(flow)`, summaries weight cases equally, and error
bars are one SEM over paired case residuals.  The exact v2 population is
retained: detected g=0, `MAG_AUTO(g=0)<25.8`, strict
`FLUX_RADIUS(g=0)>3.0` pixels, no truth cut, and no sheared-leg recut.

The boundary is resolved into 0.025-pixel (0.005-arcsec) bins from 3.0 to 3.5
pixels.  The first three bins read:

| fixed-g0 radius (arcsec) | objects | measured `R_self` | flow `R_self` | measured-flow residual +/- case SEM |
|---|---:|---:|---:|---:|
| 0.600--0.605 | 34,818 | -0.3431 | -0.0680 | -0.2751 +/- 0.0222 |
| 0.605--0.610 | 36,739 | -0.2672 | -0.0312 | -0.2361 +/- 0.0225 |
| 0.610--0.615 | 37,467 | -0.2152 | +0.0013 | -0.2165 +/- 0.0220 |

These are 12.4, 10.5, and 9.8 case-level standard errors from zero.  The
residual relaxes rapidly, becomes consistent with zero near 0.645--0.660
arcsec, and changes sign around 0.66 arcsec.  Thus the sharp failure is present
in the flow's training cases and is not a held-out-generalization effect.  The
plot does not add a measurement condition.  It deliberately shows the old
catalogue-radius/unconditional-mean extraction, so it localizes the catalogue
residual but is not the flow's generative conditional curve.  The companion
generated-radius audit in the newer entry above shows that the truth-only flow
reproduces the boundary response when its own generated radius defines the bin.

Outputs are
`plots/figures/flow_training_residual_vs_flux_radius_c20_71.{png,pdf,csv,json}`.
The two panels show a boundary zoom and the full central radius range; both
overlay object-count density, mark the strict 3-pixel boundary, and shade the
previous 0.600--0.6154-arcsec slice.  CPU job 16527536 completed in 20 seconds
with exit 0 and empty stderr.  The PNG was inspected at its native resolution
and the PDF is a valid single-page vector file.  `py_compile`, job-script
`bash -n`, `git diff --check`, and the two focused suites pass (5 tests).

## 2026-09-16 — ConstGold property panels remade for both 6x6x6 grid flows

Both experimental grid continuations completed successfully at the scheduler
level (array 16524796, exit 0:0 with empty stderr).  The truth-grid branch
selected phase epoch 20 and stopped at phase 40; its fixed validation grid
score is 45.3214 with MC jackknife SE 0.1570.  The measured-anchor grid selected
phase epoch 2 and stopped at phase 22, but its score -14.2548 has MC jackknife
SE 98.7104.  The latter selection is therefore integration-noise dominated and
must not be interpreted as evidence of a better grid objective, despite the
checkpoint and training job passing their mechanical checks.

At the owner's request, the 100-case property-binned ConstGold diagnostic was
rerun for both selected checkpoints.  Added
`jobs/job_constgold_property_bias_grid_flows.sh`, a two-task/two-GPU array that
changes only the flow checkpoint while holding cases 40--139, fixed-g0 anchors,
`R_blend` blocks, three-seed classifier ensemble, 64-draw antithetic sampling,
binning, and 10,000 common case-bootstrap resamples fixed.  Array 16527050
completed both tasks in 20m44s with empty stderr.  The plotting result now
stores and displays an optional model label so the otherwise identical panels
remain self-identifying.

The collapsed results are:

| model | matched usable `m` | actual per-leg `m` | classifier-weighted `m` |
|---|---:|---:|---:|
| staged grid-free reference | -1.0227 +/- 0.2156% | -1.2920 +/- 0.2168% | -1.7333 +/- 0.2152% |
| truth-property grid | -1.2347 +/- 0.2156% | -1.5095 +/- 0.2167% | -2.9591 +/- 0.2128% |
| measured-anchor grid | -1.0271 +/- 0.2161% | -1.2979 +/- 0.2173% | -2.5969 +/- 0.2136% |

Thus neither grid is a calibration improvement.  In the smallest fixed-g0
`FLUX_RADIUS` bin, matched `m` changes from -86.6070% to -88.3478% for the
truth grid and -81.6566% for the measured grid.  Conditioning the grid on the
measured anchor size moves the boundary-bin error by about +4.95 percentage
points but leaves it catastrophic and redistributes large errors into true-size
bins (up to a +40.8-point change for matched usability and +70.3 points for the
classifier branch).  This is consistent with the noisy measured-grid selector
and is diagnostic, not an acceptance result.

Figures and tables are
`plots/figures/constgold_property_bias_grid_{truth,measured}_c40_139.{png,pdf,csv}`
with matching `_histograms.csv` files.  The histogram tables are byte-identical
between models, all 144 response rows per result are finite and use identical
anchor counts/x coordinates, and both one-page PDFs are 771.6 x 475.15 pt.
Both PNGs were inspected at full resolution: the colorblind-safe redundant
line encodings, uncertainty bars, six panel labels, model titles, and histogram
overlays are present and unclipped.  `bash -n`, `py_compile`, and
`git diff --check` pass; the focused property suite reports 10 passed and the
full suite reports 361 passed, 2 skipped, with the existing small-bootstrap
degrees-of-freedom warning.

## 2026-09-16 — Two fixed-g0 6x6x6 physical-response grid continuations prepared for training

Added a controlled pair of experimental grid-based continuations from the
selected epoch-140 fixed-g0 NLL checkpoint.  Both branches use the corrected
v2 population (`MAG_AUTO(g=0)<25.8`, `FLUX_RADIUS(g=0)>3.0` pixels), freeze
cell membership on that g=0 leg, carry exact anchor keys to g=0.05 without a
sheared recut, and retain the existing 160/40 case split.  The `truth` branch
uses marginal train-quantile bins in true r magnitude, circularized true
radius, and cached baseline `r_blend`; the `measured` branch replaces the first
two axes with g=0 `MAG_AUTO` and g=0 `FLUX_RADIUS`.  Both are 6x6x6, with
validation tails retained in the outer cells.

`sbsi/flow_grid_physical.py` restores the archived estimator mechanics without
an archive import: per-cell forward WLS for the four sky-frame shape-response
matrix entries and two baseline-frame physical-radius response entries, joint
delete-one-case covariance, correlation-equilibrated inverse precision with no
floor, equal cell weighting, and independent stochastic cross quadratics.
`scripts/train_fixed_g0_grid_flows.py` prepares immutable cell/statistics
artifacts for both grids, checks all baseline keys exactly, hashes the domain,
parent checkpoint/optimizer, code, and prepared arrays, and continues with the
saved AdamW state at learning rate 1.5625e-6.  Flux remains NLL-only.  The
training recipe uses weight 1, 12 cells x 128 pairs per group, 16 antithetic
draws, 4M NLL rows per epoch, a deterministic cell-stratified 50k-pair
validation subset, and parent epoch zero in model selection.  This reproduces
the historical covariance-normalized objective scale and is not numerically
strength-matched to the current grid-free weight-10 loss.

Added `jobs/job_prepare_fixed_g0_grid_flows.sh` and a two-element,
two-concurrent-A40 `jobs/job_train_fixed_g0_grid_flows.sh` array.  Added six
focused tests covering grid boundaries/tails, baseline-frame rotation, exact
synthetic recovery of all six response components, response-gradient replay,
finite-integration correction, and baseline-only axis construction.

Validation: both job scripts pass `bash -n`; the new Python files pass
`py_compile`; `git diff --check` is clean for the added files; focused tests
pass (6 passed); the complete suite passes (361 passed, 2 skipped, one existing
small-bootstrap degrees-of-freedom warning).  Ruff is not installed in the
project interpreter.  CPU preparation job 16524795 completed successfully in
2m17s.  The truth grid has minimum train/validation cell counts 499/134 and
minimum case coverage 150/38; the measured grid has 794/205 and 159/40.  The
largest validation correlation condition numbers are 83.0 and 85.7,
respectively.  GPU array 16524796 then started both tasks, one A40 each, with
empty stderr.  The common parent validation NLL is -4.50820443; its initial
grid losses are 46.06766812 (truth) and 57.12940243 (measured).  These are
pre-training objective baselines, not calibration measurements.  One-shot
monitor job 16524909 is scheduled for roughly two hours after startup via
`jobs/job_check_fixed_g0_grid_flows_2h.sh`.

## 2026-09-16 — The small-FLUX_RADIUS bias is a sharp fixed-g0 selection-boundary failure, replicated on the flow's own catalogue

The owner noticed that the property-panel bias appears concentrated in the
smallest fixed-g0 `FLUX_RADIUS` bin.  It is, but the pooled `m` understates the
severity because the rest of the radius range cancels it.

### Exact contribution to the 100-case ConstGold result

The original first quantile is `0.6002 <= FLUX_RADIUS < 0.6572` arcsec,
immediately above the strict fixed-g0 cohort cut at 0.60 arcsec.  In the matched
branch it contains 1,130,618 objects (12.54%) and reads:

```text
R_measured = 0.029642
R_flow     = 0.132022
R_blend    = 0.089307
R_model    = 0.221329
m          = -86.607%
```

Its contribution to the *global* `m` is -3.5018 percentage points.  All other
radius bins contribute +2.4792 points, leaving the reported -1.0227%.  Thus the
headline is a cancellation, not a uniformly one-percent error.  Removing the
whole first bin would expose `m=+2.5836 +/- 0.2224%`; even removing only the
lowest 3.11% (`r_flux<0.6154`) gives the post-hoc value `+0.2974 +/- 0.2119%`.
Neither is a proposed cut or correction: both change the sample after seeing
the answer.

`scripts/diagnose_constgold_response_split.py` now supports
`--anchor-bin-column g0_flux_radius_arcsec`.  On 100 cases it resolves the
first old bin into four equal-count slices:

| fixed-g0 radius (arcsec) | `R_measured` | `R_flow` | `R_blend` | `R_model` | measured-model | SE of residual | contribution to global `m` |
|---|---:|---:|---:|---:|---:|---:|---:|
| <0.6154 | -0.2504 | -0.0363 | 0.0768 | 0.0405 | -0.2910 | 0.0086 | -1.3195 pp |
| 0.6154--0.6298 | -0.0624 | 0.0750 | 0.0860 | 0.1610 | -0.2233 | 0.0113 | -1.0227 pp |
| 0.6298--0.6436 | 0.1101 | 0.1910 | 0.0934 | 0.2844 | -0.1743 | 0.0143 | -0.7952 pp |
| 0.6436--0.6572 | 0.3180 | 0.2965 | 0.1009 | 0.3973 | -0.0794 | 0.0159 | -0.3645 pp |

The residual is 33.7, 19.7, 12.2, and 5.0 standard errors from zero and decays
monotonically away from the cut.  The huge ratio in the first slice is partly
because `R_model` is near zero, but the absolute -0.291 response residual and
its -1.32-point global contribution show that this is not merely a ratio
artifact.

The effect is not the faint-primary leak seen in crowded bins.  Crossing the
four boundary slices with true magnitude assigns -0.6941, -1.4684, -1.3176,
and -0.0218 global percentage points to `r<24`, `24<=r<25`, `25<=r<26`, and
`r>=26`, respectively.  The absolute residual remains negative within the
first three magnitude bands; the `r>=26` population is only 2.19% of this
radius slice and contributes 0.6% of its bias.

### Independent flow-only replication

New `scripts/diagnose_flow_self_response_by_radius.py` reads the cached staged
flow predictions on the flow's own zero/half-shear catalogue, with no blending
emulator.  All 1,807,351 measured rows join one-to-one.  On the same first four
edges, measured/model `R_self` is:

```text
<0.6154          -0.2432 / -0.0302   model-minus-measured +0.2130
0.6154--0.6298   -0.0853 / +0.0851   model-minus-measured +0.1704
0.6298--0.6436   +0.1252 / +0.1978   model-minus-measured +0.0725
0.6436--0.6572   +0.2965 / +0.3012   model-minus-measured +0.0047
```

The pattern is present in the 17 training cases as well as the three held-out
cases, so it is neither a ConstGold-only anomaly nor primarily a held-out
generalization failure.  Averaged over the old first bin, measured/model
`R_self` is 0.0245/0.1392: the 0.1147 flow excess directly reproduces about 60%
of ConstGold's 0.1917 total excess.  ConstGold also adds positive predicted
`R_blend=0.0893` while its total measured response is only 0.0296, suggesting
that the blending term supplies much of the remainder.  That last allocation
is an inference across the half-shear and ConstGold families, not a direct
measurement of the two ConstGold components; an independent blend-scene test
binned on fixed-g0 radius is still needed.

### Interpretation

The leading mechanism is post-selection conditioning.  Fixed-g0
`FLUX_RADIUS` is a noisy measured output used to define `S0`, but its realized
value is not a flow condition.  The flow represents the response marginalized
over `p(y_g | x_g,S0)`; after evaluation, selecting the narrow subset whose
realized `g=0` radius barely scattered above 0.60 arcsec conditions on
information absent from `x_g`.  The flow regresses that subgroup toward the
cohort mean and misses its sharply negative/near-zero self-response.  A sound
model-level test is therefore a joint anchor-conditioned model such as
`p(y_+,y_- | x,y_0,S0)` (or an explicit response model conditioned on the g=0
measurement), plus an independent radius-binned check of `R_blend`.  Raising
the radius cut until the pooled number happens to pass would be an empirical
sample change, not a demonstrated correction.

### Artifacts and validation

CPU merge job 16520883 and one-GPU job 16520884 produced
`$V2/constgold_small_radius_c40_139_v1/result.json`; the GPU job completed in
9m18s with only PyTorch's allocator deprecation warning.  CPU job 16520976
produced `flow_self_by_radius.json` in 11 seconds with empty stderr.  Added
`jobs/job_merge_constgold_small_radius_lookup.sh`,
`jobs/job_diagnose_constgold_small_radius.sh`, two focused flow-radius tests,
and two fixed-anchor-column tests.  `ruff check`, both job-script syntax checks,
and `git diff --check` pass; focused tests report 16 and 3 passed, and the full
suite reports 355 passed and 2 skipped.

## 2026-09-16 — Anchor-population histograms overlaid on the ConstGold property panels

At the owner's request, `plots/constgold_property_bias.py plot` now overlays a
light gray, 36-bin relative-density histogram of the fixed-g0 anchor population
behind each response-bias panel.  These are independently accumulated fine
histograms, not the eight equal-count response bins, and therefore show the
actual property distributions.  Linear, log, and symmetric-log panels bin
uniformly in their respective display coordinates.  The robust central display
ranges retain 93.2--96.9% of the 9,077,958 anchors without allowing a few
extreme tails to compress the response curves; exact counts, edges, and retained
fractions are in
`plots/figures/constgold_property_bias_c40_139_histograms.csv`.  The response
values, bootstrap errors, and original response CSV are unchanged.

The production PNG and PDF were regenerated by CPU scheduler job 16520787 in
32 seconds with empty stderr.  The histogram computation can be disabled with
`--no-histogram`, and its resolution is configurable with `--histogram-bins`.
Three display-coordinate regression tests were added to
`tests/test_constgold_property_bias.py`.

**Validation.**  A two-case histogram-overlay smoke plot was inspected, the
100-case PNG was inspected at full resolution, the vector PDF is a valid
one-page 771.6 x 475.2 pt document, `ruff check` and `git diff --check` pass,
the targeted suite reports 10 passed, and the full suite reports 350 passed and
2 skipped.

## 2026-09-16 — ConstGold property-binned `m` for matched, actual per-leg, and classifier-weighted usability

The owner asked for the actual-flag diagnostic by object properties, alongside
the matched-usable and per-leg-classifier versions, with more cases and a panel
plot.  New `plots/constgold_property_bias.py` now computes all three branches
from the same flow draws and reports eight equal-anchor-count bins in six
leg-invariant properties: true primary magnitude, true circularized size,
fixed-g0 magnitude boost, fixed-g0 measured size, brightest-neighbour flux, and
absolute blending response.  Bins are frozen before either ConstGold shear leg
is read, and the same 10,000 case-bootstrap resamples are reused across all
properties, bins, and branches.  The plotting subcommand writes a colour-blind
safe 2x3 panel as PNG/PDF plus the complete bin table as CSV.

The calculation was extended from cases 40--89 to cases 40--139 (100 cases,
9,077,958 anchors; 9,015,013 matched-usable objects, 99.3066%).  Collapsed over
properties, the 100-case results are:

| branch | `m` | case-bootstrap SE |
|---|---:|---:|
| matched usable | -1.0227% | 0.2156 percentage points |
| actual per-leg usable flags | -1.2920% | 0.2168 percentage points |
| classifier-weighted per leg | -1.7333% | 0.2152 percentage points |

The profiles are strongly property dependent.  The actual-flag curve reaches
-8.68 +/- 0.58% in the highest neighbour-flux bin and -14.83 +/- 0.44% in the
highest absolute-`R_blend` bin.  The classifier creates additional structure:
it reaches +8.78 +/- 0.92% in the faintest true-magnitude bin and +29.50 +/-
1.93% in the smallest true-size bin.  The smallest fixed-g0 measured-size bin
has measured response only 0.0283 and hence a ratio-defined `m` of -87.21 +/-
2.75%; this is a response-collapse regime, not a global calibration offset.
Only a few bins lie inside the +/-0.3% target band.

Files added: `plots/constgold_property_bias.py`,
`tests/test_constgold_property_bias.py`,
`jobs/job_build_constgold_property_rblend.sh`, and
`jobs/job_constgold_property_bias.sh`.  Results are at
`$V2/constgold_property_bias_c40_139_v1/result.json`; figure products are
`plots/figures/constgold_property_bias_c40_139.{png,pdf,csv}`.  The missing
cases 90--139 `R_blend_abs` blocks were built by CPU array 16520671, and the
one-GPU analysis/plot job 16520672 completed in 9m45s with empty stderr.  The
same implementation reproduces all three certified cases40--89 values within
1.7e-13 percentage points.

**Validation.**  The two-case compute and plot smoke tests completed; the
production JSON has 100 case reports and 144 finite summary rows, all eight-bin
populations agree within 0.51% (the measured-size property is quantized), and
all output files are nonempty.  `ruff check` passed, both job scripts pass
`bash -n`, `git diff --check` passed, the seven targeted tests passed, and the
full suite reports 347 passed and 2 skipped.

**Limitations.**  Cases 40--139 reuse finite truth scenes involved in model
development, so this is a diagnostic rather than fresh acceptance evidence.
The modeled-usable branch compares classifier-weighted model response with
actual-flag measured response.  One flow seed and one common antithetic
integration seed do not quantify training or integration uncertainty.

## 2026-09-15 — The flow, read on its own catalogue: right where it trained, ~2% high where it did not, and the zero-shear leg is truncated while the sheared leg is not

The owner asked whether the quiet-bin flow reads (+2.1% at r<23, +3.2% at
24-25, +7.7% at 25-26) were measured on the flow's own training data, and for
the direct comparison of mean predicted `R_self` against mean measured
`R_self`.  Both questions are now answered, and answering the second one turned
up a structural asymmetry in the flow's training data.

**Those reads are not on the flow's training data.**  They come from
`diagnose_constgold_response_split.py` on ConstGold cases 40-89.  The flow
trains on the half-shear g0/g05 legs of the fixed-g0 domain under the split
recorded in `$V2/domain/manifest.json` (`seed` 501, 160 train cases, 40
validation cases).  ConstGold is a different sim family on a different split and
is firewalled off from flow training, so those numbers are held out.

**The direct comparison.**  New `scripts/diagnose_flow_self_response_bias.py`
matches the two domain legs with `matched_key_indices`, projects the measured
shape difference onto the applied shear with the builder's own
`projected_response`, and joins the flow's cached `R_self_model`.  Run on the
existing `response_profile_predictions_c0_19_v2` predictions (cases 0-19):

    all           1,807,351 objects  measured 0.49157  model 0.49121  m = +0.07 +- 0.52 %
    trained (17 cases)  1,536,137    measured 0.49315  model 0.49091  m = +0.46 +- 0.55 %
    held-out (3 cases)    271,214    measured 0.48258  model 0.49290  m = -2.09 +- 0.68 %

All 1,807,351 prediction rows joined; zero rows unmatched on either side.

Two things follow.  First, **the sign is opposite to ConstGold**: on its own
catalogue the flow is not low, it is about 2% *high* on cases it did not train
on, whereas ConstGold's quiet bins say the model is low by 2-8%.  So "the flow
is low" is not a property that transports between the two sim families.
Second, there is a **train/held-out gap of about 2.5 points** -- essentially
exact reproduction on trained cases, over-prediction on held-out ones -- which
is the shape of a generalisation problem rather than a calibration offset.  It
rests on only three held-out cases (2 sigma), so it is a lead, not a result.

By true primary magnitude, pooled over all 20 cases:

    r<23    184,141  measured 1.1403  model 1.1392  m = +0.09 +- 0.47 %
    23-24   220,131  measured 0.9290  model 0.9319  m = -0.31 +- 0.92 %
    24-25   496,348  measured 0.5765  model 0.5653  m = +1.99 +- 1.30 %
    25-26   711,757  measured 0.2520  model 0.2514  m = +0.24 +- 1.86 %
    r>=26   194,974  measured 0.0432  model 0.0684  m = -36.9 +- 12.2 %

The last row is 10.8% of the sample.  Its measured response has all but
collapsed (0.043) while the flow still assigns it 0.068, which misplaces 0.0027
in `R_flow` -- 0.4% of the total R, i.e. the whole |m| < 0.3% budget from one
magnitude bin.  Splitting on crowding (`nbr_flux_max` quartiles) gives +0.88,
+0.95, +0.84% in the three quieter quartiles and -3.86 +- 1.64% in the most
crowded one; inside the quietest quartile the faint-end trend is stronger
(+2.59% at 24-25, +5.48% at 25-26, -42% at r>=26), which does resemble the
ConstGold quiet-bin pattern.

**The likely cause: the two legs are truncated differently.**  Counting rows in
the domain NPZs against the anchor rule itself (`measured_mag_auto < 25.8`,
`measured_flux_radius > 3.0` px, i.e. sampled flux > 47.863), over cases 0-19:

    g0  leg   90,817 rows/case   0.000% violate the anchor   0.000% among true r>=26
    g05 leg   90,368 rows/case   1.068% violate the anchor   3.659% among true r>=26

The zero-shear leg is exactly truncated, by construction -- it *is* the anchor.
The sheared leg is not, because the domain deliberately never recuts it
(`sheared_leg_recut: false` in the prediction manifest).  So the flow fits a
single conditional density `p(measured | truth, neighbours)` to a sample that is
hard-truncated at the zero-shear contexts and untruncated at the sheared ones,
and `physical_context_means` then averages over *every* draw at both legs.  The
measured target is a selected zero-shear mean minus an unselected sheared mean;
the prediction is unselected minus unselected.  Those are not the same quantity,
and the mismatch must be largest exactly where the cut bites -- which is where
the r>=26 row sits.

**Validation.**  `ruff check` clean on the new scripts.  The comparison job ran
on `cluster` (`--constraint=x86-64-v3`, srun 16520384) and the leg-truncation
count on srun 16520425.  Result JSON at
`$V2/response_profile_predictions_c0_19_v2/flow_self_response_bias.json`.

**Limitations.**  The held-out side is three cases only.  The comparison is at
|g| = 0.05 on the half-shear family with random shear direction per case, while
the ConstGold reads are at g = 0.02 constant shear, so a shear-amplitude or
sim-family difference is not excluded.  The truncation asymmetry is so far a
*count* of how differently the two legs are cut, not yet a demonstration that it
produces the observed response error.

**Next steps.**  (1) `scripts/build_flow_self_response_predictions.py` and
`jobs/job_build_flow_self_response_predictions.sh` are predicting `R_self` for
all 40 held-out cases and 40 fresh trained cases (jobs 16520405, 16520406 on
`cip`, ~12 s/case), which turns the 2-sigma train/held-out gap into a real
measurement.  (2) `scripts/diagnose_flow_anchor_truncation.py` (written, not yet
run) predicts the same objects three ways from one set of common random numbers
-- both legs unconditional, zero leg conditioned on the anchor, both legs
conditioned -- and puts each next to the measured response.  The anchor rule
applied there is the domain's own rule evaluated on the flow's own samples; no
offset is fitted.  It cannot apply the `detected` half of the anchor, since the
flow does not model detection.

## 2026-09-15 — Correction: the "cohort magnitude composition" row was blending-weighted, not object-weighted

The owner asked how ConstGold can hold so many primaries fainter than r = 26
when the cut is 25.8. Checking it found an error in my own bookkeeping, and
one genuine and interesting fact underneath.

**The error.** The cohort row of the primary-magnitude composition table has
been quoted in this log and in memory as `1.71 / 5.26 / 19.84 / 48.41 / 24.77`
percent for `r<23 / 23-24 / 24-25 / 25-26 / r>=26`, described as the object
composition. It is not. Recomputed directly over cases 40-49 (909,131
primaries, `rblend_v2_baseline_abs_cases40_89.feather` joined to `r_input`):

| | r<23 | 23-24 | 24-25 | 25-26 | r>=26 |
|---|---|---|---|---|---|
| **cohort, by object** | **10.11** | **12.15** | **27.46** | **39.38** | **10.89** |
| cohort, by blending | 2.83 | 5.79 | 20.18 | 47.37 | 23.83 |
| bin00 (quietest) | 29.69 | 23.53 | 26.91 | 19.36 | 0.50 |
| bin11 (crowdest) | 0.98 | 3.98 | 17.25 | 48.27 | 29.52 |

The figure previously quoted is the *blending-weighted* row. The per-bin rows
were always object-weighted and are confirmed correct (old bin00 29.67 against
29.69, old bin11 0.99 against 0.98). Only the cohort row was mislabelled.

**What it changes.** The argument that the quiet-bin flow read is measured on
an unrepresentative slice stands, but it is weaker than stated: bin00 is 29.7%
brighter than r = 23 against **10.1%** cohort-wide, a factor of 2.9, not the
factor of 17 the wrong row implied. Nothing that used the per-bin rows is
affected, including the previous entry's tail caveat, which cited bin11's 29.3%
and is right (29.52% measured here).

**Why any primary is fainter than r = 26 at all.** The cohort is a *measured*
selection. `sbsi/fixed_g0_domain.py` applies `MAG_AUTO < 25.8` and
`FLUX_RADIUS > 3.0 px` on the g=0 leg and states in its own docstring that it
is not a truth-property cut, so true magnitude is unbounded above; the cohort
reaches r = 28.817. The reservoir is large: 68.2% of every galaxy rendered in
these tiles is fainter than r = 26 (checked on both sim families, whose input
magnitude distributions are identical to r = 29.0, median 26.7). So a modest
leak rate across a measured cut fills a tenth of the selected sample.

**And the leak is not uniform in crowding.** The faint fraction rises
monotonically with blending, 0.50% in bin00 to 29.52% in bin11. That is
physical rather than incidental: a truly faint galaxy passes a measured
magnitude cut mainly when a neighbour's flux brightens its `MAG_AUTO`, so the
objects that leak in are preferentially the blended ones. ConstGold's crowded
tail is therefore disproportionately made of galaxies that are only in the
cohort *because* they are blended — which is also why the anchor population,
selected under a true r < 26 cut, cannot reproduce it.

Validation: `srun` jobs 16520205, 16520209, 16520214, 16520218 on `cluster`,
all read-only. No files were regenerated.

## 2026-09-15 — The fork lands: the summed emulator does fail in the crowded tail, and the failure is a faint-primary effect

The previous entry left one clean question open. ConstGold's top blending bin
over-predicts, and positivity forces at least part of that onto the emulator,
but nothing so far said how much. The primary-unsheared scene instrument can
answer it with no flow anywhere in the arithmetic, provided it is read as a
function of crowding rather than pooled. It now has been.

### What was built

- `scripts/dump_anchorblend_scene_primaries.py` (new). Reproduces the
  production run `anchorblend_g002_measured_cuts_v1` one row per primary
  instead of one number per case: predicted `R_scene` (signed) and
  `R_scene_abs` (sum of per-pair magnitudes), the pair count, the primary's
  true `r`/`Re`/`sersic_n`, and the selected `e1`/`e2` on each sheared leg.
  Selection, the `MAG_AUTO < 25.8` / `FLUX_RADIUS > 3.0 px` cuts, the stratum
  inverse-sampling weights and the emulator call are the production contract;
  only that run's `manifest.json` is read, never its code, and the job puts its
  `source_snapshot` on `PYTHONPATH` so the emulator is the same code that
  produced the published number.
- `scripts/diagnose_anchorblend_response_by_crowding.py` (new). Bins those
  primaries on a predicted column, pools the two strata inside a case by their
  weights, averages over cases with equal weight and bootstraps over cases —
  the production run's own estimator and error unit. `--edge` takes explicit
  edges so the scene can be read on ConstGold's edges;
  `--primary-magnitude-min/max` restricts on the primary's true magnitude,
  which is a truth property and so touches neither leg's selection.
- `jobs/job_dump_anchorblend_scene_primaries.sh`,
  `jobs/job_diagnose_anchorblend_by_crowding.sh`,
  `jobs/job_diagnose_anchorblend_on_constgold_edges.sh`,
  `jobs/job_diagnose_anchorblend_tail_by_magnitude.sh` (new). All CPU-only on
  `cluster`; no GPU was used, so the two-GPU ConstGold limit is untouched.
- `tests/test_anchorblend_response_by_crowding.py` (new, 8 tests). Weighted
  quantile edges follow the weight and not the row count; a planted per-bin
  error is recovered exactly; strata pool by their inverse-sampling weight and
  not by a row average; a large intrinsic shape cancels between the legs so a
  correct model reads exactly zero; the unmatched branch uses each leg's own
  selection; the reported share of blending is response-weighted, not
  object-weighted; an unpopulated cell is refused.

### Verification before any science was read

`dump_anchorblend_scene_primaries.py` was checked against the production
`matched/parts/*.json` for cases 400–401, both strata: identical selected
counts (3373, 2968, 3439, 2959), identical stratum weights to ten digits, and
identical weighted `e1`/`e2` sums to all eight printed digits on both legs. The
weighted emulator response sums agree to 2e-9 relative, which is summation
order. Re-pooled through the new estimator the whole matched population returns
`R_sim = 0.1413711868`, `R_emulator = 0.1379700887`, `m = +2.4650981%` —
the published production numbers, digit for digit.

Commands:

```
sbatch jobs/job_dump_anchorblend_scene_primaries.sh smoke          # 16519925
sbatch --array=0-19 jobs/job_dump_anchorblend_scene_primaries.sh run  # 16519938
sbatch jobs/job_diagnose_anchorblend_by_crowding.sh                # 16519972
sbatch jobs/job_diagnose_anchorblend_on_constgold_edges.sh         # 16519981
sbatch jobs/job_diagnose_anchorblend_tail_by_magnitude.sh          # 16520009
```

All COMPLETED; the 20-task dump took 25–43 s per task, 1000 parts under
`$DATA_DIR/sbsi_caches/fixed_g0_m258_r060_v2/anchorblend_g002_primaries_v1/`.
`pytest tests -q`: 335 passed, 2 skipped. `ruff check` clean on all new files.

### The crowded tail is present in the scene sims

The first thing to settle was whether the instrument can even reach ConstGold's
tail. It can. On ConstGold's own `R_blend_abs` quantile edges:

| ConstGold bin | objects %, ConstGold | objects %, scene | blending %, ConstGold | blending %, scene |
|---|---|---|---|---|
| bin09 (0.275–0.492) | 8.33 | 6.50 | 12.71 | 13.68 |
| bin10 (0.492–0.867) | 8.33 | 5.91 | 24.54 | 24.27 |
| bin11 (> 0.867) | 8.33 | 5.21 | 44.76 | 38.56 |

The scene population is quieter on average (mean `|R_blend|` 0.178 against
0.242) but its top bin sits on the same slice of the blending axis and carries
a comparable share of the cohort's blending. The 99th percentile of the scene's
`R_scene_abs` is 1.35 and its maximum 3.81.

### The answer

Measured response against emulator prediction, on ConstGold's edges, matched
branch, 500 cases, 3,190,026 primaries. `m = R_sim/R_emulator − 1`; positive
means the emulator is **low**, negative means it **over-predicts**.

| ConstGold bin | objects | R_sim | R_emulator | m % | blending share % |
|---|---|---|---|---|---|
| bin09 | 201,516 | 0.28274 | 0.28876 | −2.08 ± 4.77 | 13.68 |
| bin10 | 183,345 | 0.66760 | 0.59527 | **+12.15 ± 1.94** | 24.27 |
| bin11 | 161,475 | 1.03439 | 1.09840 | **−5.83 ± 1.27** | 38.56 |

So the summed emulator **does** fail in the crowded tail, with no flow involved
in the statement: it over-predicts by 5.8% at 4.6 sd in the top bin, and it is
12% **low** in the bin below. The two errors have opposite sign, which is why
the pooled scene number (+2.47%) looked healthy — it is a cancellation, exactly
like ConstGold's pooled `m`.

### And the tail failure is a faint-primary effect

Splitting the same slice of the blending axis by the primary's true magnitude
(job 16520009) turns a single number into a trend:

| primary r | objects in tail | R_sim | R_emulator | m % in bin11 | m % in bin10 |
|---|---|---|---|---|---|
| < 24 | 15,454 | 1.09210 | 1.03423 | +5.60 ± 4.66 | +15.01 ± 9.07 |
| 24–25 | 49,090 | 1.10437 | 1.11309 | −0.78 ± 2.30 | +7.71 ± 2.97 |
| 25–26 | 96,931 | 0.99358 | 1.10072 | **−9.73 ± 1.60** | +13.86 ± 2.38 |

The over-prediction is not a property of crowding alone. At the same crowding
the emulator is fine, or slightly low, on bright primaries and over-predicts by
nearly 11% on faint ones. The bin10 column shows the emulator low in every
band, so the sign flip between bin10 and bin11 is a property of the very top of
the blending range, not of magnitude.

That −9.73 ± 1.60% deserves emphasis: ConstGold's positivity bound requires its
bin11 emulator over-prediction to be at least 9.28 ± 0.89%, and the scene test
delivers 9.73 ± 1.60% for r 25–26 with no flow in the arithmetic at all. The
part of ConstGold's tail failure that positivity forced onto the emulator is
now measured directly, and it is there.

### Limitations, stated plainly

- **The scene sims contain no primary fainter than r = 26.0.** The anchor
  selection applied a true-magnitude cut the ConstGold cohort does not have:
  `anchors_case400.feather` tops out at r = 25.714 (original arm) and
  25.99992 (v2 complement), while ConstGold's cohort reaches r = 28.817,
  because `sbsi/fixed_g0_domain.py` selects on *measured* `MAG_AUTO < 25.8`
  and says so explicitly ("It is not a truth-property cut"). ConstGold's bin11
  is 29.5% fainter than r = 26 by object count and 30.3% by blending share, so
  the scene test covers about 70% of that bin, and the uncovered part is the
  faint end, which is the end where the failure grows. A transfer of these
  numbers to ConstGold's cohort is an extrapolation over that third and is not
  made here.
- The two cohorts also differ in composition inside each band, so a proper
  transfer needs ConstGold's bin11 blending share per magnitude band, which has
  not been measured. Until it is, no cohort-wide split of `R` is quoted.
- The bins are a deterministic partition of truth space (the emulator is a
  deterministic function of the truth catalogue), so binning on a predicted
  column introduces no selection noise; but a bin labelled crowded is where the
  *model* says crowded, and the per-bin `m` is conditional on that.
- `R_sim_null_21` stays within about one case standard error of zero in every
  reported cell, so the instrument's own null is clean.

### Next

- Measure ConstGold's bin10 and bin11 blending share per magnitude band, then
  transfer these per-band emulator errors onto ConstGold and read what is left
  for the flow in the tail. That is the last piece of the global split.
- The scene's bin10 says the emulator is 12% low there while ConstGold's bin10
  reads `m = +4.17%` with a model flow of 0.19168. Correcting bin10's blending
  upward would push the flow in that bin sharply down. Worth doing properly
  after the reweighting above, not before.
- Extend the anchor population below r = 26 if the tail transfer is to be
  closed rather than bounded.
- Still open from before: the r 23–24 emulator band, the orthogonal-null
  degradation, and the adjacent-in-magnitude deficit.

## 2026-09-15 — Correction: a direct scene test refutes the "blending is 15% high" split

The owner pointed at the primary-unsheared constant-shear scene test recorded
below, which measures the summed `R_blend` directly. It is right and the split
published in the entry that follows is withdrawn.

### Why the scene test outranks the remainder estimate

It measures the same model quantity, built the same way. `run.py:332` in
`anchorblend_g002_measured_cuts_v1` does
`predictor.predict_response(primary, frame).groupby(primary_column)["response"].sum()`;
`scripts/build_constgold_fixed_g0_blend_lookup.py:172` does the identical
groupby-sum, on the identical model file (SHA
`f04a19d3...5f1f8068`, verified byte-identical). The scene keeps the primary
unsheared and shears every neighbour, so the measured response IS the blending
response — no flow, no subtraction, no extrapolation:

| branch | R_sim | R_emulator | m = Rsim/Remu - 1 |
|---|---|---|---|
| matched identities | 0.14137 | 0.13797 | **+2.47 +- 2.21%** |
| original stratum | 0.11662 | 0.10822 | +7.76 +- 2.10% |
| v2_complement stratum | 0.16444 | 0.16577 | -0.80 +- 3.33% |

Positive `m` means the emulator is LOW. So on that population the summed
emulator is accurate to ~2.5%, if anything slightly under — not 15% over.

### What broke in the withdrawn estimate

The re-weighting assumed the flow's fractional error at fixed primary magnitude
is the same for quiet and crowded primaries. That assumption was flagged as the
weak point at the time; it is now refuted. The flow conditions on
`nbr_flux_near`, `nbr_flux_far` and `nbr_flux_max` (`sbsi.fixed_g0_domain`), so
crowding is inside its inputs and its error has every reason to move with
crowding. Reading the flow on the quietest primaries and applying that number to
the crowded ones is therefore not licensed.

**Withdrawn**: "flow +3.51 +- 0.46% low, blending +15.50 +- 1.64% high,
cohort-wide". The global split is NOT established.

### What survives, and one new result that needs no assumption

Still standing, all direct:

- the per-band flow read in the quiet cells (flow LOW: +2.06 +- 0.28% at r<23,
  +3.22 +- 0.86% at r 24-25, blending under 0.4% of `R` there)
- the magnitude composition of the blending bins, which is why the quiet read
  is a bright-primary number
- the scene test above

New, and forced by positivity rather than inferred. `R_flow` is a galaxy's
response to its own shear and cannot be negative, so the predicted blending
response can never exceed the measured total. In the top blending bin it does:

| bin | R_measured | R_blend model | R_flow model | forced over-prediction | if the flow model is right |
|---|---|---|---|---|---|
| bin10 | 0.76821 | 0.60163 | 0.19168 | — | 4.17 +- 1.22% |
| bin11 | 1.01092 | **1.11436** | 0.11173 | **>= 9.28 +- 0.89%** | 19.31 +- 0.89% |

bin11 is 8.3% of objects and carries **46.9% of the cohort's total `R_blend`**;
the top two bins carry 72.1%. So a failure confined to the crowded tail still
dominates the cohort's blending sum, while barely moving a population like the
scene test's (mean `R_blend` 0.141 against ConstGold's 0.196).

The two scene strata point the same way: the lower-blending stratum reads the
emulator 7.8% LOW, the higher-blending one 0.8% high.

### The consistent picture

- per-pair emulator: accurate to ~3% against labels
- summed emulator at moderate blending: accurate to ~2.5% (scene test)
- summed emulator in the crowded tail: over-predicts, at least 9.3% in the top
  twelfth, and that twelfth is half the cohort's blending
- flow on quiet primaries: under-predicts by 1-3% (bright), more toward faint
- flow on crowded primaries: unconstrained here, and NOT assumable equal to the
  quiet value

### Next steps

- Get the scene test's own blending distribution so the two populations can be
  compared on the crowded tail rather than on their means. That is the missing
  piece for a real global split.
- Separate the tail's two candidate causes with a measurement that does not
  need the flow: run the primary-unsheared scene test restricted to the most
  crowded primaries. If the emulator sum fails there, it is the sum; if it
  holds, the crowded residual belongs to the flow.
- Do not quote a cohort-wide split of `R` until one of those lands.

## 2026-09-15 — The global split, at last: the flow is ~3.5% low, the blending sum ~15% high, and they cancel

Two estimates disagreed: transferring the emulator's label accuracy said the
flow was ~2% HIGH, reading the flow where blending is quiet said it was 2.4%
LOW. Both were quoted on samples with very different primary populations. Fixing
that resolves the disagreement and gives the first cohort-wide split of
`R = R_flow + R_blend` into its two terms.

### The factor 1.93 is benign — the earlier explanation is withdrawn

The half-shear catalogue splits the input list in half, shears the second half,
and draws neighbours only from that sheared half
(`blendemu/blendemu/response.py:_primary_secondary_rows`, and `sec_pos` at line
275). Hence 766,877 labelled pairs per case against ConstGold's 1,480,423.

Measured on `case0_0.0` (699,568 rows, split exactly 349,784/349,784), the two
halves are statistically identical:

| | unsheared | sheared |
|---|---|---|
| mean `r_input` | 26.40480 | 26.40327 |
| mean `Re_input` | 0.39386 | 0.39425 |
| mean RA / DEC | 180.49940 / 0.50059 | 180.50034 / 0.50054 |
| all 7 magnitude-band shares | — | within 0.1 pp |

So the labelled pairs are a RANDOM half. Coverage never invalidated the
transfer, and the previous entry's "it validates half the sum and says nothing
about the rest" is withdrawn (amended in place above).

### What actually made the two estimates disagree

A blending bin is not a random slice of the cohort. Primary-magnitude
composition by `R_blend_abs` bin (cases 40-42):

| bin | r<23 | 23-24 | 24-25 | 25-26 | r>26 |
|---|---|---|---|---|---|
| 00 (quietest) | **29.7%** | 23.7% | 27.1% | 19.0% | 0.5% |
| 11 (loudest) | 1.0% | 4.0% | 17.2% | 48.5% | 29.3% |
| cohort | 1.7% | 5.3% | 19.8% | 48.4% | 24.8% |

The quiet read was a bright-primary number quoted as a global one.

### What changed

`scripts/diagnose_constgold_response_split.py`

- `--magnitude-edge` (repeatable) and `--primary-catalogue` cross the blending
  axis with a truth primary-magnitude axis, so the flow can be read at fixed
  magnitude. The flow is already evaluated for every object, so the extra cells
  cost no GPU time.
- Rewrote the pooling as arrays: `stack` / `group_matrix` / `group_responses`
  replace `pool` / `bin_responses`. One `tensordot` per bootstrap replicate
  covers all 78 reported groups, so 60 crossed cells bootstrap as fast as 12
  bins did. `summarize` now reports `groups` (every group), with `bins` and
  `all` kept as before.
- A crossed cell can be empty in an individual case; `sufficient_or_empty`
  records it as zero weight, and a bootstrap replicate that empties a small cell
  yields `nan` rather than failing, reported as a reduced `resampled` count.
- **Defect fixed**: the bin column was requested from every `--rblend` arm
  though only the reference is binned on, so a comparison arm built before the
  column existed could not be loaded at all.

`scripts/reweight_constgold_flow_error.py` (NEW) turns the per-band quiet read
into a cohort number. A band enters weighted by the response it contributes
(`share x R_flow`), not by object count, because the quantity being corrected is
a summed response. `R_blend_true` then follows as `R_measured - R_flow_true` —
not a second measurement, a remainder.

`jobs/job_diagnose_constgold_flow_by_magnitude.sh` (NEW).
`tests/test_constgold_response_split.py` (+5, 14 total) and
`tests/test_reweight_constgold_flow_error.py` (NEW, 5) — including one that
plants a magnitude-only flow error in a quiet bin with a skewed magnitude mix
and checks the marginals recover it, which is exactly the effect measured here.

`jobs/job_evaluate_constgold_rejection_ab.sh` and
`jobs/job_diagnose_constgold_response_split.sh` were still carrying
`--gres=gpu:a40:1 --mem=96G`, the combination that pended ten hours beside idle
GPUs; both now request `cip` + `a40-16gb` + 20G.

### Validation

    ruff check scripts/ tests/                                   -> clean
    pytest tests/test_constgold_response_split.py -q             -> 14 passed
    pytest tests/test_reweight_constgold_flow_error.py -q        -> 5 passed

Job 16519674 (COMPLETED, 4m42s, cases 40-89, 4,508,176 objects). It reproduces
the previous run exactly where they overlap: pooled `R_measured` 0.67722,
`R_flow` 0.49049, `R_blend` 0.19583, `m` = -1.325%.

### The measurement

Flow error read in the quiet cells of each magnitude band, i.e. where the
blending term is a fraction of a percent of `R`:

| primary r | objects | cohort share | cohort R_flow | flow error % | quiet \|Rb\|/R | quiet covers |
|---|---|---|---|---|---|---|
| < 23 | 458,045 | 0.102 | 1.12745 | +2.06 +- 0.28 | 0.0031 | 59.6% |
| 23-24 | 549,808 | 0.122 | 0.93140 | +0.63 +- 0.72 | 0.0023 | 46.7% |
| 24-25 | 1,240,973 | 0.275 | 0.56251 | +3.22 +- 0.86 | 0.0028 | 40.4% |
| 25-26 | 1,779,532 | 0.395 | 0.25401 | +7.72 +- 1.39 | 0.0158 | 25.6% |
| >= 26 | 479,818 | 0.106 | 0.06799 | +19.09 +- 10.90 | 0.0466 | 5.3% |

Re-weighted to the cohort — **BOTH NUMBERS WITHDRAWN the same day, see the
entry above**: they assume the flow's error at fixed magnitude is the same for
quiet and crowded primaries, and the direct scene test refutes that:

    GLOBAL FLOW ERROR  (model low by)  = +3.505 +- 0.456 %   CI [+2.63, +4.42]
    GLOBAL BLEND ERROR (model high by) = +15.50 +- 1.64  %   CI [+12.45, +18.86]
    model R_flow  0.49049 -> implied true 0.50768
    model R_blend 0.19583 -> implied true 0.16954
    R_measured 0.67722, pooled m -1.325%

Both terms are wrong by more than ten times the |m| < 0.3% budget (dR <= 0.00203)
and they cancel to -1.3%. A pooled `m` says nothing about either.

### How far this can be pushed on

Stable against every choice I can vary. Sweeping the quiet cut from the lowest
1 bin (0.4-24% band coverage) to the lowest 6 (16-73%):

| quiet bins | flow error % | blend error % |
|---|---|---|
| 1 | +3.92 +- 0.93 | +16.91 +- 3.46 |
| 2 | +4.12 +- 0.63 | +17.59 +- 2.57 |
| 3 | +3.86 +- 0.56 | +16.70 +- 2.17 |
| 4 | +3.51 +- 0.46 | +15.50 +- 1.64 |
| 6 | +3.57 +- 0.41 | +15.71 +- 1.28 |

and the relaxed-rejection arm gives the same flow (+3.49 +- 0.46%) with a
smaller blend error (+13.22%), as it must, since relaxing lowers `R_blend`.

The result is not carried by the badly-covered faint band. Contributions to the
+3.51 pp: r<23 0.48, 23-24 0.15, 24-25 1.02, 25-26 1.58, r>=26 0.28. The three
bands covered above 40% already supply 1.65 pp.

### What this says about the emulator

The per-pair emulator checked out against labels to ~3% pooled. The summed
`R_blend` is 15% high. Those are consistent only if the failure is in the SUM,
not in the per-pair prediction — and the cell table shows exactly that
signature. Within **every** magnitude band, `m` runs positive where blending is
small and strongly negative where it is large:

| primary r | bin00 | bin04 | bin08 | bin11 |
|---|---|---|---|---|
| < 23 | +1.19 | -0.66 | -12.11 | -15.69 |
| 23-24 | +2.37 | -0.46 | -5.00 | -26.04 |
| 24-25 | +1.00 | -0.81 | -2.72 | -18.33 |
| 25-26 | +10.32 | +12.81 | -0.77 | -16.74 |
| >= 26 | +45.35 | +20.10 | +18.69 | -16.97 |

~~This is the signature of a linear sum over pairs against a true response that
saturates as neighbours crowd~~ **AMENDED: the same trend is equally the
signature of a flow error that grows with crowding, and the flow conditions on
`nbr_flux_near/far/max`, so it has every reason to. The direct scene test says
the summed emulator is fine at moderate blending. See the entry above.** The
trend is at fixed primary magnitude either way, so magnitude composition cannot
explain it.

### Limitations

- The extrapolation assumes the flow's error at fixed magnitude is the same for
  quiet and crowded primaries. Nothing here tests it. Coverage says how far it
  is stretched: fine for r<25, thin at r 25-26 (25.6%, and the largest single
  contributor), very thin beyond (5.3%).
- Bands are truth primary magnitude only. A flow error tracking size or
  ellipticity at fixed magnitude is not resolved.
- One flow-training seed, one antithetic integration seed, one tile.

### Next steps

- Test the saturation hypothesis directly: at fixed `R_blend`, bin on pair count
  or on neighbour-neighbour separation. If the sum saturates, the over-prediction
  should grow with the number of overlapping neighbours at fixed summed response.
- Repeat the quiet read splitting on true SIZE as well as magnitude, to see
  whether the r 25-26 band's +7.7% is real or a size-composition effect.
- A 3.5% flow deficit and a 15% blending excess are each far outside budget.
  Neither is fixed by the other; both need their own work.

## 2026-09-15 — Reading the flow where blending cannot hide: it under-predicts by 2.4%, and the transfer argument had a hole

The previous entry concluded, from transferring the emulator's label-measured
accuracy onto ConstGold, that R_blend was right and the FLOW was about 2.6%
HIGH. A direct measurement now says the flow is about 2.4% LOW. The direct
measurement wins, and the reason the transfer failed is structural.

### What changed

`scripts/build_constgold_fixed_g0_blend_lookup.py`

- `R_blend` is a SIGNED sum over a primary's pairs, so it reaches zero by
  cancellation rather than by an absence of blending -- which is why binning on
  it never licensed a model-free read of the flow. Added two additive columns:
  `R_blend_abs`, the sum of |per-pair response|, which bounds how far any
  per-pair emulator error can move that primary, and `blend_pairs`, the pair
  count. `R_blend` itself is untouched and reproduces its previous values.
- Guard: the build fails if the absolute sum ever comes out below |signed sum|.

`scripts/diagnose_constgold_response_split.py`

- Added `--bin-column` (default `R_blend`, so existing behaviour is unchanged)
  and `load_column`, the same anchor alignment `load_rblend` does for any lookup
  column.

`tests/test_constgold_response_split.py` (+2, 10 total) pin the alignment
against anchor order and the refusal when a lookup misses an anchor key.
New job `jobs/job_diagnose_constgold_flow_on_quiet_blends.sh`.

### Validation

    ruff check scripts/... tests/...                        -> clean
    pytest tests/test_constgold_response_split.py -q        -> 10 passed

Jobs 16519101 (lookup rebuild, 5-task array, 4,539,567 primaries) and 16519502
(the split). The rebuilt lookup reproduces `mean_R_blend` = 0.198406 against the
old 0.198406, and the pooled row of the split still reads R_measured 0.67722,
R_flow 0.49049, R_blend 0.19583, m = -1.325%.

### Scheduling note worth keeping

The first submission (16519118) sat PENDING with a projected start ten hours
out while a dozen GPUs were idle. Two causes, both in the job file:
`--gres=gpu:a40:1` matches only the WHOLE cards; all of cip's idle capacity is
vGPU slices under different GRES names (`a40-8gb`, `a40-16gb`, `a40-24gb`). And
those slice nodes carry ~25G of RAM, so the inherited `--mem=96G` could not land
on one regardless. The identical earlier split job peaked at **1.2G RSS and ran
in five minutes** (`sacct -j 16516353`). Resubmitted as
`--partition=cip --gres=gpu:a40-16gb:1 --mem=20G --time=02:00:00`: started
immediately, finished in five minutes. The same `gpu:a40:1` + 96G pattern is
still in `job_evaluate_constgold_rejection_ab.sh` and
`job_diagnose_constgold_response_split.sh`.

### The measurement

Binning on `R_blend_abs`, cases 40-89. The last column is how far a 20% error on
every per-pair response could move R in that bin -- the tolerance on reading the
flow there.

| bin | mean \|sum\| | R_blend | R_measured | R_flow | m % | blend tolerance |
|---|---|---|---|---|---|---|
| 00 | 0.0113 | -0.00050 | 0.95944 | 0.93710 | **+2.437 +- 0.464** | 0.33% |
| 01 | 0.0203 | +0.00047 | 0.81808 | 0.79128 | +3.326 +- 0.741 | 0.60% |
| 02 | 0.0287 | +0.00305 | 0.71231 | 0.68733 | +3.178 +- 1.030 | 0.84% |
| 03 | 0.0386 | +0.00782 | 0.63663 | 0.60734 | +3.489 +- 1.185 | 1.14% |
| 07 | 0.1420 | +0.09263 | 0.51605 | 0.41007 | +2.657 +- 1.511 | 4.18% |
| 08 | 0.2180 | +0.15912 | 0.50436 | 0.35884 | -2.625 +- 1.418 | 6.41% |
| 11 | 1.1902 | +1.11436 | 1.01092 | 0.11173 | **-17.549 +- 0.798** | 35.01% |
| all | 0.2423 | +0.19583 | 0.67722 | 0.49049 | -1.325 +- 0.313 | |

In bin 00 the blending term is -0.0005 and could be wrong by 0.33% of R without
mattering. `m` there is **+2.44 +- 0.46%**, positive, at 5.3 sd. Positive `m`
means the model UNDER-predicts. So on those objects the flow is 2.4% LOW, read
against the simulation with no model subtracted from either side. Bins 01-03
agree (+3.3, +3.2, +3.5) while their tolerances are still under 1.2%.

The pair counts are flat across bins (13.8 at bin 00 against 16.9 at bin 11), so
the quiet primaries are not an isolated subpopulation -- they have the usual
number of neighbours, just fainter or further ones.

### Why the transfer argument failed

**AMENDED 2026-09-15, same day, see the entry above.** The coverage reading
below is wrong and the factor 1.93 is benign. The half-shear catalogue shears
one half of the input list and draws neighbours only from that half, so each
primary is labelled against half its neighbours -- and the two halves are
statistically identical (mean r_input 26.4048 against 26.4033, Re_input 0.3939
against 0.3943, every magnitude band matching to 0.1 pp). The labelled pairs
are a RANDOM half, not a distinct population, so the transfer is not
invalidated by coverage. The real reconciliation is the magnitude composition
of the quiet bins.

The corner diagnostic compares model against measured on the pairs that HAVE
labels. Its own counts: **766,877 valid labelled pairs per case, against the
1,480,423 pairs per case ConstGold's R_blend sums over** -- a factor of 1.93.
~~So the transfer validated the emulator on about half of the inference-time sum
and assumed the other half behaved the same way.~~ It validated it on a random
half, which is what a sample is.

~~If the flow is uniformly 2.4% low, the implied true R_blend is 0.1745 against
a model 0.1958, i.e. the emulator is about 12% HIGH globally -- which the
label-side comparison cannot see, because it never looks at the unlabelled
half.~~ The arithmetic stands; the "cannot see" clause does not.

### What is established, and what is not

Established:

- global `m` = -1.325 +- 0.313%, the model over-predicts R
- on the quietest twelfth, the flow under-predicts by **2.44 +- 0.46%** (5.3 sd),
  with a 0.33% blending tolerance -- this is a direct read, not a subtraction
- the loud bins over-predict hard, bin 11 by 17.5% at 22 sd
- against the pairs it CAN be checked on, the emulator is accurate to ~3%
  pooled, with all of that deficit in primaries brighter than r=24

Not established:

- the global split between the two terms. The transfer says R_blend is right;
  the quiet bins say the flow is low, which forces R_blend high. ~~They disagree
  because they cover different pair sets.~~ **AMENDED: they disagree because
  the quiet bins are 30% primaries brighter than r=23 against 1.7% cohort-wide,
  and the flow error is not constant in magnitude.**
- bin 00 is a biased subcohort: its R_flow is 0.937 against a cohort mean of
  0.490. A 2.4% deficit measured there does not automatically hold cohort-wide.

### Retraction

"The blending emulator is consistent with correct and the flow over-predicts by
about 2%" is withdrawn. The sign on the flow is wrong: where it can be read
directly, the flow UNDER-predicts. The earlier entry's per-band emulator
accuracy table stands on its own terms -- it is the transfer to ConstGold that
does not. ~~because of the 1.93x pair-set gap~~ **AMENDED: not because of the
pair-set gap** (the unlabelled half is a random half), but because the two
estimates are quoted on differently-composed samples -- see the entry above.

### Next steps

- Explain the factor 1.93. Which ConstGold pairs have no labelled counterpart,
  and is the emulator ever validated on them? This now blocks the global split.
- Resolve the flow's error on the quiet bins by primary magnitude and size and
  re-weight to the cohort, the same way the emulator was handled -- this is what
  turns "2.4% low on bright quiet objects" into a global number.
- The bright-primary (r 23-24) emulator band is still 34% low and unexplained.

## 2026-09-15 — New fixed-g0 emulator on cut primary-unsheared constant-shear scenes

The owner clarified that the requested scene-response test was for the newly
trained fixed-g0 emulator, not the configured V3.5-like trial9 artifact, and
asked for separate matched-pair and unmatched jobs with the 25.8/0.60-arcsec
measurement cuts.  The evaluated model is
`fixed_g0_m258_r060_v2`, regression SHA
`f04a19d375caa8437c7111677692f49e94e47338f4e94d161179097f5f1f8068`
and metadata SHA
`b3f251d59685e964d81dc744fd301fc9a65015e764d9da6e86de910619d7e261`.

### Population and domain

The emulator was trained on structural primaries selected once on a measured
zero-shear leg with strict `MAG_AUTO < 25.8` and
`FLUX_RADIUS > 3.0 pixels = 0.60 arcsec`; it has no truth analysis cut and no
sheared-leg recut.  The historical cases400--899 instrument has no measured
zero-shear leg: its anchor primary stays unsheared while all neighbours are
coherently sheared at `g1=+/-0.02`.  Therefore this requested evaluation
applies those same strict measured thresholds independently to the two
available legs.  The matched job then intersects selected identities; the
unmatched job keeps each leg's own selected population.  These are not exactly
the fixed-`S0` training selection convention, and the result records that
limitation rather than relabelling either population as `S0`.

All historical anchor rows are retained, without the superseded trial9 truth
cut, and the two sparse strata keep their original inverse-sampling weights.
The resulting scene population targets the instrument's narrower
`18<r<26`, `0.3<Re<1.5` primary box, which is a subset of the new emulator's
finite no-truth-cut generator domain.  On the matched selected population,
50,295,710 scored neighbour pairs contain zero nonfinite features and only
eight pair rows (`0.159` per million) outside an empirical training extremum:
two secondary circularized radii below the saved minimum by about `8.1e-10`
in scaled units and six separations below the smallest observed training
separation.  All other feature boundaries enclose the evaluation.  The
unmatched union scores 50,747,662 pairs and has the same eight extrema
warnings.  Thus the physical/pair-feature domain is consistent to numerical
boundary precision, while the absence of a measured `g=0` evaluation leg
prevents exact reproduction of the training cohort selector.

Across the two strata and 500 cases there are 3,204,159 plus-selected and
3,203,886 minus-selected anchor occurrences: 3,190,026 identities are common,
14,133 are plus-only, and 13,860 are minus-only.  After valid interior ngmix
shape measurements, the requested magnitude/radius cuts reject 247,471 rows
on the plus leg and 247,698 on the minus leg.  Exact `(0,0)` shapes remain
valid; nonfinite shapes and photometry are separately counted; unmatched rows
are never filled with zero.

### Results

The simulation estimator is
`R=[mean(e_plus)-mean(e_minus)]/0.04`.  The scalar emulator estimator uses the
same selected identities and the direct shifts `+h*R_scene` and
`-h*R_scene`.  Strata are inverse-sampling weighted within each case, then the
500 cases are equally averaged.  SBSI bias is
`m=R_sim,11/R_emulator,11-1`; uncertainty is 10,000 common case-bootstrap
resamples with seed20260915.

| branch | leg rows (+/-) | R_sim,11 | R_emulator,direct,11 | m (%) | bootstrap SE (pp) | percentile 95% interval (%) |
|---|---:|---:|---:|---:|---:|---:|
| matched identities | 3,190,026 / 3,190,026 | 0.14137119 | 0.13797009 | **+2.4651** | 2.2115 | [-1.9350, +6.7886] |
| unmatched per-leg populations | 3,204,159 / 3,203,886 | 0.14056518 | 0.13946700 | **+0.7874** | 2.2167 | [-3.6153, +5.0984] |

For matched identities, the original/complement stratum biases are
`+7.7578 +/- 2.1025` pp and `-0.8003 +/- 3.3326` pp, respectively; the
combined value again hides stratum dependence.  For unmatched populations
they are `+4.2148 +/- 2.1141` pp and `-1.3263 +/- 3.3240` pp.  The paired
unmatched-minus-matched bias difference is
`-1.6777 +/- 0.3455` pp, percentile interval `[-2.3593,-0.9996]` pp.

The matched result is the clean direct scene-response comparison because the
same identities enter both signs.  The unmatched simulation numerator also
contains detection/cut-selection response and a leg-dependent baseline-shape
mean, whereas this scalar emulator predicts only direct scene response.  Its
reported `m` is therefore a requested total-versus-direct diagnostic, not an
isolated emulator calibration bias; the paired branch difference is not a
pure selection-response estimate either.

### Execution and validation

The immutable external result root is
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/anchorblend_g002_measured_cuts_v1`.
CPU preflight job16518805, matched array16518815, unmatched array16518816,
matched reducer16518817, and unmatched reducer16518818 all completed exit0;
all stderr files are empty and the run used zero GPUs.  Production arrays had
20 tasks each, throttled to four per branch/eight total.  `matched/results.json`
and `unmatched/results.json` contain the results and exact definitions;
`cases.csv` files retain all case-level sufficient reductions.  A frozen
BlendEMU/SBSI source snapshot, model/source/input provenance, task receipts,
and scheduler accounting are retained with the run.

Independent `review.py`/`review.json` checks all 1,000 case-stratum parts per
branch, branch input/count equality, strict matched/unmatched counts, response
shift algebra, result/source hashes, and independent case-mean reconstruction;
status is `PASS`.  Login-node validation also passed Python compilation,
the evaluator boundary/algebra self-test, shell syntax, and final
`git diff --check`.  No active SBSI production code was changed; only this
work-log entry records the external validation campaign.

## 2026-09-15 — Current trial9 emulator on the primary-unsheared constant-shear instrument

The owner recalled that emulator selection had previously used scene response
and asked for the configured current emulator to be run on the corresponding
primary-unsheared constant-shear simulations.  This is a direct `R_blend`
test: anchor primaries remain at zero shear while every neighbour is sheared
coherently at `g1=+/-0.02`; no flow, classifier, measured-selection model or
inference estimator enters.

### Domain confirmation

The configured V3.5-like response artifact is ordinary-MSE trial9, model SHA
`9723589880234fa6825ae242d9b978120a6385faf5ccc9942d1993696952e46d`
and metadata SHA
`48908a9cebc7475be83eef767c95175694531d78b1dd789c512fc49c0699f2b4`.
The simulations are consistent with its declared training domain:

- evaluated primaries use the exact strict trial9 box
  `18 < r_p < 25.8`, `0.37 < Re_p < 1.5` arcsec;
- pair construction uses the trained `r_max=10` arcsec and `k=20`, with
  secondary support `13 < r_s < 29`, `0 < Re_s < 10` arcsec;
- all 2,613,893 matched anchors pass the supported-aperture control: zero has
  an out-of-support neighbour inside 10 arcsec;
- every simulation block has the same `noise.csv` SHA
  `c628e27a3db001ca60f277fd4c47832558ebd4f32eed00fa3a25d32b59e76d14`
  as the base training simulation: 0.2-arcsec pixels, 0.73-arcsec circular
  Moffat PSF, beta 2.2240680536 and pixel RMS 0.3115193694.  The model uses the
  same metadata rounding, 2.224 and 0.312, on training and evaluation sides.

An exhaustive feature-extrema audit covers 41,103,026 scored pairs.  Only 61
pairs (`1.4841e-6`, or 1.48 per million) cross an empirical finite-training
boundary: 52 primary-radius values differ from the saved extrema only at the
strict-cut/float precision boundary, one secondary radius is lower by
`8.2e-10` in scaled units, and eight pair separations are closer than the
smallest training pair.  No feature is nonfinite; the magnitude and Sersic
ranges are entirely enclosed, and all radius exceedances are at float-scale
cut boundaries.  These empirical extrema are warnings inside the declared
support rather than different population cuts.

### Result

Cases400--899 contribute 2,613,893 anchors matched usable in both shear legs.
The two historical sparse-anchor strata retain their original inverse-sampling
weights within each case, and cases are then equally weighted.  There are no
measured magnitude or radius cuts.  The estimator is
`R=[mean(e1_plus)-mean(e1_minus)]/0.04`; SBSI bias is
`m=R_sim/R_emulator-1`.  Ten thousand paired case-bootstrap resamples use seed
20260915.

| population | anchors | R_sim | R_emulator | m (%) | case-bootstrap SE (pp) | percentile 95% interval (%) |
|---|---:|---:|---:|---:|---:|---:|
| combined headline | 2,613,893 | 0.12659344 | 0.12582354 | **+0.6119** | 2.3141 | [-3.8903, +5.1816] |
| original stratum | 1,713,447 | 0.11711489 | 0.10896254 | **+7.4818** | 2.0912 | [+3.3405, +11.6459] |
| complement stratum | 900,446 | 0.14118463 | 0.15186083 | **-7.0303** | 4.2892 | [-15.4223, +1.4364] |

The global emulator-minus-simulation response residual is
`-0.00076989 +/- 0.00288908` by independent-case SEM (0.27 standard errors),
and the orthogonal null is `+0.00354967 +/- 0.00294944`.  Thus the global
response is statistically consistent with zero bias, but that agreement is a
cancellation between materially different stratum responses.  It is not a
0.3%-precision validation.

Before the two-leg intersection there are 2,892,181 eligible anchor
occurrences.  Exact per-leg reports record 6,965 unmatched plus-leg and 6,858
unmatched minus-leg usable rows, plus eight manifest-anchor occurrences absent
from a rendered input; these are reported and not filled with zero.  Three
matched anchors have no supported neighbour pair and therefore receive the
defined empty sum of zero.

### Execution and validation

The external immutable result root is
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/anchorblend_trial9_re037_v1`.
CPU-only gate job16518414, scoring array job16518427 (20 tasks, at most eight
concurrent), and summary job16518428 all completed exit0 with empty stderr.
Feature-domain array job16518499 and summary job16518500 also completed exit0.
The run used zero GPUs.  `bias.json` is the headline result,
`domain_audit.json` is the exhaustive feature audit, and `results.json` plus
`cases.csv` retain the response sufficient statistics.  The case IDs, counts,
weights, measured truth and null columns are bitwise equal to the older-emulator
control on the same instrument; only the emulator prediction changed.

### Limitations

- This historical sparse-anchor mixture is not the exact ConstGold
  population.  Its inverse-stratum weighting targets the declared primary
  box, but the scene construction and usable-pair intersection remain those
  of the anchor instrument.
- This endpoint has historical model-development/selection use and is not
  fresh acceptance evidence.
- The combined uncertainty is 2.31 percentage points and the two strata pull
  in opposite directions.  The headline near-zero mean must not be read as
  uniform response accuracy or a 0.3%-calibration claim.

## 2026-09-15 — The emulator is not the problem: resolving accuracy by primary magnitude moves the ConstGold bias onto the flow

The previous entry left a contradiction. Against labels the blending emulator
measured 0.9685 +- 0.0192, i.e. about 3% LOW, yet the indirect reading of the
ConstGold split said R_blend was roughly 9% HIGH. Both could not be true. The
reconciliation is that the two sides are different populations, and the axis
they differ on is how bright the PRIMARY is.

### What changed

`scripts/diagnose_constgold_blend_neighbour_support.py`

- The label side accumulated the raw `delta_et1` column. That column is a shear
  difference and has to be divided by `--response-shear` (0.2) before it can sit
  beside an emulator response, exactly as `diagnose_corner_extrapolation.py:229`
  and `build_fixed_g0_response_profile_predictions.py:274` already did. Added
  `--response-shear` (default 0.2) and applied it in `label_side_bands`. Pair
  counts and pair shares were never affected by this defect; the
  `summed_response` / `response_share` columns on the label side were, and every
  per-pair number read off the v1 output is superseded.
- Added a `guard_region` banding: the four cells of (neighbour brighter than the
  5x flux threshold) x (inside 3"), which is the configuration the label
  pipeline's bright-neighbour guard refuses.
- Added a `primary_magnitude` banding, importing `PRIMARY_MAG_EDGES` from
  `diagnose_corner_extrapolation` rather than restating it, so the two tables
  can be multiplied together.

`scripts/diagnose_corner_extrapolation.py`

- Added a primary-magnitude axis to the accumulation. `empty_totals` is now
  `(n_primary_bands * n_cells)`, `accumulate` takes a `primary_mag` column,
  `stack_cases` returns `(case, primary band, cell, field)`, and a new
  `collapse()` sums the band axis away. Every pre-existing report is computed on
  `collapse(resolved)`, so the two-dimensional grid and the region reports are
  bit-for-bit what they were. A new `regions_by_primary_magnitude` block reports
  each region resolved by primary band.

`tests/test_blend_neighbour_support.py` (+3, 8 total),
`tests/test_corner_extrapolation.py` (+2, 8 total). The new corner tests pin the
half-open band placement and assert that collapsing the new axis reproduces the
old grid. `jobs/job_diagnose_blend_neighbour_support.sh` and
`jobs/job_corner_extrapolation_ab.sh` repointed to fresh output paths (`_v3`,
`_v2`) so nothing was overwritten.

### Validation

    ruff check scripts/diagnose_constgold_blend_neighbour_support.py \
               scripts/diagnose_corner_extrapolation.py tests/...     -> clean
    pytest tests/test_blend_neighbour_support.py -q                   -> 8 passed
    pytest tests/test_corner_extrapolation.py -q                      -> 8 passed

Jobs 16517592 (support v2), 16517634 / 16517635 (corner v2, both arms),
16518026 (support v3). The refactor is faithful: the pooled `everything` region
still reports 0.9682 (baseline) and 0.9685 (relaxed), digit for digit what the
un-resolved version reported.

### Where the blending sum actually comes from

ConstGold scenes, cases 40-44, baseline emulator. Share of pairs against share
of the summed R_blend:

| neighbour, relative to primary | pairs | R_blend sum |
|---|---|---|
| fainter                        | 82.6% |  5.5% |
| 0-1 mag brighter               |  9.3% | 12.8% |
| 1-1.75 mag brighter            |  3.6% | 19.1% |
| >1.75 mag brighter (>5x flux)  |  4.1% | 62.6% |

8% of the pairs carry 82% of the signal. The guard-refused cell specifically --
more than 5x brighter AND within 3" -- is 0.26% of pairs and 13.45% of the sum.

### The emulator's accuracy depends on the primary's magnitude

`regions_by_primary_magnitude`, region `everything`, relaxed emulator on relaxed
labels, half-shear cases 0-19. Model over measured, case-bootstrapped:

| primary r | pairs | mean model | mean measured | ratio |
|---|---|---|---|---|
| < 23      | 1,550,712 | 0.00202 | 0.00243 | 0.830 +- 0.273 |
| 23-24     | 1,854,497 | 0.00514 | 0.00782 | 0.658 +- 0.078 |
| 24-25     | 4,192,458 | 0.00851 | 0.00842 | 1.010 +- 0.059 |
| 25-26     | 6,059,633 | 0.01388 | 0.01431 | 0.970 +- 0.028 |
| > 26      | 1,680,247 | 0.02421 | 0.02288 | 1.058 +- 0.032 |

These are not cancellation ratios -- every sum is large and positive. The
pooled 3.15% deficit decomposes as +0.36, +2.77, -0.20, +1.47, -1.25 percentage
points, so essentially ALL of it comes from primaries brighter than r=24, which
are only 10% of the labelled sum. `r_input_p_scaled` is one of the emulator's
seven features, so this is a fit failure, not a missing feature. The pattern --
large relative error where the target is small, accurate where the target is
large -- is what plain squared-error training gives on a target that grows 12x
from bright to faint primaries.

### Transferring that to ConstGold moves the bias onto the flow

ConstGold weights primary magnitude differently from the label catalogue: it
puts 24.8% of its blending sum on r>26 where the labels put 43.1%. Weighting the
per-band accuracy above by ConstGold's own shares:

| arm | R_blend error | R_flow error implied |
|---|---|---|
| baseline emulator | -1.57 +- 2.20% (0.7 sd) | +2.55 +- 1.03% (2.5 sd) |
| relaxed emulator  | -1.91 +- 2.06% (0.9 sd) | +1.88 +- 0.96% (1.9 sd) |

The two arms are independent models and they agree on the implied truth --
R_blend 0.198944 and 0.195761 against uncertainties of about 0.0042 -- which is
an internal consistency check the calculation was not built to pass.

Read globally, which is the standard the owner set: **the blending emulator is
consistent with correct, and the flow over-predicts R by about 2%.**

### Retraction

The previous entry's reading -- "flow about 2% LOW, blend about 9% HIGH" -- is
withdrawn. It was inferred from the near-zero-R_blend bins of the ConstGold
split, and those bins were already known to be cancellation-zero rather than
blend-free, so they never licensed a model-free read of the flow. The direct
measurement above has the opposite sign on both terms. `m` is negative, so the
model over-predicts R; the flow is the term that is high.

### Limitations

- The transfer assumes the emulator's accuracy measured on half-shear labels
  (cases 0-19) holds for ConstGold's anchor primaries within the same magnitude
  band. Primary magnitude is the axis the two cohorts differ on most, but it is
  not the only one, and there are no ConstGold labels to check against -- the
  R_blend firewall forbids them.
- The R_blend error itself is only 0.7-0.9 sd. The claim this supports is the
  NEGATIVE one: a +9% blend error is excluded at about 5 sd. It is not a
  measurement of a -2% blend error.
- The global picture is consistent with per-bin residuals that do not vanish.
  Bin 11 of the R_blend split still has the model 18.7% high, and the
  adjacent-in-magnitude strip is still 7% low at 6.5 sd. These cancel in the
  global number rather than being absent from it.
- The per-pair means of the scene and label sides are NOT comparable in
  absolute terms (different primary populations); only the band shapes are.
  This is now recorded in the script's `limitations` block.

### Next steps

- The flow is now the term carrying the ConstGold bias. Measure its error
  directly rather than by subtraction.
- Decide whether to retrain the emulator with a loss that does not let the
  faint-primary pairs dominate, which is what the 23-24 band needs. More
  training cases would not address it -- that band already has 1.85M labelled
  pairs and the error is a bias, not scatter.
- `retrieve_self_response` (`blendemu/response.py:458`) still hardcodes the
  5x/3" guard.
- The orthogonal-null degradation (+0.000296 -> -0.002802) is still open.

# Work log

## 2026-09-15 — The ConstGold bias is one bin: the crowded 8%, where the emulator over-predicts by 16%

ConstGold measures only `R_flow + R_blend`, so the previous entry could not say
whether the relaxed rejection removed an error or cancelled one.  Binning the
same cohort by the model's own `R_blend` separates the two terms.  Job16516353,
cases40--89, matched-usable branch, one certified flow, both arms' lookups on
identical bins.

### The split

`scripts/diagnose_constgold_response_split.py` bins on the reference arm's
`R_blend` -- a truth-side model quantity with no measurement noise in it, so the
binning cannot select on the measured shapes -- into twelve equal-count bins,
and reports the measured response, the flow term and each arm's blending term
in each.  Pooling all twelve reproduces the A/B exactly (`R` measured 0.67722,
model 0.68632, `m` = -1.325%), so the decomposition is the same number taken
apart.

| bin | `R_blend` | share | `R` measured | `R` model | residual | contribution to aperture |
|---|---|---|---|---|---|---|
| 00 | -0.0705 | 8.4% | 0.64904 | 0.66498 | -0.01594 | -0.00134 |
| 01 | -0.0088 | 8.4% | 0.77981 | 0.76611 | +0.01370 | +0.00115 |
| 02 | -0.0004 | 8.4% | 0.81062 | 0.79543 | +0.01518 | +0.00127 |
| 03 | +0.0071 | 8.4% | 0.75339 | 0.73428 | +0.01910 | +0.00160 |
| 04 | +0.0171 | 8.4% | 0.65880 | 0.63567 | +0.02313 | +0.00194 |
| 05 | +0.0322 | 8.4% | 0.57842 | 0.55544 | +0.02298 | +0.00192 |
| 06 | +0.0565 | 8.4% | 0.51802 | 0.50339 | +0.01463 | +0.00122 |
| 07 | +0.0968 | 8.3% | 0.50035 | 0.48204 | +0.01830 | +0.00153 |
| 08 | +0.1683 | 8.3% | 0.49329 | 0.49286 | +0.00043 | +0.00004 |
| 09 | +0.3169 | 8.3% | 0.55211 | 0.56462 | -0.01251 | -0.00104 |
| 10 | +0.6204 | 8.3% | 0.78200 | 0.79675 | -0.01475 | -0.00122 |
| 11 | +1.1440 | 8.2% | 1.05765 | **1.25552** | **-0.19787** | **-0.01617** |
| all | +0.1958 | 100% | 0.67722 | 0.68632 | -0.00909 | -0.00909 |

### Three things this settles

**1. The aperture bias is one bin.**  Every bin but the top three pushes `m`
*positive*; they sum to +0.0071.  The most-blended 8% of the cohort pushes it
negative by 0.0162 on its own -- more than the entire aperture residual.  Drop
that one bin and `m` changes sign, from -1.325% to **+1.213%** on the remaining
91.8% of objects.  The published negative ConstGold bias is not a property of
the cohort; it is the crowded tail outvoting everything else.

**2. The flow term is about 2% low, measured almost model-free.**  In
bins01--03 the blending term is between -0.009 and +0.007, so `m` there reads
the flow term with essentially nothing else in it: +1.79 +/- 1.15%,
+1.91 +/- 0.84%, +2.60 +/- 0.96%.  Three independent bins covering a quarter of
the cohort agree on roughly +2%.  This is a direct measurement, not the
inference from a global `m` that
`AGENTS.md` warns is not diagnostic.

**3. Neither term is wrong by a factor.**  Fitting
`residual = b * R_flow + a * R_blend` returns b = +0.0258 +/- 0.0042 and
a = -0.1327 +/- 0.0069, which look decisive and are not: the fit misses
individual bins by up to 0.063 against an rms signal of 0.059, and it gets
bin00 wrong by 0.044 with the wrong sign.  Its rms residual, 0.0289, is half
the signal it is supposed to explain.  **The fit is rejected**; the two
coefficients are recorded only because they are what a scaling model would have
claimed.  The blending term's error is structured in `R_blend`, not
proportional to it: under-prediction through the middle, over-prediction at both
ends, and a collapse at the crowded end.

### What the relaxed guard actually did

Per bin, the relaxed arm's `R_blend` is lower everywhere, but not uniformly --
by 20% at bin00, 10% at bin04, 5% at bin07 and 2% at bins10--11.  That helps
the top bins, where the model over-predicts (bin11 `m` -15.76% -> -14.12%,
bin10 -1.85% -> -0.52%, bin09 -2.22% -> -0.52%) and hurts the middle ones,
where it already under-predicts (bin05 +4.14% -> +4.65%, bin07 +3.80% ->
+4.82%).  The net +0.55 pp of the previous entry is that trade.

This confirms the previous entry's recommendation with a reason rather than a
suspicion: **the relaxed catalogue is not a fix.**  It shifts the blending term
down everywhere and profits because the single dominant error happens to be an
over-prediction.  The structural failure in bin11 survives it almost intact.

### Files

New in SBSI: `scripts/diagnose_constgold_response_split.py`,
`tests/test_constgold_response_split.py`,
`jobs/job_diagnose_constgold_response_split.sh`.  The script imports its
loaders, sufficient statistics and response estimator from
`scripts.evaluate_constgold_fixed_g0_response` rather than reimplementing them,
so the split is the certified estimator taken apart, not a parallel one.

### Validation

`ruff check` clean on script and tests; `bash -n` clean on the launcher; eight
new tests pass.  They plant known fractional errors and check the fit recovers
them, that a pure blending error does not leak into the flow coefficient and
vice versa, that bins are weighted by occupancy, that quantile edges are open
at both ends and strictly increasing, and that a degenerate quantile or fewer
than three bins is refused.  One of the eight was written with the wrong
expectation -- a single heavily weighted bin does not pin a two-parameter fit,
because the lightly weighted bins still choose the split along the constraint
line (analytically -0.04 and +0.12, which is what the code returned) -- and the
test, not the code, was corrected.  The end-to-end check is that pooling the
twelve bins reproduces the independently computed A/B `m` to four decimals.

### Limitations

Bins are equal-count in `R_blend`, so they are not a random partition and the
flow term varies across them by a factor of seven; the near-zero-blend reading
of the flow term is therefore a reading on those bins' population, not on the
whole cohort.  One flow seed and one integration seed.  Matched-usable branch
only.  The measured response in a bin is unbiased but noisy, and bin00 holds
the negative-`R_blend` tail, which is a different physical regime from the rest
and is not understood here.

### Next steps

1. **Test the additivity hypothesis in bin11.**  The emulator sums pairwise
   responses linearly over every neighbour.  In the crowded tail that sum is
   large and the measurement saturates; a 16% over-prediction is what an
   additive model does when neighbours overlap each other.  The test is to
   re-bin by *number of contributing pairs* at fixed `R_blend` and see whether
   the deficit tracks the pair count rather than the response.
2. **Confirm the 2% flow deficit against an independent cohort.**  If it holds,
   it is a separate, additive contribution to `m` of about +1 pp and it is not
   a blending problem at all.
3. Carry both items into the rejection question: with bin11 understood, the
   guard can be chosen on the merits of the labels rather than on `m`.
4. The uncommitted `blendemu/response.py` failure-mask fix still needs
   committing; `retrieve_self_response` still hardcodes 5x and 3".

## 2026-09-15 — The relaxed guard, measured: a faithful rebuild, a second bug found, and an A/B on the emulator

Acting on the previous entry's step 1.  The full 200-case response catalogue is
rebuilt at a relaxed bright-neighbour guard, a matched control is rebuilt at the
old guard with the same code, both train an emulator under the frozen trial-9
recipe, and the two are compared on held-out cases and on ConstGold `m`.

### A second defect, found by checking rather than assuming

Before regenerating anything, `scripts/check_response_catalogue_fidelity.py`
asked whether a rebuild at the *default* parameters reproduces the certified
`response_catalogue_train.feather`.  It does not: over cases0--4 production
keeps 6,212,120 rows and the rebuild keeps 6,211,287.

The cause is **not** the bright-neighbour parameterization.  `blendemu` carries
an uncommitted correction to the measurement-failure mask in `response.py`:

```python
# before (what built the certified catalogue on 2026-07-16)
fail_mask = ((e_c1[0, :] == -1.) & (e_c2[0, :] == -1.)) | \
            ((e_c1[0, :] == 0.) & (e_c2[0, :] == 0.))
# after
fail_mask = ~_valid_measured_shape(e_c1) | ~_valid_measured_shape(e_c2)
```

The old form compared one component **across** the two legs, so it only fired
when both legs showed the sentinel together and missed every one-leg failure;
the new form requires a finite, strictly interior shape in each leg
separately.  Production therefore carries 833 pairs (cases0--4) with a finite
`delta_et1` whose underlying shape is invalid, and 0 NaN labels where the
rebuild has 896.

**It never reached the model.**  `train_emulator._valid_regression_measurements`
applies the corrected rule itself, on the `measured_e*` columns, which are
written whatever the label flag says.  Measured directly: production offers
6,212,120 finite labels of which 6,211,287 have a valid measurement -- exactly
the rebuild's count -- and the surviving row keys are identical on both sides
(`only_prod=0`, `only_rebuilt=0`) with `max|delta_et1 difference| = 0.000e+00`.
The stale catalogue and the corrected one give the **same training set**.

That still matters twice over: the fix is real and currently lives only in a
working tree, and it meant the A/B had to be run against a freshly built
control rather than against the catalogue on disk.

### The rebuilds (jobs16515491 relaxed, 16515538 control)

`scripts/build_response_catalogue_rejection.py` mirrors `run_pipeline.py`
step4 -- batch in parallel, merge, drop non-finite `delta_et1` -- but builds
only the response catalogue and writes to a directory of its own, so the
certified catalogue is never touched.  200 cases in 152s and 72s of batch
work respectively.

| | baseline 5x/3" | relaxed 20x/1.5" | change |
|---|---|---|---|
| pair rows kept | 248,484,050 | 260,423,774 | +4.81% |
| primaries | 29,691,811 | 31,045,394 | +4.56% |
| corner pair rows | 254,311 | 775,072 | **+204.8%** |
| corner primaries | 245,374 | 753,109 | **+206.9%** |
| corner measured `R` | 0.49040 +/- 0.00266 | 0.53204 +/- 0.00177 | +8.5% |

The last row is the previous entry's stated limitation, now measured at full
scale: the pairs the old guard let through were the *least* blended members of
the corner, and the unbiased corner mean is 8.5% higher than the survivors
suggested, with a third less uncertainty despite the population being three
times larger.

The control is a faithful reproduction.  Its anchored training set is
149,750,534 pairs splitting 95,834,781 / 23,958,696 / 29,957,057 into
train / validation / development -- identical to the certified v2 run on all
three counts, over the full 200-case window, not just the five cases the
fidelity check could reach.

### Two emulators, same recipe (jobs16515541 relaxed, 16515542 control)

Both arms ran `scripts/train_fixed_g0_response.py` unchanged -- trial-9
parameters, 218 trees, `RESPONSE_SEED`, one A40 -- against configs that differ
from the certified v2 config only in where they read and write.

The control **reproduces the certified model exactly**: md5
`e28eb099033e9e2cb5de247005ce313d`, byte-identical to the certified v2
artifact.  The whole chain from raw legs to fitted model is therefore
reproducible end to end, and any difference in the relaxed arm is attributable
to the guard alone.

| | control = certified v2 | relaxed 20x/1.5" |
|---|---|---|
| anchored pairs | 149,750,534 | 153,309,829 |
| train rows | 95,834,781 | 98,113,936 |
| development slope | 1.040615 +/- 0.003525 | 1.037713 +/- 0.002960 |
| validation `r2` (physical) | 0.005665 | 0.006528 |
| validation RMSE (standardised) | 0.997405 | 0.996072 |
| development orthogonal null | +0.000296 | **-0.002802** |
| model md5 | e28eb099033e9e2cb5de247005ce313d | 24822f2244bd5e8eb54fac76f9acd675 |

The slope moves down by 0.0029, about one standard error on its own but the
size the corner argument predicts (the corner carries roughly 13.5% of the
response and was about 2.8% low, 0.135 x 0.028 = 0.0038).  Two things are not
improvements: the two development populations are not the same population --
the relaxed one contains the recovered corner pairs -- and the orthogonal null,
which should be zero, moves an order of magnitude away from it.

### The corner arms (jobs16515586--88)

Three runs of `scripts/diagnose_corner_extrapolation` over the same held-out
cases0--19, reporting model over measured response per pair.  `base_on_base`
is the published configuration, `base_on_relax` puts the **certified** emulator
against labels it never had, and `relax_on_relax` asks whether retraining on
those labels fixes it.  Reading down a column changes the labels; reading
across a row changes the model.

| region | base_on_base | base_on_relax | relax_on_relax |
|---|---|---|---|
| rejected corner | 0.9718 +/- 0.0199 (1.4s) | 1.0233 +/- 0.0131 (1.8s) | **0.9991 +/- 0.0131 (0.1s)** |
| adjacent in magnitude | 0.9197 +/- 0.0117 (6.9s) | 0.9468 +/- 0.0114 (4.6s) | 0.9270 +/- 0.0112 (6.5s) |
| adjacent in distance | 0.9585 +/- 0.0144 (2.9s) | 0.9741 +/- 0.0137 (1.9s) | 0.9576 +/- 0.0135 (3.1s) |
| close and fainter | 1.1590 +/- 0.1272 (1.3s) | 1.1567 +/- 0.1273 (1.2s) | 1.1949 +/- 0.1319 (1.5s) |
| everything | 0.9682 +/- 0.0204 (1.6s) | 0.9875 +/- 0.0196 (0.6s) | 0.9685 +/- 0.0192 (1.6s) |

`base_on_base` reproduces the previous entry's numbers digit for digit, so the
comparison is anchored.

Three readings, in order of how well they are supported.

**1. A third of the published adjacent-in-magnitude deficit was the labels, not
the model.**  Same emulator, same cases; only the label set changes.  The
region gains 3,069 pairs (46,267 -> 49,336) and its mean measured response
falls from 0.42313 to 0.40974 while the mean model barely moves
(0.38914 -> 0.38796).  The deficit goes from 8.0% at 6.9 sigma to 5.3% at
4.6 sigma.  The recovered pairs are real members of the region that the old
guard threw away, and they respond less than the survivors did.

**2. The certified emulator is not badly wrong in the corner it could not
see.**  Against unbiased labels it sits 2.3% high at 1.8 sigma, on twice the
pairs.  The previous entry's 1.4-sigma agreement was measured against a
survivor-biased label whose mean was 12% low; the honest test still does not
convict it.

**3. Retraining on the recovered labels fixes the corner and moves the whole
aperture the other way.**  With the labels held fixed, the corner goes from
1.0233 to 0.9991 -- exact -- while adjacent-in-magnitude goes 0.9468 -> 0.9270,
adjacent-in-distance 0.9741 -> 0.9576 and the whole aperture 0.9875 -> 0.9685.
That is not a subset effect: pairs and labels are identical between those two
columns.  Read by this diagnostic alone the certified emulator, once judged
against unbiased labels, is already consistent at the aperture (0.6 sigma) and
retraining moves it away (1.6 sigma).  That is the opposite of what ConstGold
says below, and the two are not yet reconciled.

### ConstGold `m` (jobs16515589/90 lookups, 16515591 merge, 16515592 evaluation)

Both arms were evaluated with the **same** certified flow
(`flow/paired/selected.pt`, sha `a053172c...`), the same fixed-`g0` cohort,
the same 64 antithetic draws and common latents across legs and models, so the
only thing that differs between them is `R_blend`.  Cases40--89.

| | control = certified | relaxed | difference |
|---|---|---|---|
| `R` measured | 0.677223 | 0.677223 | -- |
| `R` model | 0.686317 | 0.682507 | **-0.003810** |
| `m` (matched cohort) | -1.325 +/- 0.316 % | **-0.774 +/- 0.318 %** | **+0.551 +/- 0.004 pp** |
| `m` (actual usable flags) | -1.593 +/- 0.311 % | -1.039 +/- 0.313 % | +0.554 +/- 0.004 pp |
| `m` (modeled usable) | -2.016 +/- 0.309 % | -1.603 +/- 0.311 % | +0.413 +/- 0.004 pp |

The two arms share cases, flow and latents, so the difference is far better
determined than either `m`: a paired case bootstrap
(`scripts/bootstrap_constgold_ab_difference.py`, 20,000 replicates,
seed20260915) whose
point estimates reproduce the reported ones to four decimals gives
+0.5508 +/- 0.0039 pp, and the shift is stable to +/-0.008 pp across the
cases40--59, 60--79 and 80--89 subsets.

**The bright-neighbour guard is a systematic worth about twice the entire error
budget.**  It moves `R_blend` by 0.00381 where the budget for |m| < 0.3% is
delta-`R` <= 0.00203 -- 188% of it -- and it moves `m` by 0.55 percentage
points.  That settles the question the previous entry raised: the guard is not
a detail.

**It does not settle which arm is right.**  Neither `m` is anywhere near the
0.3% target (-1.33% and -0.77%), so this is not a fix, it is a shift.  Taking
the per-pair ratios at face value, `R_blend` at the aperture is about 0.20
(0.00381 of shift for a 1.91% change in the per-pair mean), so `R_flow` is
about 0.485; a model that is 3% low on `R_blend` and still over-predicts the
sum by 1.3% requires `R_flow` to over-predict by more than the whole `R_blend`
error.  Correcting `R_blend` upward, which is what the per-pair diagnostic asks
for, would make `m` *worse*.  Adopting the relaxed catalogue because it moves
`m` toward zero would therefore risk cancelling one error against another
rather than removing either, which is exactly what the numerical-integrity rule
forbids.  The next measurement has to separate the two terms, not the arms.

### Files

New in BlendEMU: `scripts/build_response_catalogue_rejection.py`,
`scripts/check_response_catalogue_fidelity.py`,
`jobs/job_response_catalogue_relaxed.sh`,
`jobs/job_response_catalogue_baseline.sh`,
`jobs/job_response_catalogue_fidelity.sh`,
`jobs/job_train_rejection_ab.sh`, and the two A/B configs
`configs/fs2_lsst_r_fixed_g0_m258_r060_v2_{baseline,relaxed}.yaml`, which differ
from the certified v2 config in exactly three lines (`output_path`,
`model_dir`, `model_tag`).  New in SBSI:
`jobs/job_corner_extrapolation_ab.sh`,
`jobs/job_build_constgold_blend_lookups_rejection_ab.sh`,
`jobs/job_merge_constgold_blend_lookups_rejection_ab.sh`,
`jobs/job_evaluate_constgold_rejection_ab.sh`,
`scripts/bootstrap_constgold_ab_difference.py`.

### Validation

`ruff check` clean on both new BlendEMU scripts; `bash -n` clean on all seven
new launchers.  BlendEMU has no test suite, so the pure helpers of both scripts
were smoke-checked against planted frames: `parse_ratio`/`positive` reject
zero, negatives and NaN; `corner_pairs` includes a pair exactly on the 3"
edge and one just past the 1.7474 magnitude edge and excludes a 5" pair;
`count_primaries` keeps the same truth identity in two cases distinct and
rejects a negative index; `response_summary` returns `None` rather than a NaN
standard error for a one-row region and serialises under `allow_nan=False`;
and the fidelity comparison was shown to be order-insensitive, to match NaN
positionally, and to catch both a 1e-7 value change and a NaN-versus-value
change.  One defect was caught this way: the mismatch counter used `a != b`,
which counts every NaN against itself, and now uses explicit equal-NaN
semantics.  `scripts/bootstrap_constgold_ab_difference.py` is `ruff`-clean and
reproduces every `m` the evaluation reports to four decimals and every
reported standard error to three, before differencing them.

### Limitations

The corner is defined on true flux while the pipeline rejects on measured
FLUX_AUTO, so the two differ pair by pair.  Only the response catalogue is
rebuilt; the detection and self-response catalogues keep the default rejection,
and `retrieve_self_response` still hardcodes 5x and 3".  The relaxed guard is
one setting, not an optimum -- the pilot showed 20x/1.5" recovers 99.9995% of
what removing the guard entirely does, which is why it was chosen, but nothing
here scans the two parameters.  The project filesystem is at 100% with 1.1 TB
free; the two catalogues take about 55 GB together.

### Next steps

1. **Split the aperture `R` into its two terms on the same cohort.**  Until
   `R_flow` is measured against ConstGold independently of `R_blend`, `m`
   cannot arbitrate the guard: the 0.0038 could be removing an `R_blend` error
   or cancelling an `R_flow` one.  This is now the blocking measurement.
2. **Put the corner diagnostic on the aperture footing.**  It reports a
   per-pair mean; ConstGold reports a per-primary sum.  Summing the diagnostic
   per primary before taking the ratio would let the two be compared directly
   and would show whether the sign disagreement is real or an artefact of the
   weighting.
3. **Commit the failure-mask fix.**  `blendemu/response.py` carries a real
   correctness fix that exists only in a working tree; `inference.py`,
   `nz_utils.py`, `scripts/train_emulator.py` and the untracked
   `scripts/train_fixed_g0_response.py` are also uncommitted.
4. **`retrieve_self_response` still hardcodes 5x and 3"** (`response.py:458`).
   The same survivor selection applies to the self-response catalogue and has
   not been measured there at all.
5. **The guard is one setting, not an optimum.**  20x/1.5" was chosen because
   the pilot showed it recovers 99.9995% of what removing the guard entirely
   does; the two parameters have never been scanned, and the 0.55 pp
   sensitivity above says the scan is worth running.
6. **Explain the orthogonal null.**  It moves from +0.000296 to -0.002802
   between the arms.  It should be zero, and nothing here says why it moved.
7. **The adjacent-in-magnitude deficit survives.**  Against unbiased labels it
   is 5.3% at 4.6 sigma, in a region with a thousand pairs per case and full
   training support.  A third of the published 8.0% was the labels; the rest
   is not explained.

## 2026-09-15 — The unreachable R_blend is one hardcoded line, and the emulator survives it better than feared

Follow-up to the entry below, which established that 3.206% of anchor
primaries carry 13.4% of `R_blend` and own no measured label.  This entry names
the mechanism, measures the emulator where it was said to be untestable, and
pilots the fix.

### The mechanism

`blendemu/response.py`, inside `retrieve_response`, before any pairing -- as it
stood before this session's patch, now line 250:

```python
reject_idx = utils.remove_detection_w_bright_neighbour(
    scat['X_WORLD'], scat['Y_WORLD'], scat['FLUX_AUTO'],
    ratio_max=5, r_min=0, r_max=3 / 3600)
scat = scat.drop(reject_idx)
```

A detection with a neighbouring detection inside 3 arcsec brighter than five
times is deleted, independently in each shear leg.  `scripts/train_emulator.py:338`
reads that same catalogue under the same anchor, so the rejected objects are
absent from the emulator's **training set** as well as from every measured
reference, while inference sums the emulator over the complete truth scene.

Two supporting facts, both from code rather than inference.  The identical call
sits in `retrieve_self_response` (`response.py:458`, same hardcoded 5 and
3 arcsec), so the self-response catalogue has the same hole.  Those two are the
*only* call sites of `utils.remove_detection_w_bright_neighbour` in the whole
package: the fixed-g0 v2 flow is built from
`det_meas_crowd_conc_g0.0_train_full.feather`, which comes through
`retrieve_detection`, and that path carries no rejection -- **the flow is not
affected, only `R_blend`**.  And the 3.206% are not primaries whose pairs
failed the two-leg shape rule: for case0, 2,868 of the 2,875 own no catalogue
row at all, and the shape rule drops 54 rows out of 746,014 (0.007%).

### Exposure (job16511798, cases0--19, 1,816,955 anchor primaries)

| class | n | fraction | mean `R_blend` | of which corner | pairs | bright nbr |
|---|---|---|---|---|---|---|
| all anchor | 1,816,955 | 100% | 0.19872 | 0.02685 | 16.19 | 3.93% |
| labelled | 1,758,703 | 96.794% | 0.17789 | 0.01138 | 16.17 | 1.96% |
| unlabelled | 58,252 | 3.206% | 0.82765 | 0.49404 | 16.67 | 63.57% |
| labelled, bright nbr | 34,454 | 1.896% | 0.80645 | 0.58088 | 16.87 | 100% |
| unlabelled, bright nbr | 37,030 | 2.038% | 0.95676 | 0.77705 | 16.72 | 100% |
| unlabelled, no bright nbr | 21,222 | 1.168% | 0.60234 | 0.00000 | 16.57 | 0% |

The pair-side decomposition is the sharper statement: pairs inside the
rejection corner are **0.26% of aperture pairs and 13.51% of aperture
response** -- a concentration of about 52.  The catalogue is depleted there by
a factor 2.5 (0.103% of its pairs against the aperture's 0.26%) rather than
emptied, because the pipeline rejects on measured FLUX_AUTO while the corner
here is defined on true flux; 63.57% of unlabelled primaries carry a true
bright close neighbour, and 1.896% of the cohort carries one and is labelled
anyway.  That overlap is what makes the next section possible.

### The emulator inside the corner (job16511841, same cases)

Per-pair model against measured label, pooled by summing, case bootstrapped:

| region | labelled pairs | model | measured | model/measured |
|---|---|---|---|---|
| rejected corner | 18,490 | 0.53919 | 0.55483 | **0.9718 +/- 0.0199** |
| close, 1 < dmag < 1.75 | 46,267 | 0.38914 | 0.42313 | **0.9197 +/- 0.0117** |
| 3--4", dmag > 1.75 | 45,396 | 0.64326 | 0.67109 | 0.9585 +/- 0.0144 |
| close, fainter neighbour | 1,217,630 | 0.00865 | 0.00746 | 1.1590 +/- 0.1272 |
| everything | 14,981,579 | 0.01052 | 0.01087 | 0.9682 +/- 0.0204 |

The "everything" row reproduces the 0.9682 +/- 0.0204 of the entry below, which
was computed per primary rather than per pair on a slightly different primary
set; the agreement is a check on this pipeline, not a new measurement.

**Inside the corner the emulator is 2.8% low at 1.4 sigma -- not significant,
and nothing like broken.**  The largest *significant* defect is elsewhere and
fully inside the labelled support: close pairs whose neighbour is 1 to 1.75
magnitudes brighter run 8.0% low at 6.9 sigma.  Translated through the
aperture response each region carries, against the `|m| < 0.3%` budget of
`|dR| <= 0.00203`:

| region | aperture `R` | implied `dR` | share of budget | significance |
|---|---|---|---|---|
| close, 1 < dmag < 1.75 | 0.02113 | +0.00185 | 91% | 6.9 sigma |
| 3--4", dmag > 1.75 | 0.03534 | +0.00153 | 75% | 2.9 sigma |
| rejected corner | 0.02685 | +0.00078 | 38% | 1.4 sigma |
| close, fainter neighbour | 0.01041 | -0.00143 | 70% | 1.2 sigma |
| whole aperture | 0.19872 | +0.00653 | 322% | 1.6 sigma |

The sign is that the emulator **under-predicts** `R_blend`, which pushes `m`
positive.  The whole-aperture central value is 3.2 times the entire budget
while its 20-case uncertainty (+/-0.0041 on `dR`) is itself twice the budget:
the total is precision-limited, not resolved, which is the same wall recorded
for the 20-case emulator validation two entries below.

### The fix, piloted (job16511906, cases0--4, held out of the emulator fit)

`retrieve_response` now takes `bright_neighbour_ratio_max` and
`bright_neighbour_radius_arcsec`, defaulting to 5 and 3 so the historical
catalogue is reproduced exactly; `None` disables the rejection.  Rebuilding
five cases at three settings:

| variant | ratio | radius | valid pair rows | valid primaries | valid corner rows | valid corner primaries |
|---|---|---|---|---|---|---|
| baseline | 5 | 3.0" | 6,211,287 | 741,994 | 6,295 | 6,058 |
| relaxed | 20 | 1.5" | 6,511,265 | 775,901 | 19,431 | 18,867 |
| off | none | -- | 6,511,300 | 775,904 | 19,432 | 18,868 |

Removing the rejection recovers 4.83% more valid pair rows, **4.57% more
labelled primaries (+33,907)**, and **triples the corner labels (+208.7% rows,
+211.4% primaries)**.  A much gentler guard recovers 99.9995% of what removing
it entirely does -- `relaxed` and `off` differ by 35 rows in 6.5 million, by
three primaries, and by one corner row -- so the guard can be kept in a far
weaker form at essentially no cost.  The recovered labels are not noise: the
corner's mean measured response moves 0.49806 +/- 0.01674 to 0.52000 +/-
0.01127, the standard error *shrinking* because there are three times as many
pairs, and the corner median rises 0.4012 to 0.6134 as the more strongly
blended members return.

The recovery is 6,781 primaries per case over the whole catalogue, against
2,875 unlabelled *anchor* primaries per case from the audit above; the anchor is
a subset of the catalogue's primaries, so the two are consistent in size, but
this pilot does not itself demonstrate that the recovered set covers the anchor's
unlabelled members -- that is what step 1 below has to confirm.

Regenerating does not need new images or new shape measurements:
`retrieve_response` reads shape and CrossMatch catalogues already on disk, all
400 case-legs of which are present, and relaxing the rejection is purely
additive because neighbour finding runs on the truth catalogue and no existing
label changes.  The project filesystem is at 100% with 1.1 TB free, enough for
a ~27 GB catalogue plus its per-batch intermediates, but not comfortable.

### Files

New in SBSI: `scripts/diagnose_label_rejection.py`,
`tests/test_label_rejection.py` (7 tests),
`jobs/job_diagnose_label_rejection.sh`;
`scripts/diagnose_corner_extrapolation.py`,
`tests/test_corner_extrapolation.py` (6 tests),
`jobs/job_diagnose_corner_extrapolation.sh`.  New in BlendEMU:
`scripts/build_response_rejection_variants.py`,
`jobs/job_response_rejection_pilot.sh`, and the two optional parameters on
`response.retrieve_response`.

### Validation

`ruff check` clean on all five new files and on `blendemu/response.py`;
`py_compile` clean; `bash -n` clean on all three launchers; `pytest` 7 passed
and 6 passed.  BlendEMU has no test suite, so its two pure helpers were
smoke-checked directly against planted frames, which is how the single-pair
`std(ddof=1)` NaN below was caught.  Jobs 16511798, 16511841, 16511906 and
16511930 exited 0:0 on `cluster` with `--constraint=x86-64-v3`; no GPU used.
Results in `$DATA_DIR/sbsi_caches/fixed_g0_m258_r060_v2/label_rejection_c0_19_v1/`,
`.../corner_extrapolation_c0_19_v1/`, and
`$DATA_DIR/blendemu_runs/rejection_pilot/v1/`.

Four defects were caught by guards or tests rather than by inspection: two test
expectations of mine were wrong (a primary has two neighbours inside 3", not
three, and indices present in a planted scene cannot raise "absent"); a
`searchsorted` with `side="left"` disagreed with `histogram2d` on values
sitting exactly on a bin edge; `summarize` computed `std(ddof=1)` on a
one-pair region, producing a NaN that `allow_nan=False` would have rejected
after the whole rebuild had finished; and the pilot's first submission read
`case0_0.1` because `retrieve_response` defaults to the 0.0 -> 0.1 legs while
this configuration renders 0.0 -> 0.2, now passed explicitly and checked before
any rebuild starts.

### Limitations

The corner labels exist only because the measured flux ratio fell below five
where the true one exceeds it, so they are a biased subsample and plausibly its
least blended members -- their mean `R_blend` is 0.80645 against 0.95676 for
the unlabelled ones, about 16% lower, so 0.9718 may understate the error on the
part that stays unreachable.  Per-cell ratios in sparse cells are unusable and
some are negative, where the measured mean crosses zero; only the region
aggregates are quoted.  Combining a ratio measured on catalogue pairs with the
aperture's response assumes the ratio transfers cell by cell, and the two
neighbour sets are not the same: the catalogue takes up to 20 neighbours from
the sheared half while the inference aperture takes 20 from the whole scene,
which is also why the labelled set is not a subset of the aperture (8.51
against 8.09 pairs per primary).  A primary can be unlabelled because it was
undetected in a response leg rather than rejected, and the 1.168% of the cohort
that is unlabelled with no bright neighbour still carries `R_blend` 0.60234.
The rejection is defined on measured FLUX_AUTO at the detected position; the
proxy here uses true flux at the true position.  No `m` is revised.

### Corrections

Primary counts in the pilot report's first write used `nunique()` on
`input_index` across a multi-case frame, which collapses the same truth
identity in different cases; job16511930 recounts them on `(case, index)` and
rewrites `report.json`.  Pair-row counts were never affected, so every number
quoted above except the primary counts stands as first written.

### Next steps

1. Regenerate the full 200-case response catalogue at the relaxed guard
   (ratio 20, radius 1.5") and retrain the emulator on it.  This is the clear
   win: it is additive, needs no resimulation, triples the corner labels, and
   retires the "structurally unvalidatable" objection.
2. Then re-measure the corner ratio on the unbiased population -- the number
   this entry could only reach through survivors.
3. Attack the 8.0% deficit at close separation and 1 to 1.75 magnitudes. It is
   6.9 sigma, inside the training support, and worth about 91% of the budget on
   its own, so it is a model-capacity or feature problem rather than
   extrapolation.
4. Decide whether `retrieve_self_response` should take the same parameters. It
   is unused by the current flow, so this is hygiene, not a blocker.
5. Certification precision is unchanged and unavoidable: 20 cases cannot
   resolve a 0.3% total.

## 2026-09-15 — 13% of R_blend comes from primaries no label can reach

- Added `scripts/diagnose_unlabelled_blend_primaries.py`,
  `tests/test_unlabelled_blend_primaries.py` (5 tests), and
  `jobs/job_diagnose_unlabelled_blend_primaries.sh`.  Extended
  `scripts/diagnose_blend_neighbour_set.py` to report the full-scene `R_blend`
  over every anchor primary and over each side of the labelled/unlabelled
  split, so the cost of the labelled restriction is measured rather than
  implied.  Tests still 6 passed.
- Motivation: the ConstGold `R_blend` lookup averages 0.198406 while the same
  emulator, aperture, and truth scene give 0.17789 in the validated
  comparison, an 11.5% gap that is larger than the +0.0091 producing
  `m=-1.325%`.
- **The fixed-g0 flow anchor and the response anchor are disjoint by
  construction.**  For case40 the flow anchor holds 91,457 primaries, all with
  `input_index` in 349,785--699,567, and the response anchor holds 91,071, all
  in 1--349,781; the response catalogue's primaries are likewise entirely in
  the first half.  A blend-response label measures a *stable* primary
  responding to a *sheared* neighbour, and `flow/protocol.json` records the
  response target role as `input_index < floor(rows/2)`, so every labelled
  primary is first-half while the ConstGold lookup is second-half.
  Job16510254 accordingly reports 4,539,567 of 4,539,567 ConstGold primaries
  unlabelled.  That output is correct, not a join failure; the expectation of
  a few percent was wrong.
- **The two halves are interchangeable, measured rather than assumed.**
  Job16510902 on cases0--19 gives a full-scene mean of
  `0.19872 +/- 0.00037` over all 1,816,954 first-half anchor primaries against
  the second-half ConstGold value of 0.198406 -- agreement to 0.16%.  This also
  retroactively supports the role-symmetry argument used two entries above to
  justify the `full = 2 x secondary_half` sum.
- **The 11.5% gap is entirely the labelled restriction, and it is severe.**
  Splitting the same 1,816,954 primaries:

  | partition | n | fraction | mean `R_blend` | pairs/primary |
  |---|---|---|---|---|
  | all anchor primaries | 1,816,954 | 100% | 0.19872 +/- 0.00037 | 16.189 |
  | labelled | 1,758,703 | 96.794% | 0.17789 +/- 0.00033 | 16.173 |
  | unlabelled | 58,251 | 3.206% | **0.82762 +/- 0.00315** | 16.667 |

  The unlabelled 3.206% carry 4.65 times the labelled mean blend response and
  contribute `0.03206 x 0.82762 = 0.02654`, which is **13.4% of the cohort
  `R_blend`** and 3.9% of the model response.  Their pair counts are ordinary
  (16.667 against 16.173), so this is not a neighbour-count effect: their
  per-pair response is far larger.  A primary goes unlabelled when its pairs
  fail the two-leg validity rule, which is what severe blending does, so the
  validation set is selected against exactly the objects that dominate the term
  being validated.
- **Consequence for the 0.3% target.**  For `|m|<0.3%` the whole budget is
  `|dR| <= 0.0020`.  The unvalidated contribution is 0.02654, so the emulator's
  error on that 3.206% must stay below 7.5% for those objects alone not to
  exhaust the budget, and no measurement exists that can check it.  This is a
  larger and better-localized term than the boost effect recorded in the
  previous entry.
- Corrections.  (i) The "3.18% unlabelled" figure quoted while this was in
  progress belongs to the response anchor (first half), not to ConstGold.
  (ii) A cross-simulation consistency check computed during this work added
  `R_self` on the flow anchor to `2 x R_blend` on the response anchor; those
  are different halves, so it is a population-mean comparison under the
  half-symmetry now established above, not a same-object identity.  Its correct
  form is reconstruction `0.67667 +/- 0.00467` against the ConstGold measured
  0.67722, a 0.12 sigma agreement whose precision is +/-0.0047, i.e. 2.3 times
  the entire 0.3% budget -- consistent, but far from certifying.
- Validation.  `ruff check` clean on both scripts and both test files;
  `py_compile` clean; `bash -n` clean; `pytest` 5 passed (new) and 6 passed
  (extended).  Job16510254 and job16510902 exited 0:0 with empty stderr on
  `cluster` with `--constraint=x86-64-v3`; no GPU used.  Results:
  `$DATA_DIR/sbsi_caches/fixed_g0_m258_r060_v2/unlabelled_blend_c40_89_v1/result.json`
  and `.../blend_neighbour_set_c0_19_v3/result.json`.
- Two defects were found by guards rather than inspection: a first submission
  aborted because `BLEND_RESPONSE_COLUMNS` already contains `case` and
  `input_index`, so the Arrow `select()` returned duplicate columns and
  `frame["case"]` was a DataFrame; fixed by deduplicating the scan list while
  preserving order.  A unit test also failed on my own arithmetic (2.6/6, not
  1.6/6); the test expectation was wrong, not the code.
- Limitations.  Cases0--19 for the split, cases40--89 for the ConstGold
  lookup; case bootstrap over those cases.  The unlabelled mean is an emulator
  prediction with no measured counterpart by definition, so its accuracy is
  unknown, and nothing here shows the emulator is wrong on those objects --
  only that it is untested there.  No `m` is revised.
- Next steps.  (1) Characterize the unlabelled 3.2%: truth magnitude, size,
  neighbour distance and flux ratio, to establish whether they are a coherent
  severe-blend population or a mixed failure bucket.  (2) Find a label route
  that survives severe blending -- relaxing the two-leg validity rule,
  measuring with a fixed centroid, or a dedicated simulation leg -- since
  without one this term stays permanently unvalidated.  (3) Do not remove
  these objects from the cohort: they are real, they blend, and a survey will
  contain them.

## 2026-09-15 — The flow's self-response error is a boost-conditioning gap

- Added `scripts/diagnose_selection_boost_response.py`,
  `tests/test_selection_boost_response.py` (8 tests), and
  `jobs/job_diagnose_selection_boost_response.sh`.  The diagnostic bins the
  flow self-response residual `R_self_model - R_self_measured` on a grid of
  true magnitude by measurement boost (`r_input - g0_MAG_AUTO`) and fits the
  residual-on-boost slope inside each true-magnitude bin, with case-bootstrap
  errors.  It audits one additive term against its own measured labels and
  reports no `m`.
- Motivation: the cohort selects on measured magnitude (`MAG_AUTO<25.8`) while
  `flow/protocol.json` shows the flow conditions only on true properties
  (`e1_input_p`, `e2_input_p`, `sersic_n_input_p`, `r_input_p`,
  `circularized_Re_input_p`, `nbr_flux_near/far/max`).  Mean `r_input` is 26.40
  against a 25.8 cut, so the cohort is a heavily boosted slice of the truth
  population.  The question was whether the residual is a flat conditional-mean
  offset (an objective problem) or slopes with boost (a conditioning problem).
- Job16506382 (CPU, `cluster`, 3m) on cases0--19 of `lsst_sims_fs2_25876`,
  1,807,351 objects.  **The residual slopes with boost at 8.2 sigma**: global
  slope `+0.05876 +/- 0.00715` per magnitude.  Within true-magnitude bins the
  slope is `+0.168 +/- 0.031` (`<23`), `+0.201 +/- 0.027` (23--24),
  `+0.158 +/- 0.021` (24--25), `+0.076 +/- 0.009` (25--25.8),
  `+0.046 +/- 0.016` (25.8--26.5), and `+0.002 +/- 0.018` (`>26.5`).  The
  effect is significant in every bin except the faintest, where the residual is
  instead a flat `+0.018 +/- 0.012` offset.
- The mechanism is directly visible in the grid.  At fixed true magnitude the
  measured self-response collapses as boost grows while the model plateaus or
  rises.  In true 24--25, measured `R_self` runs
  0.8514, 0.5631, 0.2953, 0.0765, 0.1241, 0.0498 across boost bins while the
  model runs 0.8126, 0.5424, 0.3453, 0.3281, 0.3557, 0.3928 -- a sevenfold
  over-prediction in the `(1,1.5]` cell.  In true 23--24 the measured value
  falls 0.9064 -> 0.3735 while the model *rises* 0.8985 -> 0.7322.  An object
  whose measured flux is boosted a magnitude or more is dominated by
  neighbour light or noise, so its measured shape barely responds to its own
  shear; the flow, blind to boost, returns close to the unboosted value.
- **The cohort mean is unbiased only by cancellation.**  The pooled residual is
  `-0.00035 +/- 0.00252` on `R_self` = 0.4916, but this is a sum of
  `-0.0112` from the 77.6% of rows with boost <= 0.5 against `+0.0108` from the
  22.4% above it.  Two errors of about 2.3% of the mean response each cancel to
  0.07%.  Holding `|m|<0.3%` through that cancellation requires the boost
  balance to be stable to roughly 18% between the population the flow is
  calibrated on and the one it is applied to.  This is the concrete reason a
  flow that looks unbiased here need not be unbiased on another simulation.
- **Correction to the preceding entry's reasoning.**  That entry inferred a
  component split "flow +3.2% high against emulator 10.7% low" inside the old
  truth support.  That combined the ConstGold `m` decomposition (job16503038,
  4,508,176 rows, 50 cases, constant-shear simulation) with the emulator
  measurement on `lsst_sims_fs2_25876` cases0--19 (1,758,703 rows, half-shear
  simulation).  Those are different simulations and different case ranges, so
  the subtraction was not valid and the "+3.2% flow" figure should not be
  quoted.  What stands from that entry is the emulator result itself, measured
  on one simulation throughout.  On the half-shear simulation the flow
  self-response is unbiased in the cohort mean, as recorded above.
- Validation.  `ruff check` clean on script and tests; `py_compile` clean;
  `bash -n` clean; `pytest tests/test_selection_boost_response.py` 8 passed,
  including recovery of a planted boost slope to 1e-9 and of a planted flat
  offset with zero slope.  Job16506382 exited 0:0 with empty stderr on
  `cluster` with `--constraint=x86-64-v3`; no GPU used.  Result:
  `$DATA_DIR/sbsi_caches/fixed_g0_m258_r060_v2/selection_boost_c0_19_v1/result.json`.
  Model and measured sides share the forward `0 -> +g` extraction convention
  (`self_prediction` in the v2 manifest is
  `dot(E_model[e_g]-E_model[e_0],g)/|g|^2`, where "antithetic" refers to the
  latent draws, not the shear legs), so the residual is not an extraction
  artifact.
- Limitations.  Cases0--19 of one simulation; case bootstrap over 20 cases.
  The diagnostic covers `R_self` only, so it does not revise any `m`, and it
  does not by itself show that this mechanism produces the ConstGold
  `m=-1.325%`.  Cells with fewer than 200 rows are not interpreted.
- Next steps.  (1) Run the same binning on the ConstGold fixed-g0 cohort: if
  its boost distribution differs, the cancellation balance predicts the sign
  and rough size of its residual, which would close the loop between this
  mechanism and the reported `m`.  (2) The information is not missing from the
  model -- the flow already emits measured magnitude as one of its four
  outputs, so the defect is a mis-modelled correlation between measured
  magnitude and shear response inside a joint distribution it already
  produces, not an absent feature.  A conditional-mean term on that
  correlation is the targeted fix.  (3) Do not tune the flow's marginal mean;
  it is already right, and moving it would disturb a cancellation rather than
  remove the underlying error.

## 2026-09-15 — The validated R_blend is half the R_blend used in `m`

- Added `scripts/diagnose_blend_neighbour_set.py`,
  `tests/test_blend_neighbour_set.py` (6 tests), and
  `jobs/job_diagnose_blend_neighbour_set.sh`.  The diagnostic evaluates two
  blend-response sums for the *same* primaries, emulator, and simulation: the
  inference-time sum over the trained KD-tree aperture in the complete truth
  scene (what `build_constgold_fixed_g0_blend_lookup.py` feeds into `m`), and
  the sum over only those pairs that exist in `response_catalogue_train.feather`
  (the only `R_blend` ever compared against measured labels).  It also splits
  the inference sum by the neighbour's structural target role.  It reports no
  `m`; it is a component audit of one additive term.
- Motivation: the ConstGold `R_blend` (~0.196, and 0.1394 on the old-support
  bridge of the 2026-09-15 regression entry above) is about twice the only
  `R_blend` that has ever been validated against labels (~0.090).  Job16505733
  (CPU, `cluster`, 8m) measured both on cases0--19 of
  `lsst_sims_fs2_25876`: full-scene 0.17789, labelled-model 0.08961,
  labelled-measured 0.09255 over 1,758,703 objects in 20 cases.
- **The factor 2 is the simulation's shear-role split, not an aperture defect.**
  Three independent lines agree.  (i) Pair counts per primary are 16.173
  full-scene versus 8.519 labelled, a ratio of 1.899 that is flat (1.88--1.91)
  across groups whose blend response spans a factor of five; a density-driven
  over-count would not be flat.  (ii) Direct inspection of the input
  catalogues shows that in every sheared leg of every case exactly the second
  half of `index_input` is sheared and the first half is not (699,568 rows,
  349,784 sheared, all with `index_input >= 349784`).  A blend-response label
  measures the primary's shape change when its *neighbour* is sheared, so only
  the sheared half can produce a label.  (iii) The two halves are a clean
  random partition: exactly equal counts and `r_input`, `Re_input`,
  `axis_ratio_input`, `sersic_n_input`, `redshift_input`, `RA_input`,
  `DEC_input` all agree within 1.4 sigma.  Job16505799 confirms the arithmetic:
  `full/secondary_half` = 2.0272 +/- 0.0023 overall and 2.01--2.08 in every
  group, with 16.173 versus 8.085 pairs.  The inference-time sum over all
  neighbours is therefore the physically correct quantity for a universe in
  which every neighbour is sheared, and the ConstGold `R_blend` magnitude is
  not a bug.
- Note that (iii) makes `full = 2 x secondary` close to automatic: the
  emulator's features do not encode role, so identical half-populations must
  receive identical predictions.  It is a consistency check that passes, not
  independent evidence.  The load-bearing test is the comparison against
  measured labels below.
- **New and unexpected: the emulator's agreement with measured labels splits
  along the old truth cuts.**  On identical pairs
  (`labelled_model_over_measured`, job16505733): inside the old truth support
  0.8933 +/- 0.0303 (-3.5 sigma, emulator 10.7% low); outside it
  1.0150 +/- 0.0267; truth-faint 1.0135 +/- 0.0270; truth-small
  1.0103 +/- 0.0340; outside-both 1.0114 +/- 0.0384.  The aggregate
  0.9682 +/- 0.0204 reported earlier as "inconclusive" was averaging a
  significant in-support deficit against a null outside, and should not be
  quoted as a single number.  The emulator is unbiased in precisely the
  population the old truth cuts discarded, and biased in the population they
  kept.
- **The sign of that bias means it masks the regression rather than causing
  it.**  Under-predicting `R_blend` makes `R_model` too small, which pushes `m`
  up, while the fixed-g0 `m` is -1.325% because `R_model` is too large.  Taking
  the old-support bridge `R_blend`=0.1394 and scaling by 1/0.8933 adds about
  +0.0167 to `R_model` inside the old support, moving that region's `dR` from
  +0.0099 to about +0.0266 and its `m` from about -1.02% to about -2.75%;
  weighted by the 46.85% in-support row fraction, all-cohort `m` would move
  from -1.325% to roughly -2.5%.  This estimate assumes the fractional
  labelled-pair bias transfers to the full-scene sum, which the role symmetry
  in (iii) supports but does not prove.  It is an order-of-magnitude
  statement, not a calibrated correction, and **no such correction has been or
  should be applied**: it is a reason not to tune the emulator against `m`, and
  it is consistent with the earlier finding that ~94% of the in-support
  regression is flow self-response.
- **Residual caveat: the two pairing rules are not nested.**  The labelled set
  has 8.519 pairs per primary against 8.085 in the sheared half of the
  emulator aperture (k=20, `r_max`=10 arcsec), so the response catalogue is
  about 5% more inclusive and the labelled set is not a strict subset of the
  inference aperture.  This accounts for nearly all of
  `secondary_half_over_labelled_model` = 0.9793 +/- 0.0001.  The clean
  statement of emulator bias is the identical-pair 0.8933 +/- 0.0303, not the
  0.8707 +/- 0.0295 of `secondary_half_over_labelled_measured`, which mixes in
  the aperture difference.
- Validation.  `/usr/bin/ruff check` clean; `py_compile` clean; `bash -n`
  clean; `git diff --check` clean; `pytest tests/test_blend_neighbour_set.py`
  6 passed.  Job16505733 and job16505799 both exited 0:0 with empty stderr on
  `cluster` with `--constraint=x86-64-v3`; no GPU was used, so the two-GPU
  ConstGold limit is untouched.  Results:
  `$DATA_DIR/sbsi_caches/fixed_g0_m258_r060_v2/blend_neighbour_set_c0_19_v1/result.json`
  (labelled comparison) and `.../blend_neighbour_set_c0_19_v2/result.json`
  (role split).
- One real defect was found and fixed by the script's own guard, not by
  inspection: job16505778 aborted with "role-split sums do not reconstruct the
  full-scene sum" because `data_utils.xgb_pred` returns float32, so summing
  ~16 float32 responses whole and then by role disagrees at ~1e-8 against a
  1e-9 tolerance.  Fixed by materializing an explicit float64 copy of the
  prediction frame before grouping, and by reporting the residual magnitude in
  the error.  The tolerance was deliberately not widened, since that would
  hide the mismatch the guard exists to catch.
- Limitations.  Cases0--19 of `lsst_sims_fs2_25876` only, one emulator
  (`lsst_r_fixed_g0_m258_r060_v2`), and case-bootstrap errors over 20 cases.
  `truth_bright` (0.09% of rows) has a near-zero measured denominator and its
  ratios are meaningless.  Nothing here measures the flow, and nothing here
  revises any reported `m`.
- Next steps.  (1) The binding constraint is the flow self-response, and the
  in-support emulator deficit makes it larger than -1.325% suggests; the
  emulator should not be adjusted to compensate.  (2) Reconcile the two
  pairing rules so the labelled set is a strict subset of the inference
  aperture, otherwise every future emulator validation carries the same ~2%
  ambiguity.  (3) Half of the `R_blend` entering `m` remains structurally
  unlabelled; the role symmetry is a strong argument that it is correct, but
  a sheared-first-half simulation leg would test it directly.

## 2026-09-15 — Replace plotting notebook with one script per figure

- Replaced `notebooks/` with `plots/`.  The former 25-cell notebook is now four
  direct entry points with concise names: `plots/raw.py`, `plots/response.py`,
  `plots/overlay.py`, and `plots/residual.py`.  Each script builds one figure
  and writes a matching short PNG/PDF/CSV basename under `plots/figures/`.
  `plots/_data.py` contains only the catalogue joins shared by the three
  response figures; it has explicit fixed paths and no CLI, compatibility
  branch, fallback, or implicit cache substitution.  The old notebook,
  generated `notebooks/figures/` products, and notebook bytecode were moved to
  the desktop trash after replacements were validated, so they remain
  recoverable.
- Preserved the scientific domains and exact-join behavior.  `raw.py` uses the
  union of complementary g=0 target halves with no truth/science cut.
  Response scripts use the measured-g0 anchor (`MAG_AUTO<25.8`, strict
  `FLUX_RADIUS>0.6` arcsec), no truth cut, and no sheared-leg magnitude/radius
  recut.  Unmatched rows are rejected or reported: the final run retained
  6,279,465 raw measurements, 1,807,351 matched `R_self` objects (8,984 g0
  rows unmatched to g05), and 1,758,703 `R_blend` objects while reporting and
  excluding 58,252 anchors without finite valid response pairs.
- Added `jobs/job_plots.sh`; final CPU job16505637 ran all four scripts
  independently in 2m41s, completed 0:0, and had empty stderr.  New response,
  overlay, and residual bin tables match the superseded notebook exactly; raw
  profile differences are below `1e-5` because the scripts retain Feather
  precision instead of applying the notebook's early float32 cast.  Python
  compilation, Ruff, launcher `bash -n`, `git diff --check`, output row/finite
  checks, and visual inspection of all four PNGs passed.  The independent
  entry points deliberately rescan the response catalogue rather than adding
  a second intermediate-cache workflow.

## 2026-09-15 — Condition response residuals on predictions and measurements

- Appended four cells to `notebooks/raw_g0_measurements_vs_truth.ipynb` for a
  two-panel response-calibration diagnostic.  Panel A bins by the v2 model
  prediction and panel B bins by the measured response; both overlay
  `R_self` and `R_blend`, use 24 equal-count bins separately for each
  component/axis, and plot the case-balanced mean `R_model-R_measured` with
  paired SEM across the 20 case means.  The panels intentionally use
  independent residual scales because the prediction-conditioned profiles
  span only about -0.045 to +0.054, whereas the measurement-conditioned tail
  bins span about -12.14 to +11.19.  The colorblind-safe component colors,
  redundant marker shapes, zero references, explicit figure legend, PDF/PNG
  exports, bin CSV, and provenance JSON were visually checked.
- The diagnostic uses the exact existing model/measurement joins: 1,807,351
  `R_self` objects and 1,758,703 `R_blend` objects in every conditioning view.
  Its domain is unchanged: selection only on the measured g=0 leg with
  `MAG_AUTO<25.8` and strict `FLUX_RADIUS>0.6` arcsec, no truth cut, and no
  sheared-leg magnitude/radius recut.  The notebook and provenance explicitly
  warn that each horizontal variable is also one term of the plotted residual;
  shared prediction noise in panel A and shared measurement noise in panel B
  can induce apparent slopes, so the strong near-minus-one panel-B trend is not
  by itself an independent calibration regression.
- Updated `jobs/job_execute_raw_g0_measurements_notebook.sh` to print the input
  notebook SHA-256 before execution.  Final CPU job16505056 completed 0:0 in
  1m45s with the submitted hash matching the local source; the executed
  notebook has 25 cells and zero error outputs.  Validation also passed for
  notebook JSON, all code-cell ASTs, unique cell IDs, direct synthetic
  execution of both response-profile helpers, 96 finite bin-table rows with
  24 bins and all 20 cases in each of four groups, exact object totals, domain
  provenance, launcher `bash -n`, `git diff --check`, output hashes, and final
  visual inspection.  No scheduler jobs remained active at handoff.

## 2026-09-15 — Overlay measured and v2-predicted response profiles

- Extended `notebooks/raw_g0_measurements_vs_truth.ipynb` with a five-cell
  measured/model comparison section.  It loads hashed predictions, requires
  exact `(case, input_index)` joins, constructs case-balanced equal-count
  profiles on identical objects, and overlays the selected v2 staged-flow
  `R_self` and v2 response-emulator `R_blend` predictions on their measured
  targets.  Each of the eight main panels now has a compact paired
  `R_model-R_measured` panel below it.  The plot uses redundant marker/line
  encodings, colorblind-safe component colors, the measured-g0 boundary, model
  case-SE bands, and paired case-SE residual error bars.
- Added `scripts/build_fixed_g0_response_profile_predictions.py`, focused tests,
  and Slurm launchers for the prediction cache and end-to-end notebook
  execution.  The flow response is the exact matched g0/g05 projected secant
  from 64 common-antithetic draws.  The blend comparison evaluates the emulator
  on the exact finite, physically valid response-catalogue pair rows used by
  the measured target and sums both labels and predictions to the same primary;
  this avoids mixing model residual with a change in pair support.  An initial
  scratch cache ending in `c0_19_v1` used the full inference-time neighbour set
  for the model side and was superseded before handoff; the notebook references
  only `response_profile_predictions_c0_19_v2`.
- The enforced population is cases0--19 on the measured-g0 anchor: detected,
  finite `MAG_AUTO`/`FLUX_RADIUS`, `MAG_AUTO<25.8`, strict
  `FLUX_RADIUS>3.0` pixels (0.6 arcsec), no truth analysis cut, and no
  sheared-leg magnitude/radius recut.  Prediction job16504256 completed 0:0 in
  3m04s with empty stderr on one A40 and wrote 1,807,351 unique finite
  `R_self` rows and 1,758,703 unique finite `R_blend` rows.  It retained
  14,981,579 valid blend-pair labels and reported 58,252 anchored primaries
  without a finite pair rather than assigning them zero.
- CPU notebook job16504274 completed 0:0 in 1m57s with empty stderr.  The
  rendered PNG and PDF, profile-bin CSV, summary CSV, and provenance JSON are
  under `notebooks/figures/`.  Visual inspection passed.  Across the 20 case
  means, measured/model/residual are
  `0.49156500/0.49121017/-0.00035483 +/- 0.00259127` for `R_self` and
  `0.09255475/0.08961381/-0.00294094 +/- 0.00201768` for `R_blend`; errors are
  paired case SEMs, not object scatter or model-parameter uncertainty.  These
  supervised target-support profiles diagnose response fit and are not by
  themselves an end-to-end `m` closure.
- Independent checks rehashed both Feather caches and their declared model,
  domain, and response-catalogue provenance, verified unique finite keys and
  exact row counts, and reproduced the manifest means.  The full suite passes
  (266 passed, 2 skipped); Ruff, Python compilation, both launcher syntax
  checks, notebook JSON/code-cell validation, unique cell IDs, and
  `git diff --check` also pass.

## 2026-09-15 — Localize the fixed-g0 ConstGold response regression

- Added `scripts/diagnose_constgold_fixed_g0_truth_support.py`, its Slurm
  launcher, and a focused strict-boundary test.  This is a diagnostic bridge,
  not a new catalogue definition: it preserves the measured-g0 anchor and the
  fixed both-leg-usable intersection, then partitions those exact identities by
  the former strict truth support (`18<r_input<25.8` and
  `0.37<Re_input<1.5` arcsec).  It permits a model result only when that
  model's R_blend lookup covers the complete requested subgroup and records all
  unmatched/out-of-coverage cases explicitly.
- GPU job16503038 completed 0:0 in 12m03s with empty stderr on one A40.  Its
  all-cohort staged-v2 result reproduces the reviewed response result exactly:
  4,508,176 matched rows, measured/model R11 0.6772227753/0.6863170922, and
  m=-1.32508967%.  Only 2,112,010 rows (46.8484%) lie inside the former truth
  support; 2,396,166 (53.1516%) lie outside it.  The mutually exclusive
  outside fractions are 3.8745% magnitude-only, 35.8317% size-only, and
  13.4454% both.  Directionally, 17.2383% are truth-faint and 44.9558% have
  semimajor truth Re<=0.37 arcsec, confirming that the measured selection is
  not equivalent to the old truth cuts.
- On the exact 2,112,010-row old-support overlap and the same measured
  numerator, historical V3.5-like gives m=-0.23820323 +/- 0.293523%, while
  staged v2 gives -1.01698156 +/- 0.292081%; the paired case-bootstrap
  difference is -0.77877834 +/- 0.00922589 percentage points.  Therefore the
  new result is not explained solely by admitting objects outside the old truth
  support.  The staged-v2 exact all-cohort m decomposition is -0.67713 points
  from the old-support rows and -0.64796 points in total from the three outside
  partitions.  The outside subgroups are heterogeneous: magnitude-only
  m=-11.1543%, size-only -2.3779%, and both-outside +2.3890%, so their partial
  cancellation must not be hidden behind one aggregate tail number.
- Added `scripts/diagnose_constgold_rblend_response.py` and its CPU launcher to
  separate the exactly additive blend-response component.  Job16503053
  completed 0:0 in 1m18s with empty stderr.  On the old-support bridge,
  historical/new R_blend responses are 0.1389504031/0.1394283540; only
  0.0004779509 of the total +0.0076144583 predicted-response regression is the
  emulator, while +0.0071365075 is flow self-response.  Replacing only the
  historical emulator changes m from -0.23820% to -0.28745%; replacing the
  flow as well reaches -1.01698%.  Thus about 94% of the clean in-support
  regression in response units is in the flow, not R_blend.  Across the full
  fixed-g0 cohort, staged v2 improves the fixed-v1 NLL flow self-response by
  -0.00678896, while the v2 emulator offsets that gain by +0.00128699.
- The common-domain response score is not a reliable proxy for catalogue m in
  its present form.  Its zero-model shape loss is 8.25944 versus 8.10528 for
  staged v2, while the radius term contributes 17.16579 of the total 25.27107;
  the top 1% of pairs supply about 41% of each shape target loss and the top
  0.1% supply 75% of the radius target loss.  This per-object squared response
  target is dominated by noisy differences and heavy tails, consistent with
  direct-from-scratch supervision lowering its nominal score while worsening
  ConstGold m.  A remaining structural concern is that one unlabelled flow is
  fit jointly to g0 rows selected on their own measured outputs and g=0.05 rows
  carried from that anchor; the selection mechanism is not represented in the
  flow context.  This is a mechanism to test, not yet a causal attribution.
- Independent validation rehashed every declared model and lookup, reproduced
  both prior full-cohort results bit-for-bit, verified the four-way count,
  response, and exact m-contribution identities, and confirmed both jobs used
  no more than one GPU total.  Python compilation, Ruff, both launcher syntax
  checks, the full suite (263 passed, 2 skipped), and `git diff --check` pass.
  Limitations remain one
  training seed, one common latent-integration seed, and partial reuse of truth
  scenes with independent image measurements.  The next discriminating test is
  a flow-only conditioning/objective ablation; retraining the classifier is not
  indicated by this diagnosis.

## 2026-09-15 — Continue fixed-g0 ConstGold model evaluation

- Two-hour checkpoint job16500292 completed 0:0 at 00:41:16 CEST and
  captured the full dependency chain. Staged-response training job16498286
  completed 0:0 in 1h46m46s with empty stderr. Its completion manifest passes
  its internal checks, records selected checkpoint SHA-256
  `a053172cb35b40181a5f82f6f63b9940e2198f3d8417f6ab9715fa4b67fcfc6e`,
  full-validation NLL -4.48963873, and fixed-subset response loss 25.27106636.
  The manifest domain hash agrees exactly with the independently hashed v2
  domain manifest. The check job's 776-byte stderr consists only of expected
  `tail` diagnostics for the not-yet-created downstream evaluation/review
  logs, not a stage failure.
- Direct-response-from-scratch training job16498288 remains healthy on one A40:
  stderr is empty, the optimizer checkpoint was updated at 00:40:52, and the
  run reached epoch78 with its learning rate at 1e-7. Its best selection
  objective so far is 228.01957921 at epoch73 (validation NLL -3.82086423 and
  fixed-subset response loss 23.18404434). Exactly one SBSI GPU is allocated
  while this job runs. The two dependent GPU jobs16499857 and16500172 remain
  unallocated; once released, their combined requests are exactly two A40s.
  CPU reviewer job16500289 remains chained after the ConstGold evaluation.
- Added CPU-only monitoring without changing the scientific chain: job16500496
  is dependency-triggered after direct training to capture downstream startup,
  and job16500497 is scheduled approximately two hours later for the next
  substantive progress/result review. Long training remains under scheduler
  control; no job was restarted or duplicated.
- Direct-response job16498288 subsequently completed 0:0 in 3h54m31s with
  empty stderr and passed completion metadata. Early stopping occurred at
  epoch93 and selected epoch73, checkpoint SHA-256
  `3f60989fd972707e10e7f29b00215c8499f77eb58f7f7667a34f586123dcbd01`.
  Its full-validation NLL is -3.79794367; its selected fixed-subset response
  loss is 23.18404434 with a much larger 2.16790427 latent-group jackknife
  error, so the low point estimate is not by itself evidence of improvement.
  Completion-triggered check job16500496 completed 0:0 and confirmed that the
  common-domain comparison and ConstGold evaluation started simultaneously at
  01:18:18 on exactly two A40s. The ConstGold run reached case45 during startup
  monitoring with empty stderr; its CPU reviewer remains dependency-pending.
- Final common-domain comparison job16499857 completed 0:0 in 1m48s with empty
  stderr. Independent artifact checks verified all seven checkpoint hashes,
  500,000 common NLL rows, 50,000 common response pairs, 64 common-antithetic
  draws, and all 21 recomputed model-pair differences; the result artifact
  SHA-256 is
  `c5dd3cfe04887fe3c8854bc87f01ad50750d97549cca0aec153d632b07196c12`.
  Corrected raw-coordinate NLL/response losses are 4.00537743/25.34235387 for
  v1 NLL, 4.00068260/25.32643543 for v2 NLL,
  4.01847089/25.27106636 for v2 staged, and
  4.68802280/23.18404434 for v2 direct. Lower is better. Relative to v1 NLL,
  v2 NLL improves NLL by -0.00469764 +/- 0.00072060 (paired case SEM), while
  its response change -0.01608935 +/- 0.02397350 is unresolved and also has a
  0.02180848 paired latent-MC error. Relative to v2 NLL, staged training worsens
  NLL by +0.01778817 +/- 0.00032097 and changes response by
  -0.05676929 +/- 0.06357122 with 0.09061037 paired latent-MC error. Direct
  training worsens NLL by +0.68735150 +/- 0.00335563; its apparent response
  change -2.06811076 +/- 2.23447744 has 2.18873676 paired latent-MC error and
  is likewise unresolved. End-to-end ConstGold m, rather than this heavy-tailed
  training score, remains the relevant calibration diagnostic.
- ConstGold evaluation job16500172 completed 0:0 in 17m38s with empty stderr,
  followed automatically by CPU review job16500289 (0:0 in 17s, empty
  stderr). The result contains exactly cases40--89, all five requested windows,
  four model stacks, three branches, 64 common antithetic draws, seed7301, and
  10,000 case-bootstrap replicates. The reviewer passed all 60 model rows and
  all 90 paired-difference rows against the independent measured and truth-pair
  audits, exact anchor/count sufficient statistics, domain declarations, and
  current model/classifier/lookup hashes. A separate read-only audit recomputed
  every m and model-pair point difference and rehashed every referenced
  artifact successfully. The raw result/review SHA-256 values are respectively
  `2b4d25bb6e469222780a6c2639a34a7160e6eae9d83cc4a62ccb90ac87024aed`
  and `2d7240e1761bd8e730aa8f72ed830e0e7aa03b70b5e0bd79b6616028fc24d751`.
- On all 50 cases, m in percent (case-bootstrap SE in percentage points) is:
  matched-both-usable `-2.1098 +/- 0.3131` old v1 NLL,
  `-1.7412 +/- 0.3141` v2 NLL, `-1.3251 +/- 0.3150` v2 staged, and
  `-2.3899 +/- 0.3118` v2 direct; actual per-leg usable flags
  `-2.3728 +/- 0.3055`, `-2.0029 +/- 0.3069`, `-1.5930 +/- 0.3077`, and
  `-2.6330 +/- 0.3047`; modeled per-leg usability
  `-2.7712 +/- 0.3058`, `-2.3815 +/- 0.3069`, `-2.0160 +/- 0.3076`, and
  `-2.8768 +/- 0.3050`, in the same model order. Thus the staged v2 stack
  shifts m toward zero relative to frozen v1 by
  `+0.7848 +/- 0.0054`, `+0.7798 +/- 0.0062`, and
  `+0.7552 +/- 0.0055` percentage points in the three branches; v2 NLL alone
  accounts for about +0.37--+0.39 points, while staging adds another
  +0.37--+0.42 points. Direct-from-scratch supervision instead shifts m away
  from zero by -0.2801, -0.2602, and -0.1055 points relative to v1.
- The staged improvement is stable across the requested windows. For actual
  per-leg usable flags, staged m is `-1.3732 +/- 0.6775` (cases40--49),
  `-1.9460 +/- 0.5650` (40--59), `-1.1590 +/- 0.4423` (60--79), and
  `-1.7539 +/- 0.5233` (80--89); its paired improvement over v1 is narrowly
  +0.7682--+0.7893 points in those windows. Nevertheless, every 50-case staged
  branch remains significantly below zero. This is improvement, not recovery
  to unbiased m and not a promotion result. The measured-g0 anchor is selected
  once with detected, `MAG_AUTO<25.8`, and strict `FLUX_RADIUS>3.0` pixels;
  there is no truth cut and no sheared-leg magnitude/radius recut. “Actual”
  and “modeled” refer only to per-leg valid-shape usability after that anchor.
  Reported m errors are case-bootstrap errors; one flow seed and one common
  antithetic integration seed leave training/integration uncertainty
  unquantified.
- Final validation: 262 tests passed and 2 optional tests skipped in 43.76s;
  Python compilation, Ruff, relevant launcher `bash -n`, strict JSON parsing,
  independent arithmetic/hash checks, and `git diff --check` all pass. No SBSI
  jobs remain running or pending. Redundant future CPU monitor job16500497 was
  verified pending and cancelled before allocation after the final review
  succeeded.

## 2026-09-14 — Launch fixed-g0 ConstGold old/new response comparison

- Added an independent measured-side audit in
  `scripts/audit_constgold_fixed_g0_measured.py` with CPU launcher
  `jobs/job_audit_constgold_fixed_g0_measured.sh`. Job16500221 completed 0:0
  in 50s with empty stderr and 392 MiB peak RSS. Across cases40--89 it found
  4,539,567 g=0 anchor identities, 4,508,176 usable in both shear legs,
  4,522,321/4,522,400 usable in the plus/minus legs, and 14,145/14,224
  plus-only/minus-only usable identities. The case-pooled measured `R11` is
  0.6772228 +/- 0.0021962 for the both-usable matched branch and
  0.6746101 +/- 0.0021335 for actual per-leg usability (case bootstrap).
  Case40 sufficient statistics agree bit-for-bit with the independent smoke
  evaluator path. Python compilation, Ruff, shell syntax, three focused tests,
  strict JSON parsing, and `git diff --check` pass. This audit contains no
  model predictions and therefore establishes the numerator of the pending m
  comparisons without prejudging them.
- Hardened the pending evaluator to record an SHA-256 digest for every exact
  per-case g=0 anchor. Added
  `scripts/review_constgold_fixed_g0_response.py` and its CPU launcher to
  reject any result whose anchor identities, catalogue counts, measured
  sufficient statistics, domain declarations, model/classifier file hashes,
  pooled responses, m recomputations, or paired-comparison coverage disagree
  with the independent audit. The reviewer also emits machine-readable JSON
  and a Markdown table for every requested case window. An in-memory review
  of the completed case40 smoke result passes all applicable checks. CPU
  review job16500289 is queued with `afterok:16500172`, so validation will run
  automatically after the GPU ConstGold evaluation and does not affect the
  two-GPU cap. A focused synthetic test proves that the reviewer accepts a
  complete internally consistent result and rejects corrupted measured
  sufficient statistics. Python compilation, Ruff, launcher syntax, four
  focused tests, `git diff --check`, and the full suite (262 passed, 2 skipped
  in 26.92s) pass after these additions.
- Added `scripts/audit_constgold_truth_pairing.py` and a CPU launcher to verify
  the simulation-side premise of the antithetic estimator. Job16500276
  completed 0:0 in 41s with empty stderr and 748 MiB peak RSS. Across all 50
  cases, every one of 34,978,398 truth rows agrees exactly between the +0.02
  and -0.02 catalogues in identity, position, redshift, intrinsic size/shape,
  Sersic index, and magnitude; only the explicitly checked applied-shear
  columns differ. All 4,539,567 g=0 anchor identities are present. The final
  reviewer now requires this audit, its per-case invariant-scene hashes, zero
  missing anchors, and matching shear/domain metadata. Compilation, Ruff,
  shell syntax, four focused tests, and `git diff --check` pass after wiring
  this additional gate into queued review job16500289. The earlier pending CPU
  review job16500270 was cancelled before allocation and replaced because
  Slurm had snapshotted its pre-truth-audit command line.
- Extended the final review artifact to retain every model-pair m difference,
  its paired case-bootstrap standard error, and paired 95% interval for all
  three response branches and all five case windows. An in-memory run against
  the real case40 smoke artifacts validates six model rows and three paired
  rows, including Markdown rendering. This preserves the same-case covariance
  needed to assess whether response supervision changed m rather than comparing
  marginal error bars.
- Closed an omission-tolerance gap in the reviewer: final acceptance now
  requires exactly cases40--89 and all five declared windows (40--49, 40--59,
  60--79, 80--89, and 40--89), 64 common antithetic flow draws, sampling
  seed7301, and 10,000 case-bootstrap replicates in every model/branch entry.
  The review artifact also records paths and SHA-256 digests of the raw model
  comparison, measured audit, and truth-pairing audit, binding the rendered
  table to its exact validated inputs. Compilation, Ruff, four focused tests,
  and `git diff --check` pass.
- Deliberately restored the useful ConstGold response-evaluation machinery as
  the new fixed-cohort scripts
  `scripts/build_constgold_fixed_g0_blend_lookup.py`,
  `scripts/merge_constgold_fixed_g0_blend_lookups.py`, and
  `scripts/evaluate_constgold_fixed_g0_response.py`. The evaluator takes exact
  keys from the corrected measured-g=0 anchor, applies no truth cut and no
  magnitude/radius recut on either ConstGold shear leg, and defines a usable
  shape as finite and strictly inside the unit disk; exact `(0,0)` is valid.
  It reports three distinct domains: identities usable in both legs, actual
  per-leg usability flags without matching, and modeled per-leg `p(U|x)` from
  the unchanged full-parent three-seed classifier ensemble. Unmatched rows and
  raw-detection/usable counts are explicit.
- Verified that the zero-shear and ConstGold simulations are identity
  compatible: case40 has the same 699,568 input identities and identical
  unsheared source properties. Each lookup worker additionally compares the
  exact anchor context against ConstGold truth before prediction. The v1 and
  corrected-v2 response emulators both declare `cuts: null`, no truth analysis
  cut, and no sheared-leg recut. CPU array job16500170 completed all five
  10-case blocks 0:0 in 2m11s--2m39s with empty stderr; dependent merge
  job16500171 completed 0:0 in 7s. Each merged lookup contains the same
  4,539,567 exact anchor keys across cases40--89.
- Added Slurm launchers for lookup construction, merge, a one-case CPU smoke,
  the final GPU comparison, and a two-hour one-shot state capture. Smoke
  job16500191 completed 0:0 in 34s with empty stderr and exercised both
  measured legs, the classifier ensemble, common antithetic flow draws, and
  the separate v1/v2 response lookups. Its single-case m values are diagnostic
  only and are not reported as calibration results.
- Final job16500172 is queued after both flow trainings (jobs16498286 and
  16498288) and the completed lookup merge. It compares the frozen v1 NLL
  stack with the v2 NLL, staged-response, and direct-response stacks using the
  same cases, common antithetic latents, and paired case bootstraps. It will
  report historical case windows40--49,40--59,40--89 plus disjoint 60--79 and
  80--89 checks. Together with already queued final model comparison
  job16499857, the dependencies keep the project at no more than two GPUs.
  One-shot check job16500292 is scheduled for 2026-09-15 00:40:59 Berlin and
  captures the full chain including both independent audits and final review.
  Its earlier pending version, job16500180, was cancelled before allocation
  and replaced after the new validation stages were added.
- Validation: 15 focused domain/evaluator tests passed, then the full suite
  passed with 261 tests and 2 skips (latest run 26.62s). Python compilation, Ruff,
  launcher `bash -n`, and `git diff --check` pass. The evaluation remains a
  diagnostic on previously used finite truth scenes, not model promotion or a
  calibration claim; final m values await job16500172.

## 2026-09-14 — Compare old and new models on the corrected fixed-g0 cohort

- Added `scripts/compare_fixed_g0_models.py`, focused tests, and
  `jobs/job_compare_fixed_g0_models_final_v2.sh` for the definitive completed-
  checkpoint comparison. It evaluates every model on identical v2 rows and
  pairs, corrects NLLs into common raw measurement coordinates, uses common
  antithetic response draws, and reports paired model differences with
  case-level sampling errors and latent-group jackknife errors. Focused tests
  initially exposed and then verified a correction to the leave-one-group-out
  minimum-group guard; the final focused suite reports 11 passed and the full
  suite reports 258 passed, 2 skipped in 27.75s. Python and shell syntax
  checks and `git diff --check` pass. GPU job16499857 is
  queued with `afterok:16498286:16498288`, so it cannot start until both flow
  trainings release their GPUs and the two-GPU project cap remains satisfied.
- Ran CPU Slurm jobs16499724 and16499800 to score frozen model snapshots on
  the same corrected-v2 validation sample: 500,000 identical NLL rows and
  50,000 identical matched response pairs, with common response seeds, four
  independent latent groups, and eight antithetic draws. The jobs completed
  0:0 in 4m21s and 2m15s with empty stderr and about 1.8/1.9 GiB peak RSS.
  Results and hashed snapshot provenance are under
  `fixed_g0_m258_r060_v2/performance_review_20260914/` in
  `common_v2_scores.json` and `common_v2_progress_scores.json`.
- The final v2 NLL epoch140 checkpoint has corrected raw-coordinate NLL
  4.00068258, versus 4.00537740 for the completed v1 NLL checkpoint and
  4.70350993 for historical lambda10 epoch154 on these exact rows. Its common
  paired response loss, 25.34136, is statistically indistinguishable from v1
  NLL (25.33974) and historical lambda10 (25.34527). The v2 staged phase1
  checkpoint gives NLL 4.00925554 and response 25.34518; the v2 direct epoch7
  snapshot gives NLL 5.08139235 and response 25.35842. Thus response
  supervision has not yet improved the common-cohort response score, while
  direct-from-scratch training gives up substantial likelihood quality.
- The apparent v1-to-v2 direct-response improvement from about 32 to 25 in
  the native logs is a validation-subset change: on identical v2 pairs the
  v1/v2 direct snapshots score 25.37482/25.37536 at epochs4/5. The response
  objective is strongly heavy-tailed: the zero-response baseline is 25.62855,
  and the largest 1%/0.1% of radius targets contribute 88.25%/75.22% of its
  radius loss. The v1 staged checkpoint is not rankable by the eight-draw
  common score because its radius integration variance explodes (mean
  quadratic 683.94 and cross-score SE 58.17), consistent with its completed
  64-draw diagnostic and catastrophic post-NLL training history.
- The corrected v2 response emulator is essentially unchanged from v1:
  validation R2 0.00566542 versus 0.00570476, standardized RMSE 0.997405
  versus 0.996653, and development vector slope 1.040615 versus 1.040998.
  The classifier was not retrained and is therefore not part of this model
  comparison. Both long v2 flow jobs remain development runs; no candidate
  was promoted, and end-to-end fixed-cohort calibration remains required.

## 2026-09-14 — Correct the fixed-g0 response cohort and launch v2 retraining

- Built a versioned `fixed_g0_m258_r060_v2` domain rather than mutating the
  earlier artifacts. The strict catalogue boundary is now represented exactly
  as `FLUX_RADIUS > 3.0` pixels, so stored `3.000` rows are excluded. The
  response anchor is restricted by the simulator's structural primary role,
  `input_index < floor(N_generated/2)`, independently of shape placeholders.
  Both flow and response anchors still use only the g=0 measured
  `MAG_AUTO < 25.8`, `FLUX_RADIUS > 0.6 arcsec` predicate, with no truth cut
  and no sheared-leg recut. The full-parent usable-event classifier was not
  retrained because its intended domain has neither measured selection nor an
  analysis truth cut.
- Fixed BlendEMU's response-catalogue writer to validate both shape components
  within each response leg. A finite response label now requires finite ngmix
  components and `g1^2 + g2^2 < 1` in both legs; exact zero is not treated as
  the `(-1,-1)` failure sentinel. The v2 response-training loader applies the
  same guard to the existing raw catalogue, records every drop, and preserves
  response-label validity as distinct from population selection.
- CPU Slurm job16498285 completed 0:0 in 28m59s. The manifest records
  18,156,749 flow-anchor objects, 18,066,441 carried usable g=0.05 rows,
  90,308 missing carried keys, and zero invalid flow rows. Relative to v1,
  exactly 7,172 flow-anchor rows were removed by the corrected 3.000-pixel
  boundary. The corrected response anchor contains 18,158,702 structural
  primaries, compared with 36,331,070 mixed-role v1 keys; the old factor of
  two was therefore chiefly a target-role error rather than a radius-cut
  effect.
- Response-emulator job16498287 completed 0:0 in 2m32s. It retained
  149,750,534 raw pair labels after dropping 21,378 invalid two-leg shape
  pairs. Its metadata records `cuts: null`, no truth analysis cut, no
  sheared-leg recut, the v2 anchor hash, validation R2 0.00566542, and
  development vector slope 1.04061529. Staged flow job16498286 and independent
  from-scratch direct-response flow job16498288 both loaded the corrected
  28,981,405/7,241,785 train/validation NLL rows and
  14,454,481/3,611,946 matched response pairs. The staged run is progressing
  normally through NLL training (validation NLL -4.27448252 at epoch36). The
  direct-response run completed epoch1 from scratch with validation
  NLL -2.28529706, response loss 27.90554776, and combined objective
  276.77018054, confirming response supervision is active immediately. A
  two-hour one-shot status capture is scheduled as job16499687. The dependency
  chain never uses more than two GPUs simultaneously.
- Re-executed `notebooks/raw_g0_measurements_vs_truth.ipynb` on CPU Slurm as
  job16499672 (0:0, 1m52s). The v2 audit verifies zero non-primary anchors and
  zero retained 3.000-pixel boundary rows; cases0--19 drop 2,276 invalid raw
  response-pair labels and retain 1,807,351 `R_self` and 1,758,703 `R_blend`
  objects. Their case-balanced means are `0.491565 +/- 0.002518` and
  `0.092555 +/- 0.002039`. The notebook has no error outputs and rewrites the
  figure, bins, audit, summary, and strict-JSON provenance under
  `notebooks/figures/fixed_g0_response_catalogue_*`.
- Validation passed: 254 SBSI tests passed and 2 skipped; the 11 notebook code
  cells compile; modified Python modules compile in their scheduled
  interpreters; Slurm launchers pass `bash -n`; the BlendEMU response-shape
  helper smoke test passed; notebook JSON/provenance parse; and both worktrees
  pass `git diff --check`. The new models remain unconfigured development
  candidates until both long flow runs complete and fixed-cohort likelihood
  validation is performed.

## 2026-09-14 — Audit the fixed-g0 self and blending response catalogues

- Extended `notebooks/raw_g0_measurements_vs_truth.ipynb` with an executed
  cases0--19 response-catalogue audit. It computes the exact matched
  one-sided `R_self = dot(e_g-e_0,g)/|g|^2` secant from the fixed g=0 and
  carried `|g|=0.05` flow rows, and the object-level
  `R_blend = sum(delta_et1/0.2)` over raw finite neighbour-pair labels.
  Unmatched keys, exact-zero-shear rows, response-pair multiplicity, and
  perpendicular-label nulls are kept separate and reported.
- Added case-balanced conditional-mean profiles with case-to-case standard
  errors versus g=0 measured `MAG_AUTO`, g=0 measured `FLUX_RADIUS`, true
  magnitude, and true semi-major radius. The PNG/PDF figure plus profile,
  per-case audit, summary, and strict-JSON provenance tables are written under
  `notebooks/figures/fixed_g0_response_catalogue_*`. Over 20 cases the
  plotted populations contain 1,808,065 `R_self` objects and 1,759,648
  finite object-aggregated `R_blend` targets. Their case-balanced means are
  `0.491248 +/- 0.002517` and `0.092221 +/- 0.002028`; the corresponding
  perpendicular means are `-0.000875 +/- 0.002970` and
  `0.004455 +/- 0.001330`.
- The audit found that the response anchor is not restricted to its declared
  primary target half: 1,817,354 of 3,634,852 anchor keys have placeholder
  primary-file ngmix shapes. Most are inert because they have no raw response
  pair, but 121 occur among the finite aggregated response targets; a further
  57,971 non-placeholder primary targets have no finite raw response entry.
  The notebook preserves and labels the as-built population rather than
  repairing it implicitly. The response anchor and response emulator must be
  rebuilt with an explicit inherited primary-target-role predicate before
  model promotion.
- The audit also makes an existing floating-point boundary explicit:
  `0.6/0.2` is stored as `2.9999999999999996`, so exactly
  `FLUX_RADIUS=3.000` pixels passes the implemented strict predicate. There
  are 715 such self-anchor rows and 1,411 such blend-anchor rows in cases0--19.
  No cohort or trained model was changed here; if exact decimal-boundary
  exclusion is intended, the domain predicate and all dependent artifacts
  need a deliberate rebuild.
- CPU Slurm job16498166 executed the complete notebook successfully in 1m35s
  with 1,763,784 KiB peak RSS. The executed notebook has no error outputs;
  every code cell parses, nbformat/JSON validation and the saved provenance
  parse pass, and `git diff --check` passes. Earlier diagnostic executions
  stopped on the missing package import, the floating-point boundary
  assertion, and the inappropriate four-output baseline join; each stopped
  before overwriting the executed notebook and directly motivated the
  self-contained matcher and explicit audit branches above.

## 2026-09-14 — Launch a from-scratch direct-response flow control

- Added `scripts/train_fixed_g0_flow_direct_response.py` and its 48-hour A40
  Slurm job. This is a second, independent flow candidate on the same fixed
  measured-g=0 cohort, grouped split, physical four-output architecture,
  4-million-row epochs, matched-pair set, CRN/antithetic response estimator,
  response weight, and validation subsets as the staged flow. It uses a fresh
  seed501 initialization and optimizes `NLL + 10 * paired(shape + radius
  response)` from epoch1 for at most 200 epochs. It has no NLL-only warm-up or
  warm-start checkpoint; flux remains NLL-only.
- Resource detection found 32 physical/64 logical login-node cores, about
  314 GB available RAM, no login-node GPU, and adequate shared storage for
  artifacts. Submitted the GPU work through Slurm while the staged flow is
  the only other SBSI GPU job, preserving the owner's two-GPU total limit.
- The first launch, job16497430, stopped before model initialization because
  of a misspelled shared learning-rate constant. Corrected the constant,
  audited every referenced shared attribute, recompiled the entry point,
  passed `bash -n`, reran the five paired-flow tests successfully, and passed
  `git diff --check`. Retry job16497446 passed data loading with
  28,992,945/7,244,582 train/validation NLL rows and
  14,460,248/3,613,344 train/validation response pairs. Its persisted
  protocol records `from_scratch: true`, `warm_start_checkpoint: null`, no
  truth analysis cut, and no sheared-leg recut.
- Retry job16497446 completed its first two epochs normally. Validation NLL
  improved from -2.21312790 to -2.83303004, paired response loss from
  32.14312028 to 32.11750728, and the combined selection objective from
  319.21807495 to 318.34204276. The first epoch used response gradients from
  initialization and 90.6% of its steps were gradient-clipped, so this remains
  an experimental schedule comparison rather than a promoted model. CPU
  job16497491 is a scheduled two-hour one-shot status/log check.

## 2026-09-14 — Define and launch the measured-g=0 fixed-cohort retraining

- Replaced the former truth-cut training-domain assumption with a fixed
  measured cohort anchored independently for the two half-shear target roles.
  An object enters an anchor only when its `g=0` measurement satisfies the
  strict cuts `MAG_AUTO < 25.8` and `FLUX_RADIUS * 0.2 > 0.6 arcsec`; exact
  `(case, input_index)` keys are then carried to the sheared leg without a
  sheared-leg recut. No truth-property analysis cut is applied. Added
  `sbsi/fixed_g0_domain.py`, `scripts/prepare_fixed_g0_domain.py`, its Slurm
  job, and focused tests for strict boundaries, matching, unmatched counts,
  shear folding, and no-cut metadata.
- Added a fresh four-output physical flow recipe in
  `scripts/train_fixed_g0_flow.py` and `sbsi/flow_paired_shape.py`. It trains
  NLL first and then a common-random-number/antithetic paired-response stage
  for shape and radius while retaining flux as NLL-only. The fixed cohort is
  already conditioned on the anchor event; it must not be passed through the
  existing likelihood's per-leg measured selection a second time.
- Rebuilt the classifier preparation separately over the complete
  half-shear target-role simulation parent. It has no measured magnitude or
  radius cut and no truth analysis cut; only inherited generator/support and
  target-role boundaries define the parent. Per-leg `U` labels are obtained
  from unique raw crossmatches and valid measurements. The new classifier
  uses eight response-independent truth/scene inputs and excludes `R_blend`,
  whose replacement emulator is trained only inside the selected fixed
  cohort. Slurm job16497155 completed 0:0 in 2m58s and prepared 27,982,713
  parent objects (55,965,426 leg labels) for cases40--119, with
  12,566,199/12,559,707 usable `g=0`/`g=0.05` labels; its manifest records
  `truth_analysis_cut: null` and `measured_selection_cut: null`.
- Updated `sbsi/forward_catalogue.py`, the API/domain documentation, and the
  corresponding BlendEMU training/inference path so a model can explicitly
  declare that no truth analysis cut was applied. Added a fixed-hyperparameter
  BlendEMU response-training entry point and configuration; it uses the
  complementary target-role anchor and does not rerun Optuna or introduce a
  response-weighted refit.
- Validation passed: the full SBSI suite reports 252 passed and 2 skipped;
  focused domain/response tests and broader API/scene tests passed; new and
  modified Python files compile; Slurm job scripts pass `bash -n`; both
  repositories pass `git diff --check`; and a CPU construction/log-probability
  smoke test passed for the new physical flow.
- Slurm domain-preparation job16497156 completed 0:0 in 16m05s. It records
  18,163,921 flow-anchor objects, 18,073,606 usable carried `g=0.05` rows,
  90,315 missing sheared-leg keys, no invalid flow rows, and 36,331,070
  complementary response-anchor objects. Response-emulator job16497159
  completed 0:0 in 2m37s; its metadata records `cuts: null`,
  `truth_analysis_cuts_applied: false`, and no sheared-leg recut.
- The first flow launch, job16497158, stopped before training because 14
  catalogue rows have exactly zero recorded shear amplitude. Added explicit
  nonzero-shear response-pair filtering and a regression test: those 14 rows
  remain in NLL training, are excluded only from finite-difference response
  supervision, are counted per case, and are not assigned an artificial
  floor. Focused tests report 11 passed. Retry job16497338 passed data loading
  with 28,992,945/7,244,582 train/validation NLL rows and
  14,460,248/3,613,344 response pairs, and is progressing through NLL
  training. Classifier array16497160 completed all three seeds 0:0, with
  tune mean-case BCE 0.23612131, 0.23617505, and 0.23616157. The selected
  checkpoints have matching hashes, load together through the production
  equal-probability ensemble loader, and return finite probabilities in a
  1,024-row smoke test. Each member records both cut fields as null. The
  dependency/array throttle kept combined GPU use at two. Flow retry16497338
  was stable at NLL epoch64 when active polling stopped; CPU job16497386 is a
  scheduled two-hour one-shot status/log check.
- All new artifacts remain development candidates: the configured likelihood
  and production model paths have not been changed, pending fixed-cohort
  likelihood/cache implementation and end-to-end validation.

## 2026-09-14 — Add an uncut g=0 raw-measurement diagnostic notebook

- Added `notebooks/raw_g0_measurements_vs_truth.ipynb`, an executed,
  reusable notebook over the owner-selected half-shear cases0--19 at `g=0`;
  the `CASES` tuple is configurable. It joins each
  one-to-one truth/detection crossmatch to the complementary primary-target
  and secondary-target shape files, requires their valid target sets to be
  disjoint, and unions them before plotting. It applies no truth-property or
  measured magnitude/radius/ellipticity selection; only finite four-output
  measurements, positive `FLUX_RADIUS`, and non-placeholder ngmix shapes are
  required.
- The audit retains 6,279,465 measurements from 6,279,928 unique
  crossmatched detections: 3,140,312 primary-target and 3,139,153
  secondary-target measurements, with 463 invalid measurements reported.
  It separately reports 7,711,428 truth rows without a crossmatched
  detection and 65,458 shape detections without a truth crossmatch; no
  unmatched row is silently treated as a measurement.
- The four-panel figure shows intrinsic `Re_input` versus convolved
  `FLUX_RADIUS*0.2 arcsec/pixel`, true `r_input` versus `MAG_AUTO`, and
  intrinsic rotated `e1/e2` versus `NGMIX_G1/G2`. Heatmaps show raw bin
  counts; equal-count truth bins show the median, central 68%/95%
  object-level intervals, and a case-median standard-error scale. The central
  99.8% heatmap windows are display-only; all retained measurements enter the
  profiles. A colorblind-safe `cividis` map and Okabe--Ito overlays are used.
- In the descriptive `0.36<=Re_input<0.38 arcsec` slice (not a catalogue
  cut), 191,997 objects have measured-radius mean/median
  0.7540/0.7032 arcsec, central 68% interval 0.5910--0.8952 arcsec, and
  0.5/0.75-arcsec pass fractions 94.24%/37.08%. The notebook notes that the
  upward-skewed distribution and the distinction between intrinsic pre-PSF
  semi-major radius and convolved image measurement make an identity relation
  inappropriate.
- Outputs under `notebooks/figures/` include 300-dpi PNG, vector PDF, exact
  profile CSV, per-case audit CSV, and JSON summary. Notebook schema and all
  code cells compile; summary counts and execution state were checked. Final
  CPU Slurm job16495685 completed 0:0 in 47s at 0.94 GB peak RAM, and the PNG
  was inspected at original resolution. Earlier 200- and 50-case executions
  (jobs16495657/16495678) were cancelled when the owner reduced the requested
  scope and did not write the final outputs. No model, cache, inference
  result, catalogue, or configured selection was modified.

## 2026-09-14 — Plot measured FLUX_RADIUS against uncut truth radius

- Added a read-only visualization over ConstGold cases120--139, pooling both
  coherent `g1=+/-0.02` legs after requiring only a usable shape measurement.
  No truth-property or measured-output cut is applied. The figure compares
  SExtractor `FLUX_RADIUS*0.2 arcsec/pixel` with intrinsic semi-major
  `Re_input` for 12,565,144 measurements.
- The upper panel shows log-count density, the per-truth-radius-bin median,
  and central 68%/95% object-level measurement intervals. These bands quantify
  conditional measurement scatter, not uncertainty on the median. The lower
  panel shows per-bin pass fractions for measured radius greater than 0.5 and
  0.75 arcsec, with 68% Wilson intervals. The old 0.37-arcsec truth boundary
  is displayed only as a reference line.
- Around true `Re=0.348` arcsec, the measured-radius median and 16--84%
  interval are 0.692 and 0.579--0.881 arcsec; 93.4%/34.4% pass the
  0.5/0.75-arcsec measured thresholds. This directly demonstrates that neither
  measured threshold reproduces the old truth-radius selection.
- A follow-up narrow-slice reducer confirms that for raw usable measurements
  with `0.35<=Re<0.39` arcsec, the measured-radius mean/median are
  0.7524/0.7030 arcsec, standard deviation 0.2604 arcsec, and central 95%
  interval 0.4360--1.3614 arcsec. Within the truth-supported `0.37<=Re<0.39`
  slice, the no-neighbour-within-3-arcsec mean/median are 0.7074/0.6684,
  versus 0.8119/0.7508 for neighboured objects. The analytic Moffat PSF
  half-light radius is 0.5268 arcsec; illustrative Gaussian quadrature with
  intrinsic `Re=0.37` already gives 0.6437 arcsec before measurement noise and
  blending. Scheduled CPU job16495552 completed 0:0 in 30 s with empty stderr;
  its result is `flux_radius_re037_summary.json` beside the figure artifacts.
- Artifacts are under
  `/home/z/Zekang.Zhang/.codex/visualizations/2026/09/14/01a09fa6-2888-7cd0-9893-fe2bb392f9d5/`:
  `flux_radius_vs_truth.png`, vector PDF, binned CSV, summary JSON, and the
  plotting source. Python compilation passed. Scheduled CPU job16495416
  completed 0:0 in 47 s with empty stderr and `FLUX_RADIUS_PLOT_COMPLETE`;
  the PNG was inspected at original resolution. No model, cache, inference
  result, or configured selection was modified.

## 2026-09-14 — Cancel the queued 500k rerun and 20k diagnostic study

- Owner requested “cancel all of them”. Cancelled all 35 remaining Slurm
  job/array IDs belonging to this conversation's 500k three-pass rerun, 20k
  numerical study, report callbacks, and scheduled checks, including the
  original 500k audit's last pending check. `scancel` returned 0 with no error.
- Verified with `squeue` that no job from those runs remains active or queued;
  saved `sacct` accounting and verification timestamps in `cancellation.json`
  under each affected run root. Previously completed preparation, the original
  inference, audit outputs, and the saved 20k baseline remain on disk.
- Added cancellation notices to
  [the 20k study](INFERENCE_CONSTGOLD_20K_DIAGNOSTICS.md#protocol) and
  [the 500k rerun](INFERENCE_CONSTGOLD_RE037_SIZE050.md#authorized-setup).
  No GPU inference from either new run had started. No restart or further
  scheduled check remains authorized; resume only on a new owner request.

## 2026-09-14 — Test and reject the minimal U-oracle response-moment classifier as sufficient

- The splitwise audit passed its prespecified generalization gate: the current
  nine-input ensemble minus the identical-moment `U` oracle has truth-response
  error `+0.00219235` on train40--99, `+0.00318211` on tune100--119, and
  `+0.00367493` on reused-screen120--139. The corresponding classifier-only
  shifts in `m` are `-0.084691`, `-0.139168`, and `-0.128813` percentage
  points. The stable sign supports one training-only response constraint; it
  does not make the screen fresh or establish total calibration.
- The first fixed-weight implementation kept the V3.5-like nine inputs,
  4,865-parameter MLP, preprocessing, optimizer, cases, batches, and three
  equal seeds fixed, changing only the loss to
  `BCE + 100*(R_truth,p-R_truth,U)^2`. Both responses use existing cut-6
  (`FLUX_RADIUS>=0.75 arcsec`) moments. It did not use measured `y`, `m`,
  cases120--139, new features, or new simulations in training/selection. A
  double-precision test matches the linearized minibatch gradient to direct
  full-batch autograd at `rtol=atol=2e-12`.
- Startup falsified lambda100: seed20260913 moved from train `dR=+0.00410689`
  to tune `dR=-0.41178304` after one epoch, then oscillated between order-0.2
  and order-1 response errors while degrading BCE. Task0 completed only to
  preserve the diagnostic; task1 was cancelled after 23 s before training and
  task2/review/screen never started. Frozen rejected root:
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_response_moment_lambda100_v1`.
- Before any screen case was read, a single stability repair divided the
  response weight by the observed approximately 100-fold overshoot. Lambda1
  otherwise retained the identical model/data/training contract. CPU
  preflight16494415 passed; serialized GPU tasks16494418_0--2 completed 0:0 in
  6m21s, 5m56s, and 6m13s; review/screen16494419 completed 0:0 in 1m01s.
  All three seeds improved tune BCE by `0.0000212`, `0.0001363`, and
  `0.0001542` and reduced tune response error from
  `0.0032473/0.0031213/0.0031775` to
  `0.0026301/0.0025188/0.0026176`. This is stable, but only an approximately
  19% response correction.
- The one-shot reused 0.75-arcsec screen gives
  `m=-0.359894 +/- 0.222734%`, versus current
  `-0.372086 +/- 0.222617%` and the identical-moment U oracle
  `-0.243273 +/- 0.220173%`. Thus the proposed proper moment term improves
  `m` by only `+0.012192` percentage points and does not reach the central
  0.3% target. Screen BCE improves from `0.17472637` to `0.17463042`.
  The truth component improves from `-0.549409` to `-0.490920` pp while the
  residual component changes from `+0.177323` to `+0.131026` pp, largely
  cancelling the total gain. Reject this lambda1 model set as a sufficient
  solution; do not promote it into the configured likelihood.
- Files are the archived lambda100 driver/wrappers plus the frozen lambda1
  wrapper and jobs ending in `_v2`. Accepted-output root for audit purposes:
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_response_moment_lambda1_v2`;
  protocol/review/screen SHA256 values are
  `fad13c8059d80a4571a0108ee6d239d4142331dc7f8aa4ab0f4d12813cfb1515`,
  `76627d884fbb3e47c1866f66cd3961482e730a17cbe2871834c02955f20d3b0b`,
  and `26b42232e0ec0da00890b8c0934cfbd267e31f20e2115a1ae9f09ba5c75ca14b`.
  The scientifically minimal conclusion is that the current classifier is a
  real but secondary lever. Any further classifier constraint needs an
  explicitly stable hard-constraint optimizer and a new acceptance set; do not
  tune another weight on the now multiply reused cases120--139.

## 2026-09-14 — Queue paired 20k sampling and full-iteration diagnostics; cap total GPU usage at two

- Owner asked to distinguish sampling from repeated updates behind the original
  500k effective bias near -1%, while retaining the original uncertainty method.
  Frozen study at
  `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/diagnostics_constgold_v35_n20k_20260914_v1`
  uses the first 20,000 rows of the already randomized original 500k sample,
  spanning all 100 cases40--139. This sample was fixed before inspecting its
  estimates; absolute row IDs preserve the original per-object random streams.
- Keep the original 12,760,990-atom prior, truth `0.5<Re<1.5`, measured
  `FLUX_RADIUS>=0.75 arcsec`, magnitude bounds, coherent-U detection, pinned
  epoch154/trial9/three-seed V3.5-like models, and original 500k raw-mean center.
  This isolates the quoted draw ladder. The concurrently queued 500k rerun
  retains the separately authorized intrinsic 0.37 / measured 0.5 arcsec floors.
- Sampling arm: nested complement M=2048,4096,8192,16384,32768,65536 with K1024.
  Add an independent proposal seed8702 and a K4096 exact-shortlist control,
  each through M32768. Recenter arm: three full passes at both M8192 and
  M32768; every new center recomputes the complete finite-prior nine-point
  selection normalization with 64 randomized-QMC draws/atom and seed8101.
  No local-quadratic normalization approximation, trimming, empirical correction,
  or tuning against injected shear is used. h=0.001 and CRN remain fixed.
- The run-specific frozen `study.py` orchestrates the already frozen production
  inference CLI; `audit_statistics.py` preserves the checked sandwich reduction.
  Each stage saves per-object moments, draw diagnostics, original sandwich SEs,
  and conditional paired changes, and updates
  [the diagnostic report](INFERENCE_CONSTGOLD_20K_DIAGNOSTICS.md#results-recorded-so-far).
  `protocol.json`, source hashes, sample identities, and `submission.json` bind
  the design. Production inference/model code was not changed for this study.
- CPU baseline job16493760 completed 0:0 in 20 s. At M8192 the chosen 20k
  gives g=(0.01989562453,0.000191346257), original SE=(0.000181320093,
  0.000167212033), effective bias -0.521877 +/- 0.906600 percentage points.
  The paired M4096-to8192 change is +0.356517 +/- 0.103490 percentage points.
  This is a nested-draw sensitivity, not independent MC uncertainty. Its
  absolute bias is too noisy to adjudicate the forward/inference gap alone.
- Python compilation, Bash syntax, `git diff --check`, and scheduled CPU
  preflight16493779 passed (0:0, 17 s): frozen hashes, actual CLI parsing of
  M/K/seed/fixed centers, full-prior normalization mode, unchanged cut/h/QMC,
  sample coverage, and the original sandwich method. GPU smoke16493764 will
  verify old-moment reproduction and nested prefixes at M8192 versus M65536
  before releasing the full study. It and the science jobs are still queued;
  no new GPU inference result is being claimed.
- Serial GPU jobs: smoke16493764, sampling16493765, low/high pass2
  16493766/16493767, low/high pass3 16493768/16493769, seed16493770,
  shortlist16493771. Bounded CPU status jobs16493772 (+2h) and16493773
  (after final stage) record status; these do not wake or notify the agent.
  Slurm currently forecasts tomorrow morning for a suitable GPU; this is
  tentative. Allow several hours of GPU work after allocation, not immediate
  results. Downstream work is guarded by successful predecessors.
- Applied the owner's total two-GPU limit immediately: all six pending 500k
  normalization/inference arrays have `ArrayTaskThrottle=1`, and the 20k chain
  can allocate only one GPU at a time. Its smoke also waits for the separately
  observed classifier audit16493776 (subsequently cancelled) to prevent a third
  known concurrent allocation. Recorded the shared limit in `AGENTS.md`;
  updated the 500k report generator and
  [timing report](INFERENCE_CONSTGOLD_RE037_SIZE050.md#status-and-timing).
  The earlier 7--9 hour four-GPU estimate is superseded; one GPU per pipeline
  implies roughly a day for the 500k three-pass run, plus queue time.
- Interpretation: the cached forward m=-0.372% used intrinsic Re>0.37,
  measured radius>=0.75, cases120--139, and `R_sim/R_model-1`. The old 500k
  number used Re>0.5, cases40--139, and `g_hat1/0.02-1`. They differ in
  population and estimand. See [bias conventions](CONVENTIONS.md#7-multiplicative-bias).
  Next: inspect GPU smoke/startup, compare paired changes and final update
  sizes in both draw-budget arms, then judge whether draw-count convergence
  is adequate. Three passes and two sampling seeds are diagnostics, not
  proof of convergence or a calibration/total-uncertainty claim.

## 2026-09-14 — Start a splitwise classifier-response generalization audit

- Added an archived read-only diagnostic that evaluates the frozen V3.5-like
  three-seed nine-input classifier on the existing cases40--139 and cached
  16-by-64 flow moments at the declared 0.75-arcsec/no-`|e|` selection. It
  reports train40--99, tune100--119, and reused-screen120--139 separately,
  including every seed, their equal ensemble, and the identical-moment `U`
  oracle. No model is trained or selected and no image catalogue is created.
- The decision rule is fixed before reading the result: only a stable,
  same-sign classifier-minus-oracle truth-response error across all three
  splits supports a training-only oracle-moment constraint. A train-only
  improvement or sign instability rejects that lever. This prevents choosing
  another loss solely because its reused-screen `m` looks favorable.
- Files: `archive/research-2026-09-14/scripts/audit_classifier_response_generalization.py`
  and its archived Slurm wrapper. Frozen-environment self-test, Python/Bash
  syntax, model hashes, and whitespace checks pass. The original P5000 request
  16493758 was cancelled while still pending and output-free because the only
  matching node is drained; an A40-slice replacement 16493776 was likewise
  cancelled pending and output-free after scheduler start-time inspection put
  it the next day. CPU replacement16493778 completed 0:0 in 4m22s and preserves
  the immutable calculation without competing for the full-inference GPU;
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/classifier_response_generalization_v1/results.json`
  was written with SHA256
  `7c14e26bdbcf18ee15a7108666dffcc3e00b7fefddee5dfe73f458db42a73492`.
- The gate passes: train/tune/screen ensemble-minus-oracle truth-response errors
  are `+0.00219235`, `+0.00318211`, and `+0.00367493`, with bootstrap 95%
  intervals `[+0.00161010,+0.00278890]`, `[+0.00209248,+0.00424116]`, and
  `[+0.00283034,+0.00454180]`. Every seed has the same sign. This supports a
  proper training/tune-only oracle-moment ablation while leaving architecture,
  features, and cases fixed; it does not support the earlier discordant-pair
  conditional loss.

## 2026-09-14 — Screen the 0.5-arcsec measured-radius cut on retained catalogues

- Added an archived, response-only sensitivity reducer and Slurm wrapper. It
  reuses cases120--139, the frozen V3.5-like classifier ensemble, existing
  16-by-64 QMC moments, and the one retained raw joint-flow draw per object.
  No image simulation, measurement, model training, or cut tuning was run.
- After usability and `MAG_AUTO<25.8`, `FLUX_RADIUS>=0.5` arcsec retains
  99.94347%/99.94382% of the two actual shear legs, versus
  78.58885%/78.57841% at 0.75 arcsec. The old-probability-weighted retained
  flow draw gives the same 99.943% retention, so 0.5 arcsec is empirically
  almost the magnitude-only sample rather than a meaningful size cut.
- The exact cached magnitude-only response screen gives current-classifier
  `m=-0.842451 +/- 0.254781%` and usable-oracle
  `m=-0.740945 +/- 0.256724%`. Anchoring the paired retained-draw change from
  magnitude-only to 0.5 arcsec gives `m=-0.896906 +/- 0.261884%` and
  `m=-0.803476 +/- 0.261955%`, respectively. For the current classifier the
  0.5-arcsec truth/residual contributions are `-0.387662/-0.509244` percentage
  points; the oracle residual alone is `-0.725701` points. Thus the lower cut
  worsens the central response screen and does not make the classifier the
  dominant remaining term.
- Approximation control: applying the identical one-draw anchor at 0.75
  arcsec gives `m=-0.565299%` instead of the exact cached `-0.372086%`, a
  `-0.193213`-point miss. This is too coarse for a precision calibration claim,
  but the 0.5-arcsec correction from the exact magnitude-only anchor is only
  `-0.054455` points and both data and model reject about 0.056% of rows there.
  Report the 0.5 number as a directional reused-data screen pending the already
  submitted full 500k likelihood/inference run, not as exact 0.3% evidence.
- Artifacts: `archive/research-2026-09-14/scripts/screen_radius050_existing_response.py`,
  its archived job wrapper, and
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/radius050_existing_screen_v1/results.json`
  (SHA256 `647cf3ee...ff8c`). Slurm job16493733 completed in 73 s with exit 0.
  Frozen-environment self-test, exact 0.75 result reproduction, component-sum
  identities, Python/Bash syntax, and whitespace checks passed.

## 2026-09-14 — Submit 500k ConstGold rerun with broader prior and three full passes

### Authorized settings and implementation

- Owner selected intrinsic semi-major `0.37<Re<1.5` arcsec, measured
  `FLUX_RADIUS>=0.5` arcsec (2.5 pixels), unchanged magnitude cuts and pinned
  V3.5-like models, 500k observations, and three full recentered passes.
  Retain the original per-object sandwich uncertainty estimator per pass.
- `scripts/prepare_constgold_inference_input.py` now reads radius and flux
  bounds from the likelihood configuration instead of hard-coding 0.75 arcsec.
  It rejects unsupported/ambiguous cuts and records exact measured bounds.
  Added coverage to `tests/test_constgold_inference_input.py`.
- Added `scripts/prepare_prior_domain_cache.py`: verify full source-shard
  hashes, all-row deterministic feature alignment and geometry; select new
  positive-mass atoms, recompute R_blend with full neighbour graphs, and merge
  uniform active masses with explicit scene/cache provenance. Reuse source
  cases20000–20199 and full neighbour features, with no new image simulation.
- `scripts/build_model_cache_from_zero_view.py` also accepts the explicit
  deterministic-zero-conditions manifest produced by that compaction, and
  recomputes all likelihood probabilities and proposal coordinates.
  Four tests in `tests/test_prior_domain_cache.py` cover broadened support,
  strict boundaries, row/value alignment, and persisted mass/response coverage.
- Documented the workflow in `doc/INFERENCE.md` and the automatically refreshed
  [run report](INFERENCE_CONSTGOLD_RE037_SIZE050.md#authorized-setup).

### Validation and scheduled execution

- Focused input/domain checks: 5 passed before the final merge regression;
  Ruff passed. The initial frozen-checkout suite found two omitted test
  resources (example catalogue and reference launcher), with 241 tests passing.
  Added those resources, preserved the first snapshot/submission hashes, and
  completed compact-response row-count metadata before any science stage ran.
  The corrected isolated scheduled suite passed **244 tests in 21.79 s**.
  Its generated pytest cache was then mistakenly included in the integrity
  manifest; removed only that mutable cache from hashing, verified every
  scientific file unchanged, and resubmitted preparation using the passed gate.
- Run root: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v35_re037_size050_n500k_20260914_v1`.
  Frozen source/config hashes, isolated editable environment, protocol, logs,
  stage records and submissions are retained there. Canonical repository
  likelihood/prior configurations and original run outputs are unchanged.
- Current chain: test16493602; input16493652; prior smoke16493653 and
  remaining shards16493654; merge16493655; fresh model/QMC cache16493656.
  Pass1 normalization/inference/combine16493657/16493660/16493661;
  pass2 16493662/16493665/16493666; pass3 16493667/16493670/16493671.
  Each pass has a normalization combine and a 256-row smoke gate. Prior
  preparation uses scheduled CPU workers; GPU arrays use up to four A40s.
- All three passes recompute all 500k observations and selection normalization
  at the current centre, preserving object IDs and CRN seeds. No bright-only
  carry-forward, fitted offsets, or uncertainty inflation. `iteration_results.json`
  records each completed pass; inspect the third update before calling it converged.
- Provisional timing: first result 3–5 h; approximately 2 h per extra full
  pass, or 7–9 h total, plus queue time. Based on prior measured 100-minute
  inference and 13-minute normalizer; revise from expanded-prior progress.
  Scheduled one-shot status16493672 is about two hours after submission;
  final status16493673 and report jobs record completion/failures and results.
- Startup verified: input preparation completed in 92 s with exactly 500k
  rows from 8,535,713 eligible observations; 322,297 unmatched shape rows
  and 400 unusable supported rows counted. First prior shard completed in
  100 s with 971,457 active atoms versus the former 638,682. Four further
  CPU shards reached source verification/response evaluation. Detailed checks
  are retained in `startup_review.json`; no inference estimate is claimed yet.

## 2026-09-14 — Complete the 500k full-prior normalizer repeat audit

- Completed the scheduled seeds8101/8102/8103 comparison at 64 QMC draws per
  atom on the same full prior. `process_audit_v1/normalization_sensitivity.json`
  records identity checks and propagation of the common gradient/Hessian term
  into the saved full-500k moments. Estimates: seed8101: [0.019776410279905596, 0.00018909575951807259]; seed8102: [0.019775930226750393, 0.00018884238801811747]; seed8103: [0.01977628170893598, 0.00018891280171030476].
- Across-seed standard deviation: `[2.4850259986329427e-07, 1.307852976320111e-07]`. With only three randomizations,
  this is a rough numerical sensitivity diagnostic, not total uncertainty or
  a bound on finite-difference/integration bias. Original outputs unchanged.
- Refreshed [the process review](INFERENCE_CONSTGOLD_500K_REVIEW.md#shared-normalizer-randomization).
  Scheduler states and log tails are saved in `followup_status.json`.
  Next steps remain full numerical convergence, pilot/iteration validation,
  and independent coverage at the declared population.

## 2026-09-14 — Retain original uncertainty and clarify PSF size-cut conversion

- Owner requested keeping the original object-sandwich SEs
  `(3.7430977e-5,3.4209518e-5)`. Recorded this decision in the full process
  review and its scheduled report generator; case/numerical controls remain
  diagnostics. No original estimate, covariance, or inference cut changed.
- Added [T-ratio conversion conventions](CONVENTIONS.md#translating-a-size-ratio-cut-into-half-light-radii):
  squared moments versus half-light radii, Gaussian intrinsic/convolved
  thresholds, Sersic index and axis-ratio dependence, and the distinction
  between fitted T and true moments. Checked local ngmix conversions,
  BlendEMU Gaussian fitting/output columns, ConstGold noise.csv, and the
  current MultiBand_ImSim PSF and galaxy renderers against primary docs.
- Light analytic calculation: for Gaussian PSF FWHM=0.73 arcsec, the
  T_gal/T_PSF>0.5 cut gives Re_intrinsic>0.258094 and Re_convolved>0.447032
  arcsec. ConstGold's Moffat beta=2.224068 has analytic Re_PSF=0.526773
  untruncated, or 0.519272 at the current renderer's 4.5-FWHM truncation.
  The approximately 0.37 arcsec same-profile scaling is not an exact
  translation to a fitted T cut on Sersic galaxies. No new simulation,
  catalogue scan, or inference job was needed for this explanation.
- Validation: Python AST and analytic arithmetic checks; git diff --check.
  An exact empirical T-ratio selection needs retained galaxy/PSF fit sizes
  and a likelihood that models that selection; the current output keeps g.

## 2026-09-14 — Audit full inference process and conditional uncertainty

### Scope and completed checks

- Owner requested a full process review and questioned the uncertainty.
  Added `scripts/audit_inference_uncertainty.py` and three regression tests in
  `tests/test_inference_uncertainty_audit.py`. The audit verifies source and
  moment hashes and exact object-row coverage, reproduces the released
  one-step estimate/row sandwich, and evaluates whole-case sandwich,
  20,000 case-bootstrap replicates, case jackknife, paired draw-ladder changes,
  influence concentration, proposal diagnostics, and classifier train/tune
  overlap. It conditions explicitly on fixed model/prior/quadrature/pilot.
- Frozen audit under the completed run's `recovery_v2/process_audit_v1/`.
  Scheduled CPU job 16493200 passed 238 tests with one optional model-path
  test skipped, then completed the 500k statistics review. Three focused
  analytic regressions passed before submission. The skipped external-model
  path check passed separately with `SBSI_CACHE_DIR` set (1 passed, 9.73 s).
  Original science untouched.
- Original row SE `(3.74310e-5,3.42095e-5)` reproduced. Whole-case bootstrap
  over 100 cases gives `(4.22876e-5,2.93572e-5)`; sandwich and jackknife agree.
  The g1 increase is about 13%; g2 is smaller. This quantifies one omitted
  dependence level and is not a total uncertainty estimate.
- Largest 1% of object contributions account for 90.46% / 84.91% of the
  estimated g1/g2 variance. 11.80% of final complements have ESS<32;
  1.65% have maximum weight fraction>0.5. These are diagnostic flags, not
  bounds on the error of the full exact-plus-complement estimator.
- Code inspection confirms a fixed 8192-draw tilted-stratified budget,
  no object trimming, and no use of legacy `max_iterations=10` in the
  single-update runner. ESS/max-weight stopping thresholds apply to the
  mixture path, not this fixed-budget mode. The release label alone does not
  establish these controls or numerical convergence.
- Added [the full review](INFERENCE_CONSTGOLD_500K_REVIEW.md#scope-and-conclusion)
  covering provenance, joins, prior/cache support, detection/selection,
  normalization, estimation, combination, uncertainty, and training overlap.
  Clarified the actual fixed-budget and conditional-error behavior in
  `doc/INFERENCE.md`; no scientific configuration or original result changed.
- Source-parent join job 16493391 completed in 25 s: 480,357 distinct parent
  templates in the 500k observations, 7.7434% of rows using repeated templates,
  maximum multiplicity five. All source joins and truth-domain checks passed.
  Repeated independent draws conditional on a fixed empirical parent do not
  by themselves justify inflating the conditional sampling error.
- Full-catalogue g1 shifts by 4.55083e-5 from 4096 to 8192 complement draws;
  paired case data SE of that difference is 3.93096e-6. This demonstrates
  unresolved draw-budget dependence, not an independent-seed MC error estimate.

### Completed probes and pending normalizer controls

- Original-normalizer seed8101 is compared with seeds8102/8103 at the same
  64 QMC draws per atom and all 12,760,990 atoms. Retained normalization job
  16493206_0 handles the first shard; unstarted tasks 1--7 were cancelled and
  resubmitted as 16493217 on available A40 16GB workers, capped at three
  concurrent workers, with 20GB host RAM (original peak was below 10GB).
  Combination/propagation job 16493301 depends on both successful groups.
- Probe job 16493229 uses 2,048 fixed observations on a V100 to vary h,
  starting centre, and proposal seed. It holds the original normalizer's
  local quadratic fixed to isolate numerator and Newton-map sensitivity.
  This is a labelled diagnostic, not an updated released inference result.
- All eight probes completed successfully in 19m08s; paired summary job
  16493438 completed successfully. The repeated baseline differs from the
  original saved subset by `(1.50e-9,-3.50e-10)`. Halving h shifts subset g1
  by 5.30034e-4, with paired case data SE 6.16260e-4; the original subset
  row SE 5.09648e-4 becomes 1.06326e-3. This small subset does not establish
  a population-wide h shift or supply a corrected full-500k error.
- Normalizer shards are progressing, with the completed first full A40 shard
  and a completed 16GB shard. Dependent summary job 16493301 will propagate
  the two additional seeds; report job 16493450 will refresh the review from
  its actual results or record failures. One-shot status job 16493451 is
  scheduled about two hours later. Submission manifests, script hashes,
  per-probe moments, and logs are under `process_audit_v1/`.
- Reported uncertainties so far exclude shared normalizer randomization,
  full numerator integration error, nonlinear pilot dependence, model
  training uncertainty, finite-parent/prior uncertainty, and population or
  one-step bias. Pending controls must not be described as completed.

## 2026-09-14 — Reject the clean lambda-1 U-transition classifier despite central 0.138%

### Frozen minimal ablation

- Trained the proposed loss-only ablation on existing catalogues: the same
  V3.5-like nine inputs, preprocessing, 64-by-2 SiLU MLP, three seeds,
  cases40--99 training data, cases100--119 tune data, optimizer, batches, and
  per-leg usable-measurement U target. The only change is lambda 1 times
  `BCEWithLogits(z_plus-z_minus,U_plus)` on U-discordant pairs, added to the
  marginal U-BCE. Response cases and moments remained absent from training and
  checkpoint selection; there was no feature, architecture, or lambda sweep.
- All three seeds improved tune transition BCE by 0.00343--0.00357 and raised
  transition-direction accuracy to 0.558--0.561, while marginal tune BCE
  worsened by 0.00255--0.00283. Selected epochs were 56, 55, and 65. The login
  resource snapshot showed no visible GPU, so array 16493160 ran on scheduler
  A40s; all seeds completed in 2m10s--2m27s with about 1.15 GiB peak RSS.
- Protocol SHA-256 is
  `41a0118ba5fa88c820fa382e0e0366f0c49d3f14aef4181583c20ea4dcaedab1`;
  tune-only review SHA-256 is
  `d7303837dfefa6c3c9d270cab65aafb190fe3058597bfb7fc61c3b49e1fcc613`.
  The selected seed checkpoint hashes are `36a03d...c745e59`,
  `0daaa2...8a120`, and `bdb52b...cb2d0`.

### Cached response screen and decision

- Job 16493191 reused exactly the cases120--139 lambda10 epoch154 / trial9
  selected-flow moments, measured magnitude/radius cuts with radius at least
  0.75 arcsec, no measured ellipticity cut, 16-by-64 stored CRN QMC mean, and
  identical 10,000 case-bootstrap weights. It completed in 32 seconds with
  1.00 GiB peak RSS and performed no image simulation, catalogue construction,
  flow evaluation, or model selection.
- The proposed ensemble gives `R11=1.00457339` and central
  `m=+0.138194 +/- 0.224551%`, with 95% interval
  `[-0.293565,+0.582663]%`. Although the central absolute value is below 0.3%,
  the decomposition is truth `+1.580206%` and residual `-1.442012%`: extreme
  opposite-sign cancellation. Relative to current V3.5-like, the paired change
  is `+0.510279 +/- 0.002945` percentage points, comprising truth
  `+2.129615%` and residual `-1.619335%` changes.
- The proposed global U fraction is closer than current (gap 0.0000768 versus
  0.0001645), but BCE and Brier are worse (0.177158/0.052080 versus
  0.174726/0.051302). The classifier therefore moves shear response strongly
  without supplying a reliable global-score diagnostic.
- Decision: reject lambda-1 discordant-transition training and do not call the
  central 0.138% a calibration result. It fails the component anti-cancellation
  gate and the full-interval 0.3% gate. Do not tune lambda on the reused
  response screen. If classifier work continues, the minimal principled test is
  marginal BCE plus a directly defined training-case response-moment constraint,
  selected on tune cases and still judged without allowing truth/residual
  cancellation; the correlated-pair conditional loss is not a proper marginal
  probability score in general.
- Canonical response result:
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_transition_lambda1_v1/response_screen.json`,
  SHA-256 `b9a26ca30de546f699a123ef395993446d1b77fb881ea14e092f43aac9ed15d9`.
  Python compilation, script self-tests, exact preprocessor identity across all
  current checkpoints, `ruff`, Bash syntax, `git diff --check`, source and
  checkpoint hashes, no-response protocol/receipt checks, response/component
  identities, exact reproduction of current/oracle result dictionaries, and
  Slurm exit accounting passed. This remains a reused one-component,
  one-amplitude development screen and cannot promote a likelihood.

## 2026-09-14 — Audit the ConstGold truth cut against the likelihood training domain

- Owner questioned the 0.5 arcsec primary truth-radius floor and recalled 0.37.
  Verified `primary_re_min=0.37` in the physical flow's inherited prepared-data
  recipe at `plain_complete_flow_response_re037_v1/grid_covariance_capped_lambda1_v1/prepared/manifest.json`.
  The subsequent circularized-condition preparation preserves population cuts.
  Trial9 response metadata also specifies a primary radius floor of 0.37.
- Checked all 20 registered inference-prior shard reports: their positive-mass
  primary domain remains `18<r<25.8`, `0.5<Re<1.5`. The new ConstGold input
  adapter and submitted stage command copied this older domain. The truth cut
  was an inference-population choice, not a requirement of the latest flow.
- The completed 500k result is internally matched to that restricted prior,
  but omits the 0.37--0.5 population. It must not be interpreted as inference
  on the full likelihood training domain or on a sample selected only by
  measured quantities. Applying a truth cut is possible for a simulation
  diagnostic; survey observations generally cannot supply that truth.
- `recovery_v2/population_domain_review.json` records the evidence and scope
  limitation beside the frozen run. No original scientific outputs changed.
  Changing the population requires a matching prior and rebuilt model/proposal/
  response caches and normalization, followed by new observation preparation
  and inference; relaxing only the observation cut would create a mismatch.
- Validation: metadata assertions passed for the training recipe, all 20
  prior shards, and the input manifest; `git diff --check` passed. No heavy
  scan or new inference was launched during this audit.

## 2026-09-14 — Reject the historical transition detector on identical cached moments

### Comparison and result

- Added an archived mechanism-screen worker and Slurm wrapper that score the
  frozen 17-input, lambda-1 transition-aware raw-detection checkpoint on the
  already materialized cases 120--139 detector frames. The comparison reuses
  the identical lambda10 epoch154 / trial9 selected-flow moments, measured
  `MAG_AUTO<25.8` and radius-at-least-0.75-arcsec cuts, no measured ellipticity
  cut, 16-by-64 stored CRN QMC mean, and 10,000 case bootstraps. It performs no
  image simulation, catalogue construction, flow evaluation, or refitting.
- The historical model gives `R11=1.01323119` and
  `m=-0.717462 +/- 0.221095%` (case-bootstrap 95% interval
  `[-1.141705,-0.277969]%`), versus the reproduced V3.5-like
  `m=-0.372086 +/- 0.222617%`. The paired transition-minus-current change is
  `-0.345376 +/- 0.006300` percentage points, with 95% interval
  `[-0.357399,-0.332871]`.
- The transition model improves the truth-component contribution by
  `+0.120595 +/- 0.002334` percentage points, from `-0.549409%` to
  `-0.428814%`, but worsens the residual component by
  `-0.465971 +/- 0.006163` percentage points, from `+0.177323%` to
  `-0.288648%`. It therefore removes the current cancellation in the wrong
  direction and does not meet the central, component, or interval 0.3% screens.
- Against 3,877,730 leg rows the old model predicts usable/detected fraction
  0.898985 versus observed `D=0.896995` and `U=0.896956`; only 152 aggregate
  labels separate the observed D and U counts. Its U-BCE is 0.176976 versus
  current 0.174726, while its Brier score is lower (0.046432 versus 0.051302),
  reinforcing that global proper scores do not determine selected response.

### Provenance, validation, and limitation

- Canonical paired result:
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/transition_aware_existing_screen_v1/results_paired.json`,
  SHA-256 `e71b5580632c19e898a7f71153b9d182bd17d9d6e12b7f20cc87a45760cb37d4`.
  Successful jobs 16493107 and 16493119 completed in 46 and 33 seconds with
  0.91 and 0.97 GiB peak RSS. Job 16493114 exposed and preserved a stale
  pre-cleanup protocol path; the evaluator now verifies the byte-identical
  archived script and job wrapper by their frozen hashes rather than restoring
  dead active files.
- Python compilation, evaluator self-test, Bash syntax, `git diff --check`,
  checkpoint/protocol hashes, cached receipt hashes, exact reproduction of the
  prior V3.5 and oracle result dictionaries, response/component identities,
  and Slurm exit accounting passed.
- This rejects the old checkpoint as a replacement, not transition supervision
  in isolation: it changes the event target (raw D rather than U), 17-feature
  semi-major/impact condition, architecture, cases, and training distribution
  simultaneously. A clean causal test would keep the current nine-input U
  classifier and vary only a paired-transition loss; it should not reuse or
  promote the historical checkpoint.

## 2026-09-14 — Complete 500k ConstGold inference with v1.2 and V3.5-like

### Completion and validation

- All twenty 25k partitions in array 16492425 and strict combination 16492426
  completed with exit 0. Combination finished at 08:42:39 UTC / 10:42:39 Berlin.
  Each partition used about 19–20 minutes; the full inference array and merge
  took about 99 minutes with at most four concurrent A40 workers.
- Result: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v35_n500k_20260914_v1/recovery_v2/combined/result.json`.
  Provenance, validation, and interpretation are recorded beside it in
  `recovery_v2/completion_review.json`; per-observation moments and the full
  draw ladder are retained. No numerical settings or model artifacts changed
  after the recovery's 236-test and real GPU smoke gates.
- Completion review checked all 20 partition result hashes, contiguous
  observation coverage 0:500000, requested `v1.2-infer` / `v3.5-like` identities,
  finite estimates and standard errors, and positive information eigenvalues.
  The combiner additionally verified moment hashes and shared input/model/code
  identities before writing the result. No new heavy computation was needed.

### Result and interpretation

- Raw plus-leg ConstGold cases 40–139, injected `(g1,g2)=(0.02,0)`, uniform
  observation seed 20260914; domain, usability, measured cuts, dropped-row
  counts, and pinned model identities are those recorded in
  [the submission entry](#2026-09-14--submit-500k-raw-constgold-inference-with-v12-and-v35-like).
- One-step estimate: `g1=0.0197764102799`, `g2=0.0001890957595`.
  Object-level robust standard errors: `0.0000374309772`, `0.0000342095177`.
  These are statistical errors for this estimator, not model or integration
  uncertainty. No empirical offsets or calibration correction were applied.
- The fractional g1 difference from injected 0.02 is `-1.11795%`.
  This is not `m=R_sim/R_model-1`, which requires the response comparison
  defined in [CONVENTIONS.md §7](CONVENTIONS.md#7-multiplicative-bias).
- Draw-count dependence remains: g1 changes from `0.0197309019551` at 4096
  complement draws to `0.0197764102799` at 8192, a shift `0.0000455083248`
  (about 1.22 times the final robust standard error). Consequently this run
  does not establish numerical convergence or a 0.3% calibration claim.
  Any next scientific assessment should quantify this draw dependence before
  attributing the full residual to the likelihood.
- Documentation-only completion update; `git diff --check` passed. The bounded
  accounting check 16492427 remains scheduled; the inference result is ready.

## 2026-09-14 — Repair physical-flow inference precision and resume ConstGold

### Failure and changes

- The original preparation chain completed: 500,000 observations from 5,148,859
  eligible rows, all trial9 responses, V3.5 model/QMC caches, and four exact
  selection-normalization shards plus their merge. The 256-row smoke job
  16491920 then failed in ragged importance scoring: `index_copy_` received a
  float64 physical-flow log density and a float32 destination. Dependent full
  inference 16491921 and combination 16491922 cancelled before execution.
- `sbsi/catalogue_likelihood.py` now promotes importance-output buffers using
  both context and target dtypes, preserving float64 physical densities on
  dense and ragged paths. Proposal logarithms use that same precision.
  `tests/test_catalogue_null.py` exercises the real physical-flow class,
  atom-dependent blend shifts, direct density agreement, and ragged padding.
- `sbsi/catalogue_null.py` accepts an optional progress callback;
  `scripts/run_inference.py` logs completed observation counts after its first
  chunk and about every minute thereafter, plus its final chunk.
- Recovery uses a separate frozen source/environment under the original run's
  `recovery_v2/`, reusing completed inputs, responses, caches, and normalization
  through explicit links. Original source hashes, logs, and failure records
  remain intact. Numerical settings and model artifacts are unchanged.

### Validation and execution

- `ruff check sbsi scripts tests`, compileall on changed Python files,
  `bash -n jobs/job_constgold_inference.sh`, and `git diff --check` passed.
- Focused regression: 2 passed (including real physical-flow float64 and the
  existing float32 padded-slot test). Isolated scheduled suite: 236 passed
  in 23.07 s; operational CLI imports and all V3.5 pinned model hashes passed.
  Real 256-row GPU smoke passed in 145.23 s total (78.58 s inference),
  retaining `v1.2-infer` / `v3.5-like`, 256 partition rows, and all 500,000
  selected input rows. The full array was released.
- Recovery Slurm chain: tests 16492423; 256-row GPU smoke 16492424;
  twenty 25k partitions 16492425 (at most four concurrent A40 workers);
  combination 16492426. Bounded two-hour status job 16492427 records
  accounting to `recovery_v2/scheduled_status.json`.
- Startup verified on all four initial full workers: each completed its
  first 128 rows; worker 0 reached 1,536/25,000 after 127.36 s of inference.
  `recovery_v2/startup_review.json` records the observed progress. The bounded
  status job writes accounting only; it does not wake an agent or repair jobs.
- No scientific result is available yet; V3.5-like remains a development
  likelihood, with the conventions in
  [CONVENTIONS.md](CONVENTIONS.md#9-reporting-checklist).

## 2026-09-14 — Submit 500k raw ConstGold inference with v1.2 and V3.5-like

### Run and scientific identity

- Owner requested 500,000 ConstGold observations with the latest numerical
  and likelihood identities. Submitted `v1.2-infer` (exact K=1024 stratum,
  complement ladder 512–8192, unchanged seeds and h=0.001) with the pinned
  `v3.5-like` epoch154 / trial9 / three-classifier ensemble.
- Input uses actual plus-leg `(g1,g2)=(0.02,0)` measurements from cases 40–139,
  independently joined to truth and crossmatches. Apply the configured finite
  prior's strict primary domain `18<r<25.8`, `0.5<Re<1.5`, then per-leg U and
  `MAG_AUTO<25.8`, radius>=0.75 arcsec, with no measured-|e| or neighbour cut.
  Uniformly sample 500,000 unique rows without replacement using seed 20260914.
  The input manifest records all unmatched, unusable, and selected counts.
- This is a one-step catalogue inference run on reused ConstGold, not a new
  image simulation or a promotion of the likelihood/calibration claim.

### Changes and execution

- Restored `scripts/prepare_constgold_inference_input.py` deliberately as a
  raw-measurement adapter; it does not import archived code. Added
  `scripts/build_model_cache_from_zero_view.py`: verify the original compaction
  report and scene hashes, reuse only deterministic neighbour-complete truth
  features, and recompute classifier probabilities and flow QMC coordinates.
- Fixed `sbsi/catalogue_likelihood.py` to preserve invariant crowding in compact
  views while reevaluating the shape-dependent classifier at every shear.
  Added circularized radius to the reusable feature set and reject attempts
  to rebuild missing conditions from a compact scene's empty neighbour graph.
- Added regression coverage in `tests/test_catalogue_likelihood.py` and
  `tests/test_constgold_inference_input.py`, the scheduler stage wrapper
  `jobs/job_constgold_inference.sh`, and workflow documentation in
  [INFERENCE.md](INFERENCE.md#constgold-observation-input). Updated README layout.
- Run root:
  `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v35_n500k_20260914_v1`.
  It contains frozen SBSI and BlendEMU sources, a private installed environment,
  source hashes, stage commands, submission manifests, resources, and logs.
  Existing caches and shared Python installations are unchanged.
- Slurm chain: final runtime gate 16491913; observations 16491914;
  trial9 response array 16491915; response merge 16491916; model/QMC cache 16491917;
  normalization array 16491918 and merge 16491919; 256-observation check 16491920;
  twenty 25k inference partitions 16491921; strict combination 16491922.
  GPU arrays permit at most four concurrent A40 workers. One-shot status
  job 16491923 is scheduled for about two hours after submission.

### Validation and next check

- Focused cache/observation tests: 5 passed. Full isolated scheduled suite:
  235 passed in 21.23 s; every V3.5-like artifact hash verified. Final frozen
  gate 16491913 also passed all 235 tests, imported each operational CLI with
  `--help`, and verified every artifact hash before releasing preparation.
- `ruff check sbsi scripts tests`, `python -m compileall -q sbsi scripts tests`,
  `bash -n jobs/job_constgold_inference.sh`, and `git diff --check` passed.
- Initial gate 16491894 found an omitted example-data fixture in the snapshot
  (234 passed, 1 failed); dependent jobs cancelled before science execution.
  Added the fixture, preserved the failed-attempt manifest/logs, and passed
  gate 16491908. Also corrected the new cache builder's coordinate import and
  the stage driver's result-count check before the final submission.
- Observation preparation completed in 80.36 s: 5,148,859 eligible rows,
  500,000 frozen observations, zero missing crossmatch-to-shape or
  crossmatch-to-truth joins, 322,297 shapes without crossmatches dropped, and
  238 unusable rows within the truth domain dropped. The first three response
  shards completed in 63–65 s each with finite atom-aligned responses.
- All 20 response shards and their strict merge completed. Model/QMC job
  16491917 passed source checks and row alignment, evaluated the three-member
  classifier across all 12,760,990 atoms, and is making regular QMC progress
  (over 1.1 million atoms in its first minute of sampling). See the external
  `startup_review.json`. A small number of response-emulator training-boundary
  extrapolations are retained and logged; no clipping or empirical corrections
  were applied.
- Yielded after verified preparation progress; status job 16491923 will record
  a bounded follow-up check. Results remain pending, and inference job
  16491921 waits for caches, normalization, and the 256-row inference check.
  Final output is
  `combined/result.json`, with `completion.json` written only after all 500,000
  observation identities combine without gaps or overlaps.

## 2026-09-14 — V3.5-like identity and repository reduction

### Outcome

- Named the development model set `V3.5-like`: lambda10 epoch154
  `physical_disk_affine`, trial9 atom-aligned `R_blend`, and the arithmetic
  mean of the coherent-`U` classifier checkpoints trained with seeds
  20260913, 20260914, and 20260915.
- Kept the configured releases unchanged: `v1.1-infer` remains the default
  estimator and `v3.2-like` remains the default likelihood. `V3.5-like` is a
  path-only development identity and is not a 0.3%-calibration claim.
- Preserved all four stochastic measurement outputs. The checkpoint target
  order is `(measured_ngmix_g1, measured_ngmix_g2, measured_flux_radius,
  measured_flux_from_mag_auto)`; magnitude and log-radius are invertible
  coordinate views, not removed variables.
- Kept the sample definition at `MAG_AUTO < 25.8`, convolved SExtractor
  `FLUX_RADIUS >= 0.75 arcsec`, and no measured-`|e|` cut.
- Pinned that selection in the V3.5-like config as the equivalent physical
  checkpoint bounds: radius at least 3.75 pixels and flux-from-magnitude above
  47.8630092322638 at zero point 30.

### Model and runtime changes

- Added `configs/likelihood_v3_5_like.json` with exact artifact paths, target
  order, ensemble aggregation, observing conditions, and SHA-256 identities.
- Reduced `sbsi.models` to the V3.5-like path preset and loaders required by
  likelihood construction. The classifier loader now returns the
  equal-probability three-member ensemble.
- Reduced the package CLI to `show-model` and `validate-model`.
- Removed training-time feature registries and unsupported realization-aware
  and spline flow families from the active measurement-model module. The
  active loader retains the affine, mean-affine, disk-affine, and
  physical-disk-affine architectures needed by configured or V3.5-like
  artifacts.
- Removed the unused PyYAML and notebook dependency groups from
  `pyproject.toml`.

### Repository cleanup

- Reduced the active surface from 48 to 28 Python package modules, 77 to 15
  Python scripts, 111 to 4 scheduler wrappers, and 81 to 26 test modules.
- Moved research-only training, response tuning, ConstGold analysis, plots,
  failed branches, tests, and notes to `archive/research-2026-09-14/`. The
  archive contains 364 files (21 MB), including 20 modules, 63 scripts, 107
  jobs, and 55 tests; it is excluded from package installation and test
  discovery.
- Replaced sprawling active documentation with the seven canonical files in
  `doc/`; the full earlier work log is preserved as
  `archive/research-2026-09-14/doc/WORKLOG_FULL.md`.

### Validation

- Baseline focused gate before pruning:
  `python -m pytest -q tests/test_release_config.py tests/test_catalogue_likelihood.py tests/test_inference_provenance.py`
  — 64 passed.
- Final active suite: `python -m pytest -q` — 232 passed, 2 skipped.
- `ruff check sbsi scripts tests` — passed.
- `python -m compileall -q sbsi scripts tests` — passed.
- `bash -n jobs/*.sh` — passed for all active wrappers.
- Active dependency audit — no missing relative modules, missing job-script
  targets, archive imports, or invalid JSON configs.
- Archive audit — all 75 tracked files removed from active paths have a dated
  archive copy (the work log uses the explicit `WORKLOG_FULL.md` name).
- `git diff --check` — passed.
- With `SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches`,
  `python -m sbsi validate-model V3.5-like` verified every pinned hash. Direct
  loading confirmed `ConditionalPhysicalDiskFlow`, the four stored targets,
  three classifier members, and the same nine features in every member.

### Scientific status and next step

- No new image simulations or catalogues were produced. All model selection
  and mechanism checks reuse existing catalogue products.
- The cached cases120--139 screen gives `m=-0.372 +/- 0.223%` for this frozen
  flow plus learned classifier, with truth-population and measurement-residual
  contributions `-0.549pp` and `+0.177pp`; it fails the anti-cancellation gate.
- The broader cases40--139 actual-usable-flag oracle gives
  `m=-0.129 +/- 0.125%`. This routes the next minimal development effort to
  usable-probability calibration, but does not by itself establish final
  image-level 0.3% calibration.
- Next work should improve or calibrate the compact nine-input classifier on
  existing catalogues, keep the joint four-output flow frozen, and evaluate
  held-out response and anti-cancellation before changing the default
  likelihood.
