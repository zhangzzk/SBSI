# Work log

## 2026-09-25 — Retain C8 by owner decision and audit emulator inputs

Owner explicitly chose to keep C8 provisionally and move to the emulator.
Recorded the working F8+C8+original-disk combination in the experiment's
working_selection.json, using first predeclared seed20260913 as primary and
preserving the second as a replicate. Historical acceptance failures remain
unchanged; no independent validation or production-preset promotion is claimed.
See [working decision and inputs](V36_FEATURE_SIMPLIFICATION.md#owner-retains-c8-provisionally-emulator-inputs).

Audited the nine emulator input names, exact size/separation and flux-noise
transformations, generated same-draw radius/magnitude conditioning, scalar pair
output and218-tree physical-moment fit against source and pinned metadata.
Tree capacity is the first simplification target; feature ablations remain
separate. No emulator training launched in this input review. Validation:
read-only source/metadata inspection, pinned C8 training receipt and whitespace
check. No model weights or existing scientific results changed.

## 2026-09-25 — Complete C8 evaluation; retain C17 and clear redundant timers

Reviewed completed C8 jobs: both training seeds, all80 evaluation scenes and
summary16695740 exited0. C8 fails the owner-amended non-degradation criteria.
Its main-cut biases are+0.020770% and+0.062423% (case SE about0.265pp), but tuning
BCE rises fromabout0.18304 to0.19362/0.19378. Paired degradations+0.010583 and
+0.010748 have intervals entirely abovezero and the0.0005 margin; constant-shear
BCE/Brier also worsen. No C8 model is adopted. The development baseline remains
F8 + retained C17 + original disk. See
[C8 completed decision](V36_FEATURE_SIMPLIFICATION.md#c8-completed-decision).

Owner asked why16695741 and16695116 were pending. Verified both were deliberately
delayed two-hour status jobs (`BeginTime`), while all scientific work had finished.
Cancelled these redundant receipts after direct review; final status16695742
already completed. Recorded completion in the C8 launch receipt. Validation:
sacct exit codes, completed comparison intervals and training receipts;
whitespace check. No weights, thresholds or production configurations changed.

## 2026-09-25 — Reject C12 and launch classifier with exact F8 inputs

Owner requested the F8 input representation on the classifier. First reviewed
completed C12 decision: rejected under the amended uncertainty-tolerant rule.
Paired tuning BCE losses+0.003296/+0.003345 are resolved and above0.0005 margin;
classification/Brier/count checks also fail. Primary m alone is similar to
matched controls. Retained C17 remains fixed; no favorable seed substitution.
See [C12 result](V36_FEATURE_SIMPLIFICATION.md#c12-completed-decision).

Addedsetup_v36_classifier_f8_inputs.py, prepare_v36_classifier_f8_inputs.py,
a scheduler launcher and three projection tests. Frozen isolated C8 checkout
and cachev36_classifier_f8_inputs_20260925_v1 reuse both completed C17 controls.
C8 takes exactly the flow's five galaxy properties and three physical Gaussian
crowding summaries. Identity-checked joins preserve all original half-shear
labels/rows and each leg's galaxy properties; format and coordinates explicitly
change. Missing/duplicate IDs reject the case. The classifier keeps original
training recipe, two seeds, fixed F8/disk, and owner-amended acceptance margins.
Full source/data/control hashes and initial core/zero-new-weight convention are
recorded before fitting. [C8 contract](V36_FEATURE_SIMPLIFICATION.md#c8-classifier-with-exact-f8-raw-inputs).

Jobs: prepare16695734; train16695735_1,3 (two GPUs); evaluate16695737_0 then
16695739_1–7; summary16695740; local status16695741/42. Eight projection and
acceptance tests passed in5.03s. Preparation completed in2m42s across80 scenes:
zero labels changed, zero rows removed, zero unmatched identities; prepared
manifest SHA256e953c2900988. Both seeds passed actual-data smoke training and
checkpoint export, then reached full-training epoch23 with finite losses.
These startup losses are not used as an acceptance result.
Ruff, shell syntax and whitespace checks pass. Source/main release installation
and production presets are unchanged. No C8 result or independent acceptance
is claimed; emulator capacity reduction remains later work.

## 2026-09-25 — Repair Boolean-label classifier scoring after training

All four C17/C12 training runs completed. C17 tuning BCE is0.183037854/0.183036001;
C12 is0.186334092/0.186381115 for seeds20260913/20260914. These are tuning
results, not a completed combined-model acceptance result.

Evaluation array16695110 failed before writing case results because the new
BCE scorer applied unary minus to Boolean usability labels. Fixed the scorer
to cast labels tofloat and added two regression tests for correct BCE/Brier and
Boolean/numeric equivalence. Original frozen training checkout/protocol/models
are preserved. The new runner is a separate pinned copy, with an exact one-line
repair recorded inevaluation_scoring_repair.json; failed status receipts remain.

Submitted repaired CPU chunk0 as16695636, then remaining chunks1–7 as16695637
(maximum four concurrent). Retargeted existing summary16695112 to both repaired
jobs, keeping the owner-amended acceptance stage16695134 and its criteria
unchanged. Added repaired launcher; no retraining or favorable model selection.
Validation: source diff asserted to be only the numeric-label cast; Ruff/shell
syntax and whitespace checks. Both regression tests passed (16.46s); the
repaired worker completed cases40/41 with retained-probability and sufficient-
sum replay checks passing.

## 2026-09-25 — Allow small classifier differences within uncertainty

Owner clarified that slightly worse point estimates are acceptable within
uncertainty. Added an explicit [acceptance amendment](V36_FEATURE_SIMPLIFICATION.md#owner-amendment-small-differences-within-uncertainty):
paired95% degradation intervals must include zero or favor improvement, with
upper endpoints still below original margins. Removed the strict point veto;
all other endpoints, seeds and controls remain mandatory. Broad uncertainty
or resolved worsening is not a pass.

Addedscripts/amend_v36_classifier_acceptance.py and five focused tests. The
experiment stores a timestamped owner amendment and a pinned copy of the new
script. Frozen training protocol/source hashes are unchanged, avoiding any
interruption to ongoing runs. Originalcomparison.json remains an audit record;
newacceptance_amended.json is the operative rule, preserving original flags.
Scheduled test16695133 and amended decision16695134 after original summary;
retargeted final status16695117. No completed candidate comparison inspected
before this user-directed amendment. No training or model selection changes.
Validation: five amended-rule tests passed in0.06s (job16695133); Ruff and
whitespace checks pass; scheduler accepted dependencies.

## 2026-09-25 — Launch matched classifier simplification with non-degradation gates

Owner authorized continuing, accepting reductions only without performance
loss. Added setup, training/reweighting, summary, preflight and status scripts;
four scheduler launchers; five gate tests. See
[C12 execution contract](V36_FEATURE_SIMPLIFICATION.md#c12-matched-non-degradation-experiment).
Fresh C17/C12 controls use two paired seeds and original data/labels/splits/loss;
F8 and disk remain fixed. Acceptance requires non-worse points plus bounded
paired uncertainty across bias, classification/calibration, counts, cross
response and regional cancellation, for both seeds versus both controls.
Frozen margins and no automatic promotion are recorded inprotocol.json.

Isolated checkout/venv at`/project/ls-gruen/users/zekang.zhang/sbsi_classifier_occam_20260925/`
and cache`v36_classifier_occam_20260925_v1`. Initial preflight16695107 caught
missing release configuration JSONs before training. Copied/pinned unchanged
configs, preserved original protocol, then preflight16695113 passed five tests,
exact paired initialization, retained both-leg probabilities oncase40, and
synthetic end-to-end summary. No acceptance thresholds or losses changed.

Training array16695108_0–3 runs at most two GPUs; moved from queuedinter to idle
cip16GB A40s. CPU evaluation16695110_0–7 reweights immutable F8 row dumps using
all trained classifiers and checks retained-probability and sufficient-sum
replays. CPU summary16695112 requires all80 scenes. Status jobs16695116/17 write
local receipts in two hours and after summary; no chat wakeup is promised.
Both initial arms passed real-data smoke training and checkpoint export, then
reached full-training epoch9 with finite decreasing tuning BCE (C17:0.187556;
C12:0.192054). These early losses are not an acceptance result.
Validation: Ruff and shell syntax checks, five targeted tests, real-data replay,
synthetic summary integration, git diff --check. No production configurations
changed. C9 and emulator capacity tests await C12 review; no performance claim yet.

## 2026-09-25 — Accept F8 for development and review remaining simplifications

Owner judged the completed F8 result good enough and asked about the other
models. Recorded F8 as the preferred flow for subsequent development comparisons,
without changing a production preset, historical test outcome or independent
acceptance status. Updated stale immediate-status text in
[the simplification plan](V36_FEATURE_SIMPLIFICATION.md#immediate-status-and-implementation-boundary)
and added [next components](V36_FEATURE_SIMPLIFICATION.md#next-component-simplifications-after-owner-acceptance-of-f8).

Source review confirms classifier17 inputs comprise core8, five annuli and four
state-major-axis Gaussian summaries; its MLP has two64-unit hidden layers and
checkpoint selection uses tuning BCE. Recommended staged17→12→9 retraining,
now holding accepted F8 fixed. Pair-emulator metadata confirms nine inputs,
218 trees and a physical-moment objective; capacity reduction should precede
removing pair/measurement conditioning. No new training or production changes.
Validation: inspected active source and model metadata; git diff --check passes.

## 2026-09-25 — Record completed F8 development comparison

All80 constant-shear scenes and row replays completed successfully; GPU array
16690166 and CPU summary16690221 exited0. Reviewed final comparison receipt,
case coverage, paired intervals, regional contributions and selected full
validation NLLs. Added [F8 results](V36_FEATURE_SIMPLIFICATION.md#f8-development-results)
and replaced the stale statement that no reduced-input result exists.

Under measured-only0.60arcsec/25.8 per-leg cuts, F8 smooth-classifier bias is
+0.138352% ±0.266521pp case-bootstrap SE, versus control+0.285858% ±0.265609pp.
Paired change−0.147506pp has95% interval[−0.167216,−0.128505]pp. Density improves
by0.087332nats/row, but the predeclared±0.05pp equivalence screen fails and
regional cancellation increases: sumabs2.817658→3.915031pp. Absolute F8 bias
95% interval[−0.395797,+0.655236]% does not establish0.3% calibration.
No production promotion or new runs. Matched training repeats, both axes,
numerical convergence and population/independent validation remain necessary.
Scheduled read-only status check16693407 also completed successfully. Validation:
sacct exit codes, final JSON80 cases/80 row replays and original NLL receipts;
`git diff --check` passes. No code or scientific settings changed.

## 2026-09-25 — Schedule requested two-hour status check

Owner requested a two-hour wake-up. Tool discovery found no callable Codex
`automation_update` or conversational wake-up capability, so no chat wake-up
was created. Explicitly disclosed this limitation and scheduled the existing
read-only Slurm status script with `sbatch --begin=now+2hours
jobs/job_check_v36_f8.sh`: job16693407. It writes a local receipt under the F8
experiment root; it cannot reopen the task or send a notification. Existing
checks and scientific jobs are unchanged. Validation: Slurm accepted the job;
scontrol confirms its future start eligibility and status-check command.

## 2026-09-25 — Move final pending F8 evaluation task back to cip

Owner requested moving16690166_7 frominter tocip. Updated the pending task
in place to `Partition=cip Gres=gpu:a40-16gb:1`; original CPU/memory/time settings,
case chunk, frozen commands and CPU summary dependency are retained. Updated
[the experiment record](V36_FEATURE_SIMPLIFICATION.md#f8-final-task-returned-to-cip)
and external `scheduling_task7_cip_20260925.json`. Slurm accepted the update;
squeue confirmscip. Sacct confirms six evaluation chunks completed with exit0,
task3 running. Task7 started oncip-cl-h01g03n3 and processed25,600 rows of
case110 without errors. Scientific settings are unchanged.

## 2026-09-24 — Expand F8 evaluation across cip and inter

Owner explicitly authorized `inter`. Updated pending tasks16690166_5/6/7 in
place to partitioninter with one full A40 each, preserving the same four CPUs,
32GB host memory, six-hour wall time, frozen commands and array throttle6.
Tasks0/3/4 continue oncip; tasks1/2 have completed. The existing array summary
dependency and scheduled check16692553 remain valid. Updated `AGENTS.md` and
[the allocation record](V36_FEATURE_SIMPLIFICATION.md#f8-expansion-to-inter);
external `scheduling_inter_amendment_20260924.json` records this authorization.
Validation: Slurm accepted all three updates and squeue confirms the new
partition/GRES requests. Both new workers5/6 started onkng-cl-nv01 and processed
their first1024 rows for cases90/100 without errors: five GPU workers total.
Task7 remains queued for priority. Scientific data, models and evaluation
settings are unchanged.

## 2026-09-24 — Increase F8 evaluation GPU parallelism

Owner requested more GPUs, superseding the earlier two-GPU experiment limit.
Updated active Slurm array16690166 in place from throttle2 to6 and moved pending
tasks3–7 to idle16GB A40 allocations. Training16690101 and refinement16690103
have completed successfully; completed evaluation task1 and running tasks0/2
are retained. A third worker, task3, started on cip-cl-h01g03n2. The partition
QoS `cip` has a hard per-user limit of three GPUs; remaining workers report
`QOSMaxGRESPerUser`. No scheduler-policy override was attempted.

Updated the scoped resource instruction in `AGENTS.md` and recorded allocation
changes in [the experiment record](V36_FEATURE_SIMPLIFICATION.md#f8-gpu-allocation-increase).
External `scheduling_amendment_20260924.json` preserves authorization and resource
evidence alongside the original immutable launch receipt. Scheduled read-only
check16692553 for about two hours later. This change affects scheduling only:
model/data identities, cases, CRN seeds, draws, batches and comparison criteria
remain fixed. Validation: scontrol confirms throttle6 and per-task resource
requests; squeue confirms three running GPUs; sacct confirms successful training
and refinement. A GPU-memory spot check on an existing worker returned716MiB,
not a peak-memory guarantee. No new scientific result is claimed.

## 2026-09-24 — Launch isolated eight-input close-crowding flow experiment

Owner authorized execution of the simplification design. F8 retains the five
target properties and three physical Gaussian crowding features, removing only
the flow's legacy near/far/max summaries. The17-input classifier and original
disk emulator remain fixed; the classifier's core features are preserved.
See [execution record](V36_FEATURE_SIMPLIFICATION.md#f8-execution-record).

Added setup/projection/stage/comparison/status scripts and five scheduler
launchers. `scripts/setup_v36_f8_experiment.py` freezes212 source/configuration
files from the active September24 research checkout into
`/project/ls-gruen/users/zekang.zhang/sbsi_flow_f8_20260924/`, with an isolated
installed venv. Only feature contract, evaluator full-context handling and
package installation differ. No archived import, shared installation change,
input zeroing, model-weight slicing, loss change or production configuration
change occurs. `experiment.json` pins seed611, split501, original training
stages, all fixed artifacts, density gate0.001 and paired development response
equivalence budget+/-0.05pp. No constant-shear checkpoint selection is allowed.

`scripts/project_v36_flow_features.py` verifies source hashes, projects columns
0–4/8–10, reloads every output and exactly checks dtype, order, identities,
targets, gamma and usability. CPU16690073 completed in1m50s;400 base plus800
harmonized files, zero altered labels and zero removed rows. New manifests
e1303a08a929/149f783002ff preserve all source counts/splits and retain receipts.
Four tests in `tests/test_v36_feature_projection.py` pass, including preservation
of unusable-row NaNs and arbitrary identity ordering, and rejection of already
projected/wrong-width inputs. The resource skill was used; the login node only
performed light setup, with projection/training/tests under Slurm.

CPU16690099 completed in5m45s: recomputed response targets and baseline moments
reproduce the full-feature banks within2e-12; the usability audit also completed.
GPU16690101 passed the inherited density-gate test plus a real
projected-data physical-density/backward preflight (8 raw /16 transformed /4
output dimensions), loaded50,246,394 training and12,563,706 validation rows,
and entered fresh NLL training with finite improving losses. Initial64GB host-
memory submission was rejected before job creation;32GB fits the40.5GB virtual
node and its24GB A40. GPU16690103 runs the original refinement chain after
successful base training and labels. Array16690166_0–7 evaluates ten fixed
constant-shear cases per task after refinement and release of existing
emulator job16689747, with throttle2 preserving two GPUs total. CPU16690221
then requires all80 cases and replays sums/rows, paired uncertainty, regional
contributions and validation density. No automatic model promotion is enabled.

Experiment outputs, frozen runner, source/job receipts and stage states:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/`.
Scheduled read-only status jobs16690225/26 write local receipts around two
hours later and after the summary ends; no chat-wakeup capability is available.
Full repository test job16690222:326 passed,2 skipped in35.24s. Ruff, shell
syntax, import identity from outside the checkout, and whitespace checks pass.

Limitations: training/evaluation pending, one training stream, historical
full-feature control, reused development scenes, g1-only initial screen.
Changing input dimension also changes random initialization layout. C12 has
not started. Next: monitor scheduled stages for failures, inspect the fixed
80-case F8 comparison, and
only then extend to repeats/transfer/independent acceptance before promotion.

## 2026-09-24 — Define controlled simplification of flow, classifier and emulator features

Owner requested consistent, simpler inputs without performance degradation.
Added [`doc/V36_FEATURE_SIMPLIFICATION.md`](V36_FEATURE_SIMPLIFICATION.md#decision-and-scope)
after inspecting active crowding/forward-catalogue/rescaling code, the newer
close-crowding feature builder, and the original feature-gain study. Gaussian
formulae are shared but physical versus sheared-major-axis coordinates and
all-neighbour versus capped-pair support differ. Legacy noise-scaled shell sums
and maximum flux are not algebraically recoverable from three Gaussian sums.
The earlier gain proxy added features and used bounded relative weights;
it did not demonstrate that the deployed legacy features can be removed.

Specified separate F8 flow (remove three legacy summaries) and C12 classifier
(remove five annuli) retraining arms, then an interaction check if both pass.
Coordinate/support changes, network sharing and pair-emulator redesign remain
separate experiments. Defined source/schema consistency, unchanged measured
cuts and no truth-property selection, paired response equivalence, density and
usability checks, cancellation diagnostics and independent acceptance. The
suggested+/-0.05pp paired equivalence budget is a proposed criterion, not an
achieved result. Classifier margin and other tolerances must be frozen before
new fits. No model or inference behavior changed and no training was launched.

Validation: read-only source/protocol inspection; documentation link/anchor
and whitespace checks. Existing full-role/emulator jobs remain within the
two-GPU limit. Limitation: there is no reduced-feature calibration result yet.
Next: matched F8 and C12 experiments using half-shear development data and
unchanged downstream components, followed by a frozen independent evaluation.

## 2026-09-24 — Independent second review of the close-crowding +0.286% result

Owner requested another thorough, honest review. Added
`scripts/review_v36_closecrowd.py` and expanded
[`doc/V36_MODEL_COMPARISON.md`](V36_MODEL_COMPARISON.md#second-review-reproducibility-reuse-and-mixture-sensitivity).
The review consumes previously replayed row sums but imports no earlier
reducer: it independently calculates each selected leg mean and their response
ratio. CPU16689895 completed in5s; final16689907, adding hashed population and
checkpoint-selection receipts, completed in4s, exit0. Output:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/closecrowd_review_v2.json`.
Initialv1 output is retained. Ruff and `git diff --check` pass; numerical checks
verify the saved headline and aligned measured sums against the control.
No inference, model, cut, checkpoint or training behavior changed.

The independent calculation reproduces+0.285857927%, delete-one-case jackknife
SE0.266597pp. An additional20000-replicate paired bootstrap, fixed seed20260925,
gives SE0.263548pp,95%[-0.237071,+0.796948]%; this is consistent with the earlier
SE0.265609pp and does not replace it with a favorable bootstrap result.
Paired change vs unbounded reproduces-2.107048+/-.014780pp. Single deletions
span+0.231241 to+0.373206%, maximum shift0.087348pp; no scene was removed.

Verified exact scene reuse: the classifier trains on40–99 and tunes on100–119
using half-shear labels. All80 coherent-shear evaluation scenes are therefore
reused, although the labels/perturbations differ. The flow split includes62
evaluation scenes in training and18 in validation. Those subsets give
respectively+0.0709+/-.3022% and+1.0281+/-.5452% (jackknife); neither is an
untouched acceptance set or sufficient evidence of overfitting. Classifier
train/tune and first/last40 splits are also retained, with no headline selection.
Parent metadata confirms no truth cut and zero truth-cut removals across
27,982,713 target-role instances, including20,141,603 g0-unusable instances.
Three out-of-parent g0 diagnostic keys were dropped/reported; those diagnostic
masks do not select the science sample. Full-rendered-role scope remains open.

Exploratory symmetric target-weight changes retain every row and recompute
both data/model leg normalizations. Bright r<24 weights0.9/1.1 give
m+0.4125/+0.1681%; faint25.5<=r<27.5 weights0.9/1.1 give+0.1754/+0.3924%.
Paired changes have SE0.012–0.013pp. These expose mixture sensitivity, but do
not resimulate the neighbour field and are not physical-domain validation or
corrections. No weight is adopted. The report clarifies that the guard's
measured baseline normalization is a surrogate that equals own-normalized
response only under matching baseline moments; the mismatch is a hypothesis
to test, not an established explanation of the full residual. Confirmed the
baseline step1 passed its0.001 NLL gate (+0.0004431); the separate guard ep03
missed its0.005 gate (+0.0055), as openly recorded in the newer worktree log.

At the final check, both full-rendered-parent workers16689736_0/1 are advancing
without errors: unbounded has completed case40 and entered41; close-crowding
is processing40. The mag26.5 emulator swap16689747_2 remains dependency-queued,
with completion reducers and the existing two-hour status receipt unchanged.
No partial science result is reported. Limitations remain development reuse,
g1 only, one saved latent/model realization and large opposing magnitude
contributions. Next: finish the existing full-parent checks, resolve numerical
and component-response transfer, then validate the frozen pipeline on new
scene-disjoint data with both axes. Robust0.3% is still not established.

## 2026-09-24 — Evaluate newer models consistently; close-crowding gives +0.286% but cancels magnitude errors

Owner requested evaluation of the retained and newer models and a robust path
toward0.3%. Added `doc/V36_MODEL_COMPARISON.md` with estimator reconciliation,
model identities, same-scene comparisons, exact normalized magnitude-bin error
allocation, revised experiments and pending full-parent checks. Linked it as
superseding the earlier coverage-first plan. No checkpoint was trained,
selected by m, rescaled or promoted; cuts remain radius>0.60arcsec/MAG_AUTO<25.8.

`scripts/compare_v36_saved_models.py` reprocesses saved per-case counts and
numerators with each leg normalized independently, h=.02g1,10000 paired
case-bootstrap replicates seed20260924. CPU16689642 completed;16689691 adds
the mag26.5 flow's no-cut-parent endpoint. On identical80-case measured sums,
predicted-smooth-U m(%,case SE pp) is: mag26.5flow+2.950293+/-.272433;
unbounded+2.392905+/-.271384; flux/size ep08+3.011971+/-.273486;
8-input guard ep02+4.697425+/-.277806; close-crowding+0.285858+/-.265609;
close-crowding guard ep03+2.180535+/-.270284. The close-crowding paired shift
vs unbounded is-2.107048+/-.014785pp. Its high-radius/shell diagnostics give
+0.42137/-0.10202%, much less cut dependence than the old flow. Source
population is the generated target-role parent with no truth-property cut,
not all rendered roles. Legacy narrow-parent controls are kept separate.
All nine distinct reported flow/classifier checkpoint file hashes replay
against saved source identities (`saved_model_checkpoint_hash_replay.json`).

`scripts/audit_v36_saved_rows.py`, CPU16689678_0–2, independently replays240
row files for unbounded/close-crowding/guard. Original per-case evaluator sums
match for measured data, predictedU and actual per-legU. The newer diagnostic
reducer used both-leg model U against per-leg data U and unnormalized numerator
slopes. Its+0.927667% is reproduced exactly; normalize with the same inconsistent
U and it becomes+0.121774%; correct to per-leg actualU gives+0.371344%; use
trained per-leg probabilities and it becomes+0.285858%. This corrects the
statistic, not the model with an empirical factor. Exact normalized regional
contributions for close-crowding are-1.198918,-0.066983,+1.459087,+0.092671pp
for r<24,24–25.5,25.5–27.5,>=27.5; all bins retained. Opposing errors remain
with actualU. Guard improves bright contributions but worsens the intermediate
region and leaves the faint residual. Historical shape/selection budgets and
native fixed-g0-lookup checks cannot automatically validate the deployed disk
emulator. All saved comparisons use disk artifact194c1d11b67d, despite newer
mag26.5 emulator existence.

Extended `scripts/audit_v36_domain_response.py` with explicit hashed candidate
flow/classifier and optional disk model, 11-input physical-distance crowding
reconstruction, and replay against cached pair identities/features. New GPU
launcher `jobs/job_audit_v36_new_models.sh`, frozen candidate JSONs, successful-
completion reducers, and a two-hour one-shot status check preserve two GPUs
total. Pending16689655 was cancelled for available24GB allocations.16689690
failed before prediction because transformed context width was mistaken for
raw feature count; fixed via preprocessor feature names. Partial16689706 was
cancelled after finding the earlier full-role pilot's k=21 query admits20
nonself neighbours, whereas saved80 uses k=20 including self (max19). The
final full-parent runs use explicit k=20 and cached-pair replay. Earlier full-
role pilots therefore also differ in neighbour aperture and must not be used
as clean population-only contrasts with the saved80. Source snapshots and
partial failure receipts remain preserved, with no partial bias reported.

Array16689736_0/1 now evaluates unbounded/close-crowding on both full rendered
case40/41 scenes,64 FP64 CRN draws. Both pass measured and context replay,
2102 cached pairs for128 probe parents; close-crowding reproduces all349784
cached three-feature extensions. Both have meaningful forward progress.
Dependent16689747_2 swaps only to mag26.5 disk modeld2dc49113ae0 on the same
close-crowding cases/draws. Reducers16689756/57 run on successful completion;
one-shot status/summary snapshot16689796 is scheduled about two hours later.
No automation tool is available to wake the agent; that check writes a local
receipt and scheduler log rather than claiming a future conversational update.
Fresh full-parent/emulator comparisons are pending, not established results.

Added `scripts/plot_v36_model_comparison.py`; CPU16689719 generated and visually
checked `doc/figures/v36_model_comparison_20260924.{png,pdf}`. Matplotlib skill
informed the standalone figure. Tests added to `test_v36_saved_models.py` and
`test_v36_domain_response.py` cover separate leg normalization, spurious raw-
mass bias, exact additive normalized response, and physical neighbour kernels.
Full suite16689712:322 passed,2 skipped in29.32s before final pair/optional-model
extension; final focused job16689797:14 passed in9.61s. Ruff, launcher syntax,
documentation links and `git diff --check` pass. All heavy work uses Slurm.

Outputs are under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/`:
`all_saved_models_v2.json`, `rows_*.json`, candidate hashes, fresh worker
snapshots and scheduled summaries. Limitations: reused development scenes,
g1 only, one latent/training stream; close-crowding95% interval[-.246,.800]%
does not bound bias at0.3%. Next: finish full-parent/emulator checks; verify
hard-cut own-normalized training response and joint near-boundary density;
use matched self/neighbour/coherent image perturbations to identify remaining
faint response; freeze and validate both axes on independent scenes. No
robustly recovered0.3% or end-to-end likelihood acceptance is claimed.

## 2026-09-24 — Correct recovery review scope after finding newer retrains

Owner asked whether the training-cut statement referred to retained V3.6-like
or newer models. The earlier review missed the September22–24 history in
`.claude/worktrees/nocut-retrain/doc/WORKLOG.md`. Read-only cache inventory,
completion/protocol/domain metadata and launchers confirm distinct unbounded
flowda73a8db1d36, classifier13cd6a81ecff and close-crowding flow1433b7217fb7.
New classifier has no truth cut; new flow r<30 bound retains the full generated
population according to the worklog (maximum below29). Close-crowding flow
adds three features. The earlier two-scene audit used the original pinned
flow5434cf/classifierad975e/emulator194c1d, not these newer models.

Added explicit correction at the top of `doc/V36_RECOVERY_PLAN.md` and
`doc/V36_DOMAIN_RECOVERY.md`; coverage-first recommendations are incomplete
for newer models. Newer logs report +2.393 +/-0.270% modeled-smooth unbounded;
close-crowding and guard decomposition JSONs give+0.927667 +/-0.260796% and
+2.739784 +/-0.265277%. Source inspection of the latter reducer finds both-leg
model usability versus per-leg simulation usability, so these numbers are not
asserted to be matched comparisons of the requested estimator. Its launcher
also supplies both old physical-disk response candidate and separate fixed-g0
lookup tables; emulator claims must be branch-specific. No new inference,
training, data replay or numerical result was produced. `git diff --check`
passes. Next: reconcile latest checkpoint/branch/population identities and
estimator definitions before updating the recovery strategy.

## 2026-09-24 — Prioritize full-domain coverage and honest recovery validation

Owner fixed the target to constant-shear simulations after measured radius
>0.60 arcsec and MAG_AUTO<25.8, without truth-property cuts, and requested a
deeper recovery strategy. Added `doc/V36_RECOVERY_PLAN.md`: fixed target,
training-label coverage, exact pilot error allocation, own-normalized response
objective, controlled coverage/objective comparisons, physical diagnosis,
scene-disjoint validation and numerical precision. Experimental-design guidance
informed the staged comparison. No training, GPU job, model change, empirical
correction, production inference or archived validation restart was launched.

Corrected `doc/V36_DOMAIN_RECOVERY.md`: the retained smooth classifier was
**also trained on strict true r<26**, contrary to the prior broader-coverage
statement. Flow refinement pins domain hash1ab650f03ae2; classifier completion
pins protocolb1ba4c56a04a and retained checkpointad975edf7020; pinned emulator
metadatae9c28e921fa0 declares true-r<26 and h=0.2 labels. Protocol/metadata and
prepared-manifest hash checks pass. Initial overbroad source-pin verification
encountered an archived launcher at its historical path; the final receipt
explicitly scopes verification to metadata lineage and prepared manifests,
not all historical source files or checkpoint/training-record bytes.
`training_coverage_receipt.json` preserves that scope and evidence.

Added `scripts/decompose_v36_population_error.py` and
`tests/test_v36_population_error.py`. CPU Slurm16689396 completed in25s:
`python -m pytest tests/test_v36_population_error.py -q` gave3 passed in10.01s;
then `python scripts/decompose_v36_population_error.py --summary
<cache>/forward_summary.json --output <cache>/population_error_allocation.json`
verified frozen source hashes, bright+faint sums and exact finite-step identities.
Predicted-U contributions to pilot m11: bright-0.106187pp, faint+0.696606pp,
selected composition+0.672686pp, migration+0.000861pp, total+1.263967%.
Actual-U total+0.990386%, with faint+0.651178pp and composition+0.283604pp.
Identity residuals are below1.3e-16. Tests cover composition-only error,
selection migration, data/model exchange symmetry and invalid mass/step.
`ruff check` on both files, `python -m py_compile` and `git diff --check` pass.

Cache is
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/`.
Limitations: two reused scenes; algebraic allocation is not causal component
identification or a population error bound. All three models lack faint target
training support, but filling coverage is not proven sufficient. Next: harmonized
full-parent label audit in BlendEMU, coverage-controlled retraining, then the
corrected objective with mass/density checks and fresh independent acceptance.
The queue is empty. No recovered0.3% result is claimed.

## 2026-09-24 — Diagnose domain transfer and expose measured-radius cancellation

Owner requested a deep investigation of recovering approximately0.3% V3.6-like
bias after a domain change. New `doc/V36_DOMAIN_RECOVERY.md` records the
working interpretation (September20–21 uncut inference parent), the distinct
measured-cut transfer test, exact response decomposition and recovery experiment.
Linked it from the retained model and inference review. The scientific-critical-
thinking and resource-check skills informed the design and bounded allocation.
No archived training/validation launcher, model tuning, empirical correction,
production inference or changed production cut was started.

Added `scripts/audit_v36_cut_transfer.py` and its finite-step migration identity
test. Reanalysed all four hash-verified retained endpoint files using paired
case bootstrap (10000 replicates, seed20260924). Main80 radius0.60–0.75 shell
has m=-6.465850 +/-1.306843%, versus+1.318989% at radius>=0.75; their union
has-0.199979%. Paired threshold shift is+1.518969 +/-0.247439pp; existing40
shift is+1.065663 +/-0.187459pp. Actual per-leg U retains essentially the same
pattern. Exact selected-fraction/migration decomposition shows opposing main80
R11 residual contributions-0.007441699 and+0.006071784. This is measured-radius
cancellation, not a truth/measurement component decomposition. Job16688906
completed;16688896 first failed only a zero-tolerance floating-point test
(3.5e-16), corrected without changing the analysis. A frozen latent-seed7302
repeat reanalysis and final suite are scheduled as16689063.

Job16689063 subsequently completed:315 tests passed,2 skipped in29.70s.
Frozen latent7302 retains main80 shell/high/union m=-6.559798/+1.333763/
-0.209397% and existing40 -3.938095/+1.466081/+0.391591%. Added the concrete
unfinished own-normalization objective correction from the historical audit:
differentiate both A and D in `[H-(A/D) outer B]/D`, with a separate probability
constraint. The old stable-gradient result (cosine0.9994166, zero updates)
used predecessor hashfade8aeb…, not V3.6-like5434cf…, so it cannot be reused
as a validated current-model direction. Both final pilot workers pass measured
and128-row context replay and show forward progress; reducer16689064 waits
for both to finish. No new training or model recovery is claimed.

Added `scripts/audit_v36_domain_response.py`, measured and GPU launchers,
`scripts/summarize_v36_domain_response.py`, and focused tests. CPU16688812
completed in4m22s after313 tests passed/2 skipped in43.08s. Across80 cases,
all original measured sufficient sums replay exactly. R11 changes from
0.685025402 on8,876,218 retained parents to0.687183145 on all17,755,228 bright
parents and0.672734807 on all55,965,430 rendered parents. Faint-only response
is0.554582307; these are measured responses, not expanded-population biases.
CPU summary16688980 completed. Unique joins, unmatched counts, same intrinsic
truth and membership across legs, hashes and per-case sums are persisted.

Initial GPU16688813_0 and receipt retry16688904_0 stopped on auxiliary
underflow:14/65536 draws, three faint objects, no nonfinite output and none
selected. Revised forward diagnostic computes zero contribution for draws
provably outside a finite measured bound while keeping the original draw
denominator; it never repairs or renormalizes them. A new test verifies that
mass accounting. Transport only selected draws is exact because it preserves
radius/flux. Any other invalid output aborts; disk roundoff is counted.
Partial16688944_0 was cancelled after1m47s to add preflight retained-context
replay and freeze matching source versions. Its incomplete output is not used.
Final case40/41 pilot jobs are16688994_0 and16688944_1, at most two GPUs total,
with dependent CPU reducer16689064. Both request oneA40/fourCPUs/48GiB;
resource discovery reports one visible GPU and process CPU affinity is bounded
to the allocation. Five focused tests pass in each final worker. Startup and
first progress are monitored. Pilot and checks completion status is updated
below when available.

Outputs live under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/`.
Ruff, compilation, launcher syntax and `git diff --check` pass. Limitations:
all response endpoints are development evidence; two-case uncut pilot cannot
establish0.3% or finite-draw convergence; moment checks do not resolve finite-
prior/derivative/normalization/inference failures. Next: evaluate uncut transfer,
localize conditional flow versus native disk-response error on matched regional
populations, then fit physically supervised regional response/selection/density
constraints on designated training data and validate on disjoint scenes/cuts.
No recovered calibration is claimed.

Final pilot completion:16688994_0/16688944_1 completed in14m40s/15m07s;
dependent reducer16689064 completed in12s. All measured/context replays pass
(context maximum<2.37e-7), with identical archived source hashes for both
workers. Pooled predicted-U m is+0.931886% retained,-0.116593% all-bright,
+1.263967% all-rendered,
and+7.489339% faint-only. Actual-U faint m remains+7.496259%. Faint selected
counts are overpredicted28.61% (plus) or12.10% with actual U; all-bright counts
agree within0.1%. The paired retained→uncut shift is+0.332081pp, not the
larger difference obtained by improperly comparing this two-case pilot to
the80-case headline. Two cases cannot supply a population calibration bound.
All179,089,280 draws remain in the integration denominator;474,893 numerical
auxiliary failures are provably outside measured selection. No non-excluded
failure or shape excursion beyond1e-12 occurs; boundary counts are retained.
Final findings and limitations are in the report's
[completed pilot](V36_DOMAIN_RECOVERY.md#completed-uncut-pilot).
All submitted work is finished, the queue is empty, and no training,
production launch or recovered0.3% result is claimed.

## 2026-09-21 — Consolidate completed plot diagnostics and current stopping point

Updated this worklog and `doc/V36_INFERENCE_REVIEW.md` after delivering the four
matched-truth corner plots. Removed stale pending-output/clarification wording
from the failed prior-mixture attempt, preserving its failure and provenance.
Added the completed matched-truth result to the review's opening decision and
linked the final figures and machine-readable report from its results section.

Current conclusion: four selected problem cases at MAG_AUTO 17.6253, 18.4570,
19.4945 and 20.5649 show no extreme individual marginal discrepancy when
conditioned on their own simulation truth/neighbours at actual shear (0.02, 0).
This is not a representative validation sample, a four-dimensional joint
calibration test, or a test of shear derivatives. The older prior-mixture slices
answer a different question and do not directly diagnose the own-truth flow.
No inference cuts, model artifacts, prior weights or production settings changed.
The 100k production launch remains withheld; no new computation was submitted
for this documentation update. Next scientific work remains the unresolved
prior/proposal, derivative, normalization and model-coverage checks described in
the [review's next decision](V36_INFERENCE_REVIEW.md#limitations-and-next-decision).

Validation: read the final JSON report and checked all four row identities,
draw counts and marginal CDFs against the recorded results; all eight final
PNG/PDF files exist. `git diff --check` passed. Documentation only; no tests
rerun (the generation job's five passing tests remain recorded below).

## 2026-09-21 — Own-truth conditional corners across MAG_AUTO17–21

Owner clarified that the requested plot conditions on each galaxy's actual
simulation truth, not a diagnostic prior-atom mixture. Added
`scripts/plot_disk_truth_corners.py`, `jobs/job_plot_disk_truth_corners.sh`, and
`tests/test_disk_truth_corners.py`. Frozen rows239507/440648/274111/10642 have
MAG_AUTO17.6253/18.4570/19.4945/20.5649: nearest each one-magnitude-bin midpoint
among the existing worst50 approximate-production curvature contributors,
chosen before evaluating predictions. They are illustrative development cases.

The script verifies source hashes, frozen row identities, truth/detection joins,
actual measurements, and intrinsic morphology against generated catalogues.
It builds each primary's eight flow features from its rendered full scene,
explicitly converts degree angles, applies actual shear(.02,0) once to intrinsic
shape, samples the retained flow in FP64, and composes the output-conditioned
disk response using the nearest20 nonself neighbours within10arcsec. Crowding
uses all rendered neighbours within7arcsec, without truth cuts. At fixed truth
the usable-event classifier is a scalar and cancels from the law conditional
on usability; no prior mixture or measured analysis cuts are used here.

262144 draws per object yield 1D marginal histograms and 2D histogram-based
68/95% contours. No KDE, model tuning or clipping. Invalid numerical draws are
counted explicitly before/after response and excluded from the displayed valid
sample law; their frequency is retained on each figure and in JSON. Samples,
contexts, pair IDs/distances, source hashes and marginal CDFs are saved.
Matplotlib/resource-check skills informed layout and the oneA40/fourCPU/64GiB
scheduled allocation; user queue was empty before submission. Ruff, compilation,
Bash syntax and `git diff --check` passed. Job16622674 completed in1m16s,
with five focused tests passing in22.02s; model loading plus generation/plotting
took31.8s. All source/identity/morphology/measurement checks passed, with no
selected-object match dropped. Source draws/report live at
`doc/figures/v36_truth_corners_16622674/`. One of262144 draws for row239507
failed the pre-transport open-disk check; all other draws were valid, and no
post-transport failures occurred. No radius/flux invalidity occurred here.
The four observed marginal-CDF ranges are35.5–88.2%,34.5–45.6%,22.5–58.0%,
and29.0–65.6%. The plots do not show conspicuous marginal outliers at actual
truth, but do not establish joint calibration, population coverage or shear
derivative accuracy. See [matched-truth results](V36_INFERENCE_REVIEW.md#matched-truth-conditional-corner-plots--2026-09-21).

Visual inspection found legend/x-label crowding. Added a hash-verified
`--render-from` mode and adjusted only figure spacing; CPU job16622696 completed
in29s, re-rendering the exact saved samples into
`doc/figures/v36_truth_corners_20260921_final/`. Final visual inspection passed.
No production code or model was changed; the earlier mixture job stays paused.

## 2026-09-21 — Marginal corner plots for fainter problematic observations

Owner requested triangular contours with diagonal marginal histograms for
fainter failures. Added `scripts/plot_disk_faint_corners.py`, the corresponding
one-GPU launcher, and `tests/test_disk_faint_corners.py`. Chosen previously
audited rows412689/120835/86325 have MAG_AUTO19.4465/20.1248/20.5881: the worst
production-curvature row in19–19.5 and20–20.3, plus the faintest of the exact
worst32. This is a selected diagnostic panel, not a representative holdout.

For each row, retain the union of the nine-node top32 atoms from completed
exact24m references. Draw8192 physical samples per atom at the current center
and center+(.001,0), with atom-identity-seeded common random numbers. Recompute
output-conditioned disk transport for every sample. Use original equal-prior
times cached usability weights, apply the unchanged measured radius>0.6arcsec
and MAG_AUTO<25.8 cuts, then normalize within the retained atom subset. No
posterior atom reweighting, KDE, smoothing, model tuning or new cuts. This
produces genuine finite-sample1D/2D marginals, unlike the earlier density slices,
but NOT the full24m population predictive law. The top32 likelihood mass at the
observation is reported separately and is not integrated predictive coverage.

The Matplotlib skill informed the corner layout; the resource-check skill
informed oneA40/fourCPU/64GiB allocation, job16622565, with an empty user queue
before submission. Three focused tests cover mixture-selection weighting,
contour mass thresholds and plotting units; the two previous plotting tests
also run before generation. Ruff, Python compilation, Bash syntax and
`git diff --check` passed. The intended output directory was
`doc/figures/v36_faint_corners_16622565/`; no completed plot products were produced.
Job16622565 passed all five tests in24.23s, then failed on a forward draw with
non-finite or non-positive radius/flux before response evaluation. No corner
plots were completed. Invalid draws were not clipped or repaired. Their
frequency and originating atom have not yet been measured. The owner then
clarified whether these predictions were conditional on each observation's
actual truth properties. They were not: both this attempted follow-up and
the earlier slice plots use data-chosen prior-atom mixtures. This mixture work
was paused and superseded by the completed
[own-truth diagnostic](#2026-09-21--own-truth-conditional-corners-across-mag_auto1721),
using actual simulation truth and neighbour context, not those prior atoms.
No core implementation, pinned artifact, inference cache or production job
changed. The100k launch remains withheld.

## 2026-09-21 — Plot joint-density slices for problematic observations

Owner requested joint likelihood plots with observed properties overlaid.
Added `scripts/plot_disk_joint_likelihood.py`,
`jobs/job_plot_disk_joint_likelihood.sh`, and two focused tests in
`tests/test_disk_joint_likelihood_plot.py`. The diagnostic uses previously
inspected bright rows142230/3563 and their fixed node-wise top32 atom unions.
Six pairwise slices hold the two other outputs at the observation; these are
not pairwise marginals or credible regions. Original uniform-prior/usability
weights and full-prior cached selection normalization are retained. Disk
response is recomputed for each radius/flux output cell. Neural evaluation,
input shear and preprocessing are FP64; classifier probabilities are the
cached FP32 values. Contours compare the existing center and center+(.001,0).
No posterior reweighting, KDE, population cut change, or model tuning is used.

The Matplotlib skill informed the OO multi-panel layout and shared sequential
color scale. The resource skill informed a one-A40, four-CPU,64GiB Slurm job
16621509, within the two-GPU combined limit; queue was empty before submission.
Whole-node resource recommendations do not supersede the allocated resources.
Python compilation, Bash syntax and `git diff --check` passed before launch.
Initial job16621488 stopped in the new test on a2.8e-17 grid-center rounding
difference. Explicitly set each production grid center to the observed value
and use an appropriate floating tolerance for the synthetic linspace test.
Immediate retry16621500 apparently reused stale shared-filesystem bytecode
(traceback displayed the new allclose call but executed the old exact-array
assertion). A job-local bytecode cache in16621509 avoids that reuse; both
focused tests then passed in15.93s. Job16621509 completed in3m25s. Output goes to
`doc/figures/v36_joint_likelihood_16621509/`, including raw grids and provenance.
All six panel centers replay the direct observed-point density within1e-9.
Visual inspection found very narrow e1 peaks at the broad display scale and
an overlapping footer. Added a fixed local-window view and a rendering-only
mode; job16621540 re-evaluates finer local grids and redraws the saved broad
grids without changing any density. Its two focused tests passed in13.64s;
the full job completed in3m07s. The local half-widths are .015/.01 in shape,
.1arcsec in radius and15% of observed flux; these are display settings only.
Details: [joint density plots](V36_INFERENCE_REVIEW.md#joint-measured-property-density-plots--2026-09-21).
Matplotlib's outside figure legend still overlapped the explanatory footer;
changed it to an explicit figure-coordinate position. CPU-only job16621572
redraws both sets from their saved grids into
`doc/figures/v36_joint_likelihood_20260921/`, preserving the original arrays
and provenance; completed in32s. Reserved more bottom margin after inspection
to separate the legend from axis labels, with final CPU rendering job16621622
writing `doc/figures/v36_joint_likelihood_20260921_final/`. These layout-only
jobs do not load models or recompute densities. Final job16621622 completed
in32s; visual inspection confirms separated labels/legend/footer. Tests passed2/2; Ruff, Python
compilation, Bash syntax and whitespace checks pass. The plot lesson is a
narrow e1 ridge with the observation on its steep right flank in both examples;
radius/flux slices are multi-ridged. This is descriptive, not model rejection.
The diagnostic unions capture93.69%/71.69% at the center observation; captured
mass elsewhere on the plotted surfaces is unknown. No full-prior surface or
physical adequacy claim follows. No production code/cache or100k job changed.

## 2026-09-21 — Diagnose likelihood failure separately from sampling

Owner asked to look deeply into possible likelihood failure. Added diagnostic
scripts `probe_disk_likelihood_failure.py`,
`summarize_disk_likelihood_failure.py`, and
`probe_disk_derivative_reference.py`, their two GPU launchers, and
`tests/test_disk_likelihood_failure_probe.py`. No core `sbsi/` implementation,
model, production cache, population, or estimator was changed.

Protocol: two previously inspected failures (rows142230/3563) and original
rows0/1, not the reserved numerical holdouts. Sum all24m atoms at the existing
nine shear nodes, measure posterior effective atom count, and compare four
disjoint index-mod4 banks and their two12m unions. Bank comparisons concern
the numerator only: subset-specific selection normalization is not computed,
and these partitions are not independently generated priors. On each fixed
union of node-wise top32 atoms, recompute genuine classifier features and
probabilities at h=.001,.0005,.00025,.000125; compare compiled/eager density,
FP64 neural evaluation, and freezing each of probability/context/transport
at its center value. Report original-node captured mass. These truncated
numerators and deliberately frozen-component arms are not production models
or full-prior derivative validation.

A separate autodiff reference uses FP64 neural evaluation AND input shear/
preprocessing; the production helper retains FP32 raw-shape arithmetic even
when neural weights are promoted. The reference freezes the classifier and
compares finite differences down to h=.00003125 against autograd derivatives.
The reducer separates posterior-averaged component information from the
covariance of component scores, reporting the finite-step identity residual.

GPU array16621068 completed in11m21s/10m39s, using two A40s total and four
CPUs/64GiB per worker. Resource detection confirms one visible A40 per worker;
scheduler allocation, not whole-node recommendations, bounded the work.
One-GPU derivative job16621104 then completed in36s, with all five new tests
passing. CPU summary/full-suite job16621083 stalled in uninterruptible I/O
before Python started on cip-cl-compute3 (process inspection: shell child,
zero CPU, about1MiB RSS). Cancelled only that job, preserving logs. A first
replacement submission specifying inter,cip plus cip-cl-nv01 was rejected
without creating a job; CPU-only16621160 on partition cip/node cip-cl-nv01
completed in43s as the replacement. The full suite passed **302 tests,
2 skipped in27.37s**, followed by successful reduction. No100k production job
is submitted. Ruff, Python compilation, Bash syntax and `git diff --check`
pass. Reports are in the existing run's `likelihood_probe_16621068/summary.json`
and `derivative_reference_16621104.json`; all scientific GPU jobs are complete.

All-atom results: bright rows142230/3563 have6.29/16.58 effective posterior
atoms at the center, versus11636.98/292088.13 for rows0/1. The12m parity halves
give g1 numerator scores19.16 versus1314.33 and-5.96 versus1534.90. Existing
worst32 records have median88.88% single-atom mass at+g1, with32 different
dominant identities. This establishes serious atom-bank sensitivity in the
selected failures, not population-level prior convergence or model bias.

Full-FP64/autodiff on the dominant unions, with center-frozen classifier,
retains I11=-1.521m/-3.232m. Freezing flow context reduces I11 to2.60/-5.64;
freezing disk transport leaves the large effect. At h=.001, g1 scores
957.16/1520.06 differ substantially from autodiff390.91/766.12. Thus sampling,
prior resolution and finite-step errors are separate observed problems;
negative curvature on these unions is not merely FP32 cancellation. The
unions capture93.69%/71.69% of bright center mass (only3.27%/.57% for rows0/1),
so no full-prior fine-step or classifier-normalized derivative claim follows.
No normalizer was extrapolated to new nodes. Findings and table are in
[likelihood-focused diagnostics](V36_INFERENCE_REVIEW.md#likelihood-focused-diagnostics--2026-09-21).

The finite-step mixture decomposition at h=.000125 gives positive mean
component I11 (~61545/~117368) but much larger component-score variances
(~1.582m/~3.349m), accounting for the negative mixture I11. The reported
finite-step residuals are nonzero and preserved, not corrected away. This
supports rapidly changing atom responsibility as the dominant g1 mechanism
on these unions. It does not imply every individual component is concave;
some also have negative information in the other direction.

The scientific-critical-thinking skill motivated separating model adequacy,
finite-prior resolution, derivative numerics, and sampler error. The resource
skill motivated the bounded allocation and dependency. Existing moment
calibration and improved NLL are not full-density/shear-derivative validation.
Next: establish converged prior/derivative integration before attributing
failed shear recovery to physical model mismatch; separately validate bright
conditional density/response against images. No archived model training or
cancelled independent validation chain was restarted. This investigation did
not establish whether the flow is physically too sharp or correctly sharp
but inadequately integrated by this prior.

## 2026-09-21 — Review end-to-end inference and gate the requested100k run

Owner requested a full likelihood/prior/sampling review, caution about empirical
tuning, and two-GPU100k inference only when robust. Added
[review and launch gates](V36_INFERENCE_REVIEW.md#decision--2026-09-21), covering
disk change of variables, full-prior measured selection, uniform finite-prior
construction, exact-plus-complement sampling, finite-log bias, state-dependent
classifier geometry, derivative precision, solver convergence and uncertainty.
Linked it from `doc/INFERENCE.md`. **Production launch is withheld.**

The inverse Jacobian/auxiliary-marginal composition and stratified reduction
are consistent on inspection and in existing regression tests. Important
unresolved issues remain: original Gaussian retrieval has demonstrated support
failure; exact bright-object curvature can still be negative;24m prior
resolution and source-generator/runtime equivalence are not established;
selection64-QMC/h=.001 convergence has not been established. The17-input
classifier contains five hard annulus bins in addition to its four smooth
crowding features, so global smoothness must not be assumed. The normalizer
called `ExactPopulationNormalization` is atom-complete but still uses a
sampled flow integral. The standardized-output density omits a fixed physical
unit determinant that cancels in same-model shear derivatives, not in arbitrary
absolute density comparisons. No implementation/cached-science change was made.

Added `scripts/audit_disk_population.py`,
`tests/test_disk_population_audit.py`, and CPU launcher
`jobs/job_audit_disk_population.sh`. Job16617937 completed in2m14s after the
full suite passed **297 tests,2 skipped in41.31s**. Resource receipt shows no
GPU; allocation was two CPUs/12GiB, with BLAS/OMP limited to one thread.
The script checks frozen input/source hashes, rejects ambiguous or unmatched
truth joins, scans all20 prior shards and reserves rows without evaluating
new likelihoods. Output is under the existing run's
`population_review_16617937/report.json`.

Actual frozen500k support audit:54231 objects (10.8462%) have true-r>=26;
all source matches succeed. Prior:16387409/24000000 (68.2809%) atoms have
true-r>=26 and account for approximately47.14% of cached usability-weighted
mass **before measured cuts**. This is not selected/posterior mass or proof
of bias. It documents a material gap from the retained true-r<26 development
evidence; no independent model acceptance exists for this uncut population.
Prior cases20000–20199 and observed cases40–139 are disjoint.

Reserved a64-object numerical holdout (seed20260922),32 uniform plus8 per
fixed magnitude stratum, excluding first32 and all50 inspected worst rows.
It is held out only from proposal development, not independent image/model
validation. Separately reserved a uniform100000-row production sample
(seed20260923), with **no tail/development exclusions**, for two50000-row
workers after acceptance;10944 of these rows have true-r>=26. Array hash and
membership are persisted. No production job or automatic production dependency
was submitted. Subsequent proposal choices must be frozen before inspecting
holdout likelihoods; further tuning retires those holdouts as validation.

Current GPU array16617850 remains the only GPU allocation (two A40s total).
At review handoff each worker has completed three exact/comparison objects
and is progressing. It and reducer16617867 remain development diagnostics,
not launch gates by themselves. Scientific-critical-thinking guidance motivated
the development/acceptance separation and prospective numerical budget; the
resource skill confirmed allocation limits. No models, response scales,
empirical shear corrections, cuts, prior weights or production caches changed.

Validation: Bash syntax, Python compilation, system Ruff and `git diff --check`
pass. The py31 environment lacks the Ruff module; `/usr/bin/ruff` is used
instead. Next: finish the existing development-tail benchmark; separately
resolve the uncut support/prior, normalization, derivative and solver gates.
If resolution requires retraining, a changed population or restarting archived
independent validation, obtain an explicit owner decision. Task-wakeup tools
are unavailable; the existing Slurm reducer runs without waking the assistant.

## 2026-09-21 — Start exact tail validation with two persistent workers

The second all-atom benchmark16617754 completed in5m13s. Row3563 remains
non-concave under exact integration: information
`[[-2454379.98,267716.02],[267716.02,-251436.95]]`. Thus a successful proposal
repair must not be confused with proving positive full-catalogue curvature at
the current expansion point.

Student(df3) union probe16617759 completed in4m09s; Cauchy(df1) union16617785
completed in4m28s. Both preserve the original1024 exact atoms and the original
defensive complement sampler. Cauchy ranking moves the previously missed
33%/18%-mass atoms to ranks58/343. AtK65536/M16384 the worst object's g1 score
error falls from1116.814 to41.388 versus exact; the second object's error falls
from72.157 to9.829. This is a useful improvement, **not adequate evidence of
integration convergence**. No proposal variant is promoted to production.

Refactored the unchanged all-atom reduction into `exact_observation` in
`scripts/probe_disk_curvature.py` and added a blockwise-normalization regression.
Added `scripts/validate_disk_tail_integration.py`,
`jobs/job_validate_disk_tail.sh`, and `scripts/summarize_disk_tail_validation.py`.
The new panel covers the32 most negative audited observations, reuses the two
completed exact references, and compares Cauchy-union K65536/K262144 with16k
complement draws against full24m-atom sums. This is a selected failure-tail
benchmark, not an unbiased population subsample. The reducer reports errors
and an explicitly diagnostic partial replacement of32 contributions; it never
reports a shear estimate or covariance.

Array16617850 has exactly **two persistent one-A40 workers**,16 objects each,
with model/prior allocations retained and exact/comparison checkpoints saved
after each object. CPU reducer16617867 depends on array success. Allow roughly
one hour plus queue time, based on the measured exact-scan rate. The initial
startup16617841 was cancelled at56s before panel evaluation after identifying
the generic prefilter-size guard forK262144. The wrapper now declares its
original131072-location shortlist separately from the larger whole-prior union;
fresh output roots and freshly loaded code are used for16617850. No old output
was deleted or overwritten.

Both16617850 startup gates pass6 tests (10.99s/10.89s), including independent
SciPy checks of Student(df1/df3) scores, union uniqueness, and exact summation
across atom blocks. Resource receipts show two full46GB A40 allocations with
four CPUs/64GiB host memory each. Ruff, Bash syntax and `git diff --check` pass.
No other SBSI GPU work is active. Both workers have completed their first
comparison and exceeded one million atoms on their next exact observation.
AtK262144 the two existing benchmarks' g1 score errors are6.1435/3.6575,
versus1116.8140/72.1568 in production. The corresponding first-object
information errors are `[[-9803.50,428.99],[428.99,-7548.77]]`; this is a
substantial reduction but does not establish catalogue-level numerical tolerance.
Task-wakeup/automation tools remain unavailable; the Slurm reducer will run
automatically but does not wake the assistant.

Next: use the exact panel to decide whether the integration proposal is accurate
enough for a targeted recompute, and separately determine whether a safeguarded
recentered solve is required. Models, cuts, prior masses, normalizer and original
production moments are unchanged; the final positivity requirement stays intact.

## 2026-09-21 — Locate proxy support failure and test auxiliary retrieval

Whole-proxy probe16617724 completed in4m19s wall time. Increasing the same
Gaussian-proxy exact stratum toK131072 still substantially disagrees with the
all-atom result for row142230. The centre's two leading atoms carry32.90% and
18.12% of exact numerator mass but have proxy ranks961486 and1648131. Neither
is shortlisted; each has proposal probability about4.17e-9, hence only6.83e-5
expected occurrences in16k draws. This explains why even64k nested draws and
two seeds can look stable while missing most of the centre's mass.

Added `scripts/probe_disk_auxiliary.py`, its scheduled launcher and
`tests/test_disk_auxiliary_probe.py`. The diagnostic unions the unchanged
original1024 candidates with radius/flux-only Gaussian-proxy candidates, using
total exact budgets4096/16384/65536. Complement sampling and weights are
unchanged; union membership is unique and preserves every original candidate.
This is a proposed integration remedy, not a new likelihood or population cut.
Job16617745 completed in4m06s wall time; its startup gate passed both union tests
(2 passed in9.23s). Radius/flux-only Gaussian ranking also fails to recover the
dominant atoms (ranks850217 and1651496), so that proposed remedy is rejected.
Added an optional four-output Student-t(df3) ranking to the diagnostic union,
leaving the actual flow density and complement proposal unchanged. Its score
ranking is checked against SciPy's independent Student density. Dependent
job16617759 is running; startup gate3 passed in9.96s. The likelihood's physical
scales and models are not modified by this proposal-only tail choice.

The exact probe now accepts an explicit row from the audited worst-object list.
Job16617754 is checking row3563, the second-worst production row, against every
prior atom as a separate numerical benchmark. Together with16617759 this
preserves the two-GPU cap. Both proposals and benchmarks keep all observations
and the same finite prior; no full rerun has been launched.

Scheduled regression16617733:40 passed in22.50s across the existing sampler,
disk likelihood/cache/runtime-compatibility tests and new stencil regression.
Ruff, shell syntax and whitespace checks pass. The original500k moments remain
unchanged. A finite-prior/non-concavity problem also remains possible after
fixing integration; an exact benchmark, not positivity alone, gates the remedy.

## 2026-09-21 — Exact bright-object probe confirms curvature and integration error

Probe array 16617682 completed: exact arm 5m49s wall time, sampled arm 3m46s.
CPU summary 16617708 completed in 21 seconds. Added the hash-checked reducer
`scripts/summarize_disk_curvature_probe.py`; its report is
`curvature_probe_summary_v1.json` under the existing V3.6 run root.

For the worst production observation (row142230, measured MAG_AUTO17.3015),
exact summation over all24million atoms gives score `(926.5168,-472.9388)` and
information `[[-1686130.55,500735.41],[500735.41,-879639.81]]`, versus production
score `(2043.3309,-1210.4676)` and information
`[[-2707895.46,452212.67],[452212.67,-2134515.99]]`. Thus **both** genuine
non-concavity at the initial centre and material finite-draw integration error
are present for this object. The exact numerator's leading atom changes across
the stencil; atom9262565 goes from8.57% posterior mass at the centre to84.59%
at the positive-g1 node and97.44% at the positive-g1/negative-g2 corner.

The32 worst objects retain a large negative curvature contribution at16k,
32k and64k draws under both seeds8701/8702. More draws alone do not resolve the
problem. This targeted selection is diagnostic, not a population estimator.
No objects were removed and no correction/offset was fitted to injected shear.

Added `scripts/probe_disk_proposal.py` and `jobs/job_probe_disk_proposal.sh`.
Single-GPU job16617724 checks each dominant exact atom's original shortlist
membership, full-proxy rank and expected proposal hits, then compares whole-prior
Gaussian-proxy exact strata K1024/K16384/K131072 with16k complement draws on
the same64 diagnostic objects. It changes numerical integration only; models,
population, centre and full-prior normalization remain fixed. Ruff and shell
syntax pass. Startup and results pending. No production inference code or
results were overwritten, and no remedy has yet been accepted. Updated stale
introductory text in `doc/INFERENCE.md` and the runtime boundary in
`doc/JOINT_CALIBRATION.md` to distinguish the dedicated disk driver from the
unchanged default/legacy path and the stopped model-development investigation.

## 2026-09-21 — Diagnose failed V3.6 production curvature

All 20 production partitions of array 16606146 completed, covering all 500,000
observations. Combiner 16606157 failed on non-positive information at the first
draw rung. A CPU audit also finds non-positive information at every subsequent
rung, including the final 16,384-draw result: eigenvalues
`[-3687071.14567819, 71478092.02541134]`. No valid final estimate or covariance
has been published; the positivity guard remains intact.

Added `scripts/audit_disk_curvature.py`, checking partition/input hashes,
identities, exact observation coverage and moment finiteness, then separating
normalization curvature and tabulating diagnostic contributions by measured
properties and source case. Job 16617664 completed in 24 seconds; report is
`curvature_audit_v1.json` under the existing 20260920 run root. The negative
direction is dominated by bright observations, not normalization: numerator-only
information eigenvalues are `[-3874527.63, 70520765.56]`. The ten most negative
observations contribute about -19.51 million along that direction. Exclusion
tables are diagnostics, **not alternative estimates or authorized cuts**.

Corrected the interpretation of saved weight arrays: they have draw-rung and
observation axes. The final-rung median ESS is 893.97, not the 161.41 obtained
by pooling all rungs; 64 final-rung Pareto-k diagnostics are nonfinite, while
all final score/information values are finite.

Added `scripts/probe_disk_curvature.py`, `jobs/job_probe_disk_curvature.sh`, and
`tests/test_disk_curvature_probe.py`. The probe revalidates production cache,
model, input and response hashes without changing inference implementation or
old receipts. One arm sums all 24 million atoms exactly for the worst object;
the other uses paired 16k/32k/64k draws and two seeds for the worst 32 objects
plus the first 32 comparison rows. Array 16617682 has two one-A40 tasks, the
only active SBSI GPU jobs, each with four CPUs/64 GiB host memory. Resource
detection confirms full 46 GB A40s. Test job 16617683 passed the stencil-algebra
regression (1 passed in 12.15 seconds); ruff, launcher shell syntax and
`git diff --check` pass. Both probes have loaded the hash-validated runtime
in 81 seconds and are compiling/evaluating. Probe results pending. Updated
the inference and catalogue-prior status sections; old outputs remain intact.

Next: distinguish finite-draw integration failure from genuine non-concavity
of the fixed finite-prior likelihood before choosing a correction. No model,
population, normalizer, production moments, or positivity requirement changed.

## 2026-09-20 — Audit full-prior pilot and eliminate response-cache thrashing

All20 preparation shards16605612 completed in7m41s–8m19s each;
assembly16605621 completed in2m08s. Full-prior pilot16605622 completed:
24million atoms,256observations,16k draws, compiled density and16-object
chunks. All moment and diagnostic arrays are finite, all objects used16384
draws, and the summed information eigenvalues are59224.20 and119798.25.
Peak PyTorch allocation19,025,835,520B; initial load82.7s, inference257.04s.
Warm throughput201.43s/240objects projects29.14h at500k/four GPUs, excluding
startup. This exposed an implementation bottleneck, not a calibration failure.

`DiskCatalogueLikelihood._coefficients` had one cache slot, while the shared
stratified estimator alternates exact-candidate and complement grids at every
shear node. They evicted one another, recomputing identical response sums nine
times. Replaced it with a bounded two-entry LRU: no density, coefficients,
draws, probabilities, stencil or numerical precision changed. Added an
alternating-grid regression test, including bounded eviction and equality to
fresh evaluations.

`scripts/run_disk_inference.py` now permits this one audited runtime-only
change over the existing preparation receipts. It requires exact old/new
density hashes, the original producer-driver hash, identical source text for
all four preparation functions, every other implementation hash unchanged,
and every scientific identity unchanged. Receipts and arrays are not rewritten;
both producer identity and current runtime identity are recorded in results.
Tests reject other implementation/model changes. This is not a general
"ignore provenance" flag.

Validation16606079: **290 passed,1 skipped in40.45s**. The same256-observation
full-prior pilot is replayed, and `scripts/check_disk_pilot_replay.py` requires
bitwise equality of every saved array before writing its validation receipt.
Resource checks again verified a full46GB A40; the conservative96GB host
allocation and16-object chunks are retained. Ruff, shell syntax and
`git diff --check` pass. Replay16606079 completed successfully: every array
(moments, all ladder levels, draw/unique counts and weight diagnostics) is
**bitwise identical** to pilot16605622. Total inference decreased257.04s to
92.69s. Excluding each first16-object compile/warmup batch, throughput
improves201.43s/240objects to51.66s/240objects, a3.90x speedup. Peak GPU
allocation remains19.03GB. Four-GPU projection is7.47hours of warm inference;
allow8–10hours plus queue delays, subject to production throughput variation.

Added production launcher for20disjoint25000-object partitions with a combined
four-GPU throttle and a dependent strict full-coverage combiner. The launcher
requires the replay receipt and refuses implementation changes since that
validated pilot. No production job was submitted before replay validation.

**Production array16606146 is submitted**:20x25000observations, capped at four
full A40s,96GB host memory and four CPUs per worker, writing
`production_lru_v1/part_00` through `part_19` beneath the existing20260920 run
root. Dependent strict combiner**16606157** runs after array success; it refuses
missing/overlapping coverage, incompatible identities, hash mismatches and
non-positive combined information. Scientific settings remain uncut24million
atoms, measured radius>0.6arcsec/MAG_AUTO<25.8, and the full512–16384 ladder.

Startup verified for all four production workers on full46GB A40s. Partitions
0–3 passed the production gate, completed cache loading/compilation and
reported16or more evaluated objects; the latest check had304,272,256,16rows
respectively. Warm first-worker rates span roughly3.8–4.7objects/s, consistent
with the8–10hour planning range. No early error was observed. The combiner is
pending array success. Scheduled task-wakeup tools remain unavailable, so no
automatic assistant follow-up is installed; the scheduler chain continues.

## 2026-09-20 — Integrate disk inference and start full-prior cache preparation

### Driver and validation

Added `scripts/run_disk_inference.py` with prepare/assemble/run stages and
`sbsi/disk_inference_store.py` for validated subset loading, frozen shear-state
classifier probabilities, global proposal summaries and memory-mapped sharded
disk responses. It calls the existing `run_adaptive_section5` estimator;
no alternate estimator or additive-response approximation is introduced.
The legacy driver still rejects the preparation-only disk config. Default
likelihood and numerical configurations remain unchanged.

The new path verifies model, subset, feature-order, pairing, input and code
hashes. Assembly requires every source shard, rejects partial/mismatched
receipts, applies proposal dispersion floors globally, and combines full-prior
selected masses. Cached probabilities cover zero plus the nine stencil nodes;
missing nodes fail closed. Inference writes additive per-object moments and
all six512–16384 draw levels for the existing strict partition combiner.
The requested500k input fixes a common starting center
`(0.013912273363531775, 0.0002826679500570267)` without truth information.

Selection uses64QMC draws per atom, proposal summaries128, step0.001,
K1024/prefilter131072, and unmodified float64 tilted-score arithmetic.
Preparation streams use `SeedSequence([8101, source_shard])` for selection and
`SeedSequence([8201, source_shard])` for coordinates, with selection reset at
every shear point for CRN. This explicit sharded stream is not bitwise replay
of the historical single global stream. The subset and scientific cuts are
unchanged; no truth cut or empirical correction is added.

Full-A40 validation16605587: **288 passed,1 skipped in24.97s**. First attempt
16605578 stopped on an overly strict exact-equality test: NumPy target-column
indexing changes float64 reduction order by1.11e-16. The test now allows
roundoff, with no change to computed summaries. Ruff, launcher syntax and
`git diff --check` pass. Resource checks confirm full46GB A40s; workers honor
their four-CPU/48GB allocations, not the entire node's detected resources.

The real-model end-to-end smoke pilot completed on16384 prior atoms and128
real observations in23.03s of inference, peak PyTorch allocation154,863,104B.
It prepared ten classifier/selection states, reloaded hashed caches and wrote
finite moments at every draw level. This is an integration check, not a shear
calibration result or a full-prior runtime estimate. Artifacts live at
`inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1/pilot_16605587/`.

### Scheduled full-prior stages

The24million subset job16603639 completed successfully in20m08s: all20shards,
24,000,000 atoms and392,263,736 retained directed pairs. The completion
manifest and per-shard receipts were checked after the owner's notification.

New jobs:

- **16605612**, `jobs/job_prepare_disk_inference.sh`:20prior shards, at most
  four full A40 GPUs total, writing `disk_cache_v1/`.
- **16605621**, `jobs/job_assemble_disk_inference.sh`:CPU assembly after all
  preparation tasks succeed, writing `disk_assembled_v1/`.
- **16605622**, `jobs/job_disk_fullprior_pilot.sh`:256observations against
  all24million atoms,16k draws,16-object chunks and compiled density, after
  successful assembly. No500k observation inference array is submitted yet.

Startup was checked on all four workers: each passed source verification and
reached at least524,288/approximately1.2million proposal-coordinate rows;
shard0 reached786,432 rows at the handoff check. All four allocations are full
A40s, split across two compute nodes. No early failure was observed.

Check full-prior memory, finite moments, compilation behavior and timing before
launching production observation partitions. Any failed predecessor blocks
its dependent stage. Automatic task-wakeup tools are unavailable in this
session; these scheduler dependencies do not themselves monitor/report results.

## 2026-09-20 — Adopt an uncut uniform 24-million-atom prior

Owner selected24million atoms instead of the full139,936,000-row store.
`scripts/subsample_disk_prior.py` samples globally without replacement,
seed20260920, assigns equal1/24million masses, records original draw ranks and
source identities, verifies source hashes and preserves every selected
primary's response neighbours (including unsampled secondary atoms).
No truth-property selection is introduced. `DiskCatalogueModelCache` accepts
explicit original identities to keep classifier pair geometry correct after
compaction. Added subsampling/identity tests and the CPU preparation launcher;
updated the16k config description and
[prior documentation](CATALOGUE_PRIOR.md#v36-like-uncut-24-million-atom-subset).
Default models, numerical defaults and the preparation-only guard are unchanged.

The earlier2–3day estimate below is superseded: it combined the enlarged prior
with a virtual24GB A40 throughput probe. A matched full-A40 probe16603474 gives
19–23hours for tilted sampling alone on140million atoms; this is not an
end-to-end ETA. New24million-atom full-A40 probe16603624 tests16/32-object
batches. Its synthetic timings must not be reported as complete inference time.
All20 full uncut source shards have completed; source preparation is reusable.

Full-A40 probe16603624 completed: warm medians0.5468s/16objects and
0.6687s/32objects, projecting1.187h and0.726h respectively for tilted proposal
sampling at500k observations/four GPUs. Peak tensor allocations are11.72GB
and22.09GB, without resident likelihood tensors. This removes the oversized
prior bottleneck; complete runtime still requires the real-model pilot.
The resource skill verified a full46GB A40 and sufficient CPU-node memory;
preparation remains sharded/memory-mapped within its48GB Slurm allocation,
not the entire node's detected memory or CPU count.

Five targeted subsampling/cache tests pass in15.04s; Ruff, shell syntax and
`git diff --check` pass. CPU job16603639 completed the full suite:
**283 passed,2 skipped in52.08s**, no regression/snapshot failures, then began
materializing `prior_subset24m`. Its startup and resource report were checked.
First shard completed with1,200,254 sampled atoms and19,625,085 retained pairs
in53.7s including source verification; the second shard is in progress at
handoff. Top-level completion manifest is written only after all20 shards.
Production integration, normalization/proposal caches and a real-model pilot
remain next steps; no full V3.6 inference has been launched.

## 2026-09-20 — Begin uncut V3.6-like 500k/16k inference preparation

### Current status and runtime decision

The500k input is complete:18,157,963 eligible measurements in cases40–139,
no crossmatch-without-shape or matched-without-truth rows. Of the eligible
population,1,977,959 (10.89%) have true r>=26 and lie beyond the V3.6 training
parent. This percentage describes the eligible population, not an exact count
in the frozen500k subsample. No selected zero shapes were found. First full
feature shard completed with6,996,800 atoms and114,354,404 directed pairs;
feature construction took219.7s before receipt hashing. Remaining CPU shards
are array16603279,1–19 throttled to four tasks, dependent on first-shard and
test success. Startup was checked through source verification and pair progress.

Density/normalizer benchmark16603280 projects4.23h for density/response work
plus3.45h for128-draw proposal coordinates and64-draw nine-node normalization
on140million atoms, on four GPUs. Crucially, full-size synthetic proposal
benchmark16603325 measured2.5847s per two observations at139,936,000 atoms:
**44.9h for tilted sampling alone** at500k observations/four A40-24Q GPUs.
An opt-in blocked float64 cumulative sum (`WholeCatalogueProxy.cdf_block_size`)
reduces the same probe to2.1235s/two observations in16603358, about36.9h.
It changes addition order, not proposal probabilities; four boundary/ragged
tests pass in16603359. Historical sampling uses the original scan by default.
The production32-object proposal batch would exceed memory on this enlarged
prior; a memory-bounded batch is required before launch. Synthetic benchmarks
exclude resident likelihood tensors and local candidate retrieval.

Thus the revised planning estimate is **2–3 days on four A40s**, not the
earlier inference-kernel-only4.4h estimate. No multi-day GPU inference or
normalization array has been submitted. Await the owner's direction on this
resource commitment; CPU truth-feature preparation may finish independently.
There is no available automation/scheduled-wakeup tool in this session, so no
automatic follow-up has been created.

### Implemented components and remaining integration

`DiskCatalogueLikelihood` now routes exact, sampled, ragged tensor and
conditional-density paths through the inverse Möbius map and its Jacobian,
with observed radius/flux-conditioned coefficients reused across stencil
nodes. `DiskCatalogueModelCache` rebuilds all17 classifier features in chunks
for every shear state, retaining probabilities instead of huge feature frames.
Tests compare direct inverse-density calculations, state-dependent mass,
padding/invalid outputs, coefficient reuse, and the original classifier feature
constructor. Full suite16603250: **275 passed,2 skipped in43.29s**; targeted
density16603200:4 passed. `configs/inference_v1_2_16k.json` records the extended
draw ladder without changing defaults; its prior remains explicitly unbound.
The disk components are not yet wired into persisted model-cache loading,
distributed preparation, the production driver or full-scale memory management.
The preparation-only guard remains essential: do not remove it until those
paths and a real-model end-to-end pilot pass. No inference result exists yet.
Final CPU regression job16603398 reports **279 passed,2 skipped in37.26s**,
with zero regression/snapshot failures. Ruff passes for all files changed by
this preparation turn; `git diff --check` and launcher syntax checks pass.
At handoff8/20 truth-feature shards are complete (55,974,400 source rows),
four CPU shards are progressing, and no GPU job is active or pending.

### Initial preparation implementation and validation

Owner requested the previous full inference with V3.6-like, no truth-property
cuts, measured radius strictly above0.6arcsec, twice the complement draws,
and up to four GPUs. This is newly authorized inference, not a restart of the
cancelled independent image-validation campaign. The four-GPU allowance applies
to this run. Current preparation uses CPUs; the bounded benchmark used one GPU.

Added preparation-only `configs/likelihood_v3_6_like.json`, pinning the three
models and strict radius/flux predicates. `run_inference.py` explicitly refuses
this config until the catalogue density/state-major classifier adapter is
integrated, preventing an accidental additive-response run. Default configs
are unchanged. Extended `OutputCut` with explicit `NAME:>LOWER:UPPER` bounds;
old half-open bounds retain their behavior and cache identity. Extended the
ConstGold adapter with explicit no-truth-cut flags and the V3.6 usability
definition, including valid zero shapes. Reports count selected true-r>=26
objects separately: the requested uncut population extrapolates beyond the
model's true-r<26 training parent. No offset or calibration correction is added.

Added `CatalogueDiskResponse`, an exact CSR pair-sum evaluator on arbitrary
atom/output triples. It pools only identical tree split cells and handles
isolated atoms as empty sums. Added an uncut, neighbour-complete truth-feature
preparer: original zero contexts are hash/identity checked; nearest20 neighbours
within10arcsec use the pinned V3.6 development pairing convention, without
truth-property cuts. This store is not yet a normalized production prior.

Validation: response pooling tests pass; Slurm16603127 reports **270 passed,
2 skipped in48.15s** for the complete active test suite. Benchmark16603029
failed on a missing PYTHONPATH;16603043 started before its parent-key correction
and failed on the old `ids` field. Both setup issues were corrected.
Successful A40-24Q benchmark16603054 verified the pinned model and source hashes:
262144 atom/output terms,3916572 pair/tree evaluations in1.06254s; nine warm
density calls in0.97518s. Linear projection is **4.422 GPU-parallel wall hours
on four GPUs for500k objects and16384 atoms**, excluding preparation, classifier
states, normalization, retrieval, I/O, queue time, and first compilation.
This is a one-scene uniform-atom eager-kernel benchmark, not a production timing
or inference result. Receipt: `logs/v36_benchmark_16603054.json`.

Preparation job16603127 first freezes500k uniformly sampled plus-leg objects
from cases40–139, seed20260914, then builds the first uncut prior-feature shard.
Run root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1`.
Startup was checked through passing tests and successful catalogue processing.
Production inference has **not** been submitted. Next, after the runtime
decision above: finish production integration, build all new
proposal/normalization caches and launch at most four GPU workers. Retain
K1024, h0.001, the one-step estimator, full-prior64-draw CRN normalization and
the paired8192/16384 draw ladder. Do not reuse V3.5 probabilities or cut caches.

## 2026-09-20 — Stop investigation, retain V3.6-like, and archive research code

Owner requested cancellation, the name V3.6-like for the retained single-flow/
disk-emulator/smooth-classifier set, and repository cleanup. Cancelled the
downstream jobs first (16595003,16602287,16602111,16595004,16595000,16594999),
then generation16593294 at about11:10 CEST. Accounting records 34 completed
scenes and six cancelled generation tasks; all pending stages are cancelled.
Queue inspection confirms no remaining research jobs. External artifacts,
partial products, original protocols, and scientific receipts remain intact.
The 40-scene validation produced no bias result and is closed by owner request.

Added `configs/models_v3_6_like.json` with the original three model identities,
four artifact hashes (including emulator metadata), source-manifest hash,
population/cuts, retained endpoints, and explicit cancellation status.
`sbsi.models` and CLI expose V3.6-like alongside the historical V3.5-like preset.
The new disk loader checks model/metadata hashes; the additive loader rejects
disk models. Extracted five unchanged feature functions into `sbsi.crowding`
and added a seventeen-input constructor. Updated model/API/README/environment/
convention/inference documentation and AGENTS.md. Inference defaults and model
weights are unchanged; V3.6-like is not a drop-in additive likelihood config.

Archived 453 retired files (150 scripts, 199 jobs, 90 tests, 7 plots, 4 experimental
package modules, 3 detailed documents) and 19 before-cleanup snapshots under
`archive/research-2026-09-20/`. Every archived file has an original path, archive
path, SHA-256 and byte count in `manifest.json`. All 472 hashes verified after
the moves. This preserves pre-existing uncommitted work and all original
scientific source bytes. The full previous work log and joint investigation
are archived; current docs summarize the retained model and link to history.
See [repository organization](REPOSITORY.md#research-archive).

Validation: active Python AST syntax and four launcher shell checks pass;
Ruff passes for the modified/new Python files. AST comparison confirms the
five extracted feature functions are unchanged. CPU verification 16602609
completes in 1m20s with empty stderr: `python -m pytest tests -q` reports
**267 passed, 1 skipped in 56.39s**, zero regression/snapshot failures.
`python -m sbsi validate-model V3.6-like` verifies all four artifact hashes.
The actual flow, classifier and disk emulator load; feature-order/training
metadata checks and a small flow/classifier prediction smoke pass. Active
imports have no missing or archived dependencies; all active document links
and their section anchors resolve. Final active counts are 31 package modules,
19 scripts, 4 launchers, and 35 test/support files. The verification used no GPU.
Final scheduler inspection is empty; `git diff --check` also passes.

Limitations: existing endpoints remain development evidence; fresh multi-axis
validation was stopped. Archived launchers need deliberate restoration in an
isolated checkout and source/hash checks before reuse. No science work or
automatic monitoring is scheduled by this cleanup.

## Earlier history

- [Complete log before 2026-09-20 cleanup](../archive/research-2026-09-20/doc/WORKLOG.md)
- [Joint calibration experiments](../archive/research-2026-09-20/doc/JOINT_CALIBRATION.md#goal-and-retained-population)
- [Earlier research archive](REPOSITORY.md#research-archive)
