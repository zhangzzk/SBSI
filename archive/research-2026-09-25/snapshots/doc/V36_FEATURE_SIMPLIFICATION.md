# Close-crowding feature simplification

## Decision and scope

Owner requested consistent, simpler inputs across the flow, classifier and
response emulator without degrading performance. The baseline is the newer
close-crowding flow1433b7217fb7, classifier13cd6a81ecff and original disk
emulator194c1d11b67d, not the original pinned V3.6-like flow. Its +0.286%
development result and limitations are in the
[model review](V36_MODEL_COMPARISON.md#second-review-reproducibility-reuse-and-mixture-sensitivity).

Simplify feature construction first, then remove information only through
controlled retraining. Shared physical definitions are the objective; identical
input vectors or neural architectures are not required for distinct tasks.
No checkpoint, active feature mapping, cut or inference configuration changes
as part of this design. The completed F8 development screen is reported in
[the results](#f8-development-results); it has not been promoted.

## Current contracts and genuine differences

| Component | Features | Coordinates and support |
| --- | --- | --- |
| Close-crowding flow | Five target properties; three legacy flux summaries; three physical Gaussian crowding features | Legacy summaries use all neighbours in0–3/3–7arcsec shells and the maximum neighbour flux, normalized by aperture noise. New Gaussians use physical0.5/1/2arcsec scales, all rendered neighbours within10arcsec, and target-relative flux. |
| Smooth classifier | Same five target properties and three legacy summaries; five annular flux-ratio sums; four Gaussian flux-ratio sums | Annuli and Gaussians use distance divided by the PSF-convolved target major-axis size. They are recomputed for each sheared state. Existing evaluations use the cached pair support, at most19 nonself neighbours. Gaussian scales are0.5/1/2/4 in these dimensionless coordinates. |
| Disk emulator | Seven pair features plus generated target log radius and magnitude | Scaled target/neighbour size and brightness, two Sersic indices, separation, and the same flow draw's output features. Predicts pairwise transport, not an aggregate usability probability. |

The Gaussian expression is shared:

`C(s) = log1p(sum_j (F_j/F_i) exp(-d_ij^2/(2s^2)))`.

Physical distance and size-normalized distance are not interchangeable.
Neither are full-neighbour and truncated-pair support. Under shear, the
classifier's major-axis scale changes even when physical positions stay fixed;
holding these features fixed would change its response. A common kernel
implementation must leave those choices explicit and versioned.

The legacy maximum is not determined by a few Gaussian sums. Those sums also
do not exactly reproduce the3–7arcsec shell, especially with a largest physical
Gaussian scale of2arcsec. Absolute noise-scaled flux can carry information
different from relative neighbour flux. These facts prevent an algebraic claim
that all old features are redundant.

## Shared representation

Use one scene/pair feature specification with explicit identity matching,
self-exclusion, units, flux zero point, noise and PSF definitions, radius/shape
conventions, aperture, neighbour cap, log transform and shear-state dependency.
Each component consumes a declared view of this representation.

For future models, the target-level starting point is the five existing target
properties plus a small smooth crowding bank. Flow and classifier should share
the feature generator, schema and preprocessing rules. A common Gaussian
family avoids maintaining independent annular and Gaussian implementations,
if retrained models demonstrate that the annuli add no required information.
Physical versus normalized scales remain an empirical choice until tested.

For the pair emulator, retain explicit target/neighbour structure and
separation. One absolute brightness plus a flux ratio is an invertible
reparameterization of two brightnesses given the declared calibration; it is
not a reduction in information or feature count. Losing absolute brightness
would discard noise dependence. Keep generated radius/flux conditioning until
its removal passes a joint test, because it retains correlations with the
science cuts. A scene-level crowding sum cannot replace pair identity and
geometry without a new model and labels.

The classifier models P(U|x), while the flow models p(y|x,U=1). Sharing x is
natural; merging their networks is a separate capacity/optimization experiment.
Keep the four joint flow outputs and physical support transforms in the first
round. Do not simultaneously simplify features, network capacity, response
transport and training objectives.

## Minimal removal experiment

Start with two independent arms, each changing only one feature group:

| Arm | Change | Unchanged components |
| --- | --- | --- |
| F8 | Retrain flow with five target properties plus its existing three physical Gaussian features:11 to8 raw inputs | Existing17-input classifier and original disk emulator |
| C12 | Retrain classifier without its five hard-annulus features:17 to12 raw inputs | Existing11-input close-crowding flow and original disk emulator |

Use the same available half-shear domain, scene splits, training budget and
predeclared training seeds as their respective controls. Train reduced models;
do not set removed checkpoint inputs to zero or interpret permutation scores
as proof of retrained-model equivalence. Prefer fresh full-feature training
controls under the same implementation for assessing training variation.
F8 may fail because it loses the far shell or strongest-neighbour information;
C12 may fail because annuli carry information the smooth bank cannot recover.
Those are useful outcomes, not reasons to quietly change the experiment.

If both arms pass independently, evaluate their combination because effects
can interact. Only then test removal of the classifier's legacy three summaries
(12 to9), and, separately, the usefulness of its largest Gaussian scale.
Test coordinate/support harmonization in a separate arm; switching to full
neighbour support or a common physical/normalized bank is not a feature-only
ablation. There is no commitment that the final smallest adequate vector has8
features. Do not launch a large combinatorial search against constant-shear m.

The previous crowding-feature screen demonstrated that adding crowding helped
a pass/fail proxy; it did not test removal of legacy summaries or the density
and usable-event models. Its kernel also used bounded F_j/(F_j+F_i) weights,
unlike the deployed log1p relative-flux sum. Its log-loss improvements therefore
do not establish that any proposed reduced input set is adequate.

## Performance contract before promotion

Keep measured radius>0.6arcsec and MAG_AUTO<25.8, independently on each shear
leg, with no truth-property selection. Report population roles explicitly.
Feature selection uses training/development data; new scene-disjoint acceptance
data must be reserved across all components after the candidate is frozen.

Evaluate full conditional NLL for the flow, usable-event BCE/calibration for
the classifier, and joint selected mass, shape/size/flux distributions and
response for the combined model. Reuse the existing0.001nats/row flow density
gate relative to a matched full-feature control. A proposed simplification
budget is a paired95% interval for the bias change wholly inside+/-0.05
percentage points, evaluated with matched scenes, common random numbers and
draw-converged predictions. Freeze the classifier BCE margin and all other
numerical tolerances before fitting the first reduced candidate; do not choose
them after inspecting outcomes. The0.05pp budget is a proposed experimental
criterion, not an already passed result or a proof of absolute0.3% calibration.

Assess both axes, finite-shear dependence, latent/training repeats, and
predeclared population/observing-condition perturbations. Compare exact weighted
regional response contributions and their changes, so a smaller aggregate m
cannot qualify solely by stronger cancellation. Truth bins remain diagnostic;
they never filter the acceptance sample. If uncertainty is too large to
establish equivalence, the result is inconclusive, not a pass. Report parameter
count, feature-building cost, inference runtime and memory alongside accuracy.

## Immediate status and implementation boundary

F8 training and the80-scene development comparison are complete; see
[results](#f8-development-results). The owner accepted this result as good enough
for the simplification effort. F8 is the preferred flow for the next development
comparisons; this records an owner decision, not independent acceptance or a
retroactive pass of the original equivalence screen. No production preset was
changed. C12 remains the next classifier experiment. SBSI owns flow/classifier
training and composition; pair-emulator training belongs to BlendEMU.

## F8 execution record

Experiment root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/`.
Isolated source:
`/project/ls-gruen/users/zekang.zhang/sbsi_flow_f8_20260924/`.

`experiment.json` freezes the input subset, source hashes, seed611, original
case split, full training recipe, fixed classifier/emulator hashes, no
constant-shear checkpoint selection, density gate0.001nats/row and proposed
paired bias-equivalence budget+/-0.05pp. `launch.json` records jobs, dependencies
and source hashes. The isolated checkout derives from the active September24
training checkout; no archived code is imported. A task-local venv installs
its packages without changing the shared environment or using PYTHONPATH.

`setup_v36_f8_experiment.py` copies and hashes212 source/configuration files.
Changes are restricted to the training feature tuple, preservation of the
full scene/classifier context in evaluation, and local package installation.
The classifier still uses its original eight core plus nine geometry inputs;
F8 removes the three legacy summaries only from the flow. Its eight raw inputs
become16 preprocessed channels. No distillation, input zeroing or model-weight
slicing is used; initialization and training start afresh.

`project_v36_flow_features.py` verifies original file hashes and writes a
column-projected view of both the original and harmonized training domains.
It reloads every output and verifies all non-context arrays and every retained
context column exactly, including dtype and row order. CPU16690073 completed
in1m50s:400 base-domain and800 harmonized-domain files, zero altered labels,
zero removed rows. Split and population metadata are unchanged. Final manifests
have SHA prefixes `e1303a08a929` and `149f783002ff` respectively.

| Stage | Job | Dependencies / scope |
| --- | --- | --- |
| Dataset projection | 16690073 | Completed; four projection tests pass |
| Recompute and replay response-label banks | 16690099 | Completed in5m45s; targets and baseline moments reproduce the full-feature banks within2e-12 |
| Fresh NLL and initial response-guard training | 16690101 | After projection; one GPU |
| Original control-variate and common-descent refinement | 16690103 | Successful base training and label banks; one GPU |
| Constant-shear evaluation | 16690166_0–7 | Successful refinement and release of the existing emulator-swap GPU job; ten cases per task, maximum two tasks |
| Paired comparison and row replay | 16690221 | Successful completion of every evaluation task; CPU only |
| Repository tests | 16690222 | Completed:326 passed,2 skipped in35.24s |
| Scheduled status receipts | 16690225 / 16690226 | About two hours after submission / after summary job ends |

The initial GPU submission requested64GB host memory, exceeding the virtual
node's40.5GB capacity, and was rejected before a job was created. The final
request is32GB host memory and one24GB A40 allocation. Base startup verifies
the frozen sources, installed package identity and data manifests, runs the
density-gate test, and exercises an actual projected-data density and backward
pass. These checks passed; initial training epochs have finite losses and
decreasing validation NLL. This is operational progress, not calibration
evidence. Each subsequent command writes a status receipt and stops on failure;
the original refinement stability and density gates remain enforced.

`summarize_v36_f8.py` requires all80 declared cases and fixed component/draw
identities, checks measured sums against the baseline, computes own-normalized
per-leg bias and paired case-bootstrap changes, replays every row dump, and
reports exact magnitude-region contributions plus full validation NLL change.
The original classifier and disk emulator remain fixed. Predictive histograms
and joint moments are retained in the evaluator outputs. The expected final
receipt is `development_comparison.json`; it always declares
`production_accepted=false` and `independent_acceptance=false`.

This first screen uses one fresh F8 training seed against the existing frozen
full-feature control. Different input dimensions change the initialization
layout; training variation is not eliminated by a shared seed. New full-feature
training controls, repeats, both axes, numerical convergence, full-role transfer
and untouched-scene acceptance remain prerequisites for a stronger claim.
Scheduled checks write local receipts; no conversational wakeup tool is
available, so these jobs do not promise an automatic chat notification.

## F8 GPU allocation increase

On September24 the owner requested more GPUs. Base training16690101 and
refinement16690103 had already completed successfully. Evaluation array16690166
was updated in place from throttle2 to6; pending tasks3–7 now request16GB A40
allocations, using idle compatible nodes. Running tasks0/2 retain their24GB
allocations and completed task1 is retained. The evaluator uses bounded batches
of1024; a running-worker GPU-memory spot check showed716MiB used, which is not
a peak-memory guarantee. The new16GB worker is monitored through initial progress.

The `cip` partition QoS enforces `gres/gpu=3` per user, so three workers run
concurrently and further workers wait with `QOSMaxGRESPerUser`. The requested
array ceiling of six does not override this scheduler limit. Model weights,
case assignments, seeds, draw counts, batch sizes and summary dependencies are
unchanged. The original `launch.json` remains a historical submission record;
`scheduling_amendment_20260924.json` records the new allocation and authorization.
One-shot status job16692553 writes a local receipt about two hours later.

### F8 expansion to inter

The owner subsequently authorized `inter`. Pending array tasks5/6/7 were moved
in place to `inter` with one full A40 each, retaining four CPUs,32GB host memory,
six-hour limits and the same frozen evaluation commands. Tasks0/3/4 continue on
`cip`, while tasks1/2 have completed. The array retains its six-worker ceiling
across both partitions. The `inter` partition QoS allows eight GPUs per user;
its use is explicitly owner-authorized. No cancelled/replacement jobs or new
summary dependencies are needed: CPU16690221 still waits for the original array.
`scheduling_inter_amendment_20260924.json` records this second scheduling change.
The scheduled status receipt16692553 continues to cover all array tasks.

Startup verified: tasks5/6 onkng-cl-nv01 each processed1024 rows of their first
case (90/100) without errors. Five workers are active across both partitions;
task7 remains pending for priority.

### F8 final task returned to cip

On September25 the owner requested moving16690166_7 back to `cip` while it
remained queued on `inter`. Updated the task in place to one16GB A40 allocation;
CPU/memory/time settings and summary dependencies remain unchanged. At this
check tasks0/1/2/4/5/6 completed successfully, task3 was still running, and task7
was the last unstarted task. `scheduling_task7_cip_20260925.json` records the move.

Task7 subsequently started oncip-cl-h01g03n3; startup verification confirmed
25,600 processed rows of case110 without errors.

## F8 development results

All eight evaluation tasks16690166_0–7 completed successfully. CPU16690221
completed in78seconds, verifying all80 cases, fixed model identities, measured
sums against the baseline, and every row dump against evaluator sums. Final
receipt: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/development_comparison.json`.
F8 flow SHA256 beginsbd51dbbbeeb7; fixed classifier13cd6a81ecff and disk194c1d11b67d.

Protocol: cases40–119, generated target-role parent, no truth-property or g0
usability cuts; separate measured MAG_AUTO<25.8 and FLUX_RADIUS>0.60arcsec in
each antithetic leg; h=0.02 on axis1;64 draws with seed7301. Means use each leg's
own selected normalization. Bias is100(R_sim/R_model−1), following
[response conventions](CONVENTIONS.md#6-response-estimators) and
[bias conventions](CONVENTIONS.md#7-multiplicative-bias). Uncertainties below
are case-bootstrap standard errors in percentage points (10,000 paired
resamples, seed20260924), not training-seed or latent-integration uncertainty.

| Quantity | Full11-input control | F8 |
| --- | ---: | ---: |
| Smooth-classifier bias, percent | +0.285858 ±0.265609 | +0.138352 ±0.266521 |
| Actual per-leg usability diagnostic bias, percent | +0.371344 ±0.263945 | +0.228311 ±0.266466 |
| Full validation NLL, nats/row | −4.656837402 | −4.744169164 |
| Sum of absolute magnitude-region bias contributions, pp | 2.817658429 | 3.915030723 |

Smooth-arm paired F8−control change is−0.147506pp, SE0.009806pp,
95% interval[−0.167216,−0.128505]pp. F8's absolute bias95% interval is
[−0.395797,+0.655236]%, so absolute0.3% calibration is not established.
The predeclared±0.05pp equivalence screen fails because the shift is larger
than that margin, even though the aggregate point estimate moves toward zero.
Do not relabel this outcome as a passed equivalence test. The density screen
passes: NLL decreases by0.087332nats/row, versus an allowed increase of0.001.

F8 exact global-response contributions across true magnitude diagnostic bins
<24,24–25.5,25.5–27.5,>=27.5 are[−0.994256,−0.894084,+1.939942,+0.086750]pp.
These sum to+0.138352%, but their absolute sum is larger than the control's.
The similar actual-usability pattern shows the increased cancellation persists
when substituting observed per-leg usability. These are diagnostic allocations,
not conditional biases, truth selection cuts, or causal neighbour effects.

Conclusion: reducing flow inputs11→8 improves the aggregate development point
estimate and validation density in this run, but does not demonstrate robust
preservation under population changes. Retain F8 as a research candidate;
production_accepted and independent_acceptance remainfalse. One training stream,
reused scenes, a historical full-feature control and g1-only evaluation limit
interpretation. Next priorities are matched training repeats, numerical and
both-axis checks, and population-transfer/untouched-scene validation, with
particular attention to the stronger magnitude-region cancellation. Classifier
and emulator simplification has not yet been tested.

## Next component simplifications after owner acceptance of F8

The next comparison should hold F8 and the disk emulator fixed and retrain the
classifier after removing its five annular summaries (17→12). This updates the
original C12 plan, which used F11, to the owner's accepted development baseline.
If adequate, remove the three legacy near/far/max summaries (12→9), retaining
five target properties and four smooth Gaussian summaries. Do not simultaneously
change the Gaussian coordinates, support or shear-state recomputation to match
the flow: those differ physically and require a separate harmonization test.
Select classifier checkpoints by tuning BCE, then assess calibrated usability,
selected counts and the same measured-cut shear response with paired draws.
The current classifier is already a small two-layer64-unit MLP; feature
construction is a clearer simplification target than network architecture.

For the nine-input,218-tree pair emulator, first test reduced tree count/depth
by retraining under its original physical-moment objective in BlendEMU. Keep
its pair geometry and generated measurement conditioning in that first test.
Its scalar pair-response output and physical transport already provide a compact
interface. Replacing two magnitudes with one magnitude plus a flux ratio improves
notation but does not reduce information or dimensionality. Removing output
radius/flux conditioning or replacing pair geometry with aggregate crowding
requires a separate ablation, because these can change joint selection response.
No classifier/emulator reduction is yet measured or launched by this review.

## C12 matched non-degradation experiment

On September25 the owner authorized continuing simplification, accepting a
reduced model only if it does not degrade performance. Experiment root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_occam_20260925_v1/`;
isolated checkout:`/project/ls-gruen/users/zekang.zhang/sbsi_classifier_occam_20260925/`.
`protocol.json` freezes the criteria and source/data identities before fitting.
F8 remains fixed, as does the original disk emulator. The retained classifier
is the reference; no preset or production artifact changes automatically.

Train fresh C17 and C12 for seeds20260913/20260914. C12 retains columns0–7 and
13–16, dropping only the five annular summaries. Original prepared files remain
unchanged and hash-checked; projection happens during loading. Labels, rows,
train40–99/tune100–119 splits, preprocessing, AdamW, batch size16384, maximum
8million pairs/epoch,100epochs and patience15 are unchanged. GPU training is
used for both matched arms; the retained historical classifier used CPU.
Original initialization gives an identical eight-input core and zero additional
feature weights in both arms; an exact function-match preflight passed.
Checkpoint selection uses minimum tuning BCE, never constant-shear response.

Both seeds must pass against their same-seed C17 control AND the retained
classifier. Every metric must have a non-worsening point estimate (numerical
zero tolerance1e-12), and its paired95% interval's upper endpoint must be below
the fixed margin. An uncertain result does not pass; there is no favorable-seed
selection. Margins bound what the comparison must rule out, not permission to
accept a worse point estimate:

| Endpoint | Upper confidence-bound margin for degradation |
| --- | ---: |
| Absolute main-cut multiplicative bias | 0.05 percentage points |
| Sum of absolute magnitude-region response contributions | 0.05 percentage points |
| Absolute relative selected-count error, both legs and all three existing cuts | 0.0005 (0.05%) |
| Absolute main-cut cross-response error | 0.0005 response units |
| Constant-shear BCE, Brier and absolute mean-probability gap, each leg | 0.0005 |
| Half-shear mean-case tuning BCE | 0.0005 nats/label |

Use10000 paired case-bootstrap samples, seed20260925. Constant-shear metrics
use all80 cases; tuning BCE uses the20 tuning cases. All gates are required.
The original80-scene F8 row dumps are reweighted by the new per-leg usability
probabilities, preserving the exact same flow/emulator draws and selections.
Reconstruct all17 classifier features in the original sheared coordinates,
then select inputs by each checkpoint's feature names. Independently replay
retained probabilities against saved values and measured/model sums against
the completed F8 reducer on every case. All truth-property bins remain in the
sample; bins diagnose cancellation only. This is a functional feature ablation;
removing unused feature-building operations comes after acceptance.

| Stage | Job | Scope |
| --- | --- | --- |
| Repaired preflight | 16695113 | Completed: five gate tests, initial-function matching, case40 both-leg probability replay, synthetic end-to-end summary |
| Matched training | 16695108_0–3 | Two GPUs maximum; C17/C12 for seed20260913, then seed20260914 |
| Cached-draw comparison | 16695110_0–7 | CPU, four concurrent tasks; all four trained models plus retained classifier on each scene |
| Non-degradation report | 16695112 | CPU, after all evaluations; writescomparison.json |
| Read-only status checks | 16695116/17 | Two hours after submission / after report finishes; local receipts only |

Initial preflight16695107 failed during imports because the copied checkout
omitted release configuration JSONs. Before any training, copied and pinned the
unchanged active configuration files; preservedprotocol_before_packaging_fix.json
and updated the training dependency to successful preflight16695113. No loss,
criterion or model setting changed. Training initially queued oninter; moved
the waiting array to idlecip16GB A40 allocations with throttle2 unchanged.

No reduced classifier is accepted yet. Reused development scenes, two seeds
and axis1-only cached draws do not establish independent calibration. C9 and
emulator capacity reduction remain subsequent stages, after reviewing this
non-degradation result. Tests exercise gate behavior and the complete summary
under identical synthetic models; runtime replays guard scientific equivalence.

Startup verified: both seed20260913 arms completed two-epoch real-data smoke
training/export, then reached full-training epoch9 with finite decreasing BCE.
C17=0.187556 and C12=0.192054 at that point; no conclusion is drawn from early
epochs.

### Owner amendment: small differences within uncertainty

While C12/C17 training was underway, the owner clarified that a slightly worse
point estimate is acceptable within uncertainty. The strict point-estimate veto
above is superseded. For each existing endpoint, the paired95% degradation
interval must have lower endpoint<=0 and upper endpoint<=its original fixed
margin. Thus an unresolved small loss may pass; a resolved loss or uncertainty
wide enough to include meaningful degradation fails. All endpoints, both seeds,
both controls and original margins remain required.

This user-directed change is recorded inacceptance_rule_amendment.json before
any completed candidate comparison was inspected. The frozen training protocol,
training/evaluation sources and original strictcomparison.json are preserved.
A separate pinned decision script writesacceptance_amended.json, retaining the
original flags beside amended flags. Job16695134 follows the original summary
and gate tests16695133; final status16695117 now follows the amended decision.
This is the operative development acceptance report; no production promotion
is automatic. The original five strict-rule tests remain valid historical tests;
five new tests cover the revised acceptance semantics.

### C12 scoring repair after training

All four training runs completed successfully. Tuning BCE (seeds20260913/20260914)
is0.183037854/0.183036001 forC17 and0.186334092/0.186381115 forC12. The initial
evaluation array16695110 failed in the new scorer: NumPy rejects unary minus
on Boolean labels. Casting usability labels tofloat restores the intended BCE
formula. No trained model or acceptance criterion changed.

Original source/protocol and failed receipts are preserved; a separate corrected
runner differs by exactly this cast. Repair hashes are recorded in
`evaluation_scoring_repair.json`. Repaired chunk0 is16695636; remaining chunks
are16695637. Summary16695112 now depends on both, followed by the unchanged
owner-amended decision16695134. Two regression tests exercise Boolean labels.

## C12 completed decision

The repaired80-scene evaluation and owner-amended decision completed successfully.
`v36_classifier_occam_20260925_v1/acceptance_amended.json` reportsfalse for overall
non-degradation. C12 is rejected; retain the existing17-input classifier.
Tuning BCE worsens by0.003296 and0.003345nats/label against paired C17 controls;
95% intervals[0.003221,0.003370] and[0.003265,0.003427] exceed the fixed0.0005
margin and exclude zero. Constant-shear classification/Brier losses and selected
counts also fail. Primary shear point estimates alone look similar within each
seed (C12:+0.144184%/+0.050214%; paired C17:+0.138352%/+0.050701%), which does not
rescue the failed probability/count checks. The retained classifier remains
unchanged; the better-looking second control seed is not substituted for it.

## C8 classifier with exact F8 raw inputs

Owner requested testing the F8 flow inputs directly in the classifier. This is
an explicit representation change: core five galaxy properties plus
nbr_close_050/nbr_close_100/nbr_close_200, physical0.5/1/2arcsec Gaussians over
all rendered neighbours within10arcsec. It removes legacy summaries, annuli and
size-normalized Gaussian features together, testing a shared representation
rather than attributing any gain/loss to one individual removed group.
Classifier preprocessing remains its original finite-input standardization;
sharing raw features does not merge the flow and classifier networks.

Experiment root:`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_f8_inputs_20260925_v1/`.
Isolated checkout:`/project/ls-gruen/users/zekang.zhang/sbsi_classifier_f8_inputs_20260925/`.
The pre-fit protocol inherits the owner's uncertainty-tolerant rule and all
fixed margins/endpoints. Compare each C8 seed20260913/20260914 against both its
already trained C17 counterpart and the retained classifier. Controls are reused
unchanged and hash-pinned. Initialization preserves their first-five target
weights and hidden layers; new physical-crowding weights start at zero.
Original training splits, labels, optimizer, budget and BCE-only checkpoint
selection remain fixed. F8 and the physical disk emulator are held fixed.

Preparation joins the verified close-crowding parent to original half-shear
classifier files by identity, requires matching base properties and complete
coverage, and preserves each leg's five original galaxy properties plus every
label/other array. The half-shear and constant-shear preparations share a pinned
source parent; the original crowding lookup verified identical positions and
fluxes across simulation sets. Physical kernels are consequently fixed under
these shear perturbations. Output files are reloaded and compared, with missing
or duplicate identities rejected rather than dropped. Raw input names, physical
units, format and source hashes are explicit in the new manifest. No truth or
measured selection is added to training.

| Stage | Job | Scope |
| --- | --- | --- |
| Identity-checked data preparation | 16695734 | Eight projection/acceptance tests passed; then80 scene files |
| Two C8 training seeds | 16695735_1,3 | At most two16GB A40 GPUs, after preparation; real-data smoke before full training |
| First evaluation chunk | 16695737_0 | Ten scenes, including retained-model probability/sum replay |
| Remaining evaluation | 16695739_1–7 | After first chunk; CPU maximum four concurrent |
| Comparison and decision | 16695740 | All80 scenes; exact saved F8 draws and unchanged uncertainty-tolerant criteria |
| Local status receipts | 16695741/42 | Two hours later / after summary; no conversational wakeup |

The new evaluation frame appends the exact physical crowding from saved F8
contexts; models select their named inputs. C17/retained models continue to
receive their original features, and their predictions and sums must replay.
C8 remains unaccepted pending all checks. This remains a two-seed development
screen on reused axis1 scenes, not independent validation. Emulator reduction
remains a subsequent experiment.

C8 startup: preparation completed in2m42s across all80 scenes, with zero changed
labels, removed rows or unmatched identities (manifest SHA256e953c2900988). Both
seeds passed real-data smoke training and checkpoint export before full fitting.

Both C8 seeds subsequently reached full-training epoch23 with finite losses;
no checkpoint or acceptance choice is made from this startup observation.

## C8 completed decision

Both training seeds, all80-scene evaluation chunks and summary16695740 finished
successfully. `v36_classifier_f8_inputs_20260925_v1/comparison.json` reports
`development_non_degradation_pass=false`. Retain the original17-input classifier;
C8 is not accepted under the owner's uncertainty-tolerant criteria.

| Seed | C17 tuning BCE | C8 tuning BCE | C17 main-cut bias | C8 main-cut bias |
| --- | ---: | ---: | ---: | ---: |
| 20260913 | 0.183037854 | 0.193620925 | +0.138352% | +0.020770% |
| 20260914 | 0.183036001 | 0.193784005 | +0.050701% | +0.062423% |

Bias uses the same measured-only0.60arcsec/25.8 per-leg cuts, generated
target-role parent and own-normalized antithetic response described above.
Case-bootstrap SE is approximately0.265pp for each C8 estimate. Small primary
bias does not establish non-degradation of the complete probability model.
Paired tuning BCE degradation is+0.010583nats/label (95% interval
[0.010424,0.010747]) and+0.010748 ([0.010592,0.010917]); both clearly exceed the
0.0005 margin. Constant-shear BCE and Brier loss also worsen in both seeds;
additional count/calibration/cancellation checks fail in some comparisons.
Thus the decision does not rest solely on the absolute-bias uncertainty gate.

The first C8 seed reduces the cancellation diagnostic from3.915030 to3.812848pp;
the second increases it from3.873877 to3.893420pp. No favorable seed is selected.
This result concerns this architecture, training recipe and representation
change; it does not prove that physical-scale inputs can never work for a
classifier. The retained development combination remains F8 flow + C17
classifier + original disk emulator. Emulator capacity reduction remains untested.

The only pending jobs at review were delayed status receipts16695741 (eligible
12:31:52 Berlin) and16695116 (eligible10:53:39). Their `BeginTime` state was
intentional, not a stalled scientific job. Both redundant receipts were cancelled
after direct result review; completion status16695742 had already succeeded.

## Owner retains C8 provisionally; emulator inputs

The owner explicitly retained C8 for now despite the broader classifier-loss
screen, prioritizing its observed shear calibration and simpler shared inputs.
The working development combination is F8 flow + C8 classifier + original disk
emulator. Use the first predeclared classifier seed20260913 (SHA256231bbbdf1f88)
for the main emulator comparison, with seed20260914 preserved as the replicate.
This is a seed convention, not selecting whichever measured bias looks best.
`v36_classifier_f8_inputs_20260925_v1/working_selection.json` records the decision.
Prior test failures and lack of independent acceptance remain unchanged; no
production preset is overwritten.

The retained emulator takes nine scalar inputs per target-neighbour pair:
Re_input_p_scaled, Re_input_s_scaled, r_input_p_scaled, r_input_s_scaled,
sersic_n_input_p, sersic_n_input_s, distance_scaled, baseline_log_flux_radius,
baseline_mag_auto. Each true effective radius is divided by its own
sqrt(Re^2+PSF_HLR^2). Separation is divided by the target's convolved radius.
Each true magnitude is transformed from flux divided by the existing aperture
noise normalization. The last two inputs are ln(measured flux radius in pixels)
and measured magnitude, taken from the SAME generated F8 draw during inference;
training uses the baseline measured values. The model predicts one scalar
physical disk-transport coefficient per pair, sums neighbours, and composes
with the generated shape using the physical disk map.

Source-verified against model metadata, SBSI output_conditioned_response.py and
BlendEMU data_utils.rescale/train_disk_response_moment.py. The current booster
has218 trees (max_depth11) fitted under the physical-moment objective. The first
emulator simplification target remains tree capacity with all nine features
retained; dropping generated radius/flux conditioning or pair geometry is a
separate scientific ablation. No emulator reduction has been launched in this
input review. Emulator fitting belongs to BlendEMU; SBSI evaluates the combined
model against the owner-retained F8+C8 working baseline.
