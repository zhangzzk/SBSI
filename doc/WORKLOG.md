# Work log

## 2026-11-08 — Retain control_nine as like-1.0-galsbi fiducial

Owner names the discussed factorized NLL-plus-response control as the current
fiducial `like-1.0-galsbi`. Add [model guide](LIKE_1_0_GALSBI.md#decision-and-identity)
and `configs/models_like_1_0_galsbi.json`; register named paths and
`sbsi.models.CURRENT_FIDUCIAL`. Exact control/nine.pt SHA9bc10a6548e79,
GalSBI C8ad5c05d2 and disk emulatorcd32781d plus metadata are preserved.
Checkpoint identity agrees with the paired-seed stage report; checkpoint
metadata confirms arm=control/use_score=false,40 additional NLL epochs and64
response updates at weight0.001. Keep initial140-epoch history and original
training measurements explicit. Pin source/evidence hashes and the four-scene
m1=+0.38554 ±0.39755%,m2=+0.04296 ±0.66616%,RMS=0.53431 ±0.25735%.

Update AGENTS, README, API and repository guide. Do not promote the research
factorized loader to the generic measurement interface, change likelihood or
estimator defaults, relabel caches or restart held evaluations. Historical
model identities and all weights remain unchanged. Add regression coverage
that the named fiducial is control_nine rather than neighbour_nine and that
published evidence, companion pins and nonacceptance status match.

Validation: all four external artifact SHA256 checks and public
`python -m sbsi validate-model like-1.0-galsbi` pass with explicit external
roots; guide links and Python AST pass. Full suite CPU17834679 completed0:0:
382 passed,3 skipped in74.24s; no regression/snapshot failures. Four-scene
reused development evidence is not independent
acceptance or a robust0.3% result; this exact checkpoint still lacks the full40
paired-seed comparison. No new training or scientific evaluation launched.

## 2026-11-08 — Localize photometry failure to conditional flux passing and crowding

Complete the [distribution audit](evidence/photometry_distribution_audit_20261108/findings.md#main-result)
on500k training/500k validation usable rows, four frozen checkpoints. Original
NLL already overselects low-neighbour-score configurations and underselects
high-score configurations. Latest highest-score joint passing deficit is
-0.8664 ±0.1319pp: radius -0.0591pp, flux -0.5660pp, within-group association
-0.2414pp. Training repeats the pattern. Close-crowding fourth quintile latest
radius bias -0.03587 ±0.00271pix, magnitude +0.00877 ±0.00088 and passing
error -1.2931 ±0.0835pp. Global averages conceal these conditional errors.

Latest selected-reference-B discrepancy0.001570 ±0.000406 versus original
NLL0.002793 ±0.000405; middle intrinsic-size/high-B cells contribute79% and66%
of the respective signed validation errors. Exact cell sums verified. These
are diagnostic B units, not response or m. Both-tail rank failures and inverse
latent cross moment show remaining distribution distortion. Association terms
include context heterogeneity; no per-object copula or unique causal training
mechanism identified. No constant-shear fitting, offsets or new loss launched.

Jobs17833751/17833754/17833774 completed0:0 in3m27s/4m36s/6s. Fixtures,
photo/full draw equality, canonical replay for every model/split, independent
row/case reductions and all decomposition identities passed. Save full JSON,
tables, findings, static PNG/PDF figures, completion hashes and job states.
Both plots visually inspected. No missing context, ties or numerical underflows.
Broad17788286 remains held; automated follow-ups disabled. Next diagnostic:
conditional CDF/quantile curves in localized cells to separate shifts, tail
shape and radius-conditioned flux before choosing a simpler one-run fit.

## 2026-11-08 — Diagnose size/flux marginals, association and conditional tails

Owner requests deeper localization of generated versus measured photometry,
without another training proposal. Add [distribution audit](evidence/photometry_distribution_audit_20261108/rule.md#authorized-question-and-fixed-scope),
source-pinned evaluator/reducer, algebra/rank fixtures and persistent launchers.
Freeze original NLL, weak-nine, neighbour-nine and latest checkpoints. Reuse
exact500k training/500k validation rows and canonical64-draw streams. Save
per-row marginal/joint cut probabilities, moments/quantiles, independent-IID
marginal ranks, inverse-flow latent diagnostics and photometry NLL. Test location,
scatter/tails and exact radius/flux/association decomposition; localize selected
neighbour-score error to25 context cells with exact normalized contributions.

Training-only context-bin boundaries, whole-scene paired errors, both original
simulation legs and half-bank checks retained. IID ranks use correct65-bin null;
antithetic ranks are not assumed uniform. Stable magnitudes retain underflowed
physical flux tails. Report will replay historical canonical masses/score sums
for all arms and check scene reductions against direct row sums. Association
terms include remaining context heterogeneity; no per-object copula, emulator
bias or full shear-error attribution claimed. B is a reference coefficient.

AST/bash checks pass; algebra/unequal-mass/rank fixtures pass in17833751. One
A40-16Q/4CPU persistent evaluator runs both splits; dependent CPU17833754
will report. Sources/models/sample provenance pinned; checkpoint every64 blocks.
No fitting, images, production changes or response corrections. Broad17788286
remains held and automated follow-ups disabled. Next: inspect evaluator startup,
then canonical replay gates and conditional error/localization results.

## 2026-11-08 — Propose one-run factor training with measured-photometry primary regularization

Owner asks for an evidence-based simpler recipe retaining response constraints
and improving conditional photometry without repeating failed auxiliary losses.
Review own-data likelihood/selection errors and executed gradient paths in the
original and photometry-only trainers. Add [proposal](evidence/training_stage_audit_20261108/simpler_recipe_proposal.md#recommendation-and-status):
photometry NLL plus conditional-shape NLL/primary-response guard evaluated at
measured training photometry/cuts, with independent factor optimization/clipping.
Both factors train throughout one fixed run with chosen inputs present from
initialization. No new probability/static-neighbour loss, distillation, constant-
shear targets or extra feature stage. This is a hypothesis, not a launched test.

The old guard can change generated photometry and selection to fit response;
this path persisted with shapes frozen. Own-data response continuation worsened
photometry NLL0.005461nat/row; selected-neighbour discrepancy0.002520→0.003148.
Factor swaps put most useful response movement in conditional shapes. These
support testing the separation, not claiming it will cure original NLL underfit.
Specify measured-pair influence-function target/replay, zero photo gradients,
matched one-run controls, hard-cut/interstitial conditional validation and
frozen-choice coherent evaluation. Preserve primary-only scope; no classifier
or neighbour-only labels are invented. Validate by source inspection and local
link checks; no simulations, training, tests or model/preset changes. Next if
implemented: numerical conditional-target/gradient pilot before a fixed-budget
matched comparison. Constant-shear data remain evaluation-only.

## 2026-11-08 — Restore measured-input localization to the training interpretation

Owner corrects the summary that primary-response regularization sometimes
worsened coherent calibration: measured radius/flux plus membership had made
neighbour predictions and full calibration substantially better. Re-read the
four-way photometry/selection intervention, fixed-cell test, full measured-input
regularized/NLL comparison and updated paired-measurement result. Add this
qualification prominently to the [prior-attempts memo](evidence/training_stage_audit_20261108/prior_attempts.md#measured-input-localization-must-qualify-the-loss-summaries).

The restricted response contrast0.006556 is almost entirely emulator-coefficient
reweighting0.006547; it persists at fixed photometry within cells. This supports
incorrect conditional photometry/scene-selection association, rather than
rejecting the response function from its generated-input aggregate. Latest
measured-input nine-cut RMS0.25525% versus0.63457% supports that priority, while
remaining radius dependence limits any all-cuts correctness claim. Good NLL-only
m may be cancellation; it is insufficient evidence to drop the useful response
constraint. No new runs or numerical calculations; source review only. Next
training design must preserve this localization and evaluation-only use of
constant-shear data while simplifying stages.

## 2026-11-08 — Review prior attempts before proposing training-stage simplification

Owner requests awareness of completed experiments. Re-read Nov6–8 log entries,
Nov7 strategy/coverage reviews, primary and selected-neighbour losses, factor
swaps, frozen-shape trial, both conditional-probability experiments, exact
selected-response accounting and historical photometric derivative audit.
Add [prior-attempts memo](evidence/training_stage_audit_20261108/prior_attempts.md#completed-attempts)
with executed versus merely proposed objectives, measurement-reference limits,
training-label coverage and the owner's constraints. The recent generic
cut-aware suggestion repeated an already tested probability objective; selected
shape supervision was also discussed previously and is not an established fix.

The successful probability continuation depends on its starting checkpoint;
the separate four-way trial improved prediction without improving its original
coherent evaluation. Freezing weights did not preserve selected response.
Full-pipeline targets cannot be inferred from the current primary-only prepared
pairs; all-parent usability labels do exist separately. Constant-shear data stay
evaluation-only. No new runs or numerical claims, no model/preset changes.
Validation: review source-linked findings and check local memo links. Next work
must state the specific difference from prior attempts; a generic new loss or
another stage is not justified by this review alone.

## 2026-11-08 — Single NLL factorized recipe remains competitive after paired measurement correction

Saved-evidence audit17833559 completes0:0 in7s. [Findings](evidence/training_stage_audit_20261108/findings.md#completed-paired-seed-audit)
compare original factorized37c4afc13b04 against latest43673e5d on all40 scenes,
both axes/all nine cuts. Initial NLL-only central m1/m2=−0.50484±0.13105% /
+0.02729±0.16848%; nine-cut RMS0.39999±0.06862%, versus latest0.63457±0.11008%.
Paired RMS change−0.23458±0.14047pp, bootstrap95%[−0.46202,+0.06884]. Smaller
absolute point bias in16/18 entries; two low-radius m1 entries worsen by0.07358
and0.15092pp, so the provisional uniform0.1pp criterion is not passed. No
statistical superiority, independent acceptance or production change claimed.

All112 historical receipts and original measured sums replay; latest new-bias/
jackknife results replay. Independent scalar point/leave-one-out checks pass2016.
Current cache hashes recorded where historical receipts lacked original hashes.
Add full40/all-cut tables, separate matched-four-stage table and interpretation.
Four-stage necessity is unproven; the final loss's incremental benefit does not
show net gain over initial NLL. Simple candidate already exists without new
training. Keep latest reference; no further runs launched. Compression cancelled,
outputs preserved. Next if continuing: assess low-radius tradeoffs/validate the
simple recipe before adding back losses. Broad17788286 remains held.

## 2026-11-08 — Correct simplification scope to the training-stage sequence

Owner clarifies that the concern is unnecessary training continuations/losses,
not network depth. Cancel unfinished minimal-flow training17833512 and dependent
17833513/17833514/17833515; sacct confirms cancellation. Preserve all partial
checkpoints, logs, numerical checks and source pins; no compression result or
production change. Update its jobs/protocol with the scope correction.

Add [training-stage audit](evidence/training_stage_audit_20261108/rule.md#corrected-question),
saved-sum reducer and one-CPU report17833559. Re-evaluate original NLL-only
factorized37c4afc13b04 on full40/both axes/nine cuts against the new paired-seed
reference, plus a separate matched-four-scene table of neighbour/response
continuations. This is distinct from the older joint Cartesian checkpoint.
Verify model/auxiliary/source receipts, old measured sums and latest replay;
independently scalar-check point/leave-one-out response diagonals. AST passes;
report queued. No new GPU prediction or training. The final cut-probability
objective has a demonstrated incremental gain, but four-stage necessity is
unproven. Next: inspect the saved-evidence audit before choosing any shortened
training recipe. Broad17788286 remains held; no automatic follow-up enabled.

## 2026-11-08 — Start controlled compression of the latest factorized flow

Owner asks to trim the latest model while retaining its performance. Add
[predeclared protocol](evidence/minimal_flow_20261108/rule.md#owner-request-and-scope),
source-pinned student assembly/serialization, transport/response distillation,
paired coherent evaluator, scalar-audited report and packed Slurm launchers.
Teacher is unchanged weighted43673e5d (1,379,880 parameters). Three candidates:
remove only extra neighbour score (1,377,320), reduce five to three couplings
in each factor (827,928), or both (826,392). Keep original crowding features,
classifier, secondary-response emulator, physical transforms and standardizers.
No new images, shape measurements, empirical response correction or presets.

Training uses only original training contexts and common antithetic teacher
outputs, matching standardized transport plus both shear-axis finite differences.
Fixed4096 AdamW updates; original disjoint validation rows supply diagnostics,
not gradients or checkpoint selection. First coherent screen uses four existing
scenes/both axes/all nine cuts, with a live teacher replay. Full40-scene paired
comparison is still required before replacing the baseline. Provisional margins
are0.1pp absolute-bias degradation per cut/component and0.05pp RMS degradation,
with paired simultaneous/bootstrap uncertainty checks; not an acceptance claim.

Validation: AST and shell syntax pass. Numerical17833501 caught a wrapper-class
identity mismatch between isolated imports before any training. Preserve its
log/source pins; replace identity check with explicit wrapper structure and
input-index validation. Retry17833508 completes0:0 in9s. All three students pass
NumPy shear replay, inverse/Jacobian consistency, physical support and exact
reload; depth-five no-score assembly exactly matches zero-score teacher outputs.
Uneven-chunk gradient relative errors2.17e-5/1.73e-6/1.62e-6 pass3e-4 gate.

Training17833512_0–2 depends on numerical success; evaluation pilot17833513
depends on all training, packed workers17833514_0–2 depend on coherent replay,
and report17833515 depends on all workers. At most three simultaneous GPUs/
12CPUs, within six-GPU/500CPU limits. Checkpoints every128 updates; fixed source
and data-receipt hashes recorded. Resource detection confirms no login GPU;
all substantial work runs through Slurm. Broad17788286 remains held, dependent
17788351 untouched, automated follow-ups disabled. Next: monitor first training
progress and dependent numerical gates; inspect fidelity and four-scene results
before full40-scene confirmation. No performance-retention result yet.
Startup: all three17833512 workers run on A40-16Q, pass repeated numerical
checks, complete128+updates and save restart states. Startup records preserve
parameter counts, exact model config, gradient checks and progress. Early
transport loss decreases strongly for the truncated students; no coherent
calibration result is available yet. Dependent stages remain gated.

## 2026-11-08 — Complete older joint Cartesian comparison: second component fails the target

All 80 case/axis evaluations and report17825579 complete0:0; six packed GPU
workers have finished. [Findings](evidence/joint_cartesian_ninecut_20261108/findings.md#completed-result-and-interpretation)
and [all nine rows](evidence/joint_cartesian_ninecut_20261108/results.md#both-components-and-all-nine-cuts)
record old joint central m1/m2=+0.65528±0.13286% /+2.73865±0.17291%, versus
latest+0.55576±0.13143% /+0.21011±0.16898%. Both-component nine-cut RMS is
1.80345±0.10883% versus0.63457±0.11008%; paired difference+1.16888±0.06961pp.
The earlier one-component comparison was insufficient to recommend the older
checkpoint. Add its superseding findings link; update completed jobs metadata.

Validation: sacct confirms pilot, both worker arrays and report COMPLETED0:0.
Report validates all80 receipts and saved measured/model sums, then audit.py
passes168 independent scalar point/jackknife checks. SHA256 matches completed
audit receipt. Source review confirms shear-axis contexts/transport and response
column assembly. Historical-to-portable central m1 shift only+0.00384±0.00715pp;
central m2 half-bank difference−0.00056±0.01031pp. No empirical correction or
scientific setting change. Errors condition on reused development scenes and
fitted models; architecture causality and independent acceptance remain untested.
Keep latest baseline; no further runs launched. Broad17788286 remains held,
dependent17788351 untouched, automated follow-ups disabled. Requested evaluation
is complete; no retraining or measurement regeneration needed for this comparison.

## 2026-11-08 — Evaluate the older joint Cartesian flow on both axes and all nine cuts

Owner explicitly requests the complete older-joint comparison using the new
paired-seed measurements. Add [protocol](evidence/joint_cartesian_ninecut_20261108/rule.md#authorized-comparison),
manifest, persistent evaluator, reducer, independent scalar audit and launchers.
Freeze d3c671a23e21, original C8/disk emulator,40 scenes, all nine cuts and
independent per-leg generated/measured selection. No training or shape refits.
Use the same64 portable PCG64 antithetic draws/1024-parent blocks as the latest
flow, preserving per-case/block seeds and adding32-draw half-bank sensitivity.

Historical joint predictions used GPU-specific CUDA draws. Pilot17825569 on
A40-8Q separately replays the old case1000/g1 central result, then evaluates g2;
legacy replay gate1e-6 normalized response, counts roundoff2e-12/2e-7. The new
portable estimate is not forced to equal the old stochastic estimate. Record
both and their difference. First-block pooled-vs-separate emulator gate passes
at6.66e-16. Initial32GiB pilot submission was rejected (8Q nodes have~25.5GiB);
24GiB resubmission starts successfully with unchanged numerical settings.

Cip array17825577_0–2 and inter17825578_3–5 depend on the pilot. Each packs13
case/axis tasks and reuses models; six GPUs/24CPUs total maximum. Report17825579
depends on both arrays and pilot. It will replay the latest model's saved m,
report paired old-joint/latest matrices, m1/m2, nine-cut RMS, cut contrasts,
half-bank sensitivity, and independently audit saved sums. Syntax and unequal-
leg-mass/nonsymmetric-matrix fixtures pass. Frozen source/measurement receipts
and job metadata saved. Broad17788286 remains held; no automated timers.
Legacy g1 pilot completes in520.9s: normalized response replay4.44e-16,
selected-count difference1.75e-10; both pass. Portable-minus-legacy flow/
transport response terms are retained separately rather than forced to agree.
The g2 pilot completes in248.5s; pilot job completes0:0 in13m16s. All six
workers start on three A40-16Q, two P5000 and one V100. Every first-block
pooling gate passes (<=6.67e-16), and all six progress beyond100 parent blocks
without errors. Startup/receipt/source verification is preserved. Original
observing-condition preset is additionally hash pinned. Readable all-nine-cut
results will be written only after the independent audit succeeds. Long
workers now proceed normally (roughly1–2h at observed rates); report remains
gated on both arrays. No automatic follow-up timer is enabled. Next: check
completion and interpret the full two-axis/nine-cut comparison.

## 2026-11-08 — Recheck simpler joint-flow baselines against paired measurements

Owner asks whether an older simple joint flow is good enough. Add small saved-
sum reducer and [findings](evidence/joint_flow_recheck_20261108/findings.md#central-cut-recheck).
On the same40 scenes, independent predicted-usability/generated-selection
radius>3px/mag<25, old joint Cartesian gives new m1=+0.65144±0.13284%, versus
latest factorized+0.55576±0.13143%. Paired difference+0.09568±0.01004pp.
Old joint intrinsic-frame gives−0.95383±0.12902%. No updated m2 or all-nine-cut
claim. Cartesian candidate is the likelihood-only140epoch FS2-rate model,
SHA d3c671a23e21; distinguish it from the retained FS2/V3.9 F8 checkpoint.

Validation: py31 `reduce.py` verifies80 checkpoint/source receipts, original
measured sums and pooled responses, all80 historical per-case biases, and
latest m1 replay. Current cache hashes saved; old receipts lacked data hashes.
Small saved-sum calculation only, no model evaluations/training, catalogue scans,
new jobs or preset changes. Next useful check is the simpler candidate on both
axes and the complete nine-cut target grid before choosing another training run.

## 2026-11-08 — Measured-input substitution reduces RMS but leaves radius-dependent response error

Complete all80 paired-input evaluations and CPU report17824974 (0:0,3s),
with168 independent scalar point/jackknife checks. New measured photometry/
actual membership reduces nine-cut RMS0.63457→0.25525%; paired change
−0.37932±0.17053pp, bootstrap95% interval[−0.62268,−0.03180]. Central m1/m2
become−0.13281±0.12329% /−0.41639±0.16471%, from+0.55576/+0.21011%.
The central mean is−0.27460±0.11311%; the input substitution overshoots zero.
[Completed interpretation](evidence/paired_measured_photometry_20261108/findings.md#completed-result-and-interpretation).

At magnitude<25, radius3.0→3.2px still shifts oracle m1/m2 by
+0.26997±0.09676 /+0.45970±0.09411pp. Thus size-dependent response error
survives actual inputs; generated photometry/classifier membership cannot be
a complete explanation within the fixed conditional-response composition.
No unique attribution to shapes, neighbour transport or measurement is made.
Actual-input RMS below0.3% is not a production or acceptance result; its
bootstrap95% interval reaches0.45306%. Retain all40 as the primary cohort,
including a noisy four-scene cohort with opposite RMS change.

Pilot full replays differ<=2.19e-9 against gate1e-6. All source/data/model
identities, exact selected counts, old/new measured sums and half-bank
identities pass. Four packed GPU jobs finish in45–71s, no active follow-up
jobs remain. Update findings/jobs/source receipts and prior next-step pointer.
Next direct diagnostic: paired-seed primary-only versus neighbour-only measured
responses on matched populations near the radius cuts; not launched this turn.
No model changes, broad-array resume, empirical correction or automated timers.

## 2026-11-08 — Launch matched measured-input test with new usable membership

Owner requests continuation. Add [paired-input protocol](evidence/paired_measured_photometry_20261108/rule.md#question-and-estimator),
preparation, evaluator, manifest, pinned dependencies, packed Slurm launchers,
report and independent scalar audit. Photometry is unchanged by the shape
refits; the oracle model expectation depends on photometry and truth context,
not measured shape. Recompute signed contributions of6,218 changed leg
memberships and add them to pinned old oracle sums. Measured shape sums are
fully replaced by the completed new measurement. No empirical adjustment.

CPU preparation17824947 completes0:0 in39s, replaying old/new selected sums,
source/hash/coverage receipts and old oracle sums for40 scenes. Preserve full
parent1024-row latent blocks,64 antithetic draws, original seeds, complete
truth neighbours, conditional shape weights, disk transport and all nine cuts.
Pilot17824967 completes0:0: independently evaluate full old/new target-domain
expectations at case1000/axis1 and1129/axis2. Old-cache replay differs by at most
2.19e-9 in normalized response; full-new vs signed-update differs by at most
2.17e-9 (gate1e-6). Counts exactly match. Model pair conditional equality,
reconstruction, feature grouping and half-bank identities pass.

Array17824973 has four persistent GPUs,19/20 case-axis tasks each, gated on the
pilot; report17824974 waits for both. No other study GPU work is active; remain
below six GPUs and500CPUs. Broad17788286 held and automated timers disabled.
Syntax checks pass. Next verify worker progress, complete full40 paired matrix/
bias/cut-contrast report and independent saved-sum audit. No production changes
or independent acceptance claim.

## 2026-11-08 — Cleaner measurements retain photometry gain and resolve radius-cut sensitivity

CPU17824924 completes0:0 in4s. Against new independently selected measurements,
frozen control→probability-loss central m1 changes0.67065→0.55576%, m2 changes
0.32372→0.21011%; paired shifts−0.11488±0.00332 /−0.11361±0.00317pp. Absolute
pooled bias improves at9/9 cuts on each axis. Joint nine-cut RMS changes
0.75735→0.63457%, paired−0.12278±0.00515pp. Average central bias remains
+0.38294±0.12187%. This is a useful partial photometry gain, not model acceptance.

At magnitude<25, radius3.0→3.2px raises m1/m2 by0.36999±0.12962 /
0.51807±0.09306pp. Measured response increases0.053356/0.054598 whereas model
increases0.049568/0.049572. Same direction at the adjacent magnitude cuts;
correlated checks do not supply independent replications or unique component
attribution. [Interpretation and next test](evidence/selected_seed_calibration_20261108/findings.md#frozen-training-comparison-and-radius-dependence).

Verify80 control cache/source/checkpoint receipts and historical control point/
jackknife replay; independent direct-sum audit verifies60 new contrast/mean/
half-difference quantities. Update completed jobs/findings, add supersession
notes to historical replication and measured-photometry findings while retaining
all original values. Next prioritize a new-outcome matched input substitution
at all nine cuts; a second paired seed bank would assess fitter robustness.
Neither follow-up launched in this report. No active study jobs, model changes,
broad-array resume or automated timers.

## 2026-11-08 — Independently selected paired-seed calibration completes; positive residual remains

All 640 extra chunks and report17824713 complete 0:0. Restore independent
per-leg selection over all nine target cuts, including recovered old-invalid
fits, and compare with the unchanged weighted model. Central m1/m2 become
+0.55576±0.13143% / +0.21011±0.16898%; old +0.80810±0.31022% /−0.33352±0.24776%.
New measured matrix is [[0.904867,0.000324], [−0.000239,0.902380]]. Measured
R11−R22 drops0.009653→0.002487, paired change−0.007166±0.003504. New component
bias difference0.34565±0.17957pp is not compelling evidence of unequal biases,
but neither component is forced equal. Nine-cut joint RMS is0.63457%; change
from old measurement+0.02934±0.21586pp. Tighter radius >3.2px increases both
positive biases. See [completed findings](evidence/selected_seed_calibration_20261108/findings.md#completed-result).

All source/hash/coverage/repeat gates and original sum/bias replay pass;
independent scalar audit reproduces126 point/jackknife checks. Retain3,189
old-invalid→new-valid and3,029 reverse leg outcomes within the loose domain.
Add comparison of the frozen training pair and predeclared adjacent-cut
contrasts (`compare_frozen.py`, CPU17824924); reuse generated/predicted-mode
caches only, preserving covariance and independently normalizing legs. Syntax
passes. No re-training, fresh acceptance claim, GPU use or broad-array resume.
Next: complete the short paired analysis and use it to prioritize localization.

## 2026-11-08 — Complete paired-seed40 and launch independently selected calibration

All2,667 common-population chunks and report17821860 complete0:0. On5,411,332
identical valid IDs, diagonal gap changes0.008625±0.003183 to0.002121±0.001439;
paired change−0.006504±0.003283. Off-diagonal gap becomes0.000119±0.001671.
[Completed evidence](evidence/measured_seed_response_20261107/completed_20261108.md#full40-result)
supersedes partial cohorts. Audit verifies all receipts and reproduces36
statistics. This is evidence of measurement sensitivity, not proof of model
calibration or independent acceptance.

Add BlendEMU `scripts/diagnose_selected_seed_calibration.py` and SBSI
[protocol](evidence/selected_seed_calibration_20261108/rule.md), manifest,
packed launcher, reduction script and estimator fixtures. Restore independent
per-leg selection at all nine target cuts; candidates depend on photometry,
not old shape validity, so recovered fits remain eligible. Reuse5,412,739 IDs;
fit1,248,893 additional IDs (4,411,805 eligible leg fits),640 chunks. Frozen
weighted model predictions with generated photometry/predicted usability are
reused, with source/data/checkpoint verification. Do not reuse oracle validity
modes. Original independently selected sums replay for all40 scenes andninecuts.
Fixtures verify nonsymmetric matrix orientation, unequal leg normalization,
independent cuts and recovery of old-invalid shapes; Python/shell syntax pass.
Preparation17824672 andpilot17824684 complete0:0;223/223 pilot fits valid,
exact repeat passes. Array17824712 packs12 allocations×40CPUs=480 maximum,
under owner's500 ceiling; every group has48–64 tasks. Report17824713 depends
on full array success and must replay original calibrated point/error values.
No GPU allocations, changes to broad-held17788286, or automatic timers.
First completed chunks pass source/hash/repeat checks. Reduce only pending
groups6–11 memory request120GiB→64GiB using preceding runs (~5.2GiB MaxRSS),
without changing40CPUs/group or scientific settings; group6 then starts.
Add a scheduled independent scalar audit of all matrices and m point/jackknife
values from compact saved sums; it runs at the end of the report stage.
Next: finish worker/report monitoring and report independently selected m1/m2
and nine cuts. Startup/resource receipts are preserved.

## 2026-11-08 — Expand paired-seed measurement to284 active CPUs under owner500-CPU ceiling

Owner explicitly requests increased CPU use, up to500. Record that authority
in AGENTS.md and the [measurement protocol](evidence/measured_seed_response_20261107/rule.md#owner-cpu-expansion-2026-11-08).
Add BlendEMU `run_seed_response_packed.py`, a scheduling-only wrapper around
the unchanged pinned measurement task, and an eight-allocation launcher.
Stop recovery17824227_1–2 after preserving1,844 completed chunks; no orphan
outputs require quarantine. Freeze a disjoint complete schedule for the823
remaining chunks, grouped by case across eight60-CPU allocations (480 requested
CPUs total, below500). Each group has67–134 chunks and uses persistent workers.

Submit17824404_0–7 and redirect final report17821860 to that array. Preserve
original model/source identities, fitting rule,2048-ID numerical chunks,
per-object seeds, pilot gate and restart receipts. Fast-I/O x86-64-v3 nodes only;
exclude the two previously slow Naples hosts. Python/shell syntax checks and
schedule uniqueness/coverage/source-pin checks pass. The scheduler estimates a
two-hour wait for seven60-CPU allocations, so cancel only those pending tasks
and retain running17824404_0. Re-submit unchanged groups1–7 as17824424 with32
CPUs each; all start promptly. Actual allocated concurrency is284 CPUs across
eight allocations, with the500-CPU ceiling retained. Final report now depends
on17824404_0 and17824424; source/task schedules remain unchanged. Startup
allocation records are preserved. All eight allocations start; first packed
checkpoints pass original source-pin, data-hash, ID and exact-repeat checks.
The broad17788286 remains held; no GPU increase or automatic follow-up timers.

## 2026-11-08 — Completed20-scene paired-seed response has no resolved diagonal gap

Owner asks what finished portions already show. Add the
[interim response analysis](evidence/measured_seed_response_20261107/interim_20261108.md#what-the-completed-scenes-say)
on whole completed scenes, excluding partially finished scenes. Main cohort:
20 scenes from completed worker groups0/3,2,707,084 identical valid IDs in both
old/new measurements. R11−R22 changes+0.006823±0.003881 to+0.001031±0.002009;
paired change−0.005792±0.004374 (1.32SE). New gap is0.51SE from zero, and its
jackknife error is roughly halved. R12−R21 changes−0.008200±0.004348 to
−0.000690±0.001865. This supports optimizer-noise sensitivity without proving
it explains the full40-scene directional gap or fixes model calibration.

681 new-invalid IDs (0.0252%) are excluded from both sides; their removal shifts
the old gap by only−0.00000489. Supplemental22 complete scenes give new gap
+0.000991±0.001883 and paired change−0.005272±0.004103. Preserve the frozen
snapshot and distinguish these reused, hardware-assigned cohorts from full40
or independent acceptance. CPU17824200 completes reduction; all source pins,
checksums, ID coverage and repeat receipts pass. Independent saved-sum audit
reproduces96 reported statistics exactly and matches all22 original sums to
the earlier membership audit.

Original worker groups0/3 complete in75m06s/54m46s. Slow group1 preserves188
chunks after2h53m, while group2 has none. Cancel only those two workers;
no orphan output files require quarantine. Restart missing chunks with the
unchanged measurement code, cases and seeds as17824227_1–2 (32CPUs each,
fast-io+x86-64-v3 constraint, excluding the slow Naples nodes). Both have begun
normal fitting on new hosts. Point final report17821860 at the recovery array;
full40 result remains pending. Add recovery/source/startup receipts. Broad
17788286 remains held, no GPUs used, automated follow-ups stay disabled.

## 2026-11-07 — Localize measurement instability to galaxy fitting; launch precise seed-paired repeat

Complete the [controlled replay](evidence/measured_fit_replay_20261107/findings.md#completed-four-scene-result):
2,048 IDs×four exposures×three modes, job17821683 completes0:0 in9m42s.
Changing only optimizer seed changes a component by>0.1 in1,042/8,189 valid
fit pairs (12.72%); tighter convergence changes323/8,165 (3.96%). All returned
valid shapes report success;30 calls fail to return shapes, leaving2,036 IDs
valid across all modes/legs. The small subset cannot attribute the full40-scene
diagonal gap to this instability: tight-minus-default gap change−0.05547±0.05378
conditional-row error (four-scene jackknife±0.04119).

The [catalogue localization](evidence/measured_gap_localization_20261107/findings.md#intrinsic-orientations-do-not-predict-the-gap)
finds ideal intrinsic diagonal gap−0.0000799±0.0001145 versus measured common-ID
+0.0086336±0.0031973. Large response excursions dominate uncertainty but do not
exclusively carry the gap. All six group partitions conserve original sums,
maximum sum roundoff3.71e-9; zero unmatched cross rows and228,683 unmatched shape
rows excluded. Job17821702 completes0:0 in1m59s. Eight isolated noiseless ring
controls have max|diagonal gap|0.0002923 (17821716,1m34s), so these simple
profiles do not reproduce the measured gap.

The enriched64-image continuation test (17821777,38s) repeats128 live default
fits exactly. PSF parameter variation is≤5.6e-17; galaxy optimizer solutions
vary. Tight fits still differ by>0.1 for19/32 deliberately unstable cases.
Early stopping contributes but is insufficient as an explanation or remedy.
Multistart diagnostic17821808 completes1m56s: independent best-of16 banks still
disagree in1/32 unstable and2/32 previously stable examples, so no replacement
measurement rule is promoted. Store flags, objectives and raw trials. Earlier
diagnostic17821736/43 fail cross-node replay (up to0.1273 difference);17821766
exposes an invalid continuation requiring explicit missing-shape handling;
pending17821749 is cancelled. Failures and the revised within-job control are
preserved in the protocol. No production source or catalogue changes.

Launch the [full paired-seed measurement repeat](evidence/measured_seed_response_20261107/rule.md#purpose)
to determine whether the directional gap survives with lower optimizer sampling
noise. Preserve the original measurement rule and5,412,739 common IDs across40
scenes; no model evaluation or rerendering. Preparation17821847 completes1m57s;
all40 data hashes and the original common matrix replay. Pilot17821859 gates
four persistent32-CPU workers17821856_0–3 (128CPU total, no GPUs), then report
17821860. Chunks are restartable and original/new comparisons use exactly the
same surviving IDs with exclusions reported. An earlier pending pilot17821851
was replaced to include the prepared-data verification gate. Estimator fixture
checks nonsymmetric matrices, unequal masses and batch/model axes. Original
broad17788286 remains held and automated follow-ups remain disabled. Full
response results remain pending; no claim of an identified absolute bias cause.
Pilot returns256/256 valid shapes with exact repeat. All four worker allocations
start; early case1000/1120 checkpoints pass ID, data-hash, source-pin and validity
checks. Two Naples nodes encounter cold-image I/O stalls; read-only sequential
warming on their existing allocations completes for the first case, without
changing data or increasing allocated CPUs. Node records and startup receipts
are retained; later checkpoints and the dependency report remain pending.

## 2026-11-07 — Start controlled measurement replay and gap localization

Owner asks to identify why measured diagonal responses differ. Add the
[fixed-image replay protocol](evidence/measured_fit_replay_20261107/rule.md#question-and-fixed-sample),
SBSI reduction/report scripts and two isolated BlendEMU diagnostic scripts:
`diagnose_constant_fit_replay.py` and `diagnose_constant_orientation_ring.py`.
No production measurement source is modified. Replay2048 common-selected IDs
across four scenes/four legs with two recorded RNG seeds and tighter LM
convergence, preserving images, PSFs, photometry and membership. Record fitter
flags that the original shape catalogue does not retain. Exact repeatability
and first-scene validity gate precede the remaining packed workers.

CPU17821683 (4CPUs/32GB) has started normally. CPU17821702 (2CPUs/16GB)
localizes the full40-scene common-ID gap by measured size, brightness, response
excursion, intrinsic ellipticity and subpixel position, comparing exact ideal
intrinsic shear response. CPU17821716 (2CPUs/8GB) isolates pixel/fitter orientation
effects using noiseless uniform-angle Sersic rings. Interpretation must retain
the population differences and unstable four-scene sampling errors; these are
exploratory diagnostics, not acceptance or empirical corrections. Syntax checks
pass; results pending. Existing broad array remains held; no GPUs or timers.

## 2026-11-07 — Measured diagonal response gap survives common-galaxy selection

Owner prioritizes measured-response symmetry. Add the
[measured symmetry audit](evidence/measured_response_symmetry_20261107/findings.md#conclusion)
and packed same-ID catalogue reduction, all-cut reports, source/config/PSF
receipts and estimator fixture. Full40 central R12−R21 is−0.002953±0.003622,
unresolved; R11−R22 is+0.009653±0.003571. The full Cartesian model does not
force either transpose symmetry or equal diagonals. Its measured-input
R12−R21=−0.002972 nearly matches the measured value.

Keeping5,412,739 identical selected IDs across all four exposures leaves
R11−R22=+0.008634±0.003197; paired change+0.001019±0.001222. Thus changing
membership does not account for most of the gap in this diagnostic. This
changed population is not a replacement estimator or causal error allocation.
All160 PSF arrays match exactly,80 paired configs differ only in paths, and
original source hashes/truth receipts pass. Unseeded fitter initialization
remains an uncertainty. PSF stamp/Jacobian centers differ by0.5px, but installed
ngmix convolution subtracts the fitted PSF centroid; no centroid bug established.

CPU17821487 completes0:0 in14s,17821503 in5m03s. Known nonsymmetric response
fixture verifies orientation and contrasts; every original catalogue sum
replays, common counts match, and accounting closes for all resamples.
228,683 unmatched shape rows are excluded; zero crossmatch rows lack shapes.
Next discriminating work: controlled fixed-seed fit replay and rotated-scene
measurement check. Neither performed here; no unique cause established.
No training, rerendering, fitter changes or broad-array restart; timers stay off.

## 2026-11-07 — Current measured-input test completes; residual component difference survives

All80 case/axis evaluations and report17821115 complete0:0 (pilot17821126,
workers17821114_0–5). At the central cut, all40 actual-input biases are
+0.11693±0.31325% /−0.95552±0.24107%. With actual usability in both comparisons,
the extra-loss endpoint's generated-input values were+1.00553% /−0.15975%.
Measured size/flux/membership shift them−0.88860±0.03704 /−0.79577±0.02703pp;
half the component difference remains+0.53622±0.19925pp. All nine measured-input
m2 estimates are negative. Joint RMS is0.58681%; its change from normal predicted-
usability extra-loss output is−0.01842±0.21708pp, not a resolved general repair.

Add [completed interpretation](evidence/current_measured_photometry_20261107/findings.md#completed-result-and-decision),
all-cut table and independent direct-sum audit; update jobs and review status.
Audit reproduces15,330 scalar values/paired/jackknife errors within8.89e-14,
verifies240 output hashes, retains matching totals and adds labelled historical
comparisons. Its antisymmetric diagnostic sign was initially reversed; matching
the original convention closes verification without changing scientific outputs.
Crossmatch-without-shape count0;228,683 unmatched shape rows excluded. Half-bank
m1/m2 differences+0.00888/+0.00032pp are small relative to the remaining bias.

Current-minus-original-nine-weak measured-input half difference grows
+0.17102±0.00083pp, but these are different training branches: current shapes
descend from neighbour-NLL12ca7856 through control6fc0c373, not from888fb044.
The fixed emulator's contribution changes with its input shapes. Do not assign
this comparison to the latest probability loss or infer a unique physical cause.
Next: directly check zero-shear conditional shapes at measured photometry.
No production changes, independent acceptance or broad-array restart.

## 2026-11-07 — Direct zero-shear shape check finds small width errors, no resolved quadrupole error

Add and complete the [fixed conditional-shape check](evidence/conditional_shape_check_20261107/findings.md#completed-result).
Use249,433 g0 rows from the existing500k validation sample across40 scenes;
119,008 pass the central cut. Four earlier/current frozen models share actual
photometry/membership and original64-draw streams. Current errors in mean e1²/e2²
are−0.0009583±0.0003962 /−0.0012653±0.0003731; their difference is
+0.0003071±0.0006186. Lower-radius-cut moment deficits are clearer, while the
quadrupole error is unresolved across all nine cuts. These are raw shape moments,
not shear-bias units or an identified explanation of the component gap.

Save per-case observed/predicted moments, independent-bank conditional moment
scores, half-bank differences and all model contrasts. GPU17821432 completes
0:0 in37s, packing four models and reduction in one allocation. Input/target
state checks, exact measured photometry, conditional reconstruction and scoring
fixture pass. Independent small-sum audit reproduces2,160 statistics exactly,
replays prior observed selected counts/shape sums and verifies quadrupole
identities. No training or measurement changes. A static width/quadrupole penalty
is not justified as the full calibration remedy; next hypothesis concerns
conditional shear response and neighbour composition with compatible targets.
Broad17788286 remains held; automated follow-ups remain off.

## 2026-11-07 — Reflect across the investigation; quantify common and component-dependent changes

Owner asks to reflect before continuing. Add the [complete-chain review](evidence/response_strategy_review_20261107/findings.md#decision-after-reviewing-the-complete-chain),
fixed joint-axis audit, source/target-coverage receipts and completed report;
mark the earlier strategy memo superseded. On all40 scenes, probability loss
changes average central bias0.35137→0.23729%, paired−0.11408±0.00262pp, while
half the component difference stays0.57190→0.57081pp, change−0.00109±0.00190pp.
Joint nine-cut RMS improves0.68902→0.60523%, change−0.08378±0.02554pp; additional36
alone improve0.69768→0.62503%, change−0.07265±0.02935pp. This is a useful partial
improvement, not every-component calibration or independent acceptance.

Separately, the original NLL→nine-weak measured-input intervention changes half
the component difference by+0.26296±0.00218pp. Do not equate that earlier pair
with current neighbour-trained checkpoints. The target audit verifies160 train
scenes/23,279,984 usable response pairs; the existing guard uses fixed measured
zero-leg normalization, excludes unmatched usable counterparts, and omits the
classifier and disk transport. All-parent usable labels do exist separately.
Do not describe another loss on those matched pairs as full coherent supervision.

Validation: CPU17821064 completed0:0 in11s,240 output files checked,1,120 saved
values/jackknife SEs replayed; unequal-count and zero-error fixtures pass;
prepared-domain/guard source hashes match. Full axes/cuts/covariance retained.
No new training or model changes. Next: close the current-pair measured-input
gap before selecting another objective. Broad17788286 held; follow-ups off.

## 2026-11-07 — Launch exact-current-checkpoint measured-photometry diagnostic

Add [fixed protocol](evidence/current_measured_photometry_20261107/rule.md#question-and-fixed-models),
source-pinned persistent evaluator, paired joint-axis report and launchers for
all40 scenes/both axes. Replace generated photometry and membership with the
actual per-leg measurements while retaining current conditional shapes and
disk transport. Prove current control/weighted nonphotometry states and
conditioning identical; require exact conditional predictions and pilot full-
field equality, then evaluate their shared measured-input prediction once.
Pilot also replays the earlier original-nine-weak measured-input field at the
unchanged1e-6 normalized response gate. Preserve64 draws,1024 blocks,seeds,cuts,
feature checks,matching receipts,half-bank checks and canonical disk boundary.

Validation: AST/bash and exact80-task coverage pass. First pilot17821113 failed
before prediction because JSON cannot serialize NumPy conditioning arrays.
Replace serialization with exact recursive array comparison; equal/unequal
nested-state fixture passes. Replacement pilot17821126 gates existing six-worker
array17821114 and report17821115; dependencies updated rather than duplicating
workers. Maximum six GPUs/24 CPUs; all six workers pack13–14 tasks and reuse
models/emulator. Replacement pilot completes0:0 in74s: all61 nonphotometry
tensors and conditions match; full-field conditional endpoint aggregates are
exactly equal; original-model normalized response replay difference0; feature
check max2.22e-16. All six workers pass startup and complete their first case.
No training, new images, production
changes or broad-array restart. Next: inspect worker startup, then compare
current measured-input residuals with both generated arms.

## 2026-11-07 — Complete 36-scene frozen-model replication: downward shift repeats, two-component improvement does not

Pilot17817800, six packed workers17817802_0–5 and report17817803 completed0:0.
The additional36 scenes give central predicted-usability m1: +0.9436→+0.8271%,
m2: −0.3689→−0.4817%; paired changes −0.1165±0.0035 /−0.1128±0.0033pp.
Absolute new-model jackknife SEs are0.3378/0.2598pp. Every additional scene
shifts downward on both axes, but absolute pooled bias improves at9/9 cuts
for component1 and3/9 for component2. Nine-cut RMS changes0.9603→0.8314%
and0.2266→0.3003%; the latter change is not resolved by scene uncertainty.
All40 central biases become+0.8081±0.3102%/−0.3335±0.2478%.

Replace stale startup findings/jobs with the [completed interpretation](evidence/conditional_cut_replication_20261107/findings.md#completed-result).
Add independent direct-sum audit, matching totals and all-nine-cut tables.
Validation: py31 audit_completed.py verifies160 files and pinned identities,
reproduces20,268 point/paired/jackknife scalar statistics within6.58e-14 in3.4s;
original-field pilot response replay difference0. Exclude228,683 unmatched
shape rows; crossmatch-without-shape count0 (totals across40 scenes/both axes/
both legs, deduplicated across model arms). Source and syntax checks pass.

The better conditional passing-probability score remains valid, but frozen
shape weights do not preserve selected response. Do not promote this endpoint
as a general calibration fix. Keep the other four-way experiment's original
888fb044 start distinct from this pair's neighbour-trained6fc0c373 start.
Next hypothesis: joint selected-shape response and passing-mass supervision,
with training-only targets and full normalized response validation. No new
training/images, production preset change or independent acceptance claim;
broad17788286 stays held and automated follow-ups remain off.

## 2026-11-07 — Selected-response localization completes; mass growth outweighs numerator gains

All frozen-model evaluations/report17818806,17818807_0–2,17818808 complete0:0. Add [completed findings](evidence/photometry_response_localization_20261107/findings.md#main-finding), interpretation/audit script, independent population-composition accounting and exported figures; update jobs/status. At the central cut extra loss changes ΔR1/ΔR2 by−0.0006224/−0.0007545. Membership contributes+0.001726/+0.001577, output values+0.000920/+0.000925, normalization−0.003268/−0.003257. Effective selected masses grow0.35–0.38% relatively. Negative size contributions concentrate at true radius0.25–0.7 arcsec; low-neighbour quartiles contribute negatively and Q4 positively. Radius-only and flux-only membership contributions are comparable.

Because common normalization can obscure interpretation, independently decompose selected group fractions versus within-group means. For20 fixed size×influence cells, composition contributes−0.000237/−0.000232 and within-cell changes−0.000385/−0.000522. Independently normalized responses decrease in Q1–Q3 and increase in Q4 on both axes. These are coupled, partition-dependent descriptions of the intervention, not causal sources of all original error. Neighbour-input-only gains in Q4 cancel lower-quartile decreases; no central calibration fix.

Validation: original aggregate replay gates pass; total-response contrast replay max1.87e-12. CPU17820925 completes0:0 in9s, independently reproducing1,512 values/SEs and both attribution closure identities. PNGs visually inspected; PNG/PDF, detailed JSON, findings and updated completion status exported to chat outputs/response_localization. No new training/images or broad restart;17788286 remains held. Next question: test conditional selected-shape response alongside passing mass, while retaining density checks and global normalized-response validation; current result remains four-scene development evidence.

## 2026-11-07 — Localize the four-way selected-response changes with exact paired accounting

Owner asks to continue after the failed shear-calibration improvement. Add the [localization protocol](evidence/photometry_response_localization_20261107/rule.md#authorized-question), evaluator, decomposition, analytic fixtures and report/plots. Reuse all four frozen endpoints, same four coherent scenes/both axes/nine cuts,64 portable draws and1024-parent blocks. Split new-minus-control response into symmetric cut-membership, output-value and normalization terms. Use full per-leg selected mass, retaining exact additive contributions across five true-size bins × four within-scene truth-neighbour-score quartiles and radius/flux/joint cut-state switches. This is common-draw accounting, not causal identification or subgroup-normalized response.

Validation: AST/bash pass; CPU17818793 completes0:0 with event assignment, numerator identity, arm-swap antisymmetry, unequal normalization and zero-change tests. Pilot17818806 gates saved four-arm aggregate replay and float64 diagnostic-response agreement≤1e-7; dependent three-worker array17818807 packs remaining seven tasks, CPU17818808 exports all-cut/component tables and plots. At submission, three older replication workers remain active; at most three new workers preserve the shared six-GPU ceiling. No new training/images, no broader evaluation restart;17788286 stays held. Sources pinned; large sums stay in project cache photometry_response_localization_20261107. Protocol/startup snapshot exported to chat outputs/response_localization. Pilot startup: all four models load on RTX2080Ti; reference-score check and pooled-emulator comparison (3.33e-16 max) pass; first diagnostic block completes. Full-scene replay/closure and remaining tasks are pending. Next: inspect pilot completion and dependent evaluations before interpreting locations.

## 2026-11-07 — Complete four-way flow test: conditional prediction improves, coherent bias does not

All fixed step64 endpoints, own-data checks, four scenes/both axes and reporting completed0:0 (17817874,17817875_0–2,17817876). Add [completed interpretation](evidence/photometry_learning_information_20261107/findings.md#completed-comparison-better-probabilities-no-shear-calibration-improvement), independent small-sum audit and summary/all-cut plots; update jobs and chat export status. Extra loss reduces validation mean nine-cut probability score by0.00011087±0.00000655, added input by0.00001088±0.00000187, both by0.00013724±0.00000785. Nevertheless, central predicted-usability m1/m2 increase0.06972±0.00494 /0.08538±0.00481pp with extra loss and0.06439±0.00746 /0.08563±0.00907pp with both. Added input alone changes central biases by−0.00502±0.00592 /−0.00227±0.00587pp. Nine-cut RMS does not improve. Input-only photometry NLL also worsens slightly despite better cut scores.

The extra-loss flow response decrease exceeds the opposing emulator increase. Fixed shape weights do not fix selected-shape response. Measured-membership diagnostics reverse the bias-change sign but retain generated model photometry and have larger absolute bias; they are not a measured-input solution. The earlier successful conditional-loss pair began from an already neighbour-trained checkpoint, so its opposite outcome is not a matched contradiction and cannot be pooled with this comparison. Four reused fields and64 updates do not establish independent acceptance, convergence or absence of useful neighbour information.

Validation: all original replay/freeze/reload gates and scheduler receipts pass. CPU17818561 completed0:0 in7s; independent direct reduction of saved sums reproduces8,160 value/SE entries across arms, cuts, axes, response terms, paired interactions and conditional scores. Summary PNG visually inspected; PNG/PDF plots and findings exported to the requesting chat's outputs/flow_experiment. Resource17817944 restored replication17817802 to six workers; broad17788286 remains held. No further training launched. Next scientific question: localize the selected-shape response change by cut boundary and neighbour environment before choosing another objective.

## 2026-11-07 — Launch matched four-way photometry learning and neighbour-information experiment

Owner asks to improve conditional learning while testing richer neighbour inputs.
Add the [four-way protocol](evidence/photometry_learning_information_20261107/rule.md#owner-authorized-four-way-comparison), source-pinned training/evaluation/reporting and scheduler launchers. All four arms start from the original nine-weak checkpoint888fb044 (not the previously neighbour-trained conditional-cut baseline). Zero-weight input augmentation exactly preserves initial density and physical draws. Cross extra neighbour score off/on with conditional-cut loss off/on; freeze all shape parameters and input-off extra weights. Use matched64 updates, rows/latent streams and training-only auxiliary weight66.86637. Report all four contrasts and interaction on own-data conditional scores and four coherent scenes/both axes/all nine cuts. No endpoint selection from outcomes.

Validation: AST/bash checks; CPU17817862 completed0:0, passing independent-bank loss/gradient, unequal-case reducers, factorial contrasts and input-freezing checks. GPU17817874 passed exact original-model replay and real-model gradients, reaching control step32/64. It packs all training, own-data evaluation and first coherent pilot. Dependent array17817875 packs seven remaining tasks into three workers; CPU17817876 exports reports. Large outputs are under the project cache `photometry_learning_information_20261107`; protocol and startup snapshot exported to the requesting chat's `outputs/flow_experiment/`.

Concurrent replication17817802 has six pending workers. Temporarily throttle it to3 so its workers plus this experiment stay within six GPUs. Resource-only CPU17817944 restores throttle6 afterany17817875. No active model sources or scientific settings of the replication changed. If an upstream failure leaves dependent arrays blocked, inspect dependencies and release unused slots. Broad17788286 remains held. Results remain pending; this bounded continuation and reused four-scene assessment are not convergence or independent acceptance tests. Next: inspect training/own-data/pilot completion and dependent evaluation/report gates.

## 2026-11-07 — Extend the frozen cut-probability comparison to the remaining36 scenes

Owner requests continuation after the completed four-field improvement.
Freeze the same two step64 endpoints and evaluate the remaining36 existing
constant-shear scenes on both axes. Add the [replication protocol](evidence/conditional_cut_replication_20261107/rule.md#authorized-question-and-fixed-models),
source-pinned evaluator/report, gate and packed launchers. Keep all nine target
cuts, original64 portable draws/1024-parent blocks, classifier/emulator,
per-leg normalization and seeds. Independently check the existing static
neighbour feature on32 IDs per field; preserve original feature checks where
available, and explicitly mark the four-field-only association diagnostic
unavailable elsewhere. Report original4/additional36/all40 separately.

Validation: all280 input/cache/verification prerequisites present; AST/bash,
imports and checkpoint/reference hashes pass. Reducer reproduces1176 saved
statistics exactly and passes unequal-count normalization; all72 new tasks
covered exactly once. Pilot17817800 starts with old-field replay then first
new-field evaluation; it gates six persistent workers17817802 and report17817803.
At most six GPUs/24 CPUs for this chain; each worker packs11--12 tasks with
both models loaded. No new training or images, no independent acceptance claim,
broad17788286 held, automated follow-ups off. Pilot has reached block100 on
A40-16Q after its feature gate and pooled-emulator check (max2.22e-16); full
known-field replay and first new field remain pending. Original classifier/
emulator identities and preset/helper hashes are verified. Next: inspect the
pilot completion and dependent workers, then the separate36-scene result.

## 2026-11-07 — Conditional cut-probability loss improves all nine pooled cuts on both axes

All planned stages completed0:0:17817062,17817064_0--2 and17817065. The fixed
probability-loss endpoint improves held-out hard-cut score by0.00014078
±0.00000887 and central predicted-usability m1/m2 by0.1033±0.0084 /
0.1146±0.0140 percentage points relative to the matched frozen-shape control.
New central biases are+0.6377%/+1.0025%, with absolute four-field jackknife
SE0.7165/0.4847pp. All nine pooled absolute biases improve on both axes with
predicted or actual usability. Nine-cut RMS falls1.0501→0.9440% and
0.9860→0.8677%; case1000 individually worsens in RMS on both axes while the
other three fields improve. No independent acceptance or resolved0.3% claim.

Add [completed interpretation](evidence/conditional_cut_probability_20261107/findings.md#completed-comparison-a-modest-improvement-across-the-nine-cuts),
all-cut/field tables and small-sum audit. The audit reproduces216 contrast
values/SEs and endpoint RMS values/SEs. All111 control state tensors match
the previous control exactly; simulated sums/counts/association are exact.
Cross-run raw-sum equality is overly strict under pooling/hardware roundoff;
the unchanged response gate passes with maximum flow difference1.20e-13.
Direct conditional shape density remains exact; photometry NLL has no detected
change at current precision. The selected flow response still changes, but
the emulator gain now exceeds the opposing flow decrease. These are prediction
changes, not physical component-error attribution. Update jobs.json to complete.
Validation: py31 audit (under one second), trusted-runtime CPU tensor comparison,
Slurm exit receipts, syntax and link checks. Next: replicate the frozen pair on
more scenes before another training change. No new jobs, images, training or
automated checks; broad17788286 remains held.

## 2026-11-07 — Separate photometry learning errors from added neighbour information

Complete the owner-requested [capacity-by-information diagnostic](evidence/flow_photometry_information_20261107/findings.md#main-conclusion) on the exact 163,840 zero-shear objects in the recent truth-property plots. Fifteen CPU diagnostic fits use 128 scenes for training and 32 for early stopping, with the 40 original validation scenes held out from those fits. Compare original eight inputs at tree depths 4/8, added fixed-photometry truth neighbour score at both depths, and a shuffled-score depth-8 control for radius, magnitude and joint-cut probability. Existing-input learning removes most of the highlighted close-crowding mean residual (radius −0.0639→−0.0106 px; magnitude +0.0222→+0.00284 mag; joint passing error −2.064→−0.369 percentage points). Added score still reduces held-out depth-8 radius/magnitude squared error by 2.48%/1.44% and cut log loss by 0.662%; shuffled-score changes are small. Thus both practical learning limitations and useful extra conditioning are supported, without proving strict input insufficiency or improved shear calibration. Full losses, paired scene SEs, per-scene gains, score-association overcorrection and limitations are retained.

Validation: source/feature hashes, unique ID matching with zero unmatched, disjoint case splits, exact serialized model replay and Python AST checks. SLURM 17817608 (8 CPU) completed all fits in 3m17s; 17817627 completed reduction/figures in 11s. Fixed flow comparisons include independent antithetic half-bank quadratic-loss sensitivity; extreme magnitude tails in scene 180 limit global flow MSE interpretation. Original flow and all production artifacts remain unchanged; no new images or broad evaluation resumption. Large data/models stay in `/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_photometry_information_20261107/`; user-facing plots are in the requesting chat's `outputs/information_test/`. Further flow training would need its own matched experiment; this diagnostic does not replace the ongoing conditional-cut study.

## 2026-11-07 — Launch conditional cut-probability supervision with shape weights fixed

Implement the owner's requested next experiment under the
[fixed protocol](evidence/conditional_cut_probability_20261107/rule.md#authorized-comparison-and-scope).
Two matched 64-step photometry-only continuations start from the same untouched
nine-cut checkpoint. Replace the previous auxiliary population-mean moment
experiment with a per-row soft-cut squared-error score; independent latent banks
share the same row, with separate row/latent seeds. The control has zero extra
weight. Fix the weighted arm's coefficient using four training-only gradient
batches at initialization, targeting 25% of the baseline photometry gradient
RMS. Preserve the original NLL/response loss, cases, nine cuts, batch sizes,
shape density, classifier and emulator. No coherent outcome selects a weight.

Add source-pinned training/evaluation adapters, resume states, analytical and
reducer fixtures, and hard/soft conditional-probability, smoothing-sensitivity,
factor-NLL and selected-shape diagnostics. Reuse the original coherent response
evaluator and paired reductions. Validation: AST and shell syntax for 13 source
files; exact independent-bank expectation/gradient, reused-bank variance bias,
conditional cancellation and row-shape checks; unequal-case score, selected-mean
normalization and paired jackknife fixtures; all reference hashes/imports pass.
Training job17817062 started on an A40 allocation. Real-model auxiliary and
response gradient checks pass at6.30e-8/8.41e-8 relative error; direct shape
density is exact on1024 inputs. Training-only calibration fixes the auxiliary
weight220.30315 before updates; control has reached saved checkpoints. It includes own-data diagnostics and a first
case/axis, gating persistent workers17817064 and report17817065. At most three
concurrent GPUs, broad17788286 remains held, no new images or automated follow-up.
No scientific improvement is established yet.

## 2026-11-07 — Review flow improvement strategy after the completed input interventions

Add an [evidence-based proposal](evidence/flow_improvement_strategy_20261107.md#status-and-recommendation)
after the owner asks how to improve the current model. Distinguish the full
measured-input bias shift from improvement of the restricted neighbour diagnostic;
the substitution does not uniformly improve coherent calibration. Inspect the
existing loss and factorization: the response penalty uses a matched-usable
flow-only target, while the failed added loss constrains population means of
selection and a static neighbour score. Propose a per-row conditional cut
probability objective, a matched frozen-shape intervention with behavioral
shape checks, and a separately gated neighbour-shear input/label extension.
Document Monte Carlo estimator requirements, full selected numerator/denominator
targets, potential emulator double counting, fit/sample precision limitations,
and train/development/acceptance separation. Validation: source/protocol review
and local Markdown link checks; no new numerical results, tests, training,
images, scheduler jobs or schedules. Next: turn the first candidate into a
bounded training-only protocol before any launch. Broad evaluation remains held.

## 2026-11-07 — Measure neighbour-only crossing response and close the frozen-shape comparison

On the existing 2,048-ID four-field g1 replay, decompose each leg's normalized fitted ellipticity into the common-ID mean plus signed contributions from entrants and exits. At FLUX_RADIUS >3 px / MAG_AUTO <25, the two contributions are +0.04595 and +0.05608: 17 distinct migration IDs add +0.10203 to physical neighbour R11 (four-scene jackknife SE 0.06792; conditional sampling SE 0.06109). +0.09550 comes from objects usable on both legs. In a paired check giving the fixed nine-cut model the actual memberships, it predicts a migration term of +0.02398; the measured-minus-model difference is +0.07804 ± 0.05837 (conditional sampling). Neither establishes a coherent calibration bias. [All nine cuts, individual cases, fit repeats, normalization and precision limits](evidence/neighbour_crossing_response_20261107/findings.md#what-the-measured-galaxies-contribute); [true membership intervention](evidence/neighbour_crossing_response_20261107/findings.md#giving-the-model-the-actual-entrants-and-exits).

The existing frozen-shape training/evaluation/report completed (17815619, all 17815620 workers, 17815621; exit 0). The added static-neighbour penalty improves the score on its own data but worsens pooled nine-cut RMS m on both axes with predicted or actual usability. Across four fields, the central predicted-usability residual rises by +0.000259 ± 0.000049 / +0.000248 ± 0.000041 (paired case jackknife R11 / R22); flow loss exceeds emulator gain. [Completed comparison](evidence/frozen_shape_neighbour_20261107/findings.md#completed-frozen-shape-comparison-on-both-axes). Validated per-case reducer 17816443 against every pooled contrast and jackknife SE. Crossing job 17816432 stopped on a NumPy corner-axis indexing error; corrected without changing data or cuts. 17816445 and the measured-membership reduction 17816462 exited 0, reproducing original responses and both uncertainty estimates for all nine cuts, both output components. Python compilation passed. No training, new images, release, or automated follow-up. The held broad array remains paused.

## 2026-11-07 — Separate primary-shear cut migration from missing neighbour-shear inputs

[Input contract and held-out paired results](evidence/primary_selection_transport_20261107/findings.md#primary-shear-a-slope-error-does-not-imply-a-target-cut-shape-error): on 40 GalSBI validation cases, compare the fixed nine-cut start and both frozen-shape endpoints with matched primary truth/photometry (5,817,708 usable pairs; model on the original 262,144-pair probe, 64 shared antithetic draws). All probe contexts replay exactly; both-leg eligibility and unmatched rows are documented. At radius >3 px/MAG_AUTO <25 the weighted flow underpredicts the global orientation-dependent magnitude slope by 0.05771 ± 0.00125, yet its fixed-intrinsic-shape selection response agrees to −0.00032 ± 0.00065 (measured minus model, case jackknife). The orientation slope of B-weighted passing mass is too large by 0.00704 ± 0.00181; the strongest-neighbour quintile shows the clearest discrepancy. Report the complete nine-cut, five-quintile grid and the matched-probe sensitivity alongside the full measured reference. A global size/flux derivative penalty is not supported as the remedy for the pooled cut. No retraining or checkpoint choice.

[Four-corner flag audit](evidence/primary_selection_transport_20261107/findings.md#physical-neighbour-only-flip-detection-as-well-as-photometry-changes): on 2,048 fixed primaries in four existing g1 image fields, neighbour-only shear flips detection for 12 distinct IDs, or the final usable-and-passing event for 17; the corresponding signed net change in selected fraction is consistent with zero. Both current models receive primary sheared ellipticity and shear-invariant crowding; the classifier cannot change usability under a neighbour-only flip, nor can the flow change selection. These are not measurements of the neighbour contribution to m. Validation: Python compilation; Slurm 17815885 / 17815894 / 17815905 for paired preparation, shared-draw evaluation and reduction, 17815927 for the neighbour event check; all completed. The independent frozen-shape constant-shear comparison passed the first field/axis gate and continues in its original persistent workers. No new images or automated checks.

## 2026-11-07 — Recover historical shape/flux/size response experiments and test their relevance

Located the Gold-V2/September 2 Flow-E orientation-slope penalty (500/500 became V3.3-like), September 10 physical eight-component runs, and the September 23 pairwise photometry/optional shape refinements. [Historical outcomes and target definitions](evidence/photometric_derivative_history_20261107/findings.md#which-experiments-actually-ran) distinguish held-out derivative loss, radius-cut migration and full coherent calibration. The old model could learn the slopes, but a later model scored better on those slopes and got the sign of a radius crossing statistic wrong.

A one-CPU reduction of existing four-corner GalSBI receipts (17815772, 0:0) verifies sample/model hashes and fixed membership predictions across neighbour-shear flips. On 1,012 common-detected finite-photometry targets, 13 distinct primaries change the central measured cut when only neighbours are sheared. The intrinsic-e1 projected response is +0.03264 ± 0.02382 conditional sampling SE; it does not establish an absolute measured-shape bias. Both frozen-shape continuations also completed with identical conditional shape density, and the weighted arm improves the held-out selection-score proxy by 0.002605 ± 0.000027 over its matched control. Joint evaluation continues. [Interpretation, exclusions and proposed test](evidence/photometric_derivative_history_20261107/findings.md#current-evidence-who-passes-versus-who-crosses). No new training or images for the historical audit.

## 2026-11-07 — Train the selected-neighbour penalty with the nine-cut conditional shape model fixed

The previous neighbour moment had exactly zero direct shape gradient, but the
other training objectives used both factors. Endpoint swaps show a selected
flow decrease from photometry alone of −0.002268/−0.002250 in R11/R22 with
actual usability and generated selection. With a common measured selection,
these changes shrink to −0.000188/−0.000146. The separate conditional-shape
weight change contributes −0.000983/−0.002980 to the flow at baseline photometry.
The original global clip was active in every step of both arms; gradient coupling
is possible, but no specific optimizer route has been established as dominant.
See [evidence and interpretation](evidence/frozen_shape_neighbour_20261107/findings.md#measured-effect-before-the-new-training).

Launch matched photometry-only continuations of the same untouched nine-cut checkpoint,
with/without the added neighbour penalty (fixed 64 steps). Only photometry parameters
enter the optimizer; require identical conditional density at fixed inputs, unchanged
nonphotometry tensors, exact reload, original selection, cases and draws.
Added a trainer, the pinned evaluation/reduction adapters and job receipts
under [the frozen protocol](evidence/frozen_shape_neighbour_20261107/rule.md#paired-optimizer-intervention).
Python compilation passed. Slurm 17815619 passed initial gradient checks
(relative errors below 9e-8), the 1,024-row conditional-density check and resumed
normal checkpointed training. It runs own-data validation and a first scene;
17815620 and 17815621 gate the remaining scene/axis checks and paired report.
The old broad array remains held; the owner-stopped follow-up loop remains off.

## 2026-11-07 — Physical primary shear carries most of the nine-cut model shift

All 4,096 mixed exposures, four model replays and the two-seed primary/neighbour report finished successfully. Previous full-parent responses replay within 2.00e-15. On the fixed pilot's 2,048 primaries in four g1 development scenes, moving from likelihood only to nine-cut regularization increases the central predicted-usability R11 residual by +0.01195 ± 0.00166. Primary shear contributes +0.01114 ± 0.00167 and neighbour shear +0.00081 ± 0.00019 (paired case jackknife SEs). The primary dominates individually in all four central-cut fields and accounts for 91–95% across the nine pooled cuts with predicted usability. Actual usability and measured selection retain that pattern. The result identifies which physical intervention changed the model prediction, not which component has the absolute error. No R22 is measured.

The absolute nine-cut predicted-usability residuals are +0.05915 / +0.09042 / +0.14956 (primary/neighbour/coherent), with conditional sampling SEs 0.08127 / 0.08860 / 0.12851. The two shape-fit repeats give opposite attributions even though *none* of the four target-cut memberships changes. Ten primaries account for 55%/68% of the absolute contributions to the primary/neighbour seed differences. Added and ran the [paired reducer](evidence/primary_neighbour_replay_20261107/paired_diagnostics.py) in Slurm job 17815268 (exit 0:0). It reproduces all 324 endpoint contrasts from the original report (two model changes, three flag modes, three roles, nine cuts and two output components) and decomposes the repeat shifts exactly. [Physical interpretation and central tables](evidence/primary_neighbour_replay_20261107/findings.md#completed-pilot-where-does-the-change-in-response-go); [all cuts and uncertainties](evidence/primary_neighbour_replay_20261107/report.json).

On the same saved centroids/pixels and original seeds, use new BlendEMU scripts `scripts/primary_replay_fit_trace{,_complete}.py` to recover deconvolved T for all central-cut fits; SBSI reduces those receipt files. First trace 17815279 stopped on a bitwise mismatch on case 1040. Job 17815288 kept nonexact fits unclassified; supplemental job 17815297 replayed just the affected mixed corners on the original CPU family. All 4,030 selected fits now reproduce exactly, zero flags, no changed scientific settings. Of 2,015 selected object–corner pairs, 141 switch T sign between the two seeds. These contribute +0.25085 / −0.16661 to the observed +0.24432 / −0.19542 primary/neighbour repeat differences. This explains substantial optimizer noise, not an absolute mean bias. [Signed decomposition and precision decision](evidence/primary_neighbour_replay_20261107/findings.md#why-two-optimizer-seeds-disagree). Both diagnostic reductions passed role/additivity checks. Broad array 17788286 remains held, no additional images or seeds and no scheduled follow-up.

## 2026-11-07 — Owner stopped automated follow-ups

Cancelled the 07:29 CET one-time check. Verified the timer and service are inactive, the timer is absent from the user timer list, and the two-hour dispatcher is disabled with no marked crontab entry. Recorded the cancellation in [job metadata](evidence/primary_neighbour_replay_20261107/jobs.json). The Slurm pilot and its dependent report continue.

## 2026-11-07 — Direct one-primary image replay clears independent pixel and coherent-detection controls

Start the owner's proposed physical attribution pilot on cases 1000, 1040,
1080, 1120, g1=±0.02. In BlendEMU, add the fixed sample, renderer validation,
native full-field measurement and operational pilot scripts. Preserve the
original entire background exposure and change one primary's rendered light
per mixed corner. This avoids confounding the neighbour intervention with
cutout boundaries, missing wings, or new noise. Select 256 detected-union
and 256 complementary truth primaries per scene with recorded probabilities.
All nine cuts are applied independently in each corner; common optimizer seeds
are used across corners. SBSI owns the frozen model predictions and uncertainty
reduction. See [the protocol](evidence/primary_neighbour_replay_20261107/rule.md#sampling-and-four-corners).

Independent pixel checks compare both previously simulated A/B splits against
coherent exposures. The first 192-pixel guard failed for one patch. Extend the
validation neighbourhood at unchanged tolerances: three of 32 patches require
384 pixels; worst final RMS difference 0.0000208 versus sky RMS 0.616.
Capture the whole native delta stamp for every sampled primary. All eight
coherent SExtractor catalogues replay byte for byte; all 4,096 target/leg
mappings replay. Central coherent R11 of the weighted pilot is 1.03394, with
four-case jackknife SE 0.08262, conditional sampling SE 0.13008, and estimated
two-seed optimizer MC SE 0.09603; these errors must not be added. This measures
pilot precision, not a model bias or an independent calibration.

All 16 single-primary operational exposures pass; projected extraction cost
for the fixed pilot is 224 CPU-hours at the slowest observed time. Release
four persistent 32-CPU allocations (array 17814392); one GPU (17814387)
replays three fixed endpoints with original latent draws. Its initial exact
comparison exposed 6.94e-18 roundoff in recomputed unchanged numerators;
reuse the native mass/shape numerator exactly when only the neighbour shear
flips. Case 1000 passes full-parent coherent model replay within 1.89e-15.
Production mixed outputs are arriving. No scientific setting or tolerance changed. Report 17814399 depends
on both stages. The reducer passed known four-corner offsets with unequal
selected masses, covariance closure and a numerical derivative with respect
to a primary weight. [Pilot receipts and limits](evidence/primary_neighbour_replay_20261107/findings.md#independent-controls)
and [job IDs](evidence/primary_neighbour_replay_20261107/jobs.json) are saved.
One same-thread check is scheduled at 07:29 CET; the old recurring loop stays
stopped and broad array 17788286 stays held. No larger image campaign or
checkpoint selection. Inspect the four-corner residuals and precision budget
when complete; no absolute physical attribution is established yet.

## 2026-11-07 — Both axes localize the regularization shift; frozen-factor follow-up complete

All eight case/axis evaluations and report 17810894 completed with exit 0:0.
At radius > 3 pixels, magnitude < 25, actual usability and generated selection,
the conditional-shape swap explains 96%/94% of the NLL-to-nine-cut joint
residual change in R11/R22. Across all nine nearby cuts, axes and flag modes,
the share is 91–97%. For the added neighbour constraint, photometry alone
raises the emulator contribution by 0.002483/0.002489 but lowers the flow
contribution by 0.002268/0.002250. The excess R22 over R11 residual shift
is +0.002188 ± 0.000155; the shape-only difference is +0.002215 ± 0.000021
and the photo-only difference −0.000024 ± 0.000160. The shape-only difference
has the same sign in every scene at every target cut for all three flag modes.
These are paired case jackknife SEs, conditional on fixed fits and draws.
The selected-neighbour proxy improves, independently of which shape factor
is attached to the photometry. It cannot diagnose the coherent shear error.

Add `scene_breakdown.py` and run it with the py31 interpreter on small saved
bin sums. It verifies 56 file digests and reproduces all 1,296 diagonal target
contrasts and their SEs in the Slurm report within 1e-11. Across eight tasks
the largest source-response replay difference is 2.93e-12; maximum pooled
emulator roundoff is 8.88e-16. Match counts and paired scene/axis tables are in
[scene and size results](evidence/neighbour_factor_swap_20261107/scene_size_breakdown.md#scene-by-scene-shape-contrast);
[cut_contrasts.tsv](evidence/neighbour_factor_swap_20261107/cut_contrasts.tsv)
retains each nearby cut. Intrinsic radii 0.15–0.40 arcsec account for 82.8%
of the added-moment shape-only axis difference and 75.7% of selected objects.

Record the [final interpretation](evidence/neighbour_factor_swap_20261107/findings.md#final-conclusion-both-shear-axes).
Four development scenes do not provide independent calibration or separately
measured physical component errors. A separate primary/neighbour image-shear
pilot could supply those labels; no new training, simulations, model choice
or broad evaluation was initiated. With the focused comparison complete, run
the owner's existing `dispatch.py stop`: state is disabled/stopped and the
marked cron entry removed. Broad array 17788286 remains held.

## 2026-11-07 — Four-scene axis1 swaps localize the regularization shift to conditional shapes

The first scheduled two-hour message arrives at01:59 CET. Record delivery
under the existing dispatcher lock and in the loop README; no duplicate
schedule. Pilot17810892 and two workers finish0:0, with seven of eight
case/axis receipts passing. Worker17810893_1 is healthy on the last task,
case1080/axis2; report17810894 waits on it. Broad17788286 remains held.
The largest unchanged-source response replay discrepancy is1.97e-12 so far;
shared-photo selection masses and reference-score sums remain exactly equal.

Add `neighbour_factor_swap_20261107/interim.py`, a lightweight reducer of
saved sums for fully completed axes only. Run with the documented py31
interpreter in under three seconds. Check receipt provenance/hashes,
simulation equality, factorial additivity and unchanged-source summary
replay:1512 point/jackknife entries agree with the prior report within
3.40e-11. No heavy numerical work, model changes or job submissions.

For all four axis1 scenes, actual usability and central generated selection,
the original response penalty changes DeltaR by+0.012049 ±0.000074:
conditional-shape-only+0.011512 ±0.000022, photo-only+0.000537 ±0.000064,
negligible interaction. Shape changes dominate all nine nearby cuts, both
usability choices, and measured selection. For the added neighbour moment,
photo-only emulator/flow contributions change+0.002483/-0.002268, leaving
DeltaR-0.000215 ±0.000088; the learned shape-factor change adds
+0.000874 ±0.000016, giving total+0.000652 ±0.000097. Photo-only effects
change sign across nearby cuts, so freezing shapes is not yet a robust fix.

Document [the interim evidence](evidence/neighbour_factor_swap_20261107/findings.md#four-scene-axis1-interim-localization)
and tables/JSON. SEs are paired whole-case jackknife estimates conditional
on fixed models/draws; absolute m1 SE remains about0.70 percentage points.
These are model contribution changes, not isolated component-error
measurements or independent acceptance. Axis2 remains pending. Leave the
healthy persistent worker/report chain and existing two-hour loop running;
check the full two-axis comparison at the next follow-up.

## 2026-11-07 — Frozen-factor checks pass; standalone shift is mainly in conditional shapes

Checksum-verified pilot17810892 assembles all seven fixed products. Direct
conditional-shape density equality is exact, source photometry and reload
checks are exact, and the three unchanged sources reproduce previous probes.
Explicit reconstruction accounts for float32 log-determinant addition
roundoff up to9.54e-7, without any prediction change. The joint pilot reaches
block100; pooled/separate emulator calls agree to4.44e-16. Full-case replay
and dependent workers remain pending; leave the healthy job running and
retain the existing two-hour same-thread follow-up.

Add the lightweight `own_probe_report.py` reducer of saved probe summaries.
Its factorial additivity checks pass. For original response regularization,
training R11 changes-0.011611: photometry-only contributes-0.000359,
conditional-shape-only -0.011264, interaction+0.000013; R22 and validation
show the same predominantly shape-factor mechanism. For the added moment,
training R11 contributions are-0.000828/-0.000984/+0.000004, and R22
-0.000795/-0.002885/+0.000002. These fixed-probe differences have no attached
case SE and are not joint calibration results. Record them in
[findings](evidence/neighbour_factor_swap_20261107/findings.md#own-data-factor-decomposition)
and the linked probe table/JSON. Histories show norm5 global clipping on all64
steps in both prior arms; note this and photometry-dependent response gradients
as possible indirect routes, without assigning their causal shares.

## 2026-11-07 — Pin executed factor-swap source to avoid stale NFS bytecode

Retry17810852 again executes the old conditional-density subtraction check:
the traceback points to old bytecode line51 while displaying the new source
line51, and the Python3.10 cache remains dated before the edit. Preserve its
outputs, cached bytecode and on-disk source under the external failed-attempt
directory. No constant-shear evaluation started.

Change the trial launcher to read each stage's source bytes, verify a pinned
SHA256, compile those exact bytes and print the digest. This bypasses
timestamp-based imported bytecode without changing model or evaluator
arithmetic. Pin preparation/evaluation/report sources in `source_pins.json`;
compilation passes. Cancel unsatisfied dependents17810853/17810854 and submit
pilot17810892, workers17810893 and report17810894. Check the executed source
digest and direct-factor runtime gate before yielding. The same two-hour
loop and scientific resource limits remain in force.

## 2026-11-07 — Fix factor-swap verification to isolate conditional density directly

Pilot17810794 reproduces all three unchanged endpoints and original probes,
then stops before constant-shear evaluation: joint-minus-photometry density
differs across the first hybrid and its shape source by at most9.54e-7.
Code inspection identifies the existing joint transport's float32 addition
of photometry and shape log determinants before casting to float64.
Subtracting separately evaluated photometry therefore does not exactly
isolate the conditional factor. This is not a new shear-response finding.

Change only `neighbour_factor_swap_20261107/prepare.py` verification: evaluate
the conditional-shape factor directly, require exact source equality, and
reconstruct the original joint density including its explicitly computed
float32 addition roundoff to the original1e-10 tolerance. Model code,
predictions, checkpoints, cuts and numerical batches are unchanged. Record
the check in [the protocol](evidence/neighbour_factor_swap_20261107/rule.md#correctness-gates).
Preserve the failed attempt's source and three prepared outputs externally;
cancel its unsatisfied dependents17810795/17810796. Replacement pilot17810852
gates workers17810853 and report17810854. Compilation passes; follow startup
and the corrected direct-factor gate before yielding.

## 2026-11-07 — Added moment improves selection but worsens joint bias; separate the two flow factors

Moment-guard pilot, all three persistent workers and reducer17809958 finish
0:0. All four scenes confirm the selected-neighbour error reduction, from
0.003112 to0.000866, but actual-usability joint m1 changes+1.0564% to+1.1300%
and m2 changes+1.3794% to+1.7015%. Paired jackknife errors on these shifts
are0.01128/0.00991 percentage points. Every one of the nine target cuts shifts
upward on both axes. The flow contribution falls by0.003246/0.005219 while
the emulator contribution rises by0.002594/0.002379. Record full results and
their four-development-scene limits in
[findings](evidence/neighbour_moment_guard_20261106/findings.md#joint-constant-shear-evaluation).
Numerical receipts pass; the largest pooled-lookup discrepancy is8.88e-16.

Prepare the frozen [factor-swap diagnostic](evidence/neighbour_factor_swap_20261107/rule.md#question-and-endpoints)
for NLL-to-response-regularized and response-regularized-to-moment-constrained
changes. Assemble four cross-products of photometry and conditional-shape
factors, retaining the three source endpoints as same-device controls. No
optimizer updates, loss tuning or model promotion. Gate matching outer
transforms/standardizers, identity reconstruction, photo-factor and fixed-photo
conditional density, exact reload, original own-data probes and source replay.
Evaluate all seven products with the original cases, cuts, per-leg flags,
1024-parent blocks and64 portable common draws. Factorial reporting separates
photometry effects, shape effects and their interaction, with paired case
errors. Same-photo products must have identical selection masses.

Submit one preparation/probe/pilot allocation, then three persistent GPU
workers for the other seven case/axis tasks and a dependent CPU report; IDs
are in [job metadata](evidence/neighbour_factor_swap_20261107/jobs.json).
The trial uses at most three16GB A40 GPUs within the six-GPU ceiling.
Resource detection finds340GB available host RAM and9.1TB available disk;
heavy work remains in Slurm. Python compilation and shell syntax checks pass.
Runtime identity and startup gates remain to be checked. The existing
two-hour loop follows this work; broad17788286 stays held.

## 2026-11-07 — Two-hour loop startup message reaches the existing thread

The queued startup test arrives at00:00 CET, confirming the local CLI
queue/resume path reaches this same thread. Inspect the existing loop prompt,
state and crontab: exactly one matching entry remains enabled, with first
scheduled run01:59 CET. Record delivery confirmation atomically under the
dispatcher lock and document
[startup verification](evidence/neighbour_moment_guard_20261106/loop/README.md#startup-verification).
No duplicate loop or new experiment is created. The first cron-triggered
delivery remains a future check; this receipt verifies the immediate startup
test, not a completed future scheduled run.

## 2026-11-07 — Configure the owner's recurring two-hour follow-up

Install a user-cron trigger on kng-sv-clhead02 for the existing SBSI thread
using the installed Codex CLI0.167.0 `queue --resume` command. The first
scheduled run is2026-11-07 01:59 CET, then every two hours. The lightweight
dispatcher, durable scope prompt and stop instructions live in
[the loop protocol](evidence/neighbour_moment_guard_20261106/loop/README.md#scope-and-operation).
It preserves unrelated crontab entries, locks concurrent invocations,
suppresses duplicate attempts, and disables itself on queue/resume warnings
or failure. The follow-up preserves held broad17788286, all scientific
settings and resource limits, and stops on owner pause, a concluded focused
investigation, or a decision requiring owner input. No additional agent
thread or permissions change is introduced.

Validation: Codex Doctor reports the persistent background server running;
the immediate startup-test message is accepted with ID01a1b314-33b3-7181-9ed9-e1d76afa0780.
End-to-end delivery is not yet confirmed while this setup turn is active.
Isolated subprocess fixtures pass unrelated-cron preservation, idempotent
install/stop, dry-run, duplicate suppression and fail-closed resume-warning
checks. Compilation, real cron readback and early dry-run pass. This is a
host-level schedule, not an app Scheduled-view automation; it needs the host,
cron and Codex background server available. It does not claim a completed
future run. Existing Slurm scientific jobs and the bounded status snapshot
are unchanged.

## 2026-11-06 — Moment-guard joint pilot passes but shows an opposing flow shift

Job17809953 completes preparation, both trainings, own-data diagnostics and
case1000 axis1. All pilot correctness gates pass; paired evaluation takes378s.
At radius>3px/magnitude<25 with actual usability and generated selection,
the added moment raises the emulator contribution by0.002610 but lowers the
flow contribution by0.003398. Single-scene m1 changes+0.4722% to+0.5608%;
predicted-usability m1 changes+0.2698% to+0.3421%. These are provisional
single-scene changes without a between-case SE, not a component-error budget
or evidence that total bias improved. Save `pilot_summary.json` and update
[findings](evidence/neighbour_moment_guard_20261106/findings.md#joint-constant-shear-evaluation).

The pilot releases the three persistent workers in array17809957 for the
remaining seven declared case/axis evaluations. The dependent reducer will
retain all four development scenes, both axes and paired uncertainties.
All three workers reach their first numerical gate and model accumulation:
pooled/separate emulator differences are at most6.66e-16. Training/pilot exits
0:0; the reducer and two-hour bounded local check remain queued. Yield with
the longer evaluations progressing normally. The original broad evaluation
stays held; no new training or model promotion.

## 2026-11-06 — Added moment improves held-out neighbour selection; joint test starts

Both fixed step64 endpoints in job17809953 complete. The matched control
replays the previous neighbour-nine endpoint exactly on1024 log densities and
validation NLL. The new moment reduces hard-cut g0 selected-neighbour error
from+0.003322 to+0.001024 in training and+0.003148 to+0.000897 in validation.
The paired validation change is-0.002251 ±0.000026 (case-bootstrap SE);
individual errors retain about0.00052 SE. Validation NLL improves from
-3.914690 to-3.918241. Record the response-probe tradeoff and limits in
[findings](evidence/neighbour_moment_guard_20261106/findings.md#training-and-held-out-selection).
This proxy improvement is not yet a joint calibration result.

Own-data photometry/joint equality, score alignment, disjoint latent-bank
identity and validation-NLL replay pass. Extend the reducer to report the
saved half-bank discrepancy and paired case uncertainty of joint NLL; verify
each new case's checkpoint, evaluator, data hash and numerical receipts before
reduction. Retain fixed endpoint identities/probes, paired axis contrasts and
the flow-versus-emulator contribution changes in the final report.
No new model evaluation or changes to frozen training/evaluation
settings. Compilation passes. The pilot's pooled versus separate emulator
calls agree to4.44e-16, and case1000 axis1 progresses normally. Its full-case
gate remains pending. Broad17788286 stays held.

## 2026-11-06 — Moment-loss gradient gate passes; share exact emulator lookups

Training job17809953 finishes the full46,833,127-row target pass with exact
cached-score alignment. Before optimization, response replay/direct gradients
agree to8.24e-8 relative error; the new moment gradient agrees to4.89e-8 and
has zero gradient in the shape factor. Photometry-only draws exactly match
the full joint draw. The matched control has started normally.

Before any constant-shear evaluation starts, add pooling of the emulator's
exact output tree cells across models and shear legs within each existing
1024-parent block. Code inspection before the pilot finds200k-row accumulation
batches can regroup floating-point sums. Gate every case/axis on first-block
agreement against separate emulator calls at2e-14 absolute/relative tolerance,
recording maximum differences. This reduces repeated work while preserving all draws,
neighbours and transport calculations. Record this in
[the execution protocol](evidence/neighbour_moment_guard_20261106/rule.md#validation-and-execution)
and numerical receipts. Evaluator compilation passes; the pooling gate remains
to be exercised by the pilot.

## 2026-11-06 — Test a selected-neighbour moment alongside the nine-cut response penalty

Owner asks to proceed with the targeted training constraint. Freeze
[the matched experiment](evidence/neighbour_moment_guard_20261106/rule.md#frozen-experiment)
and implement a differentiable selection-mass/selected-neighbour loss at the
existing nine cuts. Both arms start from neighbour NLL checkpoint12ca7856 and
receive the unchanged64 NLL-plus-response updates. The weighted arm adds weight
0.001 on independent-replica mean residuals; targets come only from all original
training NLL rows, scales are0.003 fractional mass and0.001 selected coefficient.
Measured row subtraction occurs only in the loss; no prediction offsets exist.

Add streaming target preparation with full score-column alignment checks,
direct/replayed auxiliary-gradient verification, photometry-factor parity,
per-eight-step resumable optimizer states, fixed500k-row train/validation
hard/soft-cut diagnostics, and a persistent paired-model constant-shear worker.
Cases, original classifier/emulator,64 portable draws,1024-parent blocks and
all selection/response conventions are preserved. Both fixed step64 endpoints
will be reported; no constant-shear outcome chooses a weight or checkpoint.

Job17809953 runs preparation, both trainings, own-data checks and the first
case/axis pilot in one GPU allocation. After its correctness gate, array17809957
uses three persistent workers, processing3/2/2 remaining case/axis tasks each
with both models and the emulator loaded. Report17809958 depends on both
stages. Local bounded check17809959 is scheduled for two hours. The trial uses
at most three concurrent GPUs within the study's six-GPU ceiling; broad17788286
stays held. Python compilation and shell syntax checks pass. Runtime gradient,
replay and startup gates remain to be checked before yielding.

## 2026-11-06 — Neighbour-weighted selection error already exists on training data

Persistent GPU job17809769 completes in3m33s with exit0. Six unique models
remain loaded while both500k-row splits and eight predeclared arms run. Exact
photometry-factor/joint-draw parity, score-column alignment, full/half-bank
identity and validation NLL replay pass (largest replay difference2.8e-7).
Save [findings](evidence/neighbour_flow_selection_20261106/findings.md#main-finding),
case/leg/quintile sums externally, and paired bootstrap/jackknife results in
[the report](evidence/neighbour_flow_selection_20261106/report.json).

On original g0 rows, selected-reference-score error for the augmented NLL flow
is+0.002678 ±0.000414 on training and+0.002520 ±0.000513 on validation.
Relative to the matched control, the feature improves it by0.000316/0.000261;
subsequent nine-cut regularization worsens it by0.000644/0.000628. The same
association failure therefore exists before transfer to constant-shear scenes.
Generated selection underweights the strongest-coupling quintile by about0.94pp
in training and0.93pp in validation, worsening to1.12/1.11pp after regularization.

The new input improves validation photometry and conditional-shape NLL before
regularization. Its response penalty worsens photometry NLL by0.005461 and
shape NLL by0.001198. Score-at-mean interventions confirm the input is used.
This motivates testing a training-data selection-times-neighbour-moment
constraint at the existing nine cuts; no new training or offsets were applied.
These score diagnostics are not shear responses or a complete causal budget.

The paired40-case audit gives weak-nine m2-m1=-0.7304 ±0.3985pp and trace
bias-0.3672% ±0.1961pp (jackknife); the axis contrast itself is only about1.8 SE.
Update both preceding study findings with these limits. All focused jobs are
complete; broad17788286 remains held. Future evaluations retain the new job
packing preference.

## 2026-11-06 — Completed measured-input test and neighbour-feature trial; diagnose own-data selection

Full measured-input array17807972, reducer17807973, neighbour-feature array
17807394 and reducer17807399 all complete. Record the40-scene measured-input
result in [findings](evidence/joint_measured_photometry_20261106/findings.md#full-forty-scene-result):
Cartesian NLL m1=-1.2025% ±0.3062pp and m2=-1.4070% ±0.2365pp; original
weak-nine m1=-0.0026% ±0.3097pp and m2=-0.7331% ±0.2381pp (case-bootstrap SE).
All80 numerical receipts pass; half-bank m differences are below0.0082pp.
The four-scene positive weak-nine m2 did not persist. Generated-input contrasts
remain strictly matched:40 NLL cases and22 weak-nine cases.

The new neighbour feature improves held-out NLL (-3.921348 versus matched
control -3.917680) but only modestly changes the four-scene selected-neighbour
score error and joint bias. In the regularized endpoints, true-usability m1
changes+0.0057pp and m2 changes-0.0673pp relative to the matched control.
Code audit confirms the original penalty constrains all four response-matrix
components separately; R22 was not omitted.

Launch one persistent GPU job17809769 under
[the own-data selection protocol](evidence/neighbour_flow_selection_20261106/rule.md#question-and-frozen-endpoints).
It loads six unique frozen models once, processes500k fixed training and500k
validation rows, preserves g0/g05 populations and tests the two augmented models
with their extra input at its training mean. Measure nine-cut selection/score
association and photometry versus conditional-shape NLL. The same allocation
audits paired-axis differences in the completed40-scene result. No retraining,
new images, offsets or model promotion. Four CPUs and one GPU; larger job
packing follows the owner's preference. Python compilation and shell syntax
checks pass. Monitor startup and numerical replay gates; broad17788286 stays held.

## 2026-11-06 — Remember preference for fewer, larger scheduler jobs

Owner requests persistent memory of the preference for bigger, fewer jobs
instead of small job chunks. Record it in AGENTS.md and expand the operational
guidance in [Job packing](ENVIRONMENT.md#job-packing): process multiple cases or
legs per allocation, use persistent workers or bounded pools, and reuse model
setup where practical. Retain per-case recovery and outputs, scientific seeds,
pilot gates, dependencies and resource ceilings. Distinguish allocation packing
from numerical batch sizes. This is a documentation change; existing running
jobs and predictions are unchanged. Validation: read back both instruction
sections and run git diff --check on the three edited documentation files.

## 2026-11-06 — Check CPU use and fill the existing joint-study concurrency limit

Owner asks whether more CPUs would speed the measured-input evaluation.
Read-only observation inside running allocation17808313 (array17807972_22)
finds98.7% process CPU use over100 seconds, with four CPU cores allocated;
sstat reports98 CPU-seconds. Instantaneous GPU utilization is13%, so this is
not evidence of saturated GPU arithmetic either. The evaluator caps PyTorch
and emulator threads at four and performs a serial block/leg/model loop with
CPU/GPU transfers. More allocated cores alone do not parallelize that loop.
Completed measured-input tasks take roughly49--179 seconds depending on node;
these timings are not a controlled CPU-scaling benchmark.

Raise remaining array17807972 from two to three concurrent workers, keeping
neighbour array17807394 at three and preserving the combined six-GPU cap.
Verify pending tasks show %3 despite scontrol warnings for completed elements.
Update [job metadata](evidence/joint_measured_photometry_20261106/jobs.json)
and [resource protocol](evidence/joint_measured_photometry_20261106/rule.md#numerical-and-resource-gates).
No scientific settings, source predictions or thread counts change. Existing
release17807977 still permits six measured-input workers only after the earlier
neighbour array drains; broad17788286 remains held.

## 2026-11-06 — Full joint measured-input response on four scenes

All eight case/axis tasks and reducer17807971 complete successfully. At measured
radius>3px/magnitude<25 and actual per-leg usability, original Cartesian NLL has
m1=-1.4282% ±0.5688pp and m2=-0.0930% ±0.3320pp; original weak-nine has
m1=-0.2291% ±0.5734pp and m2=+0.5960% ±0.3348pp (paired case bootstrap SE).
The four-scene jackknife errors are respectively0.6554/0.3810 and0.6608/0.3842pp.
Preserve full vectors, components, all cuts, per-case outputs and numerical
checks in [the report](evidence/joint_measured_photometry_20261106/four_report.json)
and explain scope in [findings](evidence/joint_measured_photometry_20261106/findings.md#four-scene-result).

Against generated inputs/selection on exactly the same scenes and actual
usability, biases decrease1.04--1.24pp. Emulator contributions rise0.01300--
0.01360, partly offset by conditional-flow decreases0.00164--0.00437. This is
an intervention on model contributions, not an additive attribution of true
component errors. No opposite-group selection correction is appended.

Two disjoint32-draw banks differ0.031--0.068pp in aggregate m, below current
scene SEs. Exact measured selection masses, simulation replay and all other
pilot gates pass. These four reused scenes do not establish robust0.3% bias.
The frozen full40-scene array17807972 has started; report17807973 and local
bounded check17807978 retain their dependencies. Broad17788286 stays held.

## 2026-11-06 — Measured-input joint pilot passes and four-scene test starts

Replacement pilot17808000 and gate17807964 complete. Case1000/axis1 has exact
conditional reconstruction, unused-photometry-latent independence, measured
photometry and selected-count parity, simulation-sum replay, and full/half-bank
moment identity. No clipped or discarded draws. Across both legs, only224 of
17,798,528 selected baseline draws and228 selected weak-nine draws round to the
disk boundary; maximum squared radius1.0000000000000004. The numerical support
audit is retained in the receipts.

Single-scene m1 is -1.88727% for the original Cartesian NLL flow and -0.68719%
for original weak-nine. These are pilot values without a population uncertainty,
not a calibration conclusion. The remaining four-scene/axis array17807965 has
started; expansion remains gated by its report. Add aggregate support counts,
selected counts and catalogue matching receipts to
[the reducer](evidence/joint_measured_photometry_20261106/report.py), keeping
the response estimators unchanged. Python compilation and shell syntax checks
pass. The full scientific definition is in the
[protocol](evidence/joint_measured_photometry_20261106/rule.md#authorized-diagnostic).
Only five GPU workers currently fit scheduler resources. Lower the earlier
neighbour-flow array throttle from four to three, allowing two workers for the
requested measured-input test after existing workers finish; no running job is
interrupted. Pending array tasks show the verified %3 throttle despite scontrol
warnings for already-finished tasks. The six-GPU study ceiling is unchanged.

## 2026-11-06 — Preserve canonical boundary draws in measured-input joint test

Pilot17807960 reaches measured-input sampling, then stops at a strict open-disk
assertion. The installed radial-tanh contract explicitly retains representable
boundary values when tanh saturates, without clipping/contracting them. This is
a forward-moment calculation, so replace the inappropriate strict-interior check
with finite-output and float64-roundoff support checks, retaining every draw.
Count boundary samples separately for each flow/leg and target selection; reject
nonfinite values or squared radii beyond1+16eps. No data, weights, masks or
predictions are altered. Preserve the failed evaluator externally alongside log.

Replacement pilot17808000 uses the same cases, seeds and models. Update gate
17807964 and both report dependencies to the replacement pilot; scientific
expansion stages stay gated. Record the precise convention in the
[protocol](evidence/joint_measured_photometry_20261106/rule.md#numerical-and-resource-gates).
The preceding failure produced no numerical result artifact. Broad17788286
remains held and the combined six-GPU cap is unchanged.

## 2026-11-06 — Test full coherent response using measured photometry and selection

Owner requests the complete joint constant-shear response with actual radius,
magnitude and selection. Add [explicit diagnostic](evidence/joint_measured_photometry_20261106/rule.md#authorized-diagnostic),
evaluator, paired reducer and gated launchers. Freeze original Cartesian baseline
and weak-nine models. Draw conditional flow shapes at each coherent leg's actual
photometry; feed that photometry to the original emulator and apply the full
coherent Mobius transport. Actual usability/cuts and selected denominators are
independent per leg. No common-U restriction, measured-shape substitution,
new simulations or appended opposite-group selection correction. This is a
conditional intervention rather than an additive causal decomposition.

Pilot17807960 checks both models on case1000/axis1. Gate17807964 precedes the
other seven four-scene/axis tasks17807965 and report17807971, then72 remaining
tasks17807972 and40-scene report17807973. Preserve64 portable paired draws,
full-parent block seeds, all nine target cuts, exact photometry/selection masses,
measured baseline replay, conditional reconstruction and32-draw subbank checks.
Freeze generated-reference availability before outcomes: baseline40 cases,
weak-nine22 cases complete on both axes; comparisons use matching subsets.

Both neighbour-conditioned training jobs17807379 and pilot17807385/gate17807390
completed exit0. Their remaining evaluation17807394 is progressing. Temporarily
throttle it to four and reserve two GPUs for the new test, preserving six total.
scontrol reports already-finished tasks but pending throttle4 is verified.
After that array drains, CPU17807977 raises this test's active stage to six.
Host memory request reduced32GB to24GB after inspection identified an idle26GB
A40 node; numerical settings unchanged and pilot starts there. Bounded check
17807978 in about two hours; broad17788286 remains held. Python/JSON/bash syntax
checks pass. Next: verify pilot progress/gates, then the four-scene result.

## 2026-11-06 — Runtime feature parity passes; neighbour-flow training continues

CPU17807592 completes exit0 in9s. Runtime preprocessing and serialized
reconstruction exactly reproduce1024 cached training and1024 validation rows
for both the control and augmented arms. Preserve the [startup receipt](evidence/neighbour_conditioned_flow_20261106/startup_receipt.json).
Both training workers progress through at least11/40 likelihood epochs with
finite objectives. Fixed endpoints and subsequent response guards remain
unchanged; no interpretation of individual epoch fluctuations or checkpoint
selection. The gated32-task constant-shear comparison and report are queued.
Only two study GPUs are active. Existing bounded check17807400 covers subsequent
stage status; broad17788286 remains held. Constant-shear accuracy is still pending.

## 2026-11-06 — Neighbour features complete; both matched flows start normally

All200 CUDA feature cases complete with zero unmatched training-parent IDs.
Assembly17807352 passes exact original-context alignment for46,833,127 training
rows, preserves11,703,585 validation rows and the original response-pair/probe
order, and verifies identical new features across each matched shear pair.
Training-only score normalization has mean0.1522667906474722 and
scale0.3196636294804113. These are input standardization constants, not response
corrections. Preserve original target transforms and checkpoint provenance.

Both17807379 workers start on A40-16Q with exactly matching validation NLL
-3.909359394769827 and pass same-device initial prediction replay. New initial
response replay/direct-gradient checks agree at relative1.12e-7 for both arms.
Several full likelihood epochs complete with finite losses. No calibration
conclusion is drawn from these intermediate training outcomes.

Add independent [context parity check](evidence/neighbour_conditioned_flow_20261106/verify_context.py)
for cached training/validation rows versus the runtime DataFrame preprocessor,
including control zeroing and serialized reconstruction. CPU17807592 gates
coherent pilot17807385 together with completed training; subsequent dependencies
are retained. Six-GPU ceiling, fixed endpoints, bounded check17807400 and older
broad17788286 hold remain unchanged. No model promotion or new image simulation.

## 2026-11-06 — Queue complete neighbour-flow training/evaluation chain

Feature array17807349 has successful case receipts and six workers within the
combined cap. Complete the [job chain](evidence/neighbour_conditioned_flow_20261106/jobs.json):
all-case assembly17807352, two matched training workers17807379, first coherent
evaluation17807385, strict pilot gate17807390, remaining31 evaluations17807394
throttled to six, and paired report17807399. Four endpoints x four scenes x two
axes, with original nine target cuts and three usability/selection conventions.
The evaluator also replays the reference scene score and computes the earlier
common-four-U selection association. Report both standalone proxy and full joint
responses; no arm is selected from response outcomes.

Bounded check17807400 in about two hours writes local status only, not a chat
notification. Dependencies prevent training/feature/evaluation overlap beyond
the six-GPU ceiling and prevent scaling past a failed scientific/numerical gate.
Broad17788286 remains held. All newly added Python/shell/JSON files pass syntax
checks; normalization/inverse/persistence tests already pass. Next: monitor
assembly alignment checks, matched training startup and the coherent pilot.

## 2026-11-06 — Add neighbour-conditioned flow trial and matched controls

Owner authorizes adding the neighbour information to the flow and testing it.
Create [frozen trial](evidence/neighbour_conditioned_flow_20261106/rule.md#authorized-question-and-frozen-comparison)
with a truth-only reference-neighbour coefficient, original160/40 flow split,
matched40-epoch likelihood continuations and subsequent weak nine-cut penalties.
All endpoints are fixed in advance. Preserve original classifier/emulator,
portable common random draws, both axes and nine target cuts on the four existing
constant-shear development scenes. Add true-selection and selection-association
diagnostics; no constant-shear labels enter training or checkpoint selection.

New custom artifact adds a zero-initialized term to the first layer of every
photometry/shape coupling, retaining the original linear operation exactly.
Control has the same architecture with its added input zeroed. CPU test17807336
passes exact initial densities/draws and reload, nonzero new-weight gradients,
nontrivial triangular inverse/Jacobian and physical-density consistency
(max error4.63e-6). First test17807326 reached persistence before a NumPy-array
dictionary-comparison bug; fix that assertion without altering model numerics.

Initial CPU feature pilot17807315 failed the strict prior CUDA-score replay
by at most2.79e-7 on48/290456 targets. Preserve its completed training-case cache
separately. Canonical CUDA feature pilot17807335 passes unchanged tolerance and
truth-leg parity in44s. Feature array17807349 starts six workers, at most48 CPUs
and six GPUs (three cip, three inter); early case receipts are complete. All200
training-domain cases precede assembly17807352, then matched training17807379
and the gated constant-shear pilot17807385. Older broad17788286 stays held.
Python AST, shell and JSON checks pass. No production modules/presets, images,
likelihood definitions or empirical response corrections changed.

## 2026-11-06 — Added scene information recovers selection association in four held-out scenes

All remaining CPU folds17807247_1--3 and report17807248 completed exit0 in about
one minute per fold and2s for reduction. Split, reload, validation-loss and prior
selection-baseline replay checks pass. Preserve [results and mechanism](evidence/selection_information_20261106/findings.md#four-held-out-scenes):
selected-mean reference-coefficient errors are+0.003599 +/-0.000078 for the
weak-nine flow,+0.002619 +/-0.000128 for existing-input diagnostic boosting,
+0.002584 +/-0.000125 for its depth9 capacity control, and-0.000010 +/-0.000133
after adding a truth-scene coefficient at fixed photometry. Added information
improves held-out selection log loss in every scene, paired pooled gain
-0.000247 +/-0.000024 against the capacity control. This is predictive evidence
for richer neighbour conditioning, not a new calibrated flow or proof that the
full response bias is solved. Common-all-four usability and fixed-fit uncertainty
limitations are explicit. No response-based fitting, empirical offsets or images.

Update fixed-cell findings and job receipts. All focused work is complete;
cancel redundant bounded check17806329 and preserve broad17788286 held.
Python AST, JSON and shell checks pass for9/14/7 files in the new evidence/design
folders. Next justified model experiment is richer neighbour conditioning with
full photometry and per-leg response validation; the complementary isolated-scene
image pilot is specified but not launched.

## 2026-11-06 — Four-scene fixed-photometry and context audits pass

All fixed-cell workers and reducer17807065 completed exit0. Holding emulator
radius/magnitude exactly identical within bins preserves the coefficient gap:
+0.006547 +/-0.000172 originally and+0.006547 +/-0.000162 after intervention.
Within-bin contribution remains+0.005929 +/-0.000244; all four scenes agree in
sign. Preserve [full findings](evidence/photometry_fixed_cell_20261106/findings.md#four-scene-fixed-photometry-result).
Selection audit17807129 finds nearly matching overall fractions but systematic
underselection of high neighbour-response configurations. Context audit17807192
recomputes all eight raw features exactly for1024 targets and verifies the
training/runtime crowding contract on a fixture. No measured response offsets.

Add a [disjoint-case selection-information diagnostic](evidence/selection_information_20261106/rule.md#split-and-fixed-arms):
two training scenes, one validation scene, one test scene, rotating over four.
Compare existing eight inputs, a depth9 capacity control, and an added truth-scene
reference coefficient. Fixed logistic boosting recipe; validation log loss
alone chooses the iteration. No production flow/classifier/emulator changes.
Preparation17807215, pilot17807226 and reducer17807232 pass exact reload and
split checks. Remaining folds17807247 start with meaningful learning progress;
reducer17807248 checks all folds and replays the preceding selection baseline.
Use at most24 CPUs and no GPUs. Extend bounded status check17806329, keeping
broad17788286 held. The common-all-four usability conditioning and fixed-model
four-development-scene uncertainty remain explicit limitations.

## 2026-11-06 — Fixed-cell pilot passes; primary/neighbour replay design prepared

Pilot17807045 completed exit0 in9m14s, reducer17807054 exit0 in2s. Per-bin original
counts/coefficient moments, per-target sums and every bootstrap replay pass.
The within-bin coefficient contrast persists at exactly fixed photometry:
0.005567(original) to0.005609(fixed); total0.006056 to0.006102. Preserve
[interpretation](evidence/photometry_fixed_cell_20261106/findings.md#pilot-identical-photometry-preserves-the-mismatch).
Remaining1040/1080/1120 workers all reach meaningful progress. Add a predeclared
CPU selection audit at common reference photometry, with fixed coefficient
intervals and coarse intrinsic/crowding controls. Pilot17807128 and full17807129
use saved target moments; no additional GPU sampling or fitted correction.

Prepare the owner's [isolated-scene replay design](evidence/primary_neighbour_grid_20261106/design.md#status-and-purpose),
including four shear corners, exact population-ratio closure, a separate coherent
grid replay control, positive-probability source sampling for population closure,
full neighbour/guard handling, paired noise and optimizer seeds, and a2048-target
operational pilot whose variance would set a larger fixed budget. No image job
is launched while the owner has left its timing tentative. Analytical fixtures
pass four-corner additivity with changing selected denominators and demonstrate
why axis-only finite differences need not sum exactly. Existing frozen renderer
and fitter paths are identified; no PSF or measurement-definition change.

## 2026-11-06 — All response-bin scenes pass; fix photometry inside bins next

Owner requests the remaining scenes and deeper input localization, and suggests
a possible primary-versus-neighbour isolated-scene replay. All three response-
bin workers and reducer17806590 completed exit0. Preserve
[four-scene interpretation](evidence/photometry_response_bins_20261106/findings.md#four-scene-replication-complete):
fine-bin fraction term-0.000204 +/-0.000281, within-bin+0.005938 +/-0.000241,
exclusive-support+0.000822 +/-0.000011, total+0.006556 +/-0.000171. Within-bin
effects are positive in every case and persist with intrinsic-size strata.

Add fixed-cell rule/evaluator/reducer/launchers and numerical fixtures. Hold
emulator radius/magnitude exactly fixed within each fine bin, retaining original
membership and truth features. Original per-bin masses and coefficient sums
must replay before interpreting any intervention. Preserve per-target moments,
flow contexts and a fixed reference-photometry coefficient for later audits.
Analytical fixtures pass original-array preservation, identical representative
values and paired-moment conservation; Python/bash checks pass. Resource skill
records336GB available login memory and9TB disk, but all substantial work uses
Slurm. No other study GPUs are active; ceilings remain six GPUs/500 CPUs.

Pilot17807045 has meaningful progress on V100. CPU17807054 gates three remaining
workers17807060/17807061/17807062 and full report17807065. Extend existing bounded
check17806329 to include these stages. Broad17788286 stays held. No new images,
model training, offsets or modifications to the configured likelihood.

## 2026-11-06 — Four-scene factorial confirms selected-response coupling

A40 worker17806261 completed exit0 in17m32s; full reducer17806280 completed
exit0 in2s. All four original generated branches replay, all factorial/size/
coefficient identities pass in every bootstrap, and uncut selection branches
agree. Preserve [four-scene findings](evidence/photometry_selection_factorial_20261106/findings.md#evidence-and-subsequent-localization)
and update the earlier localization summary.

Shape errors for G/G,G/M,M/G,M/M are+0.005217,+0.005391,+0.004562,-0.001339,
with paired case-bootstrap SE0.001492,0.001546,0.001500,0.001549. Joint
substitution changes model response+0.006556 +/-0.000171; photometry-only
+0.000655, membership-only-0.000174, interaction+0.006075 +/-0.000032. Scalar
coefficient change+0.006547 dominates finite-transport geometry+0.000009.
All scene changes are positive; jackknife SE0.000199 supplements bootstrap,
half-bank contrast difference-0.000005, previous-bank M/M difference+0.000002
+/-0.000011. No training-uncertainty or independent-acceptance claim.

Response-bin pilot/reducer and prose refresh17806691 all completed exit0. Three
replication workers17806587/17806588/17806589 start with original dependencies
and hardware; full reducer17806590 remains gated. Broad17788286 stays held.
Check17806329 includes both diagnostics. Syntax/JSON/bash checks pass; no
production code, model, simulation or configured likelihood changed.

## 2026-11-06 — Binning pilot points to a within-bin neighbour-response mismatch

GPU17806566 completed exit0 in7m10s; CPU17806572 passed all raw-moment and parent
bootstrap replays. [Pilot interpretation](evidence/photometry_response_bins_20261106/findings.md#pilot-the-within-bin-term-dominates)
retains all fixed resolutions: of the0.006089 input-response change, within-bin
effects are0.005588–0.005987; distribution terms-0.000317–0.000141. Exclusive
support is separately retained (0.000040–0.000787), including endpoint cut
crossers. Intrinsic-size strata do not remove the dominant within-bin term.
One scene only; not proof of missing conditioning features or a calibrated fix.

Reducer computes each distinct whole-case bootstrap weight vector once, then
restores all10000 original ordered resamples for uncertainty and replay checks;
this exact reuse changes no estimator or seed. Correct report limitation prose
to cover either pilot or four-scene output. Remaining three GPU workers are
gated on the successful pilot and respective parent cases. No broad restart.

## 2026-11-06 — Response-bin dependent stages and bounded check prepared

Pilot17806566 reaches multiple blocks without failure. Its report17806572 gates
remaining A40/V100/RTX jobs17806587/17806588/17806589, each also gated on its
parent factorial case. Full reducer17806590 additionally waits for the parent
four-scene report17806280 so every bootstrap prediction can be replayed.
Record IDs in jobs.json. Extend existing read-only check17806329 to include
the new jobs, completion receipts and reports; it never releases held17788286.

Add factorial findings.md explaining the joint selected moment, the meaning
and limits of crossed inputs, and the additional uncertainty controls. Preserve
paired draw-level membership and endpoint support in response-bin accumulation.
No change to numerical models or earlier running evaluators. At most three
response-bin workers plus any remaining factorial worker, within six GPUs.

## 2026-11-06 — Matched cancellation budget complete; response-bin pilot started

Baseline1080 job17805335_4 and full reducers17804354/17805824 completed exit0.
Preserve the [matched four-scene budget](evidence/alignment_regularization_shift_20261106/findings.md#completed-matched-own-group-and-neighbour-budget):
actual-U own/opposite errors move-0.008736/+0.006989 to+0.001025/+0.007538.
Their globally normalized sum changes-0.001747 to+0.008563. Paired total error
shift+0.010310 +/-0.000031 is a model contrast, not an absolute calibration SE.
Both flows' neighbour predictions recover with measured photometry/selection;
the generated-input deficit predates regularization.

Factorial pilot17806179 and reducer17806215 passed all original-output replay
gates. Enhance report.py with scalar-coefficient versus finite-transport
contrasts, individual-scene results, antithetic half-bank differences and a
delete-one-scene jackknife supplement to the fixed bootstrap. Keep original
bootstrap NPZ axes intact. Updated pilot reducer17806542 completed exit0; the
one-scene diagonal shift0.006089 is almost entirely scalar coefficient0.006056.
All three remaining workers have meaningful progress; V1001080 already passed.

Add [response-bin protocol](evidence/photometry_response_bins_20261106/rule.md#question-and-fixed-comparison),
evaluate.py/report.py and launchers. Retain complete endpoint support and exact
factorial diagonal replay; split response changes into photometry-bin mass and
within-bin response effects, with exclusive support separate. Coarsening and
known analytical decomposition fixtures pass; AST/bash checks pass. Pilot
17806566 starts on V100 with first block complete; reducer17806572 gates later
work. No scientific bin result yet. Keep broad17788286 held, fixed cases/draws/
models, and at most six GPUs. No new images, retraining or empirical correction.

## 2026-11-06 — Factorial validation fixtures pass; dependent stages prepared

[Synthetic fixtures](evidence/photometry_selection_factorial_20261106/verification.json)
pass known two-target2x2 means, intervention order, interaction and preservation
of draw-level membership covariance. Factorial pilot17806179 is progressing
normally on V100. Its strict original-output replay and CPU reducer17806215
gate A40/V100/RTX remaining jobs17806261/17806262/17806263; full reducer17806280
reports all four scenes with shared whole-case bootstrap resamples, both
intervention orders and intrinsic-size contributions. Crossed branches retain
their own selected denominators. The generated/generated branch must reproduce
old opposite responses in every resample. Reports distinguish the new random
bank's M/M difference from scientific shifts and retain half-bank diagnostics.

Add bounded read-only check.py/run_check.sh; schedule17806329 for about two
hours later, preserve it in jobs.json. No release of paused17788286. At most
three new GPUs plus the older baseline job, below six authorized. Numerical
pilot outcomes and four-scene conclusions remain pending; no fitted correction.

## 2026-11-06 — Cross photometry inputs with selection membership

Owner asks to continue. Preregister a fixed weak-nine four-scene2x2 experiment
in [protocol](evidence/photometry_selection_factorial_20261106/rule.md#four-crossed-input-choices):
generated/measured photometry, independently crossed with generated/measured
selection weights. Conditional shapes follow their input photometry using the
same latent draws. Distinguish membership from input-value weighting without
claiming the artificial crossed branches are corrected generative predictions.
Retain both intervention orders, the symmetric effects and their interaction;
measured neighbour selection response stays separate.

Add evaluate.py, report.py, run_eval.sh, run_report.sh and verify.py under the
new evidence directory. Full-parent1024 blocks and original64 portable draws
replay the old generated/generated branch; hardware families match each case.
Actual common-U measurement sums and generated per-group mass/numerators have
strict replay gates. Two independent antithetic half-banks diagnose draw noise;
log-photometry summaries track their positive-output denominators without
altering zero-underflow draws. Syntax/bash checks pass. Synthetic fixtures test
known2x2 effects and draw-level shape/selection covariance.

Pilot17806171 was cancelled during startup to eliminate a local-name shadow
before predictions; corrected pilot17806179 runs on one V100 and has meaningful
progress. Reducer17806215 gates remaining cases. The earlier baseline1080 job
continues independently; broad17788286 remains held. At most three new workers
after the pilot plus the one existing worker, below six GPUs. No training,
image changes, empirical offsets or checkpoint selection.

## 2026-11-06 — Four-scene nine-cut residual localized to photometry/selection inputs

All conditional jobs17805816 and reducers17805817/17805818 completed exit0.
Extend report_fourstate.py with an explicit single-arm option so the completed
nine-cut arm uses all four predeclared scenes while baseline case1080 finishes;
no incomplete-case substitution. CPU17805954 completed exit0 in3s; all measured
replays and edge identities pass. Extend synthesize_neighbour.py with --nine-only
and selected-mass ratios. CPU17805962 completed exit0 in2s, verifies shared
cases/bootstrap seed and identical measured reference in every resample, and
writes [paired comparison](evidence/alignment_regularization_shift_20261106/neighbour_input_nine_results.md).

Common-U equal-group neighbour shape errors: generated inputs+0.005217
+/-0.001492; measured photometry with conditional nine-cut shapes-0.001337
+/-0.001546; fully measured inputs-0.001274 +/-0.001458. Photometry/selection
substitution changes prediction+0.006554 +/-0.000175; replacing conditional
shapes changes only-0.000063 +/-0.000115. Missing neighbour selection response
is+0.002925 +/-0.000418. Selected mass ratio0.998324 +/-0.001624. Separately,
actual-U global own/opposite/total errors are+0.001025/+0.007538/+0.008563,
with case SE0.004583/0.001627/0.005466. Do not combine these normalizations.

Interpretation is photometry/selection input misweighting plus missing neighbour
selection; no detected mean emulator shape error on measured inputs. Distinguish
remaining radius/flux, membership/weighting and conditioning/underfit hypotheses.
AST/bash checks pass; numerical identities and all staged pilot gates pass after
the documented dtype fix. Baseline case1080 has meaningful progress; complete
two-arm report17804354 and synthesis17805824 retain its dependency. Broad
17788286 remains held. Bounded check17804408 remains scheduled; no new training,
simulation, offset or model selection.

## 2026-11-06 — Pilot isolates photometry/selection input effect

Conditional pilot reducers17805812/17805813 completed exit0 in3/4s. Preserve
both conditional_transport_*_pilot_report.json files. For case1000, replacing
generated photometry/selection with measured photometry while drawing shapes
from each flow raises the opposite-group model response0.084603->0.090387
(baseline) and0.084072->0.090128 (weak nine-cut). Measured shape response0.090202;
fully measured-input emulator0.090354. The final conditional-to-measured shape
substitution changes response by only-0.000033/+0.000226. Nearly all pilot
input sensitivity is therefore photometry/selection. Remaining17805816 keeps
all three other cases, fixed checkpoints and64 common portable draws. Do not
claim population confirmation until the full reductions pass; radius versus
flux and density underfit versus missing scene conditioning remain unresolved.

## 2026-11-06 — Conditional sampler pilot passes; preserve casewise robustness

Retry17805843 completed exit0 in3m24s after fixing index dtype. Both flows
passed conditional joint-draw reconstruction, photometry preservation and
measured-sum replay;64 antithetic draws share portable latent banks across
arms/endpoints. CPU pilot reducers17805812/17805813 gate the other three cases.
Casewise reducer17805868 completed exit0 and independently reproduces the pooled
measured-input values. Selection terms are positive in all four scenes
(0.00182–0.00418); measured-input shape errors range-0.00616 to+0.00140.
Preserve [casewise values](evidence/alignment_regularization_shift_20261106/casewise_measured_transport.json)
instead of treating the favorable aggregate as uniform accuracy.

Code inspection identifies an untested information limitation: the flow's three
crowding summaries omit neighbour size/profile/separation information used
explicitly by the emulator. A calibrated marginal output distribution may still
miss its association with those variables. Document as a hypothesis, distinct
from density underfit; no causal attribution or improvement claim.

After V100 task17805336_5 and RTX task17805337_7 completed successfully, raise
A40 array17805335 to three workers and restrict remaining conditional17805816
to inter. With one remaining V100 and one conditional worker, at most five
GPUs can run; the user's six-GPU ceiling and held broad array are preserved.
Final synthesis17805824 remains gated on all four-scene reports. Existing bounded
check17804408 reads all new stage identifiers. No new images or retraining.

## 2026-11-06 — Four measured-input scenes consistent with zero shape bias

All measured-input jobs17805747/17805767 and reducer17805770 completed exit0.
Preserve [report](evidence/alignment_regularization_shift_20261106/measured_transport_report.json)
and [interpretation](evidence/alignment_regularization_shift_20261106/findings.md#four-scene-measured-input-neighbour-test).
Target-cut opposite-group shape error=-0.001274 +/-0.001458; missing selection
response=+0.002925 +/-0.000418; conditional total error=+0.001651 +/-0.001762.
All use equal A/B within-group normalization, common-all-four usability and
paired four-case SEs. Do not double them or substitute them into the coherent
full-population budget. Endpoint-anchor and finite-transport differences are
small; conditional size-bin residuals cancel, consistent with owner's acceptance
of internal cancellations. No new model or calibration correction.

Conditional pilot17805804 reached READY after its conditional-sampler checks,
then failed before predictions because group indices retained an unsupported
integer dtype. Make the index tensor explicitly torch.long; retry17805843.
Update pilot reducer dependencies17805812/17805813 to successful retry. Preserve
failure and all original settings. Broad evaluation remains held. Remaining
conditional and synthesis jobs stay behind the pilot/report validation gates.

## 2026-11-06 — Matched measured inputs remove pilot shape residual; conditional follow-up

Measured-input pilot17805747 completed exit0 in1m45s; all measured cache replays,
finite-output and transport inversion checks passed. CPU17805749 passed exact
shape/selection identities. On case1000, equal-group opposite response measured
shape0.090202 versus anchored emulator0.090354 gives error-0.000153. Endpoint
predictions differ0.000088; exact-versus-infinitesimal prediction differs0.000012.
The independent selection term remains+0.002893. This is preliminary evidence
for an input-distribution effect; a one-scene result has no population SE.
Three remaining measured-input jobs17805767 run sequentially; full reducer17805770.

Preregister and add evaluate_conditional_transport.py and its launcher to
separate measured photometry from measured shape: both fixed flows sample
conditional shapes at identical measured radius/flux, with64 portable paired
draws, common actual usability and endpoint-average selection weights. Check
conditional joint-draw reconstruction and measured-source replay. Extend
report_measured_transport.py to reduce these outputs and retain matched bootstrap
arrays; report_fourstate.py also stores its paired opposite-response resamples.
Add synthesize_neighbour.py for paired input-substitution changes, exact matched
measured references and explicit cohort limitations. AST/bash checks pass.
Pilot17805804 follows the measured-input reducer; CPU17805812/17805813 gate
remaining17805816 (one GPU at a time), then reducers17805817/17805818 and
synthesis17805824, also gated on four-state report17804354. Check.py and jobs.json
record the chain. No broad evaluation resumed, no model weights changed, at most
six concurrent focused GPUs. Monitor pilot startup; four-scene scientific results
are still pending at this entry.

## 2026-11-06 — Separate neighbour selection; test transport on measured inputs

Owner requests deeper localization. Extend
[four-state reducer](evidence/alignment_regularization_shift_20261106/report_fourstate.py)
to reuse the original common-usability measured edge sums and report exact
shape/selection terms with paired case resampling. Measured sums replay and all
edge identities pass in pilot CPU17805748 (exit0,2s). In case1000 the equal
average of within-group opposite responses is0.093095, consisting of
shape0.090202 plus selection0.002893. Model responses are0.084603 baseline and
0.084072 weak-nine; remaining shape errors0.005598/0.006129. These are one-scene
diagnostics with no population SE, not global coherent contributions.

Preregister measured-endpoint conditional transport in
[rule](evidence/alignment_regularization_shift_20261106/rule.md#deep-neighbour-diagnostic-fixed-before-results-2026-11-06).
Add evaluate_measured_transport.py, report_measured_transport.py,
run_measured_transport.sh and verify_measured_transport.py. Use the unchanged
GalSBI-dev emulator (cd32781d...), actual common-usable shape/radius/flux,
and fixed A/B neighbour groups. Invert endpoint model transport then predict
the opposite-group shear change with fixed endpoint coefficients; retain both
anchors, their average, the infinitesimal comparison and intrinsic-size
contributions. This is a conditional test, not a deployable prediction or
empirical correction. Analytical known-transport fixture and syntax checks
pass. Case1000 GPU17805747 started and reached meaningful progress; CPU17805749
gates remaining cases17805767 (one GPU at a time). Together with the five
existing focused workers this respects six GPUs. Broad17788286 stays held.
Extend check.py for the new stages. Full four-scene results remain pending.

## 2026-11-06 — Distinguish emulator pair calibration; recover pilot metadata

Owner notes the neighbour-weighted standalone emulator residual is consistent
with zero. Clarify in
[findings](evidence/alignment_regularization_shift_20261106/findings.md#why-the-standalone-emulator-null-does-not-close-this-budget)
that this projected pair statistic does not directly measure coherent selected
R11. The descriptive joint-minus-standalone gap persists near0.015 for baseline
and nine-cut endpoints; different experiments prevent interpreting that gap as
a measured emulator error. No claim of uniquely localized component bias.

Focused V100 pilot17805334 passed replay gates and saved its NPZ, then failed
on receipt writing because a replay-loop variable shadowed device metadata.
Rename that variable in evaluate_fourstate.py. Add recover_pilot_receipt.py
to audit existing arrays against measured and historical coherent sums, verify
checkpoint/latent hashes and existing physical bounds, and reconstruct only
the missing metadata with explicit recovery provenance. Both scripts pass AST
syntax checks. CPU audit17805618 completed exit0 in3s; strict saved-output
replay passed. Pilot reducer17804353 completed exit0 in2s. Rebind remaining focused arrays
17805335–17805337 and pilot reducer17804353 to its successful completion.
Keep original failed job/log and arrays unchanged. Broad17788286 remains held;
focused work permits at most five GPUs, below six authorized. Full four-case
inference remains pending; do not promote a one-case pilot to a population result.
Preserve [pilot report](evidence/alignment_regularization_shift_20261106/fourstate_pilot_report.json):
case1000 actual-usability own/opposite/total errors change from
-0.014142/+0.007196/-0.006946 to-0.004323/+0.007748/+0.003425.
Own-group includes same-group neighbours; opposite-group includes neighbour
selection effects. No case SE exists from one scene. The persistent opposite-group
point residual is preliminary evidence, not a full component attribution.

## 2026-11-06 — Compile available component table; keep missing flag rows explicit

Owner asks for the earlier six-row error table for current models. Preserve
[available Cartesian table](evidence/alignment_regularization_shift_20261106/available_error_table.json):
five of six rows exist for the improved unregularized checkpoint. Flow-generated
selection DeltaR11=-0.015366 +/-0.003347 on40 validation cases; joint40-case
predicted/generated,-0.002492 +/-0.002724; true/generated,-0.000607 +/-0.002730;
true/measured,+0.027384 +/-0.002781. Unchanged emulator projected-pair diagnostic
is-0.002556 +/-0.002293. The quoted historical flow rows used half-trace;
the new table explicitly uses R11 for flow/joint and labels emulator separately.
Missing standalone measured-selection-mask row is not filled with measured-reference
normalization or conditional-photometry numbers. New regularized measured-selection
rows also remain missing. No broad held evaluations resumed.

During status audit, strict same-device mixed replay17804916 passed but failed
its historical-cache mass-fraction gate on mag25:1.92e-7 versus1e-7, while
total response difference was2.07e-6. No physical prediction altered. Optional
V100 probe17805079 completed; compare17805080 failed the pre-existing per-draw
tolerance in4/262144 elements. Descriptive CPU17805316 retains those failures,
identical portable latent hashes and numerical differences; it does not relax
that tolerance or constitute a pass. Retry on historical GPU families instead:
pilot17805334 V100; remaining17805335 A40 (2workers),17805336 V100(2),17805337
RTX(1); previous pending17804352 cancelled and report dependencies updated.
All broad workers are now finished and pending work remains held. At most five
focused GPUs after the pilot, within authorized six. No further gate widening.

## 2026-11-06 — Owner pauses earlier coherent evaluation expansion

Owner asks to pause earlier evaluations while localization continues. Hold
pending regularization array17788286 tasks221-399; scontrol verifies
JobState=PENDING/Reason=JobHeldUser/Priority0. Preserve all completed outputs
and let running217-220 finish; no active worker cancelled. Keep final reducer
17788351 dependency-pending. Cancel automatic resource-restore job17804373 so
the focused study cannot inadvertently resume the held evaluation. Record the
pause in both jobs receipts; resumption requires an owner request. Completed
alignment diagnostics and focused four-state retry17804916/remaining17804352
continue within the six-GPU ceiling. Initial scontrol message about already
finished subtasks is harmless; the pending user hold is explicitly verified.
Persist the scoped pause in AGENTS.md. Retry17804916 has reached READY and
block0 with the strict independent same-device check. Current running coherent
worker220 has a V100, so queue optional device probe17805079 after it exits and
CPU comparison17805080; these do not block the focused mixed-state diagnostic.
No held work is released automatically.

## 2026-11-06 — Add strict same-device replay after mixed-state numerical mismatch

Four-state baseline17804351_0 passed all historical-cache checks and completed0
in6m55s. Weak-nine17804351_1 failed its raw-sum gate by0.00016172; cached
coherent run used V100, new mixed evaluation A40. Same pinned weights and
portable latent stream; measured and split-neighbour checks passed. Preserve
failure; do not treat it as calibration evidence. A40 probe17804825 completed;
cancel V10017804826/compare17804827 after scheduler predicts multi-day wait.
No two-device cause established by these incomplete probes.

Amend [numerical validation](evidence/alignment_regularization_shift_20261106/rule.md#numerical-replay-amendment-after-pilot):
every block now evaluates all-neighbour and split-neighbour response independently;
strict same-device aggregate replay retains2e-7 absolute/2e-10 relative tolerance.
Measured replay unchanged. Separately bound historical-cache differences in
physical units: selected-mass fraction<1e-7, normalized leg means<1e-7, component
and total response differences<5e-6. Record actual differences; no empirical
offset or replacement of predictions. These pre-rerun numerical bounds are far
below the observed response shift. Weak-nine retry17804916 gates existing
remaining17804352 and pilot reducer17804353. Syntax passes; awaiting startup.
Existing six-GPU ceiling, release17804373 and bounded check17804408 preserved.

## 2026-11-06 — Full validation confirms shape-response shift and retained alignment recovery

All six standalone17804090 evaluations completed0; CPU17804172 completed0 in43s.
Preserve [full-validation report](evidence/alignment_regularization_shift_20261106/findings.md#completed-full-validation-replay).
All40 measured sums replay, finite-pair shape/selection identities close in all
bootstraps. At the hard target cut weak-nine DeltaR11 improves-0.015438 to-0.006063,
but DeltaR22 worsens+0.000997 to+0.006572. Model R11 decreases0.0093755, closely
matching the0.0092820 coherent flow shift. Shape-change term accounts for93.1%;
cut-crossing term6.9%. Measured-reference normalization does not remove the shift.

New portable-bank alignment errors (intrinsicRe0.15-0.25,mag25,radius2.7-2.8)
are-0.004247 +/-0.004253 for improved Cartesian and-0.014892 +/-0.004312 after
weak-nine regularization. Preserved FS2 NLL is+0.032291 +/-0.004626; old GalSBI
NLL+0.413230 +/-0.004720. Just-above-cut Cartesian alignment improves+0.027125
to+0.013163. The old large alignment failure does not return; modest profile
changes coexist with a substantial response change. FS2 is an independent-family
reference and alignment errors are not m. Four-state pilot17804351 is running,
has passed measured-cache replay and split-neighbour checks and reports block
progress. Full same-image attribution remains pending. Bounded2h check17804408.

## 2026-11-06 — Trace regularization shift; recheck alignment and same-scene neighbours

Owner requests localization and comparison to FS2 alignment. Added
[shift study](evidence/alignment_regularization_shift_20261106/rule.md#questions-and-fixed-evidence)
with frozen checkpoint/case manifest, cached coherent component decomposition,
full40-case standalone alignment/response evaluations, and reuse of four-state
images for original Cartesian versus weak nine-cut. No training or new images.

CPU17804158 completed0 in20s; all19-case interim sums and exact decomposition
identities replay. Flow contributes89-94% of regularized R11 change. Weak
nine-cut flow shifts-0.009282 +/-0.000030, emulator-0.001121 +/-0.000002;
0.15-0.4arcsec intrinsic sizes carry88.3% of the flow shift. Conditional-mean
term-0.009658, selected-size-fraction term+0.000376. Paired change precision
must not be described as absolute calibration precision. See
[findings](evidence/alignment_regularization_shift_20261106/findings.md#the-coherent-shift-is-mostly-in-the-flow-contribution).

Array17804090 evaluates all six endpoints with portable CRNs and exact
finite-pair shape/selection decomposition; CPU17804172 will reduce all40 cases
and independent FS2 NLL reference. Existing evidence already shows improved
Cartesian GalSBI near-zero alignment error in the original problematic bin;
above-cut errors and full response remain distinct. New results test whether
the guard changes alignment or mostly response sensitivity.

Four-state pilot17804351, remaining17804352 and reports17804353/17804354 reuse
existing mixed images and frozen groups. Require measured replay, new coherent
model replay and zero common-U within-group opposite flow response. Original
emulator/C8 unchanged. Four cases are diagnostic, not independent acceptance.
Syntax/bash checks pass; first standalone workers progressed/completed normally.
CPU resource audit~696GB free host memory/~14TB free project space; allocations
8GB for CPU reports,24GB/one GPU per evaluator. Main coherent array throttle4
plus diagnostic throttle2 preserves six-GPU cap. Temporarily Nice100 for pending
main tasks; no running jobs interrupted. Release17804373 restores throttle6/Nice0
after mixed-state work; superseded17804091 cancelled. scontrol's finished-task
warning did not prevent throttle update, verified by scontrol/squeue.

## 2026-11-06 — Interim regularization results shift coherent m1 upward

Owner requested status. All80 portable baseline evaluations and reducer17787293
completed0; preserved [baseline report](evidence/alignment_target9_20261106/findings.md#completed-40-case-baseline).
Main-cut predicted-usability m1=-0.2552 +/-0.3019%, m2=-0.5153 +/-0.2471%;
new random realization changes old central results by about+0.02pp. All ten
endpoint pilots and reducer17788287 completed0. At check194/400 endpoint
evaluations completed, five active, none failed; six-worker ceiling unchanged.

Add optional interim phase to report_constant.py/run_report.sh, intersecting
completed case/receipt sets across every model and both axes before reading
numerical outcomes. Full/pilot scientific logic unchanged. Python AST and bash
syntax pass. Scheduled CPU17803908 completed0 in21s, freezing19 common cases.
Preserve [interim report and interpretation](evidence/alignment_regularized_20261106/findings.md#interim-constant-shear-comparison-19-common-cases),
jobs metadata and updated baseline findings. On the matched19 cases central
m1 moves from-0.2763 +/-0.4436% to+0.8735/+0.8779% for one/nine-cut weight0.001,
and+1.0817/+1.2642% for weight0.01 (each SE about0.45pp). Paired shifts are
+1.15 to+1.54pp with0.007-0.008pp SE; do not confuse this with absolute precision.
True usability preserves the upward shift, and nine-cut RMS also worsens.

These fixed64-step trials improve standalone proxy loss but do not yet preserve
joint calibration. No arm/checkpoint selection or early stopping; finish the
planned40-case comparison. Reducer17788351 remains queued. New bounded check
17803918 in2h; recent throughput suggests roughly4-5h remaining if allocations
hold. Existing models, cuts, seeds and simulation settings unchanged.

## 2026-11-06 — Complete one-cut and nine-cut continuations; joint transfer pending

All five fixed-step64 continuations17788119 and training reducer17788251
completed exit0. Preserved [training report](evidence/alignment_regularized_20261106/training_report.json)
and updated [findings](evidence/alignment_regularized_20261106/findings.md#completed-fixed-step-continuations).
Common initial predictions match, replay/direct gradients agree at~1e-7 and
checkpoint reloads are exact. One/nine-cut weights0.001 cost+0.000880/+0.000704
validation NLL nat/row; weights0.01 cost+0.002966/+0.002764. The common nine-cut
validation proxy score falls from10.2099 to3.7259/3.8198 and2.4406/2.0632.
NLL-only control costs+0.018848 and scores11.4363. All64 updates in every arm
used the declared norm5 clipping. No arm/checkpoint selected from these probes;
they do not prove coherent calibration preservation or converged performance.

Portable baseline axis2 pilot17787232_1 completed0 in5m13s; axis1 has regular
block progress on slower A40-8Q. Remaining78 baseline tasks and then400 endpoint
tasks preserve the six-GPU ceiling and dependency/pilot gates. Update baseline
findings to distinguish measured replay from the changed latent realization.
The user's constant-shear question remains pending the paired40-case, both-axis
reports. Existing bounded checks17787294/17788449 cover later stage status.
Documentation-only closeout; no additional numerical changes or tests needed.

## 2026-11-06 — Verify regularizer gradients; fix cross-GPU latent reproducibility

Preparation17788350 completed0 in2m53s after correcting raw8/transformed16
context width. Targets cover23.28M training pairs/160 cases,5.82M validation
pairs/40 cases. Three A40-16GB continuation workers fit24GB host/16GB device;
replay versus direct-autograd gradients agree~1e-7. Fixed endpoint jobs are
progressing; NLL control and first regularized arms completed exact reload.
Full A40 allocations were queue-delayed; resource-only update kept job IDs and
scientific settings. No fitted-model selection performed.

The broad radius scan's later tasks exposed strict replay failures across GPU
families; do not loosen tolerance. Case1083 old A40/new RTX model sums differed
by18.13 despite matching seeds. Explicit device probes17788989/17788990 and
CPU compare17789022 establish different CUDA latent hashes for1024-row banks,
identical hashes for528-row tails; TF32 off. New NumPy PCG64 CPU float32
antithetic banks give bitwise-identical physical draws on both GPUs at both
sizes. [Verification](evidence/alignment_target9_20261106/device_verification.json).

Stop incomplete broad array17786000 and full reducer17786002; preserve partial
outputs/failures. Resource-drain requirements satisfied, so train17788119 now
depends only on successful preparation; cancel obsolete release17788490.
Exact-grid pilot17787232 waits on training and device check17789022. Baseline
and all regularized evaluations use the same verified portable sampler and
record its SHA. This is an explicitly new MC realization, with same models,
64draws, cases and seed formula. Model replay against old CUDA draws is replaced
by an explicitly reported legacy-realization comparison; measured-catalogue
replay remains required. No empirical offsets or scaling. Full baseline reducer
will quantify paired sampling changes on six overlapping old cuts.

Update protocols/findings/jobs receipts. Next: complete continuation report,
watch portable baseline pilot and full-stage startup, then all endpoint joint
calibrations. The local-response preservation question remains open pending
coherent simulations; NLL/proxy improvements alone are not its answer.

## 2026-11-06 — Test whether one-cut and nine-cut penalties preserve Cartesian calibration

Owner asks whether the improved NLL-only flow retains performance with response
regularization. [Fixed experiment](evidence/alignment_regularized_20261106/rule.md#fixed-comparison)
compares NLL continuation and one/nine-cut penalties at weights0.001/0.01,
all starting from Cartesian seed611.64 matched AdamW1e-5 updates; same8192-row
NLL batches and independent32768-pair x4-draw guard banks. Full2x2 selected
response, scale0.003 x training half-trace per cut, softness0.05px/0.05mag.
Fixed endpoints, no validation/coherent-bias checkpoint or arm selection.

Added CPU preparation, GPU continuation with direct-autograd/replay gradient
check and exact reload, train/probe report, coherent evaluator and paired
nine-cut reducer. All targets from160 flow-training cases;500k fixed validation
NLL and262144-pair probes. User's nine joint cuts are preserved exactly. No
production model changes or new images. Constant-shear all40 scenes/both axes
queued for every endpoint, fixed classifier/emulator/CRNs; matched NLL control
separates continuation effects. Scientific result remains unknown until tested.

Syntax/bash checks pass. First prepare17788090 failed at first transformed
context block (raw8 versus transformed16 columns); replacement17788350 infers
actual preprocessor width. Failed partial directory preserved, no weights
changed. Resource audit on output filesystem:~14TB free; scheduler-only compute.
Train17788119 five tasks%3, with old active-scan drainage dependencies; broad
scan17786000 throttle3. Exact-grid baseline17787232 now waits for training too.
Release17788490 restores scan throttle6 after training. Train report17788251;
constant pilot17788252, remaining17788286, reports17788287/17788351; check17788449
in2h. Combined six-GPU ceiling preserved. Next: verify corrected preparation,
training gradients/first updates, then dependent validation and calibration.

## 2026-11-06 — Accept internal cancellation; refocus on the owner's nine target cuts

Owner clarifies that internal cancellations are acceptable and local response
robustness is the objective. Persist this scoped preference in AGENTS.md;
0.75-arcsec/subpopulation diagnostics are not automatic rejection criteria.
[New protocol](evidence/alignment_target9_20261106/rule.md#owner-objective) fixes
radius>2.8/3.0/3.2pixels AND magnitude<24.9/25.0/25.1, exactly nine joint cuts.
The improved Cartesian flow has no response penalty yet (NLL-only training).
Earlier old-architecture nine-cut runs used24.8/25.0/25.2 and are not this trial.

Prepare exact-grid unchanged-model baseline on both constant-shear axes, all40
scenes, original C8/emulator and CRNs. Add14-cut evaluation (nine target plus five
replay-only selections), intrinsic-size sums, individual response components,
and paired nine-cut RMS/max/range summaries. No images or model weights changed.
Syntax/bash checks and exact Cartesian-product assertion pass. Six-cut overlap
replay required for every case, using the already validated radius-scan code.

Queued pilot17787232 after current scan17786000, remaining17787261 afterpilot,
reports17787262/17787293, bounded2h check17787294. Six-GPU ceiling preserved;
all new work is dependency-pending, so startup will be checked when eligible.
Regularized training is not launched: specify NLL-plus-nine-cut full2x2 loss,
160 training-case targets, independent-pair gradient checks and checkpoint
criterion before training/validation comparisons. Held-out flow and coherent
constant-shear checks test transfer; never fit to coherent-test outcomes.

## 2026-11-06 — Complete axis/seed/recipe checks; expose radius cancellation and scan it

All prior pilot, packed simulation, matched-control, seed-replication and g2
GPU jobs completed0. Packed simulation jobs finished in1h40m..3h37m; full g2
report17772097 available. Preserve four reports and update study findings:
[axis and cancellation](evidence/alignment_axis2_constant_20261105/findings.md#completed40-scene-results),
[seeds](evidence/alignment_cartesian_seeds_20261105/findings.md#completed-seed-replication),
[matched controls](evidence/alignment_matched_constant_20261105/findings.md#completed40-case-matched-controls).

Main-cut factorized Cartesian m1=−0.2739±0.2995%,m2=−0.5354±0.2466%; extra
seed m1=−0.3786±0.3026% and−0.6322±0.2992%. Case uncertainties condition on
fixed models. Retained model m1=+2.4288±0.3102%,m2=+0.1681±0.2492%; original
axis inconsistency shrinks in Cartesian. At mag<25.8,r>=0.75 the new model gives
+2.3464±0.3713% and+2.3759±0.3365%, so broader closure is still absent.
Matched full-dimensional Cartesian gives+0.9040±0.3029%; its paired difference
from factorized is−1.17794±0.00885pp. Do not equate paired precision with total
calibration precision or attribute all retained-to-new change to architecture.

Added nested_selection.py: exact cached per-leg count/shape decomposition by
measured-radius region and intrinsic magnitude. Initial CPU17785969 failed on
wrong limited-cut cache; corrected to pinned grid cache,17785999 completed0 in10s.
All additive/normalization checks pass. For mag25.8,r>.6, Cartesian DeltaR11
splits−.011130+.009364=−.001766; DeltaR22−.011981+.009457=−.002524. This proves
cancellation within the selected sample. Symmetric mean/fraction decomposition
is descriptive, not a causal component error budget. Preserve JSON receipts and
aggregate unmatched-row counts in findings.

Launch [radius scan](evidence/alignment_radius_scan_20261106/rule.md#question-and-fixed-settings):
23 fixed cuts, five intrinsic-size bins, two usability modes, both axes, all40
existing scenes. Same64 antithetic draws/CRNs and frozen seed611. Per-case six-cut
replay gate; no empirical offsets. Syntax/bash checks pass. Pilot17785984_0/1
completed0 in5m12s, all six replay cuts passed; pilot reducer17786001 completed0
in4s. Remaining17786000 started six workers across cip/inter; all six reached
PROGRESS block0 or later. Full report17786002;
one-shot status17786134 in2h. One-scene pilot SE explicitly null. No science code
or production model preset changed. Next: check full-stage progress and use full curves to localize conditional response
errors before attempting model changes or independent acceptance.

## 2026-11-06 — Owner authorizes500 CPUs; pack all72 remaining axis2 legs

Persist the owner ceiling in AGENTS.md. Add BlendEMU launcher
`jobs/job_constant_axis2_packed.sh` (receipt in
[evidence](evidence/alignment_axis2_constant_20261105/findings.md#owner-authorized-cpu-expansion)):
nine8-CPU/160GB allocations, eight unchanged singleton per-leg workers each.
Array17774673 replaces pending17772073, preserving pilot17772072 prerequisite.
Rewired evaluator17772096 before cancelling old array; seed-constant dependency
and six-GPU ceiling unchanged. No science, catalogue, seed or model changes.

Validation: bash -n; mapping covers indices8..79 exactly once; CPU smoke17774672
COMPLETED0 in3s, all eight workers rank0,size1 and correct IDs. First smoke17774658
failed in environment activation with nounset; using original launcher shell
semantics fixed it. Production waits for the running pilot; cannot yet claim
72-worker throughput.72 useful workers fit within500; allocating the remainder
would not accelerate this unchanged case-parallel implementation. Existing
bounded check17772183 reads amended jobs.json. Rule/findings/jobs receipts updated.

Runtime audit17774601 COMPLETED0 in2m29s: all80 sersic_n arrays exactly equal to
original g1 inputs,122 rounded values each; grid roundoff at most8.9e-16.
Eight rendered pilot truth catalogues also pass. Preserve runtime_audit.json.
Next: inspect completed pilot timings and packed-stage startup, then axis2 report.

## 2026-11-06 — Trace slow axis2 ETA to CPU concurrency; retain rounded Sersic grid

Owner asks why new simulations take longer and whether n rounding is active.
Read original scene writer: sersic_n uses122 logarithmic nodes[0.3,6.2], native n
is a separate field; renderer reads sersic_n. New g2 prep preserves every non-
shear input column. Added [runtime audit](evidence/alignment_axis2_constant_20261105/audit_runtime.py),
CPU17774601, comparing all80 input arrays with old g1 and the exact grid, plus
rendered inputs. No rounding or renderer change made.

Earlier job17643819_0 allocated10 MPI workers for10 cases and finished4h37m
(render/detect1h36m, shape3h01m). Historical campaign used280 constant workers,
up to500 with training. New rendering logs show50–62min/image and2–3min detection;
the35–55h estimate included shape fitting and an agent-imposed eight-job cap.
Removed that unnecessary bottleneck: remaining array17772073 now throttle40,
verified live in scontrol. Cluster limits50jobs,3500CPUs,20000GB/user permit it.
Keep the eight-leg pilot gate, job IDs, seeds, catalogues and all scientific
settings unchanged. Amend rule/findings/jobs receipt. Provisional revised ETA
12–18h remaining, dependent on actual fit runtimes and scheduling; monitor the
first completed pilot before treating this as a measured throughput estimate.

## 2026-11-05 — Seed and axis2 startup checks; preserve combined resource ceiling

Seed612 job17772044_0 reaches finite improving NLL through initial epochs;
seed613 remains queued for an A40. Pending matched-control tasks now use inter
so cip A40 slots can serve training; running allocations unchanged, four controls
plus two training/standalone GPUs maximum. Record exact updates in seed jobs.json.
All40 g2 catalogue pairs prepared with source/non-shear checks; all40 new C8
probability caches completed, including original g1 probability replay. Both
axis2 reducer checks17772095/17772200 pass known response and3% m2. Added paired
axis changes using common case-bootstrap weights (m2−m1, diagonal/cross errors).
All eight pilot image jobs started and wrote rendered input catalogues; images
and measurements remain running, so no g2 calibration result exists yet.

Saved startup receipts for both studies. Resource discovery17772184 runs on a
CPU allocation, not the login node; project filesystem has~14TB free. Bounded
read-only combined status job17772183 is scheduled about2h later and writes
scheduled_check.json; it does not wake the chat. Existing matched status17771762
remains intact. No model/image settings adjusted from pilot outcomes.

## 2026-11-05 — Launch Cartesian seed repeats and second-component simulations; check existing null component

Owner authorizes the proposed continuation. Added [seed-replication study](evidence/alignment_cartesian_seeds_20261105/rule.md#scientific-choices)
with seeds612/613, fixed Cartesian factorization and original NLL checkpoint rule.
Training17772044 -> standalone17772045 -> report17772053; constant17772054 also
waits for matched controls17771754, then report17772055. Reduced matched-control
throttle to4 (bulk update warns about completed members; live state verifies4).
Training waits for the recorded active control allocations, keeping totalGPU<=6.
Only seeds/initialization/batches change; no selection by constant-shear results.

No existing GalSBI g2 constant suite found. Added [axis2 protocol](evidence/alignment_axis2_constant_20261105/rule.md#predeclared-experiment),
manifest, C8 prep, frozen joint evaluator and full reducer. BlendEMU owns new
scripts prepare_constant_axis2.py, verify_constant_axis2.py and job_constant_axis2.sh
under /home/z/Zekang.Zhang/blendemu_galsbi_resim. Preserve original40 scenes,
all non-shear properties, noise seeds, PSF/detection/centered65px measurement;
only set g1=0,g2=+/-0.02. Reuse exact-image control after source-hash checks.
Original ngmix initialization remains unseeded; no claim of deterministic shapes.
Prepare17772071 -> eight-leg operational pilot17772072 -> remaining72 legs17772073,
CPU array cap8. C8 axis2 probabilities17772094 include axis1 replay; synthetic
reducer17772095 passes R12=.03,R22=.8,m2=3% with unequal counts. GPU17772096
waits for these prerequisites and seed GPU array17772054; report17772097 follows.
AST/shell checks pass; monitor startup and use bounded follow-up for long jobs.

Owner asks about g2~0 on the existing g1 scenes. Added null_component.py and
[paired null results](evidence/alignment_factorized_constant_20261105/findings.md#existing-g1-scenes-second-component-null-checks),
CPU17772085 completed. Cartesian DeltaR21=−0.000218±0.001936; even e2 residual
+0.000203±0.000120, about1.7sigma. Frame values−0.003430±0.001934 and
+0.000045±0.000115. Even e2 is not an inferred g2. Existing g1 data test R21
and even offset; R22 multiplies zero and requires a nonzero second-component
perturbation. No production models or original simulation products changed.

## 2026-11-05 — Measured-photometry diagnostic leaves conditional-shape residual

Both conditional40-case evaluations17771721 completed exit0 in4m04s, paired
reducer17771722 in51s. Added matrix-error combinations in reducer17771808
(47s,exit0) to preserve covariance when comparing diagonal components; no new
sampling. [Findings](evidence/alignment_conditioned_photometry_20261105/findings.md#result-and-interpretation)
and completed receipt retain full absolute response/size/alignment evidence.
At radius>3/mag25, generated-to-measured-photometry DeltaR11 changes from
−0.015366±0.003347 to−0.013609±0.003470 Cartesian, and−0.009345±0.003358 to
−0.005823±0.003419 frame. Thus measured photometry and exact selection do not
remove standalone shape error. Frame Re0.25–0.4arcsec contributes−0.005422±0.002048;
Re0.15–0.25 contributes+0.000584±0.002563. Contributions add, not subgroup means.

Conditional (DeltaR11−DeltaR22)/2 is−0.008410±0.002400 Cartesian and
−0.004637±0.002405 frame; measured-photometry intervention barely changes this
imbalance. Scalar alignment does not test the full response. These are existing
validation cases and fitted-model case-bootstrap errors, not training uncertainty
or independent acceptance. No claim that physical R11 must equalR22. Matched
constant controls continue on six GPUs; both newly admitted workers also pass
first-block progress. Full report remains17771756, bounded2h check17771762.

The completed constant-shear plot was rendered on CPU17771710, legend placement
fixed on17771751, both exit0, and visually inspected. All modified scientific
source AST and shell checks pass. No production preset, retained checkpoint,
measurement or emulator changes.

## 2026-11-05 — Add matched full-dimensional constant-shear controls

The completed factorized-vs-retained comparison changes architecture, learning
schedule and response-penalty history together. To separate these, added
[matched protocol](evidence/alignment_matched_constant_20261105/rule.md#question-and-predeclared-scope),
frozen checkpoint manifest, adapted validated constant evaluator, paired five-arm
reducer, scheduler launchers and bounded status check. Joint-Cartesian FS2-rate
and joint-frame controls already trained under the same140epoch budget and
selection rule as the factorized arms; images/C8/emulator/CRNs are unchanged.
Parameter allocation differs slightly; this is an architecture comparison, not
an exact equal-capacity intervention or independent acceptance.

Array17771754 evaluates both controls on all40 scenes with four workers. Alongside
two conditional-photometry workers, total concurrency is at most six authorized
GPUs. Pilot17771755 depends on first four scenes for both arms; full17771756
on all80 tasks. Status check is recorded in the study jobs.json, scheduled about
two hours after submission. AST/shell checks pass, checkpoint hashes pinned.
Both original control arms pass first-block progress (A40 and RTX2080Ti);
startup receipt saved. Conditional array17771721 completed both arms exit0 in
4m04s, all40 cases each; reducer17771722 started normally. Increased the control
array to six only after those two GPUs were released, retaining the combined
six-GPU ceiling and unchanged science settings.

## 2026-11-05 — Complete40-case factorized constant-shear comparison; test conditional shapes

All80 evaluations17768588 and reducers17768589/17768590 completed exit0.
Updated [constant-shear findings](evidence/alignment_factorized_constant_20261105/findings.md#completed40-case-comparison),
saved completed receipt/accounting and added plot.py (CPU17771710).
At radius>3/mag25 predicted flags, retained m=+2.429±0.310%, factorized Cartesian
−0.274±0.299%, frame+0.437±0.304%. At mag25.8: +2.610±0.278%,−0.216±0.269%,
+0.877±0.275%. Tighter radius>=3.75/mag25.8 worsens to+2.346±0.371% Cartesian,
+2.597±0.370% frame. Actual-usability main-cut Cartesian is−0.067±0.301%.
No accepted replacement; one seed, development scenes, conditional case errors.
92.2% of the Cartesian model R11 increase is in its flow term. Paired change
m=−2.70271±0.01371 percentage points is not total calibration uncertainty.

Clarified that actual selection times unconditional generated shapes removes
shape–photometry dependence; its large residual is not a clean selection-error
estimate. Added [conditional-photometry protocol](evidence/alignment_conditioned_photometry_20261105/rule.md#predeclared-scope),
evaluator, paired reducer and Slurm scripts: freeze both selected checkpoints,
condition shapes on each measured leg's radius/flux, use all40 standalone
validation cases and original CRNs/cuts. No new images/training/preset changes.
Checks reconstruct joint samples exactly for both arms, preserve measured
photometry, unused photometry-latent independence, disk support and exact leg CRNs.
First attempt17771679 stopped at a bitwise GPU cross-sum assertion (max2.39e-15);
added exact per-draw membership assertion and1e-12 aggregate tolerance, keeping
original measured-replay tolerances. No science setting changed; cancelled
blocked reducer17771681. Replacement array17771721 has two one-GPU tasks,
reducer17771722 dependent on both. Syntax/shell checks pass; monitor startup,
then compare conditional vs generated selection responses before further fitting.

## 2026-11-05 — Expand factorized constant-shear evaluation to six GPUs

Owner asks to use more GPUs. Verified partition limits: cip3 GPUs/user,
inter8 GPUs/user; user has inter access. Increased existing array17768588
from2 to6 concurrent one-GPU tasks and made pending tasks eligible on
cip/inter. A40 GPUs were fully occupied, so requests permit other compatible
GPU types while excluding the old P5000 node. Kept running allocations and original job IDs, so all
pilot/final/status dependencies remain intact. No duplicate cases or resubmission.
Bulk scontrol update reported finished members but live task metadata verifies
ArrayTaskThrottle=6 and updated partition/GRES. Saved
[resource receipt](evidence/alignment_factorized_constant_20261105/gpu_expansion.json),
amended study rule/findings and recorded the scoped authorization in AGENTS.md.
Scientific code, checkpoints, cases, seeds, draws and evaluation scope unchanged.
Verified six live workers: threecip and threeinter (RTX2080Ti nodes); runtime
CUDA build supports sm75. All four added workers reached finite first-block
progress successfully; saved startup logs in resource receipt. Full
report remains17768590 after all80 tasks.

## 2026-11-05 — Factorized shape recovers alignment; launch paired constant-shear validation

Owner requests continuation. Factorized Cartesian/frame training17764099_0/1,
all40-case standalone evaluation17764100_0/1, paired report17764101 and status
17764102 all exit0. Update [completed evidence](evidence/alignment_factorized_20261105/findings.md#what-changed),
compact receipt and inspected comparison figure (CPU17768598). Exact model
reload and measured replay pass. Common500k NLL is−3.90471 Cartesian and−3.93722
frame, compared with original−3.69413 and joint-frame−3.88700.

At intrinsicRe0.15–0.25arcsec/mag25, conditional-frame alignment error is
−0.00246±0.00480 at radius2.7–2.8 and−0.00111±0.00417 at3.0–3.1, resolving the
large opposite-side tradeoff in the joint-frame model. Mag25.8 errors remain
−0.00813/−0.00983; full curve/counts are not exact. Main-cut standaloneDeltaR11
is−0.00935±0.00336,DeltaR22−0.00063±0.00385; mag25.8DeltaR22 remains+0.00909.
No claim of a calibrated replacement or complete population/band explanation.
One-seed architecture/allocation intervention supports conditional learning
as a substantial cause of alignment underfit with fixed data/measurements.

Added [constant-shear protocol](evidence/alignment_factorized_constant_20261105/rule.md#purpose-and-fixed-choices),
frozen helper extraction, two-candidate manifest, evaluator, paired reducer,
synthetic unequal-count verification and scheduler launchers under
`doc/evidence/alignment_factorized_constant_20261105/`. Freeze Cartesian epoch139
and frame epoch138, original C8/emulator and64-draw latent streams; same40
existing constant scenes as grid comparison. Three explicit modes: predicted
usability/generated selection; actual usability/generated selection; actual
usability/measured selection while keeping generated emulator photometry.
The last is a diagnostic, not predictive calibration. ReportDeltaR,m,R21,
flow/emulator terms, counts, paired-case SEs, per-case m and leave-one-out ranges.

Retained case1000 replay17768579 gates two-GPU-throttled80-task array17768588.
Pilot17768589 after first8tasks; full17768590 after all; bounded2h status17768591.
Retained replay completed: measured sums exact, model maximum absolute
difference2.62e-10; saved replay receipt. First two candidate
tasks started with finite first-block progress on A40-8Q GPUs; startup receipt
saved. Long array continues under the bounded2h status audit. No expansion/retuning based on interim m: all40 cases predeclared.
Every new case must exactly replay measured sums; measured-final-mask masses
must equal measured counts. Original installed F8 runtime, no sys.path hacks
or archived launcher restarts. All outputs external in matching cache root.

Validation: small CPU reducer recovers knownm3%/DeltaR11.024 with unequal leg
counts, additive flow/emulator components and exact-zero paired identity;
AST/bash/JSON checks pass. This developmental joint test is warranted by the
large relation/density improvement despite residual standalone responses; it
is not acceptance. All SEs condition on model fits and measurement realizations.
No production files, retained presets/weights, simulations or emulator changes.

## 2026-11-05 — Complete full-budget frame comparison; test photometry-conditional shape

Owner requests continuation. Full-budget intrinsic-frame training17762865,
selected-model evaluation17762866 and paired report17762867 all exit0. Update
[completed findings](evidence/alignment_frame_full_20261105/findings.md#completed-result),
compact receipt and density/alignment plot (CPU17764011, reference-line correction
17764103). Inspect plot; cancel superseded pending status17762868. Best epoch136,
exact reload and measured replay. No joint constant-shear calibration claimed.

With fixed160-case data, seed/batches, original standardizers/network,140×4m
budget and FS2-rate schedule, frame output improves common500k NLL−3.83296 to
−3.88700. Alignment error at intrinsicRe0.15–0.25/radius2.7–2.8/mag25 decreases
+0.25333 to+0.09174, paired change−0.16159 ±0.00149; just above the selection
cut at3.0–3.1 it worsens+0.03438 to+0.11492, paired change+0.08054 ±0.00087.
These are absolute alignment, not m; uncertainties condition on fitted models.
Main-cut r>3/mag25 DeltaR11 worsens−0.00497 to−0.01741 while DeltaR22 improves
+0.02770 to−0.00705. Re0.15–0.25 contributes−0.01016 of the−0.01244 R11 change (~82%).
Off-diagonalR12 remains near−0.010. Full relation/matrix
preclude acceptance from low-radius alignment alone. Population/measurement
origin of the FS2/GalSBI difficulty remains incompletely isolated.

Added [factorized study](evidence/alignment_factorized_20261105/rule.md#evidence-and-hypothesis):
exact triangular density p(radius,flux|context)p(shape|radius,flux,context),
two5-coupling affine flows with original physical maps and standardizers.
Cartesian/intrinsic-frame paired arms retain the absolute context orientation,
identical initial parameters and full original budget/samples/FS2-rate schedule.
New1,377,320 parameters vs original1,390,160; allocation differs, so this is an
architecture/allocation test rather than strict capacity matching. Custom
artifact format prevents accidental ordinary-flow loading. Retained model,
production code, simulations, measurements and emulator are unchanged.

Verification: nontrivial triangular inverse/autograd determinant, full physical
density Jacobian(max8.49e-7), support, frame inverse/norm/determinant, paired
initial weights, photometry independent of shape latent, exact save/load all
pass; AST/bash/JSON checks pass. Train17764099_0–1%2 pending priority; afterok
full40-case eval17764100_0–1%2 and paired CPU report17764101 preserve two-GPU
limit. Scheduled bounded2h check17764102. Sources/hashes/rule/job receipt in
`doc/evidence/alignment_factorized_20261105/`; outputs external in matching
cache directory. Check startup when allocated; evaluate both sides of the cut,
likelihood/counts and full matrix before further constant-shear work.

## 2026-11-05 — Complete learning-rate diagnosis; launch matched full-budget frame test

Owner requests continued explanation and improvement of GalSBI relative to FS2.
Read all completed trajectory jobs and update
[findings](evidence/alignment_trajectory_20261105/findings.md#what-is-established)
plus compact `completed_receipt.json`. Both native replays match historical
validation NLL exactly at all140 epochs; full-state fork profiles agree to
1.35e-13. Training, evaluation and reports17758686/17758694/17758743/17758753/
17758754 exit0. Cancel superseded pending check17758885. Inspected final figure.

Exact-state schedule intervention establishes a partial optimization cause:
selected-model alignment error at intrinsicRe0.15–0.25arcsec/radius2.7–2.8px
falls+0.41323 to+0.25333 atmag25 and+0.16668 to+0.08846 at25.8. Same500k-row
GalSBI NLL improves−3.69413 to−3.83296. Training and validation both improve;
coarse48-cell common-input mixture retains a large FS2/GalSBI gap. Earlier
rate reduction is not its sole origin; unique population/band attribution is
unresolved. No calibration success: r>3,mag25 DeltaR22 worsens+0.00537 to
+0.02770 while DeltaR11 improves−0.00943 to−0.00497. Use full response matrix,
not only R11 or local alignment. Case SEs omit retraining/shared-noise issues.

Added [full-budget representation experiment](evidence/alignment_frame_full_20261105/rule.md#motivation-and-controlled-difference)
with isolated trainer, reused verified frame wrapper, selected-model evaluation,
paired full-matrix/alignment reducer, launchers, checks and provenance in
`doc/evidence/alignment_frame_full_20261105/`. One frame arm versus completed
Cartesian FS2-rate control: all160 cases, original configuration/preprocessing,
seed611,140epochs×4m, identical batches/optimizer/rates. Only output-shape
coordinates change. This removes the smaller-budget/scaling confounds in the
previous40-case frame pilot. Absolute input orientation remains conditioned;
no rotational symmetry assumption or measurement intervention.

Validation: CPU `verify.py` passes inverse/norm, standardized determinant1,
physical support, exact reload, exact Cartesian identity, density/Jacobian
maxerror5.08e-7 with original unequal scales. Python AST/bash/JSON checks pass.
Resources: available project disk15TB; one GPU64GB host-memory training,
dependent one GPU24GB evaluation, CPU8GB reduction. Train17762865 running;
eval17762866/report17762867 after-success dependencies; bounded check17762868
scheduled2h later. Startup verified through epoch7, finite losses and all expected rows
on NVIDIA A40; saved startup receipt. Follow scheduled status.
Large outputs external under `sbsi_caches/alignment_frame_full_20261105/`.
No production code, retained preset/weights, emulator or simulation changes.
Next evaluate the representation intervention on likelihood, full alignment
curves/counts and full response matrix before any joint constant-shear test.

## 2026-11-05 — Replay FS2/GalSBI learning and fork GalSBI rate schedule

Owner requests continued work toward a GalSBI flow improvement, prioritizing
why FS2 learns the relation better with original measurements fixed. Added
[trajectory study](evidence/alignment_trajectory_20261105/findings.md#objective-and-fixed-boundary),
predeclared native replay/rate-intervention rules, training-time diagnostics,
complete intermediate optimizer states, selected-model response/NLL evaluation,
case-bootstrap reports, plot scripts, checks and job provenance under
`doc/evidence/alignment_trajectory_20261105/`. All large outputs live externally
under `sbsi_caches/alignment_trajectory_20261105/`. No measurement simulation,
production model/preset change, empirical calibration or sub-agent work.

Native array17758686 uses two A40 GPUs, original160 training/40validation cases,
all50,246,394 FS2 or46,833,127 GalSBI usable leg rows, original standardizers,
seed611,140epochs×4m and original adaptive NLL recipe. FS2 uses original base
labels; GalSBI its harmonized labels. Startup inspected through first training
progress and later diagnostics. At saved startup receipt, native historical
validation NLL matches **exactly through56 FS2/57 GalSBI epochs**. Record
fixed160k intrinsic-small train/validation profiles at21 diagnostic epochs,
with two16-draw CRN banks, counts/negative-dot fractions/local NLL/48 intrinsic
strata; diagnostics do not affect gradients or checkpoint selection.

Interim mag<25/radius2.7–2.8 alignment errors at epoch20 are+0.28491 FS2 and
+0.48835 GalSBI, before GalSBI's first rate reduction27. The reduction cannot
be the sole origin of this particular gap. Atmag<25.8 the early ordering is
reversed(+0.31117 FS2,+0.19589 GalSBI), so no universal early FS2 advantage is
claimed. Native label-history differences are explicit. These are fixed-sample
trajectory point estimates, not calibration acceptance or full uncertainty.

Queued after native replays:17758694 forks GalSBI epoch20 including AdamW
moments, uses FS2 replay's consumed rates for21–140 with identical GalSBI batches.
This tests whether the earlier native rate reductions prevent later catch-up.
Selected-model evaluation17758743 follows with three tasks throttled to two
GPUs: all40 harmonized validation cases/seven response cuts/two16-draw banks,
and fixed500k common-target NLL against original NLL and retained checkpoints.
Reports17758753(native)/17758754(final) save full trajectories and comparisons,
10000 case bootstraps, measured-replay/fork-state checks and figures. Scheduled
status17758885 runs about2h later and writes a local receipt; not a chat wakeup
or automatic repair. Dependencies preserve the combined two-GPU ceiling.

Validation: resource skill reports abundant host memory/disk; cluster allocations
reserve64GB per native worker. Python AST/JSON and four bash launchers pass.
Frozen-runtime `verify.py` passes known signs/radius/magnitude bins, case totals,
draw normalization, reducer pooling, JSON-null empty bins and bootstrap identity.
Native startup/progress passes; downstream stages are pending, no new flow
improvement claimed. Next inspect full local/global learning curves and causal
rate fork; only a candidate retaining density quality and improving selected/
neighboring-cut responses merits joint constant-shear evaluation.

## 2026-11-05 — Isolate PSF-model pathology and test intrinsic-frame flow

Owner clarified that FS2/GalSBI populations should not be too different and
requested explanation/improvement. Added
[completed frame/PSF findings](evidence/alignment_frame_20261105/findings.md#what-is-established),
predeclared rules, invertible intrinsic-axis output wrapper, matched training,
response/profile evaluation, paired-case reducer, verification, three image
interventions, compact reports/CSVs and inspected PNG/PDF figures under
`doc/evidence/alignment_frame_20261105/`. Raw artifacts are external under
`sbsi_caches/alignment_frame_20261105/`. Simulation/measurement lives in new
BlendEMU `scripts/audit_identical_galaxy_conditions.py`; production measurement
and retained model presets/weights remain unchanged. Experimental frame
checkpoints have a custom format to prevent accidental world-frame loading.

Fixed216-context grid×2PSFs×3noise levels×4paired seeds=5184fits per image
intervention. Moffat array17757925 all succeeded. At intrinsic Rc0.20arcsec
with no added noise, alignment changes-0.83170→-0.10551 when replacing only
FS2PSF with GalPSF. Whole-grid PSF-only alignment change+0.44422; negative-T
fractions79.63%/64.24%. Gaussian-rendered control17758019 all succeeded and
removed negative T without added noise, alignment+0.88003/+0.87277.

Sharper same-Moffat-image control17758135 changes only diagnostic PSF ngauss1
→3, retaining galaxy fit/prior. On identical successful no-added-noise rows,
FS2 alignment-0.36190→+1.03593 and negativeT79.83%→0%; Gal alignment
+0.07033→+0.97009 and negativeT64.58%→0%. This implicates the one-Gaussian
Moffat-PSF approximation in the targeted-grid pathology. However918/5184
three-component attempts fail PSF convergence (17.71%); raw failures are
preserved and comparisons explicitly condition on paired success. These are
isolated, zero-shear grids without detection/FLUX_RADIUS cuts, not estimates
of selected constant-shear bias. All12 image tasks completed exit0.

Matched fresh flow array17756866 completed using at most two GPUs: same40
training cases/all usable leg rows,120epochs×2m,initiallr3e-5/common scheduler,
fixed500k validation NLL selection. Both selectepoch119; NLLCartesian-3.531801,
frame-3.545000, versus retained-3.693633 on identical rows. Original training
had a larger budget; pilots are not a capacity-limit test. Evaluation17757916
completed40validation cases/seven cuts/two16-draw CRN banks; reduction17757938
completed10000 paired whole-case bootstrap resamples. At intrinsicRe0.15–0.25,
measured/generatedradius2.7–2.8,mag<25, alignment error retained+0.38163±0.00479,
Cartesian+0.31488±0.00471,frame+0.25007±0.00486; paired frame-minus-Cartesian
-0.06481±0.00119. But full-population standalone DeltaR11 at radius>3/mag<25.8
worsens from retained+0.01352±0.00298 toCartesian+0.02552±0.00300 and
frame+0.02753±0.00300. No candidate accepted and no joint validation launched.
Improving this one alignment statistic is insufficient. SEs exclude retraining.

Validation: rotation/inverse/norm/determinant1, density identity maxerror
5.0821e-7(tol3e-5), support and exact reload pass. All Python AST/JSON and bash
launcher checks pass; known-response/additive-cell check passes. Measured
replay maximum absolute moment difference6.37e-12. Largest draw-bank R11
difference0.00108 at tighter neighboring cut, below its caseSE0.00474.
Figures used Matplotlib skill and were visually inspected. All science jobs
finished exit0; status-only17757962 cancelled as superseded by final receipt.
Next investigate robust multi-component PSF fitting on actual selected scenes;
new measurement labels would require flow/emulator retraining and full joint
validation. Existing-label flow optimization remains unresolved.

## 2026-11-05 — Localize GalSBI alignment failure to density underfitting

Owner identified the FS2/GalSBI difference in shape–size alignment as the
specific learning problem. Added [alignment-learning findings](evidence/alignment_learning_20261105/findings.md#frozen-checkpoint-result-density-underfitting-already-on-training-data),
predeclared rules, frozen checkpoint and training-history audits, stratified
case-bootstrap reports, train/validation figures, and a bounded two-arm NLL
continuation. Artifacts reside under external `alignment_learning_20261105/`;
all new diagnostic code and compact evidence are in the matching doc folder.

Evaluated NLL-only/adopted FS2/GalSBI on40 training and40 validation cases each,
intrinsic Re0.15–0.25 arcsec, matched usable zero-shear rows. GalSBI adopted
alignment errors at radius2.7–2.8px/mag<25 are+0.37896 +/-0.00443(training)
and+0.38140 +/-0.00482(validation); NLL-only is worse(+0.41067/+0.41339).
FS2 errors are+0.03245/+0.03638. The GalSBI predicted bin count agrees within
0.5%, but negative shape–intrinsic dot products occur74.09% in simulation
versus49.96% in the flow. A Sérsic×true-magnitude decomposition assigns
+0.38572 to within-cell error and-0.00432 to bin composition. Error persists
across orientation/Sérsic strata and is large for bright, weak-crowding objects.
This is an NLL fitting failure already on training data, not solely validation
transfer or a response-penalty artifact. No proven architecture-capacity cause.

Same architecture/features/recipe; actual LR first halves at27 GalSBI versus
111 FS2. Predeclared20-epoch NLL continuations share starting weights/AdamW
moments, resetlr1e-4,40 training cases and four million rows/epoch: full inputs
versus intrinsic-small only, without measured training cuts. Local validation
error improves from0.41339 to0.37928/0.31265, but full-population validation
NLL worsens by0.10396/1.30433. Training is unstable, especially the targeted
arm. These diagnostic endpoints are not accepted replacements or constant-
shear improvements; the reset test does not identify the original cause.

Validation: GPU17756442 tasks0/1 and CPU17756457 completed0:0; learning
array17756566 and reducer17756597 completed0:0, with at most two GPUs active.
Two16-draw banks, checkpoint CRN,10,000 whole-case resamples; uncertainty
excludes retraining/shared noise. Every intrinsic partition and signed-shape
contribution closes; measured validation profiles replay earlier results
within3.4e-16. AST/shell/strict-JSON/link checks passed and figures inspected.
Next discriminating test is a stable, controlled Cartesian versus intrinsic-
frame density fit; not performed. The link from this diagnostic to the full
joint constant-shear bias remains subject to independent response validation.

## 2026-11-05 — Render standalone FS2 alignment panel at MAG_AUTO below25.8

Owner requested the FS2 panel at MAG_AUTO<25.8. Added
`doc/evidence/fs2_galsbi_transfer_20261105/plot_fs2_mag258.py` and
[PNG](evidence/fs2_galsbi_transfer_20261105/fs2_mag258_alignment.png)/PDF outputs.
Uses the preserved `profile_comparison.json` FS2/25.8 values and case-bootstrap
SEs without recomputation; retains intrinsic circularized Re0.15–0.25 arcsec,
zero shear, radius3px marker and the earlier axis limits. Lightweight render
with py31 completed successfully; figure visually inspected. No science
inputs or model outputs changed; uncertainty limitations remain as documented
in the transfer findings.

## 2026-11-05 — Audit FS2-to-GalSBI V3.9 transfer with matched flow and stamp experiments

Owner requested deep localization of the failed GalSBI/i-band retraining of
the successful FS2/r-band recipe. Added [transfer findings](evidence/fs2_galsbi_transfer_20261105/findings.md#conclusion-and-scope),
predeclared rules, frozen-flow evaluators, CPU reducers, compact reports,
provenance receipts and a shape–size figure. BlendEMU owns two new measurement
diagnostics in `/home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts/`:
`audit_population_stamp_transfer.py` and `audit_transfer_psf.py`.
Production models, presets, catalogues and inference behavior are unchanged.

New matched 40-case validation checks show main-cut standalone Delta R11
-0.013984 +/-0.001551 for FS2 versus +0.013518 +/-0.003032 for GalSBI.
True Re0.15–0.4 contributes -0.007416 versus +0.014605. Coarse common-support
size/crowding standardization reduces GalSBI's standalone m_R11 from2.294%
to0.962%; it is not a correction or causal partition of joint bias.
FS2's two accepted refinement updates change model R11 by only-0.002009.
GalSBI's measured/intrinsic shape alignment near2.7–2.8px differs from its
flow by+0.3816 at mag<25, versus+0.0361 for FS2, at true Re0.15–0.25.

Same-pixel paired48/65-stamp fits on16,727 objects in four scenes shift this
alignment by only+0.0091(FS2)/+0.0280(GalSBI), retaining the transition.
Negative-T fits dominate this deliberately targeted bin; branch contributions
are descriptive, not proof of signed shear bias. Actual fitted PSF T differs
by15.38% (about8% linear size), in addition to nearly doubled pixel noise.
The historical mag25 test reduces GalSBI joint m only2.74%→2.47%.
Replayed FS2's near-zero joint result and documented its internal cancellation,
reuse of training galaxies/noise, and differing measurement-label history.

Validation: `reduce_flow.py --verify` reproduces a known2x2 response;
disjoint-bin bootstrap sums and T-branch alignment sums close. Measured
GalSBI hard half-trace replays to9e-16. Frozen identities checked; all
unmatched rows reported. Two16-draw antithetic banks, CRN between shear legs,
10,000 whole-case resamples; errors exclude training/shared-noise uncertainty.
GPU17754234/17754248/17754364 and CPU reductions completed0:0; stamp array
17755520 plus reports17755548/17755726 completed0:0. Initial log-radius
underflow, environment startup and empty-cell reduction failures are preserved
and repaired explicitly; common support excludes0.0006253% of FS2 pairs.
Scoped AST/shell checks, JSON/source receipts and figure inspection completed.
Next causal discriminator is population×PSF/noise with fixed measurement and
paired latent scenes, separating SED band flux from observing conditions.
That factorial rerender/retraining has not been run; no independent acceptance
or complete numerical attribution of the joint bias is claimed.

## 2026-11-05 — Deepen intrinsic-small/measured-large constant-shear localization

Owner requested confirmation and investigation of noise versus blending.
Added [mechanism findings](evidence/small_large_mechanism_20261105/findings.md#confirmation-and-scope),
CPU analysis, exact case sums, JSON report, provenance and PNG/PDF figure.
Reused the original four constant scenes and saved per-object four-state
GalSBI-dev moments; no training, correction, measurement changes or GPU jobs.

Confirmed Re0.15–0.4 arcsec contributes +0.021526 +/-0.007451 of total
+0.021850 +/-0.005689 after FLUX_RADIUS>3/MAG_AUTO<25. Finer crowding split
reveals +0.021705 +/-0.002470 at log1p weighted neighbour flux0.01–0.1,
versus -0.004531 +/-0.002981 below0.01. Intrinsic mag24–25.5 contributes
+0.013838 +/-0.006308;88.14% of selected objects there have photometric S/N<20.
These correlated populations do not identify causal noise/blend fractions.

Small galaxies have opposite-group error+0.006291 +/-0.002571 and4090
radius-threshold crossings with target shear/noise seed fixed (0.488% of
relevant union-selected edge comparisons). Common-U within-group measured
neighbour-selection terms+0.003234/+0.003402 are absent from model opposite-
group selection. Constant-scene shape alignment flips sign around2.9px and
rises sharply across3px, supporting the previously observed conditional
shape–size problem. The actual truncated PSF half-light radius is0.482arcsec
before pixel convolution: intrinsic versus measured radius alone cannot
identify inflation. Negative-T optimizer scatter remains distinct from a
proved mean-bias mechanism.

Validation: AST checks; CPU17754665/17755490 completed0:0 in30/23s;
every original coherent selected sum replays, and global contrasts/bootstrap
SEs agree within2e-12. Disjoint cell and exact shape/selection identities pass.
10,000 paired case bootstraps, seed20260925; sparse undefined ratios reported
explicitly. Unique joins and all unmatched counts retained in receipts.
Figure inspected; exact PSF parameters checked against noise.csv and renderer.
Next discriminating work: matched noise on/off and neighbours on/off images,
with original photometry replay and unselected controls; not performed here.
No independent acceptance claim from these four development cases.

## 2026-11-05 — Verify V3.9 guard training and correct common-descent attribution

Traced the pinned F8 artifact through its external experiment manifest,
training protocols, target bank and completion receipts. Added the
[V3.9 provenance correction](evidence/grid_regularization_history_20261105.md#v39-training-provenance-measured-cut-grid-and-retained-step-zero):
16 measured-cut guards, no property strata, harmonized refinement step 2,
then common-descent stage retaining step 0. The latter stage made no adopted
update; its directory name alone does not establish an optimization gain.
Frozen trainer/target-builder/refinement-runner hashes match the manifest;
final receipt identity matches the retained V3.9 manifest. Scoped whitespace
check passes. Documentation only; model identities and weights unchanged.

## 2026-11-05 — Correct grid-history coverage to include successful constant-shear models

Owner pointed out that the history omitted good constant-shear models. Expanded
[the index](evidence/grid_regularization_history_20261105.md#successful-constant-shear-results-omitted-from-the-initial-index)
with the August 24 original E 6x6x5 full-matrix grid recipe and four-seed
ConstGold result (+0.23433% +/-0.15040 pp), its frozen V3 comparison, and the
retained V3.6/V3.9 development results. Distinguished truth-restricted older
tests from later per-leg measured selection, and grid training from later
common-descent refinement. The earlier failure-heavy summary was incomplete;
it does not establish that grid regularization never worked. Validation:
checked archived recipes/results, all index file links and section anchors,
and scoped git diff whitespace. Documentation only; no experiments restarted.

## 2026-11-05 — Retrieve recent and historical grid-response regularization results

Owner requested recent and earlier grid-based regularization evidence. Added
[history index](evidence/grid_regularization_history_20261105.md#scope-and-terminology)
covering November 4 GalSBI measured-cut-inside-property-grid, September 16
fixed-g0 truth/measured grids, September 11–13 circularized shape/shape+radius
grids including the failed coherent follow-up, and August 10 multi-seed no-size
grids. Separately identified the measured-threshold 9/16-cut guards. Preserved
distinct catalogue families, cohort/cut definitions, response proxies versus
coherent m, and case versus seed uncertainties. Checked November JSON result
identity and noted its 16-guard RMS differs from the 21-cut target RMS in the log.
Sources are linked by section. Documentation only; no training, simulations,
archived launchers or stopped experiments restarted. Local link targets and
scoped whitespace checks validate the index; no scientific recomputation.

## 2026-11-05 — Compare standalone flow/emulator errors for intrinsically small galaxies

Owner asked whether the intrinsic-small-galaxy bias is visible in each
standalone check. Added `standalone_size_comparison.py` and its JSON report
under `doc/evidence/joint_residual_localization_20261105/`, reducing existing
replay-verified sums only. CPU17753827 completed; all original global emulator
values and bootstrap SEs replay within1e-12, and disjoint size contributions
add in every bootstrap. No image/model evaluation or model changes.

For circularized Re0.15–0.4 arcsec after measured radius>3/MAG_AUTO<25, flow
R11 error contribution+0.010572 +/-0.003621; halftrace+0.012887 +/-0.002464.
Emulator baseline shape contribution-0.001793 +/-0.002036, omitted selection
response+0.003478 +/-0.000511, net selected error+0.001686 +/-0.002083. These
are globally normalized contributions, with neighbour pair multiplicity already
included for the emulator, not per-bin m. Same intrinsic-size range at all
crowding levels, distinct from the previous narrower low-crowding subset.
Flow40 held-out training-family cases and emulator40 development cases use
different shear experiments; no cross-experiment error-budget addition.
Updated [findings](evidence/joint_residual_localization_20261105/findings.md#standalone-intrinsic-small-galaxy-comparison).

## 2026-11-05 — Complete mixed-shear reduction after derived-column roundoff check

Owner asked whether negative-T fits can explain the calibration bias. Checked
the completed matched-image chain: all eight simulations17750385 completed0:0
in3h00m–5h31m; control17750387 and check17750442 completed0:0. The control
reproduced science/PSF pixels and input, detection, and crossmatch tables exactly.
Initial reducer17750443 failed because a derived intrinsic ellipticity column
differed by at most1.11e-16 on535/883944 rows, consistent with node-dependent
trigonometric roundoff. No measurement or model discrepancy was involved in
that stopped comparison.

Updated `doc/evidence/constant_split_20261105/reduce.py` to permit absolute
5e-16 tolerance only for the renderer's derived `e1_input_rot*`/`e2_input_rot*`
columns and record their maximum discrepancies. Original q, angle, radius,
positions, IDs and other non-shear inputs still require exact equality; expected
mixed shears and original model/simulation response sums retain their checks.
Submitted CPU reducer17753575. Analytical reducer fixture still passes. This
repair does not change scientific sums, selection, model parameters or images.

Reduction completed with all original coherent totals replayed. Actual-U,
radius>3/MAG_AUTO<25 error splits exactly into own-group+0.013958 +/-0.004683
and opposite-group+0.007892 +/-0.001789, total+0.021850 +/-0.005689. Own-group
also includes same-group neighbours; no empirical doubling of opposite-group.
Within-group common-U opposite-shear errors are+0.007056 (A) and+0.010091 (B),
including measured selection terms+0.002715/+0.003136 absent from the model
opposite-shear selection. The matched test localizes missing neighbour-related
response, but does not identify negative-T switching as its cause. Original
four-case paired bootstrap and conditional limitations retained. Updated
[findings](evidence/constant_split_20261105/findings.md#completed-matched-shear-response-decomposition),
job ledger and source manifest; no model or fitter change.

## 2026-11-05 — Start matched constant-scene decomposition and audit optimizer randomness

Completed optimizer follow-up: audit17750449 finished all eight tasks0:0;
reducer17750462 completed0:0. Identical-pixel independent fits have component
RMS change0.22005 across62767 usable selected case/leg rows;12.10% have vector
change>0.1. Radius3–3.2px accounts for35.65% of squared g1 changes. Seeded
replay17750506 completed0:0 on1021 case/leg rows; all2042 fits flags=0, same
seed reproduces to1e-10, different seeds still give0.20836 RMS. Summary17750514
completed0:0. This demonstrates stochastic instability among nominally successful
fits, not a mean calibration-error explanation. Detailed parameter stress
inspection17750533 targets the16 largest changes; no fitter settings changed.
Single-fit response-noise projection is0.007437 for the four full scenes;
conditional scaling to140 cases is0.001257, similar to the old0.001274 case SE.
Do not double-count optimizer noise already present in case scatter. The sampled
repeat response differences are too noisy to establish a systematic shift.
See [findings](evidence/constant_split_20261105/findings.md#optimizer-instability-confirmed)
for repeat changes, uncertainty definitions, and limitations. Environment
preflight17750489 completed0:0; final status snapshot17750490 depends on the
main reduction. Actual image decomposition and rendering control remain pending.

Parameter stress inspection17750533 and uniform replay17750545, followed by
summary17750548, completed0:0. In1001 selected both-success case/leg rows,
7.94% of fits have negative ngmix deconvolved T;5.29% switch T sign between
starts and account for75.52% of squared g1 changes. All16 deliberately selected
largest-change examples switch sign and reverse ellipticity; PSF parameters
replay within5.6e-17. Original FlatPrior(-10,1.e6) permits negative T, and
success/disk-usability checks retain these solutions. This localizes optimizer
scatter, not the mean joint calibration bias. Selection remains FLUX_RADIUS>3
and MAG_AUTO<25; no positivity cut or production fitter modification. Companion
diagnostics also include `scripts/audit_constant_fit_flags.py` and
`scripts/audit_constant_fit_modes.py`. Source hashes are saved in
`evidence/constant_split_20261105/source_manifest.json`.

Owner requested a deep autonomous investigation and correctly challenged
whether standalone cancellation should transfer for the same population.
Qualified the preceding [localization finding](evidence/joint_residual_localization_20261105/findings.md#main-finding):
the highlighted cells do not explain the excess joint residual. No model
training, tuning, single-cut restart or empirical response correction.

Predeclared [four-state rules](evidence/constant_split_20261105/rule.md#four-state-image-experiment)
and added evidence scripts under `doc/evidence/constant_split_20261105/`.
Split every original input galaxy in cases1000,1040,1080,1120 into fixed random
A/B groups (seed202611050001+case); retain original all+/-0.02 scenes and render
two opposite-sign mixed states with identical positions, intrinsic catalogue,
noise seed, conditions and centred65px measurement. Exact own/opposite-group
contrasts add to the original coherent response. Own-group includes same-group
neighbours and is not pure self-response. Report predicted/per-state actual/
common-all-four usability, four cuts, global and within-group normalizations,
core/complement contributions, separate flow/emulator predictions, and an exact
shape-versus-selection response decomposition on common usability only.

Simulation-owned preparation and fitter audit live in BlendEMU's companion
checkout: `scripts/prepare_constant_split.py`, `scripts/audit_constant_fit_randomness.py`,
`jobs/job_constant_split.sh`, `jobs/job_constant_fit_audit.sh`. New catalogues
use `/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/constant_split_20261105`.
Original measurements remain unchanged. Original shape code uses an unseeded
RandomState per object; sample8000 IDs/case from the union of detected
radius>3px/MAG_AUTO<25 objects without conditioning on old shapes, and repeat
both legs twice with independent recorded fitter streams. Quantify matched
response changes, usability changes and conditional subsampling/fitter noise.

Preparation17750383 completed0:0 in17s. Model array17750397 completed all four
cases0:0 in5m21s–5m46s, with at most two GPUs. It replays every original +/-
predicted/actual-U mass and shape numerator for all four cuts, with the original
64 flow draws and common random numbers. All first-block A/B neighbour sums
replay the unsplit emulator. Mixed image array17750385 (eight CPU allocations)
and unchanged-image control17750387 started normally and are long jobs.
Control check17750442 requires exact original pixels and input/detection/match
tables; reducer17750443 depends on it, all mixed measurements, and model replay.
Optimizer array17750449 and dependent reduction17750462 started; status snapshot
17750478 is scheduled for two hours later. Jobs are recorded in
`evidence/constant_split_20261105/jobs.json`; results are pending, not a closed
error budget or independent acceptance evidence.

Validation: AST and shell syntax checks; analytical contrast tests cover known
own-only/opposite-only/nonlinear response, normalization, population additivity,
and exact shape/selection identities. Full reducer executes on a synthetic
fixture and recovers known errors and zero opposite-group flow response on
fixed usability. Four-case paired bootstrap is conditional on these scenes,
models and flow draws; separate optimizer estimates are not total uncertainty.
Next: inspect fitter audit, exact rendering control, and matched mixed-state
response errors before assigning a mechanism. Follow long-job policy rather
than polling the image simulations until completion.

## 2026-11-05 — Localize original joint residual and expose cancellation in the standalone flow check

Owner requested further localization after measured-flag substitutions failed
to explain the original discrepancy. Added diagnostic scripts, rules, case sums,
reports and [findings](evidence/joint_residual_localization_20261105/findings.md#main-finding)
under `doc/evidence/joint_residual_localization_20261105/`. Original older16-cut
GalSBI-dev model unchanged; each side retains its own shape/radius/flux selection.

Four constant cases1000,1040,1080,1120, original64 draws/CRN, actual usability:
radius>3px/MAG_AUTO<25 error+0.0218503 +/-0.0056889. Fixed truth-cell response
contributions sum exactly to this original error. Re0.15–0.4 arcsec with
nbr_close_100<0.1 contributes+0.0171740 +/-0.0041490; complement+0.0046763
+/-0.0024197. On the40 held-out training-family flow cases, original16 draws,
the same cells contribute+0.0162250 +/-0.0031093 to R11 error; complement
-0.0085934 +/-0.0016459, leaving only+0.0076316 +/-0.0034100. Thus a conditional
flow error was hidden by cancellation in its standalone average. This locates
a shared failure population, not a causal79% attribution across two estimands.

Intrinsic Re0.15–0.25 and0.25–0.4 bins contribute+0.0089682 and+0.0125580 to
joint error while selected fractions closely match. Low-crowding alone contributes
+0.0175272. Moderate crowding0.1–0.3 has standalone flow contribution-0.0090897
but joint+0.0048044: the reason cancellation changes remains unresolved. No
opposite-diagonal cancellation: standalone hard R11/R22 errors+0.0076316/+0.0083432.
Radius-cut dependence and earlier shape/size-alignment evidence identify a
specific mechanism to test, not a demonstrated correction.

Also measured the neighbour-induced selection term omitted by the old baseline
shape-only emulator check. On its40 cases at h0.2, exact selection contribution
+0.0045188 +/-0.0005661 changes the emulator error from-0.0025563 +/-0.0022935
to+0.0019625 +/-0.0023175 per represented primary. Matched-object h0.05 versus
h0.2 comparison gives total response change-0.0002176 +/-0.0085673: no detected
shift, but inadequate precision to rule out a material finite-step effect.
Half-neighbour random-direction results are not scaled/pasted into the coherent
constant-shear budget. Actual usability migration is not tested by that check.

Validation: CPU17750063 completed0:0 in8m30s, replaying all40 original target-cut
pair counts/labels/model sums; GPU array17750068 completed0:0 in3m12s–3m14s per
case, at most two GPUs, replaying all four original cut totals; reducer17750080
completed0:0 in3s. Matched-step CPU17750106 completed0:0 in8m38s, verifying shared
truth identities/directions and h ratio4. Flow17750130 completed0:0 in1m45s,
replaying original soft/hard totals and cell sums; reducer17750131 completed0:0
in1s. Pending24GB jobs17750127/17750128 were cancelled before startup and replaced
with an available16GB GPU. All scripts pass AST checks; scoped git diff --check
passes. Whole-case bootstrap10000, seed20260925, fixed models/draws. Four-case
joint estimates remain diagnostic. No training, simulations, production changes,
or restarted single-cut experiment; no jobs remain.

## 2026-11-05 — Substitute measured usability and selection in joint and standalone checks

Owner requested the absolute response-error table with actual usability and
measured selection, including recomputation of the standalone flow. Added rules,
evaluation/reduction scripts, case sums, reports, receipts and
[findings](evidence/true_selection_flags_20261105/findings.md#definitions-and-scope).
The requested mask is per-leg usable U times measured FLUX_RADIUS>3 pixels and
MAG_AUTO<25, replacing generated selection. Shapes and emulator auxiliary inputs
remain generated. Older 16-cut GalSBI-dev checkpoints and production code unchanged.

Joint replay array17750012 used four matched constant-shear cases, 64 original
draws/CRN, and at most two GPUs. All tasks completed0:0 in3m30s–4m00s and replayed
all four old substitution branches, original true-U sums, and measured counts and
shapes within declared tolerances. Requested-mask counts match simulation exactly.
Reducer17750016 completed0:0 in2s. Matched four-case errors (simulation minus model):
predicted-U/generated-cut +0.019721446 +/-0.005763188; true-U/generated-cut
+0.021850320 +/-0.005688872; true-U/measured-cut +0.042211577 +/-0.005503867.
The last substitution increases the error by0.020361257 +/-0.001133698: flow
contribution falls0.020493037, emulator contribution rises only0.000131780.

Standalone flow job17750022 depended on the joint array and completed0:0 in3m45s.
All40 held-out training-family cases, 5,817,708 usable pairs, original16 draws and
block seeds. The original soft-cut simulation/model responses replay within6e-15.
Hard-cut errors are +0.007987374 +/-0.002743722 with generated selection and
+0.029663156 +/-0.002779413 with measured selection; paired increase0.021675782
+/-0.000342597. Bootstrap resamples whole cases and recomputes baseline mass,
mean and shear normal matrix. Soft-cut original error remains+0.005698836 with
SE0.002486060. Standalone neighbour-weighted emulator remains-0.002556339
+/-0.002293462 because its existing40-case check already uses measured selection.

Corrected [earlier localization interpretation](evidence/constant_error_localization_20261104/findings.md#finding):
the measured-auxiliary-input/selection substitution raises the emulator contribution
but lowers the flow contribution more; full joint error increases to0.028370090.
It did not explain a reduction in the total error. Actual membership applied to
independent generated outputs breaks the original shape/selection correlation;
even a correct conditional model need not have zero error under this intervention.
Four-case joint SEs are conditional diagnostic estimates, not the old140-case SEs
or total uncertainty. Component checks are different estimands and are not additive.

Validation: saved flow statistics independently reproduce report points within1e-14;
all three scripts pass AST syntax checks; scoped git diff --check passes. No jobs
remain, no training/recovery chain restarted, and no further runs launched. The
original generated-selection joint discrepancy remains unexplained by this test.

## 2026-11-04 — Localize constant-shear response discrepancy

Owner requested localization of the error left unexplained by standalone flow
and emulator checks. Diagnostic sources, rules, receipts, and reports are under
[constant error localization](evidence/constant_error_localization_20261104/rule.md).
No models, inference settings, or training runs changed; the single-cut experiment
remains stopped.

Completed CPU diagnostics: Slurm17734831 re-reduced saved40-case constant outputs,
replaying the original common-selection responses and uncertainties; Slurm17734824
rebuilt40 emulator-development cases with bright-neighbour exclusions recorded
and bypassed, reproducing the original included pair counts/labels/predictions;
Slurm17734961 evaluated measured-input emulator sensitivities on8 constant cases,
reproducing selected counts and measured shape sums. Runtime6s,6m18s,2m52s,
respectively; all0:0. Exact symmetric magnitude-bin decomposition separates
+0.0229711 within-bin and +0.0028782 population-mixture errors in the +0.0258493
common-selection deficit. Final total discrepancy on these40 cases is
+0.0234197 +/-0.0027575; selection remainder -0.0024297 +/-0.0008792.

The measured-input8-case emulator contribution is0.1850917 vs0.1713221 with
generated outputs, difference+0.0137696 +/-0.0001687. This combines inputs,
selection, and shapes; it is not a deployable correction. Measured-primary
distances add0.0038067, nearest40 vs production nearest19 add0.0024230, and
linear vs finite Mobius adds0.0000965 on this measured-input diagnostic.
Original training labels use measured-primary distances; production uses true
positions. Previously excluded bright-neighbour rows show overprediction
(-22.53% +/-3.32), opposite to the joint deficit.

Completed follow-up: four-case64-draw CRN replay17735430 (array%2, two GPUs
maximum), all0:0 in4m21s–4m34s; original true-flag sums reproduced in every case.
Report17735553 completed0:0 in2s and independently checked CPU/GPU measured-input
branches. Emulator response0.171240 ->0.172668 when only measured radius/flux is
substituted, ->0.185213 when measured selection is also used, ->0.185131 when
measured shapes are also used. Successive shifts+0.001428 +/-0.000066,
+0.012546 +/-0.000135, -0.000083 +/-0.000266. The large effect is selection
coupled with emulator inputs; ordered substitutions are not unique causal shares.
The flow uses flux-weighted crowding summaries, not simple neighbour counts.
The missing conditional dependence on detailed neighbourhood remains a hypothesis,
not an identified feature or training fix. See [findings](evidence/constant_error_localization_20261104/findings.md#finding)
for exact scope, the remaining discrepancy, and next discriminating checks.
Scripts compile and git diff --check passes. No further jobs remain.


## 2026-11-04 — Weight emulator pair residual by measured neighbour multiplicity

Owner requested the per-pair emulator discrepancy weighted by the average number
of neighbours. Added [counting script and report](evidence/neighbour_weighted_response_20261104/report.json)
under `doc/evidence/neighbour_weighted_response_20261104/`. On the same 40 development
cases, radius>3 pixels/MAG_AUTO<25: 29,148,629 retained pairs / 2,699,377 distinct
represented primaries = 10.798280 neighbours per primary. Summed pair-response
simulation/model averages are 0.079403706/0.081960045 per represented primary;
simulation minus model is -0.002556339 +/-0.002293462. The paired whole-case
bootstrap (10,000 resamples, seed20260925) recomputes the primary denominator,
retaining covariance with neighbour multiplicity. Primaries with no retained valid
pair are absent; original bright-neighbour and label-validity exclusions remain.
This is a summed projected pair diagnostic, not a direct coherent-shear or
Möbius-composed response; it does not by itself close the joint error budget.

Validation: CPU Slurm17734647 completed 0:0 in44 seconds. Every per-case selected
pair count exactly matches the saved emulator sums; multiplying the previous
per-pair point discrepancy by the measured multiplicity reproduces the new result
within1e-15. Script syntax and `git diff --check` pass. No new model evaluations,
GPU jobs, or restarted experiments. No pipeline/model changes.


## 2026-11-04 — Report component and joint discrepancies in response units

Owner requested absolute errors for the preceding relative-error table. Added
[report](evidence/absolute_response_errors_20261104/report.json) and a small
saved-sum reducer under `doc/evidence/absolute_response_errors_20261104/`.
Sign is simulation minus prediction. Older16-cut flow on40 training-family
validation cases: +0.00569884 +/-0.00248687 per galaxy. Emulator on40 development
cases: -0.000236736 +/-0.000212399 per neighbour pair. Joint140 constant-shear
cases: +0.02185523 +/-0.00127417 with C8 and +0.02371895 +/-0.00128022 with actual
usability flags. All use radius>3px/magnitude<25 under their original selection
conventions. The emulator's per-pair denominator prevents adding these component
numbers directly into a per-galaxy joint error budget.

Validation: lightweight arithmetic on small saved case-sum arrays only;
10,000 paired case bootstraps, seed20260925. Flow and emulator relative point
values and SEs replay saved reports within1e-10; joint discrepancies reuse the
verified constant-component receipt. No model evaluations or new scheduler jobs;
the cancelled single-cut experiment remains stopped. No pipeline/model changes.


## 2026-11-04 — Constant-shear component accounting with predicted and true flags

Owner requested flow-only, emulator-only and joint errors on constant-shear
simulations with predicted/true usability flags. Reduced existing140-case
GalSBI-dev mag25 sums only: original16-cut flow, fixed C8/disk emulator,
g1=+/-0.02, independent own-leg radius>3px/MAG_AUTO<25 selection. CPU17730923
completed successfully; no new flow/emulator evaluations or restarted experiment.
First reduction17730900 failed because old NPZ files omit the newer `cuts` key;
retained failure/source and used the documented frozen row1 with exact schema
assertions. Point m and bootstrap SE replay existing summaries within1e-10.

Predicted C8: R_flow0.71284354 +/-0.00011455, actual Mobius shift contribution
R_emulator0.17266942 +/-0.00007641, joint0.88551296 +/-0.00013137; measured total
0.90736819 +/-0.00127436; joint m+2.468087% +/-0.143909pp. True usability:
R_flow0.71269670 +/-0.00011443, R_emulator0.17095255 +/-0.00007744,
joint0.88364924 +/-0.00013378; joint m+2.684204% +/-0.144918pp. Whole-case
bootstrap10,000, seed20260925, paired components and refitted leg normalizations.
Saved [receipt](evidence/constant_component_accounting_20261104/report.json)
and reducer under `doc/evidence/constant_component_accounting_20261104/`.

These model contributions add exactly, but independent component calibration
errors are not measured: whole-scene-shear images provide only total truth.
Flow-without-emulator compared to total truth yields +27.28855%/+27.31477%,
which are missing-component ablations, not isolated flow error. Emulator-only
shift ratios in the receipt are likewise not emulator calibration errors and
still condition on generated flow measurements. Do not substitute earlier
training-family component checks for constant-shear component truth. Conditional
case errors omit full retraining and independent finite-draw uncertainty. The
cancelled single-cut experiment remains stopped; no new diagnostic proposed.


## 2026-11-04 — Stop single-cut constant-shear evaluation at owner request

Owner: "we should stop this and reflect a bit". Cancelled both GPU arrays
17730541/17730542, pending dependent reducer17730543 and two-hour status17730548.
Preserved all source, manifests, logs and partial case outputs in
`sbsi_caches/galsbi_target_guard_constant_20261104`; saved
[stop receipt](evidence/galsbi_target_guard_constant_20261104/stopped.json).
No full-cohort m was produced or claimed; no partial science result used for
selection. This supersedes the pending-run status below. No automatic restart
or replacement experiment is authorized by this stop.

The proposed run could compare checkpoints but would not isolate why the
flow-only half-shear response and combined constant-shear response disagree.
Reflection should first distinguish population/selection/shear-estimator changes
from flow, classifier and neighbour-response composition. No new jobs launched.
Validation: scheduler cancellation and accounting checked; requested follow-up
jobs cancelled with the workers; original models/data and completed evidence
remain unchanged.


## 2026-11-04 — Launch the single-cut response-trained flow on constant shear

Owner requested constant-shear m for a flow trained with only the radius>3 px /
MAG_AUTO<25 response term and validated on that same cut. Fixed the discussed
lambda0.01, optimizer seed1, step128 checkpoint from the target-guard repeat
study before new outputs (SHA256 dfc8d59d378ab7e76e49cfbb7862bd2512ae461deb50e1122ff88850880780e1).
Its full training trajectory endpoint has m_eq+0.00106%; full validation
m_eq+0.87481%. This is the single-cut continuation, not the original 16-cut flow.

Prepared [frozen protocol](evidence/galsbi_target_guard_constant_20261104/rule.md#data-estimator-and-unchanged-components)
and isolated study `sbsi_caches/galsbi_target_guard_constant_20261104`.
Reuse the established constant-shear evaluator unchanged, GalSBI-dev C8/emulator
and parent/C8 caches. Candidate flow draws and output-conditioned emulator
responses are new. Forty fixed cases1000–1009/1040–1049/1080–1089/1120–1129,
g1=+/-0.02; primary independent own-leg measured radius>3px/magnitude<25 cuts;
64 antithetic common-random-number draws, original seeds/batch size; no truth cut.
Report all existing15 cuts, C8 primary and true usability flags secondary,
m=100*(R_sim/R_model-1), own-leg normalization,10,000 case bootstraps.
Reducer checks exact simulation replay and saved baseline m/error, then reports
paired change. This is a combined forward-model test, not inference or isolated
flow-self calibration. These constant-shear cases were used in development;
no untouched acceptance or production promotion.

Syntax checks passed for the frozen evaluator, new wrapper and shell launcher;
all40 parent/C8/reference-output paths exist and candidate/source/preset hashes
are pinned. Submitted cip array17730541 (chunks0,4, throttle2), inter array17730542
(chunks8,12, throttle2), dependent CPU summary17730543 and two-hour status17730548.
No other SBSI GPU jobs were active at submission; maximum four GPUs. Both cip
workers loaded the model and report NVIDIA A40-16Q; inter workers initially
queued. First cases1000/1040 completed in321/322seconds with finite outputs and exact simulation replay against the baseline; saved startup_receipt.json. Inter worker8 loaded the model and is running; worker12 remains queued. No full m result yet; dependent reduction and the two-hour scheduled check remain in place. Final receipt
will be `comparison.json`, primary key `C8/mag25_radius_gt0.60`.


## 2026-11-04 — Audit whether target-response error bars are too small

Owner questioned both the fixed-checkpoint validation m=0.874807% +/-0.347154 pp
and measured train/validation response difference 0.00699076 +/-0.00277436.
CPU Slurm 17730387 used saved case sums only plus 200 scene JSON metadata files;
no model evaluation, catalogue scan, training or production changes. Preserved
[audit receipt](evidence/response_error_audit_20261104/report.json) and executable
source in `doc/evidence/response_error_audit_20261104/`.

Fixed-checkpoint m replays exactly, including its bootstrap error. Delete-one-case
jackknife SE=0.347012 pp. Measured train/validation response jackknife SEs
0.00126908/0.00247873 agree with bootstrap 0.00127799/0.00243655. A 20,000-draw
random 160/40 split test of the pooled 200 cases (seed2026110407, full baseline
and normal-matrix recomputation) gives null SD0.00284133 and two-sided p0.01445.
This is an exchangeability diagnostic, not a new independent validation.
All 200 scene seeds are unique and follow [20261101, case]; the simulation rule
uses per-case pixel noise shared only by shear legs. No demonstrated large
case-sampling variance underestimate. Largest case influence variance shares
are 16.9% for the validation simulation response and14.2% for fixed-model m;
bootstrap/jackknife agreement cannot establish unseen-tail coverage.

Scope correction: the m error conditions on the trained checkpoint, fixed
measured-baseline influence definition and shear design. It does not include
retraining/model-selection variability or full inference/model uncertainty.
The target difference does recompute baseline mass, selected mean and shear
normal matrix per bootstrap. These two discrepancies share validation measured
responses and are not independent evidence. Repeated optimizer seeds do not
resample training catalogues or the common starting fit. Next, if total pipeline
uncertainty is required, use independent simulation splits with refitting and
untouched evaluation; do not inflate reported errors by an arbitrary factor.


## 2026-11-04 — Case uncertainty of the single-cut simulation response targets

Owner asked for errors on training response 0.715244 and validation response
0.722234 quoted in the response-regularization review. Recomputed the GalSBI
training-family measured half-trace R=(R11+R22)/2 at the same soft FLUX_RADIUS
>3 pixels / MAG_AUTO<25 cut (widths 0.05), matched usable g0/g05 pairs and
original 160/40 case split. CPU Slurm 17730360 completed 0:0 in 2m36s;
no model evaluations, retraining or preset changes. Whole-case bootstrap
(10,000 independent within-split draws, seeds 20260925/20260926) recomputes
baseline mass, baseline selected mean and the shear normal matrix each time.

Training R=0.7152435188 +/-0.0012779868; validation R=0.7222342837
+/-0.0024365465 (one-standard-error case bootstrap). Difference
0.0069907649 +/-0.0027743609, 2.52 standard errors; relative difference
0.977396% +/-0.388737 percentage points. Case jackknife SEs are 0.00126908
and 0.00247873. This modest significance does not establish a population shift
or a model-generalization cause. It is distinct from flow integration error
and uncertainty on simulation/flow response ratios.

Validation: point targets replay within 3.11e-15; original pair counts
23,279,984 / 5,817,708 and disjoint case membership verified. Preserved script,
matching receipts, sufficient statistics, bootstrap draws and
[report](evidence/target_response_uncertainty_20261104/report.json) under
`doc/evidence/target_response_uncertainty_20261104/`; external run uses the same
basename in `sbsi_caches`. No new simulations or independent acceptance claim.
Next: case-level simulation-minus-flow residual analysis if investigating the
origin of the split difference; these target errors alone do not isolate it.


## 2026-10-18 — Correct review scope to the latest galsbi-tempering branch

Owner clarified that the review must use the most up-to-date branch. The prior
entry reviewed this older `dev` working tree and is historical background only.
Verified latest local/remote `worktree-galsbi-tempering` at21ed49e and wrote
[the corrected review](../.claude/worktrees/galsbi-tempering/doc/INFERENCE_DIAGNOSIS.md#scope-and-conclusion--october-18-2026)
in that worktree, with its own evidence receipt and work-log entry. Added an
explicit supersession notice to this checkout's `doc/INFERENCE_DIAGNOSIS.md`.
The current branch includes later successful numerical improvements and an
unlocalized1M paired-mock c2 residual; the earlier review is not its status.
Validation: remote/local branch identity and local document links checked;
no code, models, scheduler jobs or historical experiment outputs changed.

## 2026-10-18 — Review inference context, evidence and limits of localization

Owner requested a fresh review of recent logs: problem context, experiments
and results, and what has actually been localized. Added
[inference diagnosis](INFERENCE_DIAGNOSIS.md#review-scope-and-conclusion--october-18-2026),
covering September25–October2 science evidence, with separate finite-catalogue,
continuous-sampling, population-consistency and optimizer questions. October3
entries concern the README figure. This is a retrospective review, not a new
sampling result or current scheduler-status check.

Rechecked seven small saved JSON reports and recorded SHA256 identities and
extracted results in `doc/evidence/inference_review_20261018.json`. Bright32k
full convergence remains0/128 in both overlapping cohorts; local-ratio checks
pass35/128 and10/128. Retrieved completed reports beyond earlier pending-status
text: the four-replica million-step pilot passes combined checks for three
objects, but obs1 remains at worst Rhat1.04247/minimum bulkESS298.50, with both
local-ratio Rhats above1.01. The two-replica64k measurement-initialized pilot
still fails for all four objects. None is an accepted recovery result.

The review distinguishes established numerical failures from causal claims:
conditional curved paths support proposal development but do not measure
posterior mass or prove global connectivity; wide-shear overlap is measured
from unconverged reference samples; target generation/density consistency and
finite-prior resolution remain unresolved. Proposed next work is a bounded
target-consistency check and frozen curved-proposal comparison with independent
validation, followed by shear bridges and only then recovery/population tests.
No new experiment, closed model investigation, or large run was launched.

Validation: lightweight Python JSON assertions/hash extraction passed for all
seven receipts; local Markdown file/anchor check passed33 links. Patch whitespace
checks passed. No raw chains, large atom sums or model evaluations were rerun;
no inference code, defaults or models changed, so no numerical suite was run.
Existing unrelated workspace changes are preserved. See
[next discriminating tests](INFERENCE_DIAGNOSIS.md#where-to-look-next-and-what-each-test-would-decide)
and [receipt scope](INFERENCE_DIAGNOSIS.md#saved-receipts-and-validation).

## 2026-10-03 — Thicken README blending ellipses and arrows

In `scripts/plot_readme_figure.py`, increased dashed ellipse strokes from
1.1 to 2.0 points and curved arrows from 1.25 to 2.2 points, with proportionally
larger arrowheads and ellipse outlines. Regenerated
`doc/figures/measurement_flow_contours.png` and its receipt. Layout, neighbour
choices and scientific settings are unchanged.

Validation: syntax compilation passed; Slurm CPU job 16805406 completed with
exit 0 in 32 seconds. Visually checked the regenerated figure. Scoped
`git diff --check` passed. No outstanding work for this styling change.

## 2026-10-03 — Simplify README figure labels and confine blending guides to the first stamp

Updated `scripts/plot_readme_figure.py` and regenerated the README PNG/JSON:
removed the overall model title, retained all eight neighbour ellipses only
in the bright stamp, and added two curved-arrow labels beside that stamp.
The arrows identify recognized neighbour 184965 and unrecognized neighbour
629505, with their catalogue recognition states checked before rendering.
The typical and faint stamps have no ellipses. Moved the measurement and
input-truth legend inside the ellipticity panel and removed the separate
legend strip. The receipt distinguishes displayed overlays from the full
neighbour inventory. Updated `README.md` and
[figure conventions](README_FIGURE.md#figure-conventions) accordingly.

Validation: syntax compilation and scoped `git diff --check` passed. Slurm
CPU job 16805381 completed successfully in 46 seconds. Visually checked the
PNG for label placement, curved arrows, legend placement and absence of an
overall title. Receipt checks confirm displayed ellipse counts `[8, 0, 0]`
and two callouts with opposite recognition states. The model, examples,
random seed, draw count and PSF reference method are unchanged. These remain
illustrative guides, as described in the figure conventions; no further work
is required for this layout revision.

## 2026-10-03 — Restore the README figure with v3.9-like contours and blending guides

Restored the original three ConstGold case 40/+0.02 examples and plotting layout
from Git `18b4218` into `scripts/plot_readme_figure.py`. The README now embeds
`doc/figures/measurement_flow_contours.png`; `.gitignore` admits only this new
PNG and its JSON receipt within the otherwise ignored figure directory.
`jobs/job_readme_figure.sh` provides a bounded CPU regeneration job.

The figure samples the pinned F8 checkpoint (SHA-256
`bd51dbbbeeb72763f6982a958cb8f3a0b2d3906d4f78935d9d959fa85a17797e`),
rebuilding its eight physical inputs from the complete scene. It uses the
requested lowercase `v3.9-like` display label and removes `R_blend` from stamp
titles. Dashed cyan/gold ellipses mark recognized/unrecognized neighbours by
unique detection crossmatch, with illustrative PSF-broadened footprints and
the renderer's pixel-coordinate orientation. The three stamps contain 8/11/10
neighbours, of which 3/7/2 have their own matched detections.

Input-truth radius markers now integrate isolated sheared Sérsic profiles
convolved with the simulation's truncated Moffat PSF and pixel response.
The bright/typical/faint circular half-light radii are 0.972757/0.811249/0.784618
arcsec; halving numerical sampling from 0.05 to 0.025 arcsec changes them by at
most 0.003475 arcsec (required tolerance 0.005). Total input flux and the
sheared pre-PSF truth appropriate to PSF-corrected ngmix shapes are retained.

Validation: `python -m py_compile scripts/plot_readme_figure.py`,
`bash -n jobs/job_readme_figure.sh`, local Markdown-link checks and scoped
`git diff --check` passed. Final Slurm CPU job 16805330 completed with exit 0
in 41 seconds. The launcher uses a job-local bytecode cache after shared stale
bytecode delayed an overlay-orientation edit; final receipt angles confirm
the corrected source executed. Visually inspected
the final PNG. All 314,869 crossmatches joined to shapes and truth; 3,168 shape
rows lacking crossmatches were reported and dropped. The run rejects ambiguous
identities and invalid draws, checks artifact identity, and verifies radius
integration convergence. No GPU allocation or model training was needed.

Limitations and reproduction are documented in
[README figure conventions](README_FIGURE.md#figure-conventions). Ellipses are
guides, not segmentation or measured contamination. The three examples are
illustrative; contours use the conditional F8 flow alone, without C8 weighting
or disk-response transport. No likelihood setting, model artifact, archived
investigation or inference cache changed. No further work is required for
this figure revision.

## 2026-10-02 — Deep MH convergence audit isolates geometry and shear-overlap failures

Owner requested a deep investigation. Added `scripts/diagnose_galsbi_convergence.py`,
`probe_galsbi_geometry.py`, `plot_galsbi_convergence.py`, and `probe_galsbi_ridges.py`;
recorded methods, limitations, receipts and next proposal design in
[convergence audit](MH_CONVERGENCE_AUDIT.md#finding--october-2-2026). Linked bright
results and continuous-sampling status. All238 saved observations were audited;
post-hoc geometry panel10, conditional optimized-path panel4. Original bright
runs, immutable source snapshot, frozen models, observations and seeds remain intact.

Exact80-state prior/kernel replay differences0; summary error2.84e-14. Median
morphology acceptance about30% conceals poor movement. In one difficult object,
a straight conditional path drops52.27 log-density units but a curved route
between the same endpoints has no sampled drop. This is geometry evidence,
not posterior samples/global connectivity. Larger Gaussian proposals do not
uniformly improve mobility. Near-shear weights are healthy despite chain
disagreement; atg1=.010,78/128 brightest objects have raw weight ESS<400,
and at.040,89/128. Broad reweighting has a separate overlap problem. No recovered
shear, hybrid acceptance or10-second convergence claim is warranted.

Fixed `scripts/galsbi_bright_math.py::profile_fit` to use the same interpolated
curve for peak and interval crossings; added a narrow off-grid-peak regression
to `tests/test_galsbi_bright.py`. Derived corrected intervals are separate from
original reports and remain scientifically unvalidated. Five focused tests pass;
full scheduled suite16794283:383 passed,1 skipped in101.46s. Ruff passes for the
four diagnostics, helper and test. CPU audit16794252 completed1m49s, GPU geometry
16794276 completed1m54s, profile audit16794287 completed52s, GPU curved paths
16794310 completed1m14s; all0:0. Audit used one GPU at a time and all jobs finished.
Artifacts and executed-script checksums live in `galsbi_convergence_audit_20261002`.
Next: discarded-pilot frozen curved proposals with exact correction and independent
validation, then multiple reference-shear bridges. No20k or automatic longer
sampling run submitted; no failures removed from either cohort.

## 2026-10-02 — Bright MH jobs finish but convergence and shear recovery fail

Owner requested status. Read scheduler accounting and saved reports: both
bright replicas16791908_0/_1 completed0:0 in1h35m10s/1h35m09s, report16791910
in19m13s, and independent throughput16788845 in3h26m. No jobs remain active.
Updated [completed bright results](BRIGHT_MH_TEST.md#completed-result--october-2-2026).

At32k steps, full convergence passes0/128 in both cohorts; both local shear-ratio
checks pass35/128 below20 and10/128 below22. Exploratory conditional fits are
g1=.01410259 and.01077937 against injection.02, but convergence and broader
profile overlap fail, so these are not accepted recovery or bias measurements.
Conditional MH shear MCSE7.66e-4/1.48e-3 dominates measured normalization MCSE
2.75e-5/1.99e-5; selection ladders pass. Noted that the coarse report's mixture
of quadratic peaks and linear interval crossings yields inconsistent intervals
for some replicas. Those intervals are not validated and must be refined before
any future acceptance; original frozen reports remain intact.

Bright sampler cost22.53/22.51 GPU seconds per distinct galaxy at32k; complete
allocated workflows about24.0s including wider profiles/selection. Representative
1024-object16k full-process benchmark costs12.0659s per galaxy, above10s.
Validation: read-only JSON/accounting audit and compact diagnostic count/quantile
calculations. Documentation-only status update; no tests rerun, jobs submitted,
settings changed, or observations removed. The20k run remains paused.

## 2026-10-02 — Bright cohorts and numerical smoke pass; both replicas start

Preparation16791905 completed0:0 in58s:128 observations per measured cut,
238 distinct rows,18 shared. Read-only cohort audit confirms strict<20/<22
membership and no dropped failures. Smoke16791907 passed MH validation,
saved-summary profile replay (maximum error0), and both forward selection
integrals. Both main array tasks16791908_0/_1 are running on inter A40s.
Measurement initialization completed in277.41/275.22s, both initial likelihood
replays have maximum error0, and both reached step128/32768. This is startup
success, not convergence or shear recovery. See
[receipts](BRIGHT_MH_TEST.md#preparation-and-startup-receipts).

Added a bounded CPU status check16792113, scheduled in about2h, to write
`galsbi_bright_mh_20261002/scheduled_check.json`. It reads scheduler state,
progress, failures and any completed report; no chat wakeup tool is available.
The helper compiles and the new protocol's local document anchors validate.
Scientific source snapshot and plan remain unchanged; operational job metadata
is in `jobs.json`. Startup monitoring is complete; long sampling proceeds with
the scheduled check and dependent report, without occupying a chat polling it.

## 2026-10-02 — Launch bright-only continuous MH and selected shear-recovery test

Owner requested tests below measured magnitudes20 and22, including convergence
and shear recovery. Implemented optional cut-specific cohorts in
`scripts/prepare_galsbi_population.py`; added `scripts/galsbi_bright_math.py`,
`scripts/run_galsbi_bright.py`, `scripts/report_galsbi_bright.py`,
`jobs/job_galsbi_bright.sh`, and `tests/test_galsbi_bright.py`. Documented the
protocol in [bright MH test](BRIGHT_MH_TEST.md#owner-request-and-scope--october-2-2026)
and linked the hybrid assessment. No model or inference default changes.

Use128 fresh observations per measured cut, deduplicating shared rows without
altering either cohort; two independent four-chain32768-step runs with8192
warmup. Recovery profiles are conditionalg1 atg2=0, with independent cut-matched
selection populations and explicit covariance/MCSE, overlap, prefix and
selection-ladder checks. Failed objects stay included; unsupported profiles
cannot pass. This is GalSBI model closure, not FS2 recovery or a0.3% claim.

Validation: four focused tests passed; full CPU suite16791497 completed with
382 passed,1 skipped in65.77s. `ruff check` passes (system executable; py31
does not contain the Ruff module); Python compilation and bash syntax pass.
Tests cover strict measured selection and overlap, an analytic selected
Gaussian likelihood, invalid/boundary fits and correlated peak sensitivity.
Frozen225-file source snapshot and immutable plan are under
`catalogue_prior/galsbi_bright_mh_20261002`; job IDs are separate in `jobs.json`.

Submitted preparation16791905, smoke16791907, two-replica array16791908_0–1
(throttle2), and CPU report16791910 with prerequisite dependencies. Preparation
is initially Priority-pending; main jobs cannot start before numerical smoke.
At most two new GPUs plus the existing throughput GPU overlap, within the
four-GPU cap. The20k population run remains paused. Monitor preparation and
smoke through startup/completion; numerical or scientific failures are results
to diagnose, not permission to discard galaxies or force shear recovery.

## 2026-10-02 — Evaluate atom integration with a continuous-MH fallback

Owner requested a deep assessment while the throughput study runs. Added
`doc/HYBRID_SAMPLING.md` with the common-prior derivation, observation-dependent
prior-splice normalization, separate retrieval/discretization/MH errors,
brightness-plus-diagnostic routing, shared-bank/selection error accounting,
cost arithmetic and a bounded proposed comparison. Linked it from
`doc/CONTINUOUS_SAMPLING.md`. Reusing the old algorithm with a new common-prior
bank is distinguished from directly joining FS2 and GalSBI populations.
No inference code, model, routing default or scheduled science settings changed.

Evidence review: actual20k has135 measured-mag<20 and909 mag<22 rows; at the
unverified10 GPU-second target, their two-replica MH components cost0.75 and
5.05 GPU hours, excluding atom integration, bank preparation and normalization.
Light inspection of the1024-row saved GalSBI measurement file finds5 and51
rows respectively. Historical bright-extension results are labelled V3.6 and
three-case diagnostics, not V3.9 validation. Reviewed pop-cosmos v2 inference
and the rank-normalized Rhat reference; both are linked in the assessment.

Scheduler accounting confirms ensemble16788846_0–3 and report16788848 all
completed0:0. At16k steps, control obs1 ratio Rhats1.07745/1.04877 improve to
1.00445/1.00710 with stretch, but stretch obs0 worsens to1.05724/1.05999 and
obs1 bulk ESS remains346/265. No uniform mixing or shear-accuracy acceptance.
Throughput16788845 completed measurement initialization in1191.16s, passed
initial likelihood replay with maximum error0, and reached step346/16384;
full-process timing remains pending. Existing scheduled check16788875 remains;
no additional jobs launched and20k remains paused.

Validation: small saved-JSON/Parquet inspections and explicit count/cost
arithmetic; local Markdown link/anchor checks and `git diff --check` pass.
Documentation-only change, so no numerical tests rerun. Next scientific step
is common-prior bank consistency and paired method validation, including a
representative faint audit; this assessment does not certify either sampler.

## 2026-10-02 — Distinct-observation cache replay passes; main benchmarks released

Paired128-observation job16788841 completed0:0 in6m44s. All retained latent
states, MH decisions, log priors/kernels and all256 local shear profiles agree
exactly between cache-off/on. Maximum summary difference4.26e-14 is within
the unchanged1e-10 tolerance. Initial replay is exact in both arms. Sampling
time is25.097s versus18.525s (26.2% less); full256-step processes are204.806s
and194.07s. Baseline measurement initialization costs156.11s, or1.22s per
galaxy. This short run does not establish the full16384-step throughput target.

Four-arm startup16788842 completed0:0 in1m56s; both independent control and
stretch seed pairs passed numerical receipts. Main control16788846_0 has
passed warmup and made substantial retained-sampling progress. Remaining
ensemble tasks and full1024-observation throughput16788845 are scheduler
Priority-pending, with successful prerequisite dependencies cleared. Queue
waiting has become prolonged; no duplicate jobs or scientific changes made.
CPU ensemble report16788848 remains dependent. One bounded Slurm CPU status
check16788875 is scheduled about2h later to inspect accounting/progress/errors
and write `galsbi_throughput_20261002/scheduled_check.json`. No Codex chat-wakeup
tool is available, so this records cluster status without promising a chat
notification or autonomous repair. Both check files pass syntax validation.
The20k array stays paused. See
[bounded experiment](CONTINUOUS_SAMPLING.md#bounded-throughput-and-hybrid-ensemble-experiment--october-2-2026).

## 2026-10-02 — Launch bounded full-process throughput and hybrid-ensemble comparison

Owner authorized proceeding. Added complementary morphology stretch proposals
to `sbsi/mcmc_proposals.py` and opt-in `--ensemble-morphology` plus stage timings
to `scripts/check_galsbi_local_sampler.py`. The3D stretch move uses frozen
complementary walkers and the exact2log(z) MH correction, retaining full-support
local and original neighbour moves. Models, target density, support, numerical
tolerances and default proposals are unchanged. Interacting walkers are flagged;
`scripts/report_galsbi_sampler_runs.py` rejects independent-chain treatment.
`scripts/report_galsbi_ensemble.py` instead computes expectation diagnostics
after averaging walkers within each independent ensemble. Added mathematical
closure and covariance tests; fixed the scalar-conversion warning in
`scripts/report_galsbi_population.py` without changing its calculation.

Added `scripts/run_galsbi_throughput.py` and `jobs/job_galsbi_throughput.sh`
for full subprocess timing, distinct-observation cache replay, four-arm smoke,
and a matched-seed original/hybrid proposal pilot. Preserved sources and plan
under `catalogue_prior/galsbi_throughput_20261002` (156 files). Preparation
16788816 generated1024 independent selected observations in26s; geometry
error2.22e-16 and full/partial/order likelihood replay errors all zero.
Initial suite16788817:376 passed,1 skipped; final suite16788840 including
new ensemble-report tests:378 passed,1 skipped in55.21s. Ruff and Python/shell
syntax checks pass. No heavy work ran on the login node.

Submitted paired cache16788841 and startup16788842, both observed running
on inter A40s; first control startup completed with valid numerical checks.
Throughput16788845 follows successful cache replay; ensemble16788846_0–3
follows successful startup with throttle2; CPU report16788848 follows any
ensemble outcome. At most three GPUs can overlap, below the four-GPU cap.
Full run16384 steps/4096 warmup/four chains; timing includes initialization,
sampling, output and local profiles. The two-replica proposal pilot uses
the original four-observation stress panel, including the unresolved obs1.
The20k launch remains paused. No measured10-second full-process result or
accuracy acceptance yet; monitor dependent stages through startup. See
[bounded experiment](CONTINUOUS_SAMPLING.md#bounded-throughput-and-hybrid-ensemble-experiment--october-2-2026).

## 2026-10-02 — Assess pop-cosmos paper and adopt initial10 GPU-second target

Owner set10 seconds per galaxy per GPU as the first target and requested
reading arXiv2602.03930. Reviewed its v2 inference section, initialization
and convergence-sensitivity footnotes, and the earlier method paper
arXiv2406.19437v2. Recorded applicability and limits in
[throughput assessment](CONTINUOUS_SAMPLING.md#october-2-ten-gpu-seconds-per-galaxy-and-the-pop-cosmos-comparison).
The new milestone supersedes the interim0.1 core-second target. Retain the
earlier per-replica interpretation and count initialization/output in the
budget. The20k population array remains paused and was never submitted.

Recomputed costs directly from `galsbi_cost_20261001/cuda/result.json`:
at1024 observations,16384 steps give9.24s per galaxy in measured components;
cached-component subtraction estimates7.27s. Neither includes the entire
run or establishes mixing. CPU response and scene inputs account for58.5%
and17.5% of baseline component time, versus2.1% for the neural likelihood.
Affine-invariant target proposals are a possible future controlled comparison,
not an implementation change or permission to replace the frozen models.
Plan receipt updated with the initial speed target. Scheduler check shows no
active/pending user jobs; no new jobs or code changes, so no tests rerun.

## 2026-10-01 — Prepare 20k population diagnostic; pause for owner throughput target

Recent-job check: both64k measurement-start replicas16788678_1/_2 and
report16788679 completed successfully. Combined worst Rhat/minimum bulk ESS
at65536 steps for obs0–3 are1.09042/67.0,1.07064/93.7,1.02013/441.1,
1.03448/289.9; all numerical checks pass, but none passes all exploration
checks. Long array16781571 and report16781574 also completed; at1048576
steps combined obs0/2/3 pass, obs1 fails atRhat1.04247/ESS298.5, including
shear-ratio Rhat1.01513/1.01799. GPU cache replay16787894 passes with identical
latent states and MH decisions. GPU cost16787881 passes the baseline replay;
the alternative CUDA response predictor fails log-kernel replay at512/1024
observations (9.13e-5/4.93e-4), despite small coefficient differences.

Owner requested20k observations and selected16384 steps,4096 warmup, four
chains per observation and two independent replicas. Added
`scripts/prepare_galsbi_population.py`, `scripts/run_galsbi_population_shard.py`,
`scripts/report_galsbi_population.py`, and `jobs/job_galsbi_population.sh`.
The existing mock generator gains optional full measured-selected population
retention; the local sampler accepts a separately validated population
reference with observation checksums and fresh initialization banks. Original
pilot defaults, model artifacts, likelihood, proposal mixture and tolerances
are unchanged. The planned population is newly model-generated with the same
measured cuts, without truth cuts or magnitude-stratum selection.

The reporter compares independent-replica numerator log profiles atg1=.018/.022
and8k/16k prefixes, retaining within-shard and between-shear-point covariance
in conditional MCSEs. It rejects missing shards, coverage mismatches and failed
numerical receipts. It does not infer selected-population shear: the identical
selection normalizer cancels only in replica differences. Two replicas and
prefix agreement cannot rule out shared missed-mode or finite-log bias.
Tests in `tests/test_galsbi_population_report.py` check correlated-noise MCSE,
nonfinite draws, incomplete coverage and full-population measured selection.

Validation: Python compilation, bash-n and Ruff pass. Full suite16788794
completed0:0:374 passed,1 skipped in65.26s, with two NumPy scalar-conversion
deprecation warnings in the new report. Frozen153-file preparation snapshot
and plan are under `catalogue_prior/galsbi_population20k_20261001`.
Preparation job16788795 started and was cancelled after16s when the owner
rejected the36–48h estimate on four GPUs. No main array, smoke or report jobs
were submitted; no completed20k reference or sampling results exist.

Owner subsequently clarified the speed target to approximately10 galaxies
per second per core, replacing the interim10 seconds per galaxy per replica
interpretation. CPU-core versus GPU-worker scope is pending clarification.
Keep population execution paused while reassessing throughput and the method;
do not present tiled component estimates as full-process performance. See
[population throughput target](CONTINUOUS_SAMPLING.md#population-diagnostic-and-throughput-target--october-1-2026).

## 2026-10-01 — Authorize and launch 64k measurement-initialized MH comparison

The 16k initialization comparison completed all four workers in two waves,
about13min total; report16788283 completed. Across two replicas, prior-bank
versus measurement-start worst Rhat by observation was1.4427/1.1643,
1.6090/1.4959,1.1245/1.0703,1.0874/1.0494. Minimum bulk ESS remained below400
for all four observations in both arms. Better starts help but do not establish
convergence. No accepted shear estimate or sampler promotion follows.

Owner authorized two measurement-start replicas at65536 steps, retaining4096
warmup, the same four observations, four chains per observation, MH seeds
2026100141/142 and bank seeds2026100241/242. These replay the original starts
and then extend the trajectory; they are not additional independent replicas
of the earlier same-seed runs. Original frozen150-file source snapshot is
reused unchanged; models, proposal mixture, adaptation and cache-off setting
are unchanged. New external root `catalogue_prior/galsbi_measurement_init_64k_20261001`
contains the plan, launcher, snapshot link and automatic report launcher.
No active library code changed. Both shell launchers pass bash-n.

GPU array16788678 (tasks1/2, throttle2) requests one inter A40 per task and
40min walltime, without unrelated dependencies. No other user GPU jobs were
active at submission, within the combined four-GPU cap. CPU report16788679
on cip-cl-compute3 follows after any array outcome and checks16384/32768/65536
prefixes, including shear likelihood ratios. Expected elapsed is roughly25min
on two GPUs based on the six-minute16k runs; this is an estimate, not a bound.
Both tasks started on inter kng-cl-nv02, passed initial likelihood replay
with exactly zero error, and reached896/65536 steps without startup errors.
Final convergence results remain pending.

## 2026-10-01 — Release initialization comparison alongside old MH runs

Owner authorized running the two inter initialization workers alongside the
two old cip workers, raising this continuous-prior study's combined cap to
four GPUs. Recorded the scope in `AGENTS.md` and both external study plans.
Removed the artificial old-long/cache dependencies from array16788277;
throttle2 remains. Smoke16788261 was independently verified COMPLETED0:0.
Slurm rejected reattaching this already-finished smoke dependency, so cleared
the dependency field after verifying its successful accounting record.
The released array initially showed Priority rather than Dependency.

Cost job16787881 still waits for old task16781571_2 to release one slot;
cache16787894 follows the cost job. With at most two initialization workers
and the remaining old task3, this schedule stays within four GPUs. Report
dependencies, seeds, sources, models and numerical settings are unchanged.
Validation: sacct smoke status and scontrol/squeue dependency/resource audit;
no code changes or tests required. Tasks0/1 started on inter kng-cl-nv02;
both initial likelihood replays have exactly zero error (tolerance1e-6), and
both reached step1920/16384 without errors. Both old cip tasks were still
running, confirming four simultaneous GPUs. The second pilot pair and final
comparison remain pending; startup success is not a convergence result.

## 2026-10-01 — Move queued continuous-sampler GPU stages to inter

Owner requested trying `inter`. Scheduler inspection found four unallocated
A40s on kng-cl-nv02; the queued study stages were dependency-blocked rather
than waiting on GPU capacity. Updated existing jobs16787881 (cost benchmark),
16787894 (paired cache replay) and16788277 (initialization array) in place to
`Partition=inter TresPerNode=gres:gpu:a40:1`. Verified all three partition/GRES
requests and existing dependencies with scontrol; array throttle remains2.
Job ids, seeds, frozen sources, scientific settings and dependent reports are
unchanged. Running cip long jobs are not interrupted, and the combined two-GPU
limit still holds through the preserved dependencies.

Updated both study plan receipts under `galsbi_cost_20261001` and
`galsbi_measurement_init_20261001`. The first TRES spelling with a slash was
rejected after Slurm applied the partition change; corrected immediately to
the colon form and verified the final resource requests. No duplicate jobs
were submitted. All moved jobs are still pending upstream completion; check
their resource receipts and numerical replay when they start. Benchmark
timings on inter full A40s must not be presented as a controlled speedup over
cip virtual8GB A40 allocations. No code or model changes; no tests rerun.

## 2026-10-01 — Measurement-informed versus prior-bank initialization pilot

Owner explicitly requested trying starts near the measured properties. Added
`sbsi/galsbi_initialization.py` and the optional `--initialization measurement`
flag to `scripts/check_galsbi_local_sampler.py`. Candidate initial magnitude
comes from measured flux; radius uses a rough Moffat-PSF quadrature subtraction;
shape uses approximate inverse shear plus broad epsilon scatter. Four streams
have different Sérsic centres. Native redshift/SED and neighbour scenes come
from fresh prior banks. Candidate selection uses target-native log prior plus
the unchanged full likelihood; generating truth is never read. No guess
floor/clamp/scatter restricts subsequent MH support. The default initializer
is unchanged. New `initial_state.npz` and initialization receipts preserve
coordinates, summaries, log densities, neighbour counts, candidate counts and
preparation time. Reporter compatibility includes the initializer setting.

This compares initializer packages: the new arm changes both candidate
centring and ranking from the original highest-likelihood prior-bank control.
It cannot isolate those contributions. Both arms score2048 candidates per
chain and retain the same target, original proposals, warmup/adaptation,
4096-row density calls and numerical tolerances. Invariant-work reuse stays
off in both arms. Four fixed observations; four chains per observation;
MH seeds2026100141/142; bank seeds2026100241/242;16384 total steps with4096
warmup. Task order counterbalances the two arms over two seed pairs/waves.

Added `tests/test_galsbi_initialization.py` for measurement units, inverse
shape conventions, unchanged nuisance coordinates and boundary/support
handling. Full suite16788224:370 passed,1 skipped in67.59s. Ruff, Python
syntax and shell syntax checks pass. CPU startup16788261 exercises all four
arm/seed combinations at64 steps/32 warmup: completed3m47s, peakRSS1.04GB.
All four initial replays are exact; SED logK replay is exact and summary
differences are at most2.84e-14. Short-chain convergence is deliberately not
a startup gate.

Frozen150-file snapshot and plan:
`catalogue_prior/galsbi_measurement_init_20261001`. Launcher
`jobs/job_galsbi_initialization.sh` runs the study; GPU array16788277 has
four tasks with throttle2 and waits for successful startup and completion of
long array16781571 and cost/cache GPU continuation16787894. This preserves
the two-GPU study cap. CPU report16788283 runs after any pilot-array outcome.
`scripts/report_galsbi_initialization.py` generates within-arm prefix reports
and cross-arm likelihood-ratio agreement, initial-state and timing evidence.
It does not assume independent arm MCSEs under matched seeds. Startup has
been monitored through completion; GPU stages remain dependency-pending.

See [initializer comparison](CONTINUOUS_SAMPLING.md#measurement-informed-initialization-comparison--october-1-2026).
Higher initial likelihood is not posterior convergence. Retain Rhat<1.01,
bulkESS>400 and all numerical checks, including shear-relevant ratios; no
automatic promotion, population-scale launch or selected-shear claim.

## 2026-10-01 — Cost audit and validated CPU savings from invariant MH work reuse

Owner requested a deep efficiency investigation after the million-observation
cost estimate. Read the existing long-chain/mixing evidence and profiled the
actual model workload. Added `scripts/benchmark_galsbi_sampler.py` and
`jobs/job_galsbi_benchmark.sh`: controlled tiled-scene component timings,
identical-proposal MH replay, fixed4096 density batches, model/source receipts,
and CPU1/CPU4/CUDA response comparisons in the pending GPU arm. Warmup-only
cProfile does not contaminate reported timing repeats. These probes do not
establish convergence or representative million-observation throughput.

CPU job16787846 completed in2m09s, peakRSS928MB. For4/32/128 observations
(four chains each), weighted baseline component cost is249/276/369ms per
step. CPU neural density stays231ms due to padding, while response prediction
grows4.66/20.76/76.12ms and scene-input mapping6.66/12.24/31.39ms. All batch
and invariant replay checks pass, with no changed MH decisions. GPU timing
cannot be inferred by dividing this CPU measurement by an assumed factor.

Added opt-in `--reuse-invariant-work` to `scripts/check_galsbi_local_sampler.py`:
reuse K on z/SED-only moves (25% of the original mixture), reuse target prior
on neighbour-only moves, and avoid copying unchanged neighbours after target
moves. Full native support/Jacobians/Hastings factors remain active. Guard
requires the established persistent/joint-morphology configuration; invariant
blocks are periodically fully replayed at1e-10 summary/1e-6 logK tolerances.
The flag remains off by default. Reporter protocol compatibility now includes
the flag. Existing long jobs use frozen sources and are unchanged.

Added a full-scene redshift/SED invariance test with nontrivial photometry,
plus `scripts/check_galsbi_cache_replay.py` for paired512-step real-model runs.
Full suite16787870:368 passed,1 skipped in60.77s; Ruff and launcher syntax
checks pass. CPU replay16787880 failed safely against old GPU-derived starts
(2.35e-5 versus1e-6 tolerance). Retried with a fresh bank seed2026100133
evaluated on the same device; no tolerance was relaxed and failed outputs
remain preserved. Retry16787889 passed: exactly identical retained latent
states, MH decisions, log priors/kernels and local profiles; summary error
at most2.84e-14. Whole processes164.41s→132.82s; steps128–512 give249.74ms→
187.80ms, a24.8% loop-time reduction on CPU. The first frozen wrapper's
`sampling_seconds` includes initialization; current source instead derives
loop cost from progress differences. No GPU speedup or mixing improvement
is claimed from this test.

Study root `catalogue_prior/galsbi_cost_20261001` contains frozen sources,
plans, profiles and receipts. GPU benchmark16787881 waits for long task
16781571_2, preserving the two-GPU cap, and compares batch sizes through1024
observations and response devices. GPU paired replay16787894 follows the
benchmark and successful CPU paired replay. Startup was verified for the CPU
jobs; GPU stages remain dependency-pending. No automation wake-up tool is
available; dependencies provide execution, not an automatic agent review.

See [cost audit](CONTINUOUS_SAMPLING.md#cost-and-scalability-audit--october-1-2026)
for measured component costs, limitations and proposed next experiment.
Curved morphology and conditional environment trapping remain sampler issues;
prior pooled covariance/broad joint moves did not solve them. A small frozen
mixture independence proposal with an explicit MH correction is the proposed
next morphology test, not an implemented or accepted replacement. Production
also needs streamed summaries and a shear-level Monte Carlo error budget,
including finite log/ratio bias. No population-scale launch or0.3% claim.

## 2026-10-01 — Long-chain interim check: three observations pass within both completed replicas

Owner requested status. Tasks16781571_0/_1 completed with exit0 in5h10m32s
and5h10m57s. Their1,048,576-step results pass individual exploration gates
for obs0,2,3, but fail obs1 (measured MAG_AUTO21.2551). Seed2026100121
worst Rhat/minimum ESS by observation is1.00292/842.9,1.12420/24.7,
1.00280/855.4,1.00139/2664.3; seed2026100122 gives1.00328/703.2,
1.05566/54.1,1.00107/2429.8,1.00775/732.9. These are separate four-chain
replica checks, not the pending combined-replica result. New initial banks
also differ from the shorter historical runs, so this is not a paired
causal comparison of chain length alone.

Obs1 remains slow in radius/Sérsic/magnitude/crowding. Its two local log-ratio
Rhat values are1.0180/1.0257 and1.0336/1.0397 in the completed replicas;
the shear-relevant functions therefore also fail. Initial replay is exact.
SED invariance remains within the declared tolerance: maximum summary
change4.2633e-14, maximum log-kernel change9.3599e-7 in seed121 and zero in122.
Do not describe every SED evaluation as exactly unchanged.

Tasks2/3 are running normally at about685k/683k of1,048,576 steps, with
exact initial replay. Current throughput implies roughly two more hours,
including final diagnostics. Report16781574 still waits on the array.
No intervention, acceptance, or new job submission; final combined checks
and prefix diagnostics remain pending.

## 2026-10-01 — Owner-authorized million-step original-MH replicas

Owner explicitly requested longer chains after the runtime discussion. Use
the original per-chain3D morphology tuning, corrected SED jump and persistent
Poisson refresh; disable pooled covariance, joint-environment and local-
position variants. Fixed four-observation panel, four chains per observation,
four replicas with MH seeds2026100121–2026100124;1,048,576 total steps and
32,768 discarded warmup per replica. Run two GPUs concurrently, in two waves.
Measured original runtime implies roughly5.2 hours per wave and10.4 hours
overall, excluding queue/preparation/report time. Six-hour limit per GPU job;
24GiB host memory allows the longer unthinned traces. No batching change is
mixed into this numerical convergence experiment.

Added optional `--initial-bank-seed` to the existing runner. Each replica
draws8192 new prior scenes, splits them into four2048-scene streams and
chooses the highest-likelihood scene from each stream for each observation.
Initial bank seeds2026100221–2026100224 differ across replicas; generating
truth and mock observations are never regenerated or used as starting
states. The full-bank likelihood is replayed after selecting/reconstructing
the16 starting scenes, at the unchanged1e-6 tolerance. Save bank seed,
selected indices and likelihoods in `initialization.json`. Default behavior
still reconstructs the original reference bank.

Existing launchers now accept seed base, initial-bank seed base and warmup;
the report launcher accepts replica count and includes longer prefix checks.
No proposal, density or acceptance changes. Ruff/shell syntax pass. Full
suite16781551 submitted on compute3. Startup array16781558 depends on its
success, checks all four new banks with128 steps/64 warmup, and is throttled
to two GPUs. Smoke convergence is not an acceptance condition; execution,
likelihood replay and SED invariance are. Long-run submission follows these
checks. Source snapshot manifest SHA256:
`098c3993f7c6c1601d131b16ffcdbdd7b9d6db97e67e1bc0f5af8526630d4b8e`.
Protocol/output root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_long_mh_20261001`.
Post-run prefix diagnostics use65k/131k/262k/524k/1,048,576 steps; these are
computed after completion, not live convergence stopping. Gates remain
Rhat<1.01 and bulkESS>400 with independent and combined checks. No automatic
promotion, population run, selection normalization or shear-bias claim.

Full suite16781551 passed367 tests/1 skip in71.40s. All four smoke tasks
16781558 completed with exit0: starting likelihood replay error is exactly
zero for every bank; SED log-kernel change is zero and maximum summary
change2.8422e-14. The new-bank path therefore passed its scheduled runtime
validation in addition to the existing suite. GPU array16781571 is running
tasks0/1, with2/3 throttled behind them. Both running replicas passed exact
initial replay and reached768 steps with finite progress. Afterany CPU
report16781574 has32GiB/45min for the four larger traces. Sources and
scientific settings are frozen. No application wakeup tool is available;
Slurm enforces the sequential two-GPU waves and runs the completion report.

## 2026-10-01 — Pooled and joint-environment pilots completed; neither accepted

Owner requested a check. Scheduler accounting confirms pooled array16781077
completed in20m27s–20m31s and joint-environment array16781117 in20m41s–21m01s,
all exit0. Reports16781079/16781121 and comparison16781124 completed; user
queue is empty. Both variants and both seeds pass initial/SED numerical
checks, but every individual run and combined observation fails at least
one exploration gate. No sampler promotion or further job submission.

At matched65536 total steps/32768 warmup, worst combined Rhat for obs0–3:
original1.3009/1.3116/1.0489/1.0264;
pooled1.1175/1.4171/1.0324/1.0438;
pooled+joint1.2918/1.2254/1.0221/1.0211.
Minimum bulkESS respectively20.8/20.3/145.5/422.4,
46.0/17.0/213.9/146.4, and22.0/25.9/323.3/346.2.
Pooled tuning gives mixed results; joint moves improve several Rhat values,
but the faintest ESS remains below the original matched-budget result.
These two-seed development runs do not establish general superiority.

The joint move is often inefficient: faint-observation per-chain acceptance
is about0.1–3.0%. For obs1 one seed2026100101 chain accepts0/1966 joint
attempts. This supports the limitation of broad Gaussian target moves plus
an independently redrawn environment; formal invariance alone does not make
the proposal effective. Bright radius/Sérsic and obs1 magnitude/shape remain
slow. Joint-run profile Rhat atg1=.018/.022 is1.0527/1.0482 for obs0,
1.1026/1.1106 for obs1,1.01014/1.00986 for obs2 and1.00333/1.00334 for obs3.
Passing one ratio diagnostic does not override posterior failures.

Preserved comparison:
`galsbi_joint_env_mh_20261001/report/comparison.md` and `.json`, under the
catalogue-prior output root. No selected-population shear estimate or0.3%
claim. Before another scale-up, any next proposal should address the observed
curved morphology and conditional target/environment dependence, with a
small fixed-budget test and the unchanged acceptance gates.

## 2026-10-01 — Conditional environment trapping and a joint-move pilot

Owner requested continuation. Pooled array16781077 is still running; do not
interpret warmup acceptance as its result. During that run, inspected the
overnight chains using scheduled plot16781089 and tail audit16781093. Scripts,
figures (`geometry.png`, `traces.png`) and JSON are preserved in
`galsbi_mixing_audit_20261001`. Used the Matplotlib skill for these static
diagnostic plots. Plotting thins every32nd draw and trace panels show512-step
means; neither reduction is used for inference or convergence diagnostics.

The observed bright morphology distributions are curved, with directions
changing across Sérsic index; a single larger covariance is not guaranteed
to resolve them. Fainter observations have long excursions toward targets
much fainter than their measured magnitudes. Audited the diagnostic region
true r > measured MAG_AUTO+1 (no samples removed or new population cut).
For obs2, full environment refresh acceptance in this region is2/638 and
1/104 across the two seeds, versus roughly28–29% outside. Obs3 has1/41 and
3/418. Thus the earlier high aggregate acceptance conceals conditional
environment trapping. Tail occupancy ranges0.1–9.3% across obs2 chains and
0.16–11.0% across obs3 chains; these are unconverged occupancies, not inferred
population fractions. Absolute weighted neighbour-flux diagnostics remove
the target-flux denominator algebra: log(expm1(crowding))-0.4log(10)m_r.
Residual magnitude correlations remain substantial (roughly0.3–0.65), so
the apparent relationship is not solely the definition of flux ratios.

Added opt-in `--joint-environment` to the existing pilot, requiring the joint
morphology setup. A fixed20% of morphology attempts use a4-times broader
symmetric target step together with a full independent environment-prior
draw. They count within the existing block budget, use one likelihood
evaluation, and do not tune the scalar scale used by local moves. The
environment proposal cancels its own prior; acceptance keeps the full
target-coordinate prior and likelihood ratio. Whole scenes are retained on
rejection. No analytic neighbour-density approximation, truth cut, or
state-dependent decision to activate the move is introduced.

Save joint-move indicators and separate local/full/joint acceptance. Reporter
compatibility now includes the joint-move protocol. The existing launcher
accepts an optional joint-environment flag with defaults unchanged. Added a
joint target/Poisson-environment closure test against numerical quadrature,
including target/count cross moments and off-support rejection. Ruff and
shell syntax pass; full suite16781100 runs on compute3. Planned second pair
uses the same65k/32k protocol and seeds with pooled tuning plus this move,
after the first pair/report and passing tests, within two concurrent GPUs.
No acceptance or population-sized run follows automatically.

Suite16781100 completed:367 passed,1 skipped in53.58s. Frozen joint-pilot
source manifest SHA256 is
`bdb370a1d9cfcdab6b00816dca40461059495f16d23fa41dd5e598c1079c19d8`.
Queued array16781117 after the passing suite and pooled report16781079;
the two study GPUs cannot overlap the preceding pair. Afterany report16781121
and read-only fixed-budget comparison16781124 follow. Joint output root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_joint_env_mh_20261001`.
Its preserved `compare_reports.py` writes `report/comparison.md` and JSON
from the three source reports at65k/32k. Missing runs are recorded and no
sampler is promoted. The original and pooled pilots remain untouched;
all variant sources and random seeds are frozen separately.

## 2026-10-01 — Audit slow directions and test staged pooled morphology tuning

Owner requested a deeper sampler investigation and improvement. Overnight
receipts show thousands of full environment proposals, accepted roughly
27–75% for seed2026100101; simple failure to replace neighbours is therefore
not an adequate explanation for the remaining slow exploration. This does
not rule out rare unvisited environmental configurations. Started a scheduled
read-only audit of morphology proposal covariance relative to sampled
posterior covariance, and within-chain target/environment correlations.
Audit script and output are preserved under `galsbi_mixing_audit_20261001`.

Prepared opt-in `--pooled-morphology` in the existing local sampler and
`pooled_block_cholesky` in `sbsi/mcmc_proposals.py`. Pool retained warmup
states across the four chains of each observation only, using at most8192
steps, update every1024 steps, and freeze covariance halfway through warmup.
Continue scalar step-size adaptation through the remaining warmup, then
freeze all proposal parameters. The physical block remains magnitude,
log circularized radius and Sérsic logit, with5% diagonal shrinkage. No
likelihood, prior, environment law, block probability or acceptance rule
changes; no shared covariance between observations. This tests the suspected
short-window/per-chain tuning limitation, not a new population model.

Added an independent covariance calculation checking between-chain variation
and observation separation, plus a correlated-Gaussian MH closure test using
displaced warmup chains. Ruff passed. First CPU audit16781055 and suite16781064
were stuck before Python in `autofs_wait` on cip-cl-compute1; cancelled only
these two jobs and resubmitted on compute3 as16781065/16781066.

Predeclared comparison: if tests pass, run65536 steps/32768 warmup on all four
reference observations with seeds2026100101/2026100102 and original starting
scenes, matching the existing overnight65k prefixes. At most two GPUs.
Compare posterior and likelihood-ratio diagnostics for both seeds, not a
favorable shear point estimate. No automatic promotion or population run.

Audit16781065 completed in8s. It compares actual frozen proposal covariance
(including scalar multipliers) with each chain's observed post-warmup
morphology covariance, using eigenvalues after posterior whitening. For
obs1 the smallest ratios are0.00220–0.01777, with several0.002–0.007:
proposal standard deviations can be only5–8% of the sampled posterior width
along a slow combination. For obs0 the smallest ratio reaches0.00247.
This supports inefficient local proposal geometry, but sampled posterior
covariance is itself not a converged reference. Bright radius/Sérsic
correlations reach−0.81; faint magnitude/crowding reaches0.97 (partly the
algebra of target-relative flux). No claim that high environment acceptance
rules out rare tails or that covariance tuning alone solves the problem.

Full suite16781066:366 passed,1 skipped in77.16s. Ruff and launcher shell
syntax pass. Reused the overnight launcher with optional step count and
pooled flag, preserving defaults. Frozen143-file checkout manifest SHA256
`6f65101d430c897e5323dd28d0b113a62db597ca58bbdf17b60da416cf0f9a4e`.
GPU array16781077 runs the declared pair on two cip A40 GPUs; dependent
afterany report16781079 is assigned compute3, avoiding the observed mount
problem. Output root is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_pooled_mh_20261001`.
Separate seeds share covariance only among warmup chains for the same
observation within a run. Each proposal is symmetric after freezing, so no
new Hastings factor is required. The fixed experiment remains unaccepted
pending its individual/combined diagnostics.

## 2026-10-01 — Overnight comparison completed; convergence still fails

Owner requested status. Scheduler accounting confirms both tasks16778955
completed with exit0 in1h18m31s/1h18m35s; dependent report16778957 completed
in2m57s. No jobs remain in the user queue. Read the preserved report JSON
and Markdown under `galsbi_overnight_20261001/report`; both numerical audit
receipts pass and the runs are compatible, but neither individual run nor
the combined panel passes all exploration checks.

At262144 total steps, combined worst Rhat/minimum bulkESS by observation:
0:1.0301/381.9;1:1.0622/165.9;2:1.0112/552.2;3:1.0170/508.6.
Required Rhat<1.01 and bulkESS>400 remain unchanged. Main slow summaries
are circularized radius for obs0, Sérsic/magnitude for obs1, and
crowding/magnitude for obs2–3. The endpoints improve over the65k prefix,
but the faintest observation worsens at131k before improving: short-run
agreement cannot establish convergence.

Combined likelihood-scale ratio Rhat atg1=.018/.022 passes for obs0,2,3
(approximately1.006/1.003/1.003), but fails for obs1 (1.0125/1.0162).
Conditional relative MCSE is0.14–0.17% for the brighter pair and
0.008–0.014% for the fainter pair. These are errors of individual-observation
numerator ratios, not shear bias or guaranteed error bars for unconverged
chains. Largest individual draw fraction is at most5.51e-6 in the final
combined profiles: no observed single-draw dominance, without ruling out
unvisited tails. Bright obs0'sg1=.022 log mean ratio changes+0.00215 at131k
to−0.000944 at262k; its small profile slope remains numerically uncertain.

No acceptance, model change or population-scale launch. Next bounded
diagnostic should examine the slow magnitude/size/Sérsic and environment
directions on this existing panel before increasing observation count.
The missing selection normalization and population-law limitations remain.

## 2026-10-01 — Frozen overnight sampler comparison launched

Completed the optional local-position pilot16778924 in270.97s after
suite16778902 passed362 tests/1 skip. Bright maximum Rhat1.3893 and
numerator-ratio Rhat1.0537; faint maximum1.0634 and ratio1.0261. High local
position acceptance does not establish convergence. Neither this variant nor
the five-coordinate block consistently improves the simpler three-coordinate
joint morphology proposal. No pilot passes the full convergence gates.

The overnight protocol therefore retains the SED jump, three-coordinate
joint morphology, separate shape block, exact type sum and persistent Poisson
environment refresh. No local-position or five-coordinate proposal is used.
Added saved local profile draws and the read-only
`scripts/report_galsbi_sampler_runs.py`, with likelihood-scale MCSE and
largest-draw contributions, individual and combined-seed diagnostics at fixed
prefixes, and explicit incomplete-run handling. Tests cover missing/corrupt
receipts, disagreeing replicas, an IID lognormal MCSE oracle, and the complete
combined-prefix report path. Latest suite16778930:364 passed,1 skipped
in51.77s; Ruff and launcher shell syntax passed.

GPU array16778955 started both tasks on separate cip A40 allocations, after
the passing suite. Each uses four observations, four chains per observation,
262144 steps/32768 discarded warmup, with seeds2026100101/2026100102.
The seeds share the same dispersed reference-prior starting scenes; they do
not independently regenerate initialization. Both initial likelihood replays
have zero error, and both passed8576 steps with ongoing accepted SED-region
transitions. Adaptation is restricted to warmup. Two concurrent GPUs maximum,
four-hour allocation limit, fixed run length and no favorable-result stopping.

Run root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_overnight_20261001`.
`plan.json` records the fixed protocol. A143-file source snapshot is checked
at startup; SHA256 of its manifest is
`245f0e8696512dfbed8997cbb289490dc7e270424b8c561d1143824f0bb9fa6a`.
Launchers are `jobs/job_galsbi_overnight.sh` and
`jobs/job_galsbi_overnight_report.sh`. CPU report16778957 depends on
**afterany:16778955**, so incomplete runs are reported rather than silently
omitted. It writes `report/report.md` and `report/report.json`, comparing total
step prefixes65536/131072/262144. No automatic promotion or population run.

These are posterior-integration stress tests under the fixed F24 target and
implemented Poisson generator. They do not supply the measured-selection
normalizer, an accepted shear estimate, or evidence of0.3% calibration.
Mixing and likelihood-ratio overlap must be assessed separately. Slurm and
the dependent report provide the overnight continuation; no agent wakeup
automation is available. Next: inspect the completed independent-seed report
before deciding whether to change proposals or expand the observation panel.
See [overnight protocol](CONTINUOUS_SAMPLING.md#overnight-comparison--october-1-2026).

## 2026-10-01 — Overnight continuation: joint morphology proposals

Owner explicitly requested autonomous overnight work. Continue small controlled
sampler tests, preserving the posterior, V3.9 artifacts, measured-cut contract
and convergence gates. No population-scale inference until numerical checks
pass. Use at most two study GPUs concurrently for the planned independent
replicates, within existing limits; check the queue before submission.

Added `sbsi/mcmc_proposals.py` and opt-in `--joint-morphology` in the existing
pilot. Magnitude, log angular radius and Sérsic logit now move jointly with
per-chain Gaussian covariance. During discarded warmup only, recompute
covariance every128 steps from at most1024 retained states, including rejects,
shrunk5% toward the original diagonal proposal. Freeze both covariance and
scalar scales after warmup. Remove Sérsic from the separate structure block;
the corrected SED jump, exact type sum and persistent-neighbour law remain.
This is a symmetric proposal after freezing, with unchanged target density.

Added an independent correlated-Gaussian posterior moment test and a stuck-
chain covariance support test. Ruff passed; full CPU suite16778860 submitted.
Initial joint pilot will use the same two observations, four chains each,
seed2026093061 and16384 steps/4096 warmup, output
`galsbi_joint_mh_20261001_v1`. Longer independent-seed diagnostics will be
predeclared only after this implementation test; no shear claim follows from
one favorable run. Native-density/generator approximation and F24 versus FS2
population limitations remain unchanged.

Full suite16778860 passed355 tests/1 skip in53.08s. Joint-morphology GPU
16778863 completed in289.83s script time. SED likelihood replay stays exactly
unchanged. Bright Sérsic Rhat improves1.5730→1.1415, but radius1.2836 and
shape1.2165/1.2797 remain poor; its shear-ratio Rhat is1.0581. Faint maximum
Rhat1.0625 remains in crowding; ratio Rhat1.0094 passes but overall exploration
does not. Do not select this run from its favorable small shear-ratio means.

Extended the same Gaussian block to optionally include distortion logit and
wrapped phase (`--joint-shape`). Total physical-proposal probability stays0.6,
previously split between morphology and shape. Estimate phase covariance in
the circular chart centered on each chain's last warmup phase; Gaussian
increments are wrapped afterward. With fixed covariance the wrapped proposal
is symmetric. Added phase-origin invariance and correlated wrapped-Gaussian
moment tests. Full suite16778895 submitted. The next16k pilot keeps all other
settings, output `galsbi_joint_shape_20261001_v1`.

Also added explicit reference-observation indices and saved local shear-ratio
draws to the existing driver for the forthcoming independent-seed/prefix
checks. Default observations and scientific settings are unchanged. These
diagnostic options were added after the running v1 source snapshot was taken;
that completed run retains its exact executed source under its output folder.

Full five-coordinate physical pilot16778896 completed in278.61s, after
357 tests/1 skip passed (16778895). It does not consistently improve the
bright result: maximum Rhat1.3724, ratio Rhat1.2389; faint maximum1.0646 and
ratio1.0045. Both fail overall. Keep it optional rather than choosing it from
the favorable faint result. All starting/SED numerical replays pass.

Added a read-only independent-run/prefix reporter, including between-seed
Rhat/ESS and likelihood-scale MCSEs of numerator ratios. Missing/incompatible
runs cannot pass, and no sampler is promoted automatically. Tests explicitly
include two individually well-mixed but mutually disagreeing replicas.
Report test16778898 initially failed on a script-only import; the reporter
now has no model imports. Full suite16778899 then passed360 tests/1 skip;
removed its NumPy scalar-conversion deprecation afterward.

Next small test: optional local neighbour-position moves, retaining the
three-coordinate joint morphology block. `GalSBIScenePrior.move_positions`
proposes independent Gaussian steps for all positions and retains each point
whose proposal leaves the aperture. Each bounded uniform-disk kernel is
prior-reversible, as is their product; outer MH uses only the likelihood
ratio. Counts, marks, target and population law stay unchanged. Added
stationary-disk and analytically tilted-disk posterior tests. The new block
gets0.05 probability, taken from the SED jump (now0.10); original refresh
probability0.15 stays fixed. Position scale0.2arcsec, same16k/4k protocol.
Full suite16778902 submitted; output `galsbi_local_positions_20261001_v1`.

## 2026-10-01 — Add a corrected SED independence jump to the continuous sampler

Owner authorized the proposed red/blue SED move. Added `sed_log_proposal`
and `propose_sed` to `sbsi/galsbi_target_coordinates.py`. At fixed redshift,
propose from an equal mixture of the two known Dirichlet laws and preserve
apparent r magnitude, angular radius, Sérsic n, distortion and phase. Decode
new intrinsic luminosity/photometry and evaluate the complete native prior.
The proposal density includes the simplex-to-ALR Jacobian prod(c), and MH
uses log q(old|z)-log q(new|z). Equal proposal weights do not change target
population weights. Off-support proposals are rejected without redrawing.

The existing driver gains opt-in `--sed-jump`, requiring persistent scenes
and exact type summation. Block probabilities become0.3 photometry/size,
0.3 shape,0.1 local structure,0.15 environment and0.15 SED independence.
Every finite-prior SED proposal verifies all nine physical summaries remain
fixed within1e-10 and log likelihood within1e-6; failures stop the pilot.
Output now includes native log prior, SED-region crossings, conditional red
probabilities and separate full/partial environment acceptance. No model,
prior density, inference default or measurement-selection rule changes.

Added independent SciPy proposal-density/Jacobian checks, an analytically
soluble unequal-mixture posterior with a likelihood tilt, and a photometric
replay/support test. Ruff and shell syntax pass. Full Slurm suite16778800
submitted with artifact roots configured. Planned pilot keeps the same two
mock measurements, four chains each, seed2026093061,16384 steps/4096 warmup,
per-chain frozen scales and existing neighbour law. It will depend on the
suite passing. Source/protocol receipts and results are preserved under
`galsbi_sed_jump_20261001_v1`. Full suite16778800 completed:353 passed,1
skipped in52.28s. Dependent GPU16778801 started and exact initial likelihood
replay passed; all eight chains have made SED-region crossings during warmup.
GPU16778801 completed in294.96s script time (5m24s allocation). Across19,808
finite-prior SED proposals, maximum summary change is4.2633e-14 and maximum
log-likelihood change is exactly zero. CPU audit16778804 completed in2s.

The SED-region trapping improves markedly, but **neither observation passes**.
After warmup the four bright chains make24/28/28/86 region crossings, with
mean conditional red probabilities0.9912/0.9793/0.9804/0.9561; previously two
chains stayed red and two blue. Full environment refresh acceptance is now
75.3/86.1/77.7/75.3%, versus a zero-acceptance bright chain previously.
Bright SED ALR rank-Rhat is1.0029–1.0106, bulkESS525–1821. Its largest remaining
Rhat is1.5730 for Sérsic index (bulkESS6.94), and circularized radius is1.3641
(ESS9.13). Theg1=.022 numerator-ratio Rhat improves from1.4675 to1.0489,
but its per-chain log ratios still range−0.01379 to+0.00780 and are not accepted.

Faint maximum Rhat is1.0629 and minimum bulkESS63.07; magnitude/crowding
remain slow. Itsg1=.022 ratio Rhat is1.0106 withESS445.94. Conditional red
probability differs0.0004–0.0744 across faint chains, so crossings alone do
not establish correct component mass. Shape/SED diagnostics improve, but
joint posterior convergence is unresolved. This remains a two-observation
diagnostic, without a selection integral, shear estimate or0.3% claim.

Read-only correlation audit16778808 examines the remaining structural
coupling, using saved traces; it completed in1s with no extra inference run.
Within bright chains, log angular radius and Sérsic logit correlate−0.43 to
−0.80, but current proposals put them in different blocks. Faint magnitude
and close-crowding correlations range0.55–0.98; crowding uses target-relative
flux, so this is partly structural and is not by itself proof of trapping.
The next minimal test is a joint magnitude/size/Sérsic proposal with covariance
adapted during warmup and frozen afterward, retaining the corrected SED jump.
Do not infer that this will resolve environment exploration or authorize a
population-sized run from the favorable SED result. All jobs are complete.
Results and next step are recorded in
[SED-jump pilot](CONTINUOUS_SAMPLING.md#sed-independence-jump--october-1-2026).

## 2026-10-01 — Retain neighbour environments during local MH target moves

Owner authorized the proposed next sampler step. Added a prior-reversible
partial marked-Poisson refresh and complete ragged-scene selection/retention
to `sbsi/galsbi_scene_prior.py`. Extended the existing local pilot with
`--environment-update persistent`: one explicit environment per chain,
unchanged during target moves, with separate refresh moves. A refresh retains
each neighbour with probability0.9 and adds an independent0.1-intensity
population;10% of refresh proposals instead redraw the complete environment.
The fraction choice is independent of state. Random order within each patch
preserves the ordered-iid convention. Acceptance uses the likelihood ratio
for a refresh, and the target-prior times likelihood ratio for local target
moves. Rejected states retain every target and neighbour field.

Bounded protocol: same two independent model-generated bright/faint mock
observations and four chains each;4096 steps including1024 warmup, exact
red/blue target sum, fixed4096 density batching, seed2026093061. Initial scenes
are the best measurement-likelihood candidate in each previous prior stream;
their original environments are retained, and no generating truth is used.
This changes initialization, environment update and run length together; it
is an algorithm refinement, not an isolated causal comparison. Initial full
scene replay must agree with saved reference log kernels within1e-6.
All native coordinates except wrapped phase, neighbour counts and all three
crowding summaries are now monitored in addition to the existing diagnostics.
Rank-Rhat<1.01 and bulkESS>400 remain necessary exploration gates.

Added tests of ragged-state retention, refresh prior stationarity, count
cross-covariance, and an analytically tilted marked-Poisson posterior whose
count and mark laws are known. Also fixed empty-environment field creation:
neighbour fields come from the galaxy generator, not a potentially different
supplied target schema. Ruff and launcher shell syntax pass. Full CPU suite
16778740 passed:348 tests passed,3 skipped in55.48s. Dependent GPU16778741
failed before scientific work because the resource detector's output directory
did not exist; the launcher now creates it. Identical pilot resubmitted as
16778742. Source snapshots,
protocol hashes, final native scenes and resource receipts are preserved.

No production default, model identity, selection integral or shear estimate
changes. The native analytic target prior and numerical environment generator
retain their existing approximation difference.

First pilot16778742 completed in89.68s script time (2m03s allocation).
Starting replay is exactly equal to the reference. Both observations fail:
bright maximum Rhat2.860/minimum bulkESS4.775; faint1.097/34.88.
Bright magnitude/size acceptance ranges0.99–27.92% across chains, despite
using a shared step scale, and the local numerator ratios disagree strongly.
Faint ratios atg1=.022 span0.00414–0.00670 in log ratio; this is neither a
selected likelihood nor a shear estimate. The change removes the independent
likelihood-estimator noise but does not establish posterior exploration.

Added the conventional per-chain warmup-scale option to the existing driver
and launched one bounded follow-up16778765, output
`galsbi_persistent_mh_20261001_v2`:16384 steps/4096 warmup, same eight chains,
observations, seeds, posterior, starting scenes and environment-refresh law.
Scale adaptation freezes after warmup. This tests whether proposal scales
better match the different chains; it also increases length, so is not a
single-factor causal comparison. Ruff passes after this script-only option.

Follow-up16778765 completed in272.29s script time (4m45s allocation), with
exact starting replay. **Neither observation passes**. Bright maximum
Rhat2.2307/minimum bulkESS5.896; faint1.0921/39.20. Bright magnitude/size
acceptance is now18.8–32.1%, but shape acceptance is5.1–33.3% and chains still
disagree on the local shear numerator ratios. Faint ratio bulkESS493/511 is
better, but ratio Rhat1.0137/1.0129 and other diagnostics still fail.

Read-only CPU audit16778767 completed in2s. Its script and
`chain_mode_summary.json` are preserved in the v2 output. Bright chains0/3
have mean conditional red probability>0.99999998; chains1/2 have probabilities
near zero. Summing type analytically did not remove the separated SED regions.
Chain1 samples a much broader intrinsic-magnitude range (10–90% quantiles
19.35–22.39) than chain0 (19.71–19.75). It accepts0/181 full environment
refreshes, despite89.3% acceptance of partial refreshes. The other bright
chains accept67–77% of full refreshes. These are nonconverged chain summaries,
not estimated posterior population weights; do not select the chain with the
largest likelihood and discard the rest.

Conclusion: explicit scenes remove independent likelihood-estimator noise,
but local MH still fails to explore the joint target/environment posterior.
No accepted sampler, selected shear estimate or0.3% claim. No jobs remain.
The next smallest targeted change is an independence proposal for SED from
the known red/blue Dirichlet mixture, at fixed z and likelihood-visible target
properties, with its exact proposal-density/Jacobian correction. This can
jump SED regions without fitting a new prior. Environmental trapping remains
a separate check, potentially requiring local moves of existing neighbours.
See [persistent-scene MH](CONTINUOUS_SAMPLING.md#persistent-scene-mh-pilot--october-1-2026).

## 2026-09-30 — Implement fixed-summary likelihood and test standard MH variants

Owner requested rethinking the sampler around the likelihood's fixed summaries
and then explicitly asked to continue. Added `sbsi/galsbi_summary.py`,
`sbsi/summary_sampling.py`, `scripts/check_galsbi_summary_sampler.py` and
`jobs/job_galsbi_summary_sampler.sh`. For fixed observed radius/flux, the eight
F8/C8 features plus the summed disk response reproduce the full retained
likelihood. The response coefficient is observation-indexed, and stale
auxiliaries are rejected. All neighbours enter crowding; the existing response
uses nearest20 within strict10arcsec, without truth cuts. No model retraining,
emulator simplification, fitted summary density or production default change.

Established the previously unresolved native radius mapping by inspecting the
installed UFig photon-rendering matrix: semi-major=R_native*sqrt(1+a),
circularized=R_native*(1-a^2)^(1/4), with native distortion converted to epsilon.
Independent SVD replay of the actual rendering matrix agrees to2.22e-16.
This explicitly adopts the zero-flexion rendered ellipse, not an empirical
calibration or direct identification of native and circularized radii.

The induced nine-summary prior is available by fresh scene generation and
compression. It is not generally a smooth Lebesgue density: the tree response
can have point masses, as can the no-neighbour event. Ordinary independence
MH with fresh prior proposals avoids evaluating that density because proposal
and prior cancel. Rejections remain in each chain; no generating truth is
inserted and no finite catalogue is resampled indefinitely.

Completed summary pilot16776331, artifact `galsbi_summary_mh_20260930_v2` under
the external catalogue-prior root, script time28.50s.8192 independent model
inputs yielded3188 usable and1961 measured-cut-passing mocks. Four measured
magnitude strata were represented, and each observation used four independent
2048-draw fresh-prior streams. Full-pair versus compressed likelihood,
reversed order and padded partial batches agree exactly. Initial job16776323
stopped before MH because C8 in8192 versus4096-row batches differed by3.98e-6
in log probability; rerun matched classifier batching, retaining the1e-6 gate,
seeds, observations and proposal streams. No threshold was loosened.

All four independence-MH probes **failed exploration**. Prior importanceESS
per2048 draws was1.00–3.08 at measuredmag19.745,1.05–3.76 at21.255,
2.90–11.32 at23.971, and20.45–24.86 at25.118. Bright chains were nearly frozen;
faint acceptance was0.68–2.05%. These are not recovered shear estimates.

Continued with standard pseudo-marginal MH, adding
`sbsi/galsbi_target_coordinates.py`, `sbsi/pseudo_marginal.py`,
`scripts/check_galsbi_local_sampler.py`, and `jobs/job_galsbi_local_sampler.sh`.
Refactored the existing neighbour generator to expose `sample_given_targets`,
preserving its original full-scene RNG sequence. Ten transformed target
coordinates retain the evaluable native prior and all Jacobians; fresh
neighbours supply a nonnegative likelihood average. Rejection retains the
current average and compressed bank. Bank moments use likelihood weights,
and shear numerator ratios use ratios of bank sums, not equal scene weights.
Measurement-selection normalization remains separate and uncomputed here.

First local pilot16776398 completed in51.50s script time: same bright/faint
measurements, four chains each,1024 steps/512 warmup, four environments per
proposal. Local moves improve acceptance but convergence still fails; two
bright chains stick. Independent fixed-target audit16776440 used1024 fresh
environments per chain: those two chains' stored log-likelihood estimates
exceed the audit means by7.59 and8.63; four-environment log-estimate SD is16.86
and17.86. At64 environments SD falls to2.31 and1.76. The1024-draw audit itself
has only2.82 and4.47 effective environments at those targets, so its mean is
not treated as exact. This is demonstrated estimator noise, not evidence of
recovered posterior mass.

An exact additional simplification sums red/blue native prior intensities:
type is not a direct F8/C8/disk input at fixed physical properties, so it need
not be a discrete MH variable. Added `log_mixture_prior` and a test against the
explicit native sum. The bounded refined pilot16776459 uses64 environments,
the exact type sum, and separate proposal-scale adaptation per observation
during warmup only. It keeps the same two measurements and1024 steps. Its
protocol and source snapshot are in `galsbi_local_mh_20260930_v2`. It completed
in330.48s script time, but both observations still fail. Bright rank-Rhat is
1.79–3.11 and bulkESS4.56–55.91 over monitored functions; faint is1.07–1.50
and7.81–51.42. Faint photometry/size and shape acceptance reaches23–44%, but
the chains still disagree. At two bright final targets the independent
1024-environment audit has only5.73/12.17 effective environments, and retained
log-estimates exceed its mean estimates by8.36/3.89. Faint M64 log-estimate SD
is0.13–0.22, showing that target mixing also needs work. This is an algorithm
refinement, not an isolated causal comparison of environment count alone.
Retain exact summary compression, reject both short sampler pilots as inference
replacements. No additional or large jobs remain. Next is proposal design
that can retain useful target/environment configurations; do not interpret
acceptance or a favorable shear proposal as convergence.

Validation: analytic Gaussian posterior/evidence-ratio and point-mass tests,
independent full ten-dimensional Jacobian determinant, noisy latent-Gaussian
pseudo-marginal closure, rejected auxiliary retention, neighbour law with fixed
targets, and likelihood replay. Full Slurm suite16776297:339 passed/1 skipped;
16776388:343/1;16776457 after type marginalization:344/1 in43.53s. Ruff and shell
syntax checks pass. Each main GPU pilot uses one GPU, one CPU,16GB and a20min
cap; the separate noise audit uses8GB/5minutes. No study GPU jobs overlap.
Resource receipts are captured in the main pilot allocations.

Limits: native target density and numerical environment generator retain their
audited grid approximation; fixed F24 population/uniform neighbours differ
from FS2. No representative-population closure, measured-selection integral,
accepted shear estimate or0.3% claim. Design, equations, protocols and results:
[continuous sampling](CONTINUOUS_SAMPLING.md#fixed-summary-prototype--september-30-2026).

## 2026-09-30 — Design continuous-scene posterior sampling before implementation

Owner requested thinking through the next sampling algorithm. Added
[the proposed design](CONTINUOUS_SAMPLING.md#status-and-recommendation--september-30-2026),
after inspecting the native prior, V3.9 disk density/response code, selection
contract and earlier profile/batch failures. This is documentation only;
no sampler, model, inference default or job changed.

Recommend tempered SMC with one blocked Metropolis kernel, posterior scene
reuse for nearby shear likelihood ratios, and an independent CRN forward
selection integral. Derived the partial Poisson refresh proposal: independently
retain1-rho of neighbours, add a fresh rho-intensity population by envelope
thinning, randomize list order. Its prior reversibility cancels prior/proposal
terms, leaving the tempered likelihood ratio; no separately estimated surface
density or custom birth/death factor is needed. Local target/neighbour moves
still require native/transformed-coordinate Jacobians and support checks.

Recorded the full q_U*f_disk posterior target, the population Z(g) selection
factor, reuse of observed-radius/flux response across shear nodes, the remaining
native-to-SBSI mapping, common-support and batching checks, and why simple
gradient methods or independently noisy Newton stencils are poor first choices.
Validation progresses from analytic/replay tests to two observations,32 stress
probes, then representative continuous-model mocks. Proposal budgets are
diagnostic; neither high ESS nor favorable shear is an acceptance guarantee.
The prospective global numerical target is1e-5 absolute shear versus6e-5
corresponding to0.3% atg1=0.02, not a claimed achieved precision.

Validation: checked formulas against the local likelihood contracts and primary
SMC/Poisson/MCMC-diagnostic references; reviewed the Poisson exchangeability
argument and posterior-ratio/selection identities. Documentation links and
whitespace checked. No numerical experiment or speed claim; next step is a
small arbitrary-scene likelihood adapter with pointwise numerical replay.
## 2026-09-30 — Add uniform GalSBI neighbour environments

Owner authorized adding neighbours to the continuous prior, retaining the
previously requested uniform, unclustered approximation. Added
`sbsi/galsbi_scene_prior.py`: independent target-centred 10-arcsec patches,
native galaxy marks, uncapped Poisson neighbour counts, Cartesian sky offsets,
and explicit scene density. Targets and neighbours use the same fixed F24
population/support. Neighbour sampling thins the pre-photometric Poisson
population; no Monte Carlo post-cut normalization enters the sampler.
`GalSBIContinuousPrior.sample` now returns retained `proposal_index` values
to preserve patch membership through this thinning.

The ordered-list density includes the count factorial and per-steradian to
per-arcsec-squared measure conversion. `log_kernel` omits only constants at
fixed population/aperture; `log_prob` accepts an explicit uncertain surface
density. Derived physical Gaussian crowding matches the F8/C8 definition,
using all nonself neighbours within10 arcsec. Native pair marks and offsets
are retained; emulator radius mapping and predictions are not implemented.

Added `scripts/check_galsbi_neighbours.py`, CPU launcher
`jobs/job_galsbi_neighbours.sh`, and `tests/test_galsbi_scene_prior.py` covering
Poisson measure/count changes, rejection thinning, zero-neighbour scenes,
support and existing V3.9 feature replay. The diagnostic compares4,096
continuous patches with disjoint patches made by uniformly placing the
previous independent GalSBI catalogue marks/count, without row reuse. This
reference deliberately does not use its original HEALPix sky positions.

CPU job16775918 passed the full suite:331 passed,3 skipped in59.83s.
Comparison16775919 passed all eight declared gates. After randomizing the
reference list order (GalSBI stores types in blocks), final replay16775946
completed exit0 in48s and again passed all gates; counts, marks, spatial
checks and gate decisions are unchanged. No seeds/thresholds were tuned.
Artifacts: `galsbi_neighbours_20260930_v1` under the external catalogue-prior
root. Resources recorded in the allocation; one CPU/16GB requested for the
comparison despite greater host-wide availability. No GPU or inference job.
Scientific seeds are2026093021 (continuous scenes),2026093022 (reference
positions). Ruff, shell syntax and AST checks pass; final source snapshots
and comparison hashes verified, comparison figure inspected.

The4,096 scenes contain87,622 neighbours (mean21.392 versus21.540 predicted,
-1.62 standard deviations including normalization uncertainty), with count
variance/mean0.99429. The424 disjoint reference patches contain9,134 galaxies;
2,535 square-cell-corner rows are excluded and every retained row is used once.
Minimum spatial raw/adjusted p is0.03464/0.13854; minimum neighbour-mark raw p
is0.08022. Paired crowding means are compatible at allthree scales, and first64
scenes replay production crowding within4.44e-16. Every scene density is finite.
Detailed evidence and usage:
[neighbour report](GALSBI_PRIOR_FEASIBILITY.md#uniform-neighbour-pilot-results--september-30-2026).

Limitations: unclustered independent patches at fixed population parameters
and declared intrinsic support; the reference reassigns uniform positions.
Absolute scene normalization still carries the isolated Monte Carlo error,
while fixed-population density ratios and neighbour sampling avoid that
estimate. No emulator input mapping, posterior sampling, measured-selection
normalization or shear validation. Next is coordinate mapping and a small
inference closure test, not a0.3% claim.

## 2026-09-30 — Implement an isolated GalSBI continuous-prior pilot

Owner authorized an isolated-galaxy density implementation and comparison
against a GalSBI-generated catalogue, without neighbours. Added
`sbsi/galsbi_prior.py` with explicit F24 native-coordinate density factors,
support checks, continuous luminosity/redshift sampling, and an explicitly
uncertain population normalization. Added the diagnostic
`scripts/check_galsbi_continuous_prior.py`, its CPU Slurm launcher
`jobs/job_galsbi_prior_check.sh`, and measure/support tests in
`tests/test_galsbi_prior.py`. No V3.9 model or inference default changes.

The reference uses installed GalSBI0.3.0, F24 posterior row0 (accepted row,
source id8), HSC r2/i2 filters, zero foreground extinction, float64 catalogue
output and intrinsic reference-i limits14–28. It includes no measured cuts,
images, neighbours or shear inference. The analytic adapter is in the native
latent variables; it is not yet a marginal density in SBSI's five features.
The source package differs from the latest upstream checkout, so the actual
installed source and data hashes are recorded by the comparison.

Artifacts are under `galsbi_continuous_prior_20260930_v1` in the external
catalogue-prior root. Preflight CPU job16775773 generated11,669 reference
galaxies successfully and recorded resources. Initial comparison16775826
failed before generation because SBSI was not installed into py31 for script
imports; installing this checkout editable follows ENVIRONMENT.md and avoids
adding PYTHONPATH workarounds. Job16775840 generated the reference, then
exposed the installed photometry wrapper's keyword-only interface; repaired
the callback. Job16775846 completed the calculations but exposed NumPy-bool
JSON serialization; explicitly converted gate values to Python bool. No
scientific settings, seeds or acceptance thresholds were changed after seeing
results. Final comparison16775868 completed exit0 in40s on one inter CPU,
about0.80GB reported peak RSS. No GPU used.

All nine pilot gates passed: 11,669 GalSBI galaxies versus70,507 independently
retained continuous draws from131,072 proposals; magnitude replay max1.42e-14
mag; every density finite; smallest raw two-sample p0.07295 across15 marginal,
five coarse joint and one type-fraction checks. Morphology conditional-CDF
minimum raw p0.04835 (shape), Holm-corrected0.19338. Expected reference count
11,652.83 versus11,669 actual (0.144 standard deviations). Population
normalization is2.91704e9 sr^-1 with0.256% Monte Carlo relative SE, unrelated
to shear accuracy. Luminosity grid change1.81e-9 and magnitude inverse-CDF
error2.35e-8. Results, support and usage are recorded in
[the pilot report](GALSBI_PRIOR_FEASIBILITY.md#isolated-galaxy-pilot-results--september-30-2026).

Validation: full-suite job16775847 gives328 passed,3 skipped in43.31s;
`ruff check sbsi/galsbi_prior.py scripts/check_galsbi_continuous_prior.py
tests/test_galsbi_prior.py` and `bash -n jobs/job_galsbi_prior_check.sh` pass.
Final source snapshots and comparison hashes verified, both plots inspected.
Generation metadata must match comparison arguments; any failed diagnostic
writes evidence and exits unsuccessfully. Tiny samples may omit a galaxy
type; the sampler handles this and rejects nonpositive proposal counts.

Limitations: fixed F24 posterior row and native latent density only; shared
nearest-grid photometry; no five-feature marginal, SBSI radius/shape mapping,
neighbours, measured selection, MCMC or shear calibration. The matched
M_B<=-4 population boundary carries GalSBI's own incompleteness warning;
support completeness and rare bright tails remain unvalidated. The next
inference step needs explicit coordinate mapping and sampler/selection
closure. No sub-percent equivalence or0.3% calibration claim is established.

## 2026-09-30 — Audit GalSBI as a continuous population prior

Owner asked whether GalSBI can evaluate the density of arbitrary true
galaxies, allowing uniform neighbour positions and excluding its clustering
extension. Added [the source-level feasibility audit](GALSBI_PRIOR_FEASIBILITY.md#scope-and-verdict--september-30-2026).
Inspected the public implementation at commit
`df802e1db2f2a4247955720aea227d5b65cdf975` and the non-clustering F24 model,
alongside the primary paper and official documentation. The checkout is
`/tmp/sbsi-galsbi-audit-20260930`; no upstream modules were executed or
dependencies/data installed. No production code, model, prior or job changed.

The package returns ABC population-parameter samples and generates galaxies;
no full galaxy `log_prob` interface was found. Explicit Schechter, Dirichlet,
lognormal, scaled-beta and truncated-beta-prime factors support constructing
a native joint density. A five-dimensional SBSI truth-density evaluator
additionally needs marginalization over redshift, type and SED. The proposed
alternative keeps these nuisance variables in the sampler and uses the
existing generative factors rather than first fitting a Gaussian mixture.

Recorded the uniform marked-Poisson neighbour factorization, shared population
parameters, intensity versus probability normalization, Sérsic truncation,
shape/size/filter conventions, and F24 default intrinsic i-band limits.
Source inspection also finds grid-discretized redshift/magnitude generation
and nearest-grid spectral lookup: continuous-model closure cannot silently
assume the released sampler is exact. GalSBI is a different parent from FS2;
success on GalSBI mocks would not itself establish recovery on ConstGold.

Validation: source API searches and AST inspection of nine upstream Python
modules passed; documentation whitespace and relative links were checked.
This is feasibility analysis only: no complete density normalization,
posterior bundle inspection, generated-catalogue comparison, MCMC benchmark
or calibration result. Next step is a bounded native-density adapter and
generator-consistency audit before an inference pilot.

## 2026-09-30 — Trace the expanded-prior large Newton proposal to cancelling noisy curvature

Owner requested a deep investigation of the full matched-mock proposals
0.18813134 and0.14730972. Added the read-only saved-moment reduction
`scripts/audit_v39_expanded_step.py`, frozen with outputs under
`inference_v39_expanded_step_audit_20260930_v1` in the external catalogue-prior
root. No model, prior, observation, sampler, or production default changed.
CPU job16773458 completed exit0 in19s on one inter CPU with8GB requested;
no GPU jobs were submitted. The resource receipt is retained in the run root.

The audit verifies50 source hashes, original20k row coverage and exclusion of
the3686 mock padding rows, primary/maximum-rung equality, and all six saved
mock and actual aggregate scores/information. Independent2x2 solves and
influence covariance reproduce all six positive mock proposals and their
reported standard errors. Independent differences of the saved selection-mass
nodes reproduce the normalization gradient/Hessian. Exact row-attribution
identities reproduce the512→8k,8k→16k and512→16k changes. Ruff and AST parsing
pass; numerical validation is against the frozen scientific arrays rather
than new tests duplicating the reduction.

At16k the mock weak-direction information is the residual of+5.200966m and
−5.088162m: only1.096% of the sum of absolute contributions remains. This is
scientific/numerical cancellation, not evidence of a floating-point failure
in the2x2 solve (its condition number is about15.3). The134 measured-mag<20
rows account for82.58% of the512→16k g1 movement, while774 mag20–22 rows
supply80.98% of the final g1 score. Thus bright integration errors change the
curvature used to convert other observations' scores into a shear step.
The16k conditional g1 influence error is0.90423; the0.14356 step length is
143.6 times the0.001 derivative stencil. No valid maximum was located.

Replayed the earlier three exhaustive worker files (complete28,012,032-atom
coverage) and both normalizers from raw arrays; receipt
`exact_probe_replay.json` preserves source hashes. For influential mock4460,
the valid new bright atoms supply86.246% of exact centre mass. Sampled16k
g1 information is−233,919.29, versus exhaustive+165,073.89 under the same
expanded prior. The known399k entry error is material to the whole-sample
solve. This demonstrates numerical misrepresentation of the expanded prior,
not failure of bright support alone. Dropping selection normalization only
as an algebraic diagnostic still gives g1=0.12554; it is not the main cause
of the large proposal and is never removed for inference.

Documented the full attribution and limitations under
[expanded-step forensic audit](V39_POPULATION_AUDIT.md#expanded-step-forensic-audit-on-september30),
linked from the original comparison. All numbers remain conditional on the
unconverged old K1024 sampler. The matched inputs are already image selected;
finite-prior resolution, double selection, normalization uncertainty, and
the global likelihood maximum remain unresolved. Next informative work is
an accurately evaluated small-cohort profile, including magnitude20–22
observations, before another large run or further prior expansion.

## 2026-09-29 — Exact group profile completes; A/B peak disagreement persists

At the owner's status check, all four workers16746033 completed in22m35–39s
and reducer16746034 completed in41s. No study GPU jobs remain. All four
workers replayed the source exact centre with maximum error7.8613e-11 at
the unchanged1e-7 tolerance. Added completion_audit.json to
inference_v39_independent_exact_group_20260929_v1: verified ten pinned source
hashes, all four worker hashes, exact49-node coverage and cross-worker centre
agreement, and independently reproduced the replacement/aggregation within
1e-10. All128 fresh-parent rows remain: four exact,124 sampled; normalizers
are unchanged.

The coarse peaks remain A=(-0.00625952,0.01735242) and
B=(0.02374048,-0.00264758). Each prior prefers its own peak over the other's
by3.2310 and5.5163 log-likelihood units, respectively. Seed/draw profile
variation now passes0.05 under both priors (A0.04542; B0.02861), but this
is **not** an oracle accuracy guarantee. Exact replacement changed the B
curve by0.25055 at(-0.00625952,-0.03264758), although the previous saved
variants there differed from their primary by at most0.02088. Common error
can survive seed/draw agreement. Changes at the competing peaks were small:
A+0.000473/-0.000229 and B-0.000040/-0.001878 at own/other peak.

This particular numerical repair does not explain away the prior-realization
sensitivity. Only one group is exhaustive, the grid remains coarse, and the
A-generated bright control retains strong intrinsic overlap across priors.
No accepted shear estimate or0.3% result. No new GPU jobs submitted during
this status check. See
[exact group profile result](V39_POPULATION_AUDIT.md#completed-exact-group-profile).

## 2026-09-29 — Independent profiles complete; target the remaining numerical failure

All v4 jobs completed: preflights16743015[0–1] in13m39–40s, four profile
workers16743017 in3h37–3h52, reducer16743018 in66s. Both exhaustive preflight
errors passed0.001 (maxima0.000390 and0.000361). All87 groups passed full
candidate-density and response replay. No study GPU jobs remained at this
check. Added scripts/audit_v39_independent_profiles.py; CPU audit16746010
verified every group/profile hash and reproduced all four aggregate curves
within1e-10, saved per-observation contributions and coarse-grid figures.
Ruff and syntax checks passed. These are small saved-array reductions, not
a new likelihood calculation or accepted estimator.

The A-generated cohort has the same coarse peak under both priors:
(g1,g2)=(0.02374048,-0.00264758). Near-mode integration differences are
0.00933 (A) and0.00745 (B), and normalizer differences0.01191/0.01790,
all within the fixed0.05 limits. The fresh-parent cohort instead peaks at
(-0.00625952,0.01735242) under A and(0.02374048,-0.00264758) under B.
A fails integration convergence (0.12726); B passes (0.02760). Both
fresh-parent normalizers pass. These are grid points spaced0.01 apart,
not refined shear estimates or0.3% evidence.

Fresh-parent row47 (measured magnitude20.76839) dominates the draw instability:
its maximum8k-to16k change is0.18904. For attribution only, removing its
original numerical group leaves maximum summed integration drift0.04542;
no observations are removed from inference. Other numerically stable rows
also contribute strongly to the A/B likelihood difference. Submitted an
exhaustive full49-node calculation for group66, rows44–47, across both priors:
four one-V100 workers16746033, reducer16746034, frozen in
inference_v39_independent_exact_group_20260929_v1. Each worker replays the
old exact centre, then handles12 other nodes. Only these four rows will be
replaced in the128-row sums; normalization and all other rows stay fixed.
Estimated20–30 minutes based on the completed three-node preflights.
All four exact workers passed startup and reached10.5m of20.23m atoms in
their initial centre scan without errors. Status-only audit16746082 is
scheduled for about two hours later; it is not an agent wakeup or notification.

CPU context-overlap audit16746043 found exact five-intrinsic-feature matches
in B for19/128 A-generated inputs, and no exact eight-feature F8-context
matches. Disjoint scenes share an empirical source catalogue; this is not a
fully independent intrinsic-galaxy holdout. All three measured-bright inputs
have intrinsic matches in B (3,1,2 for rows116,208,245); the common coarse
peak therefore does not cleanly test removing their intrinsic support.
Follow-up ID audit16746067 and saved centre candidates quantify the effect:
the matched B atoms carry68.63%,81.32%,99.973% of the three objects'
posterior mass at the fixed initial centre. Their original generating A atoms
carry66.71%,99.23%,38.38%, respectively. These are centre-only posterior
fractions, not prior weights or inferred shear probabilities; source arrays
and exact centre denominators provide the calculation in
audit/bright_intrinsic_support.json. No extra GPU inference was used.
The complete response neighbour
graphs were not compared. See
[completed independent profiles](V39_POPULATION_AUDIT.md#completed-independent-profiles).

## 2026-09-28 — Padded replay passes; launch independent-profile preflights

V4 preparation16742975 passed all11 targeted tests (0.20s), source receipts,
scene disjointness, prior masses and unchanged cohort checks. Bounded GPU
replay16742976 passed all ten cases exactly: ordered/permuted first, middle,
padded-final blocks, mixed subsets and duplicated draws, for both cohorts.
Maximum coefficient and density differences were0 throughout. The padding
repair therefore removes the reproduced partial-batch discrepancy in these
probes without relaxing any tolerance; the original six candidate differences
still require full-centre replay and are not assumed resolved.

Submitted two one-V100 exhaustive preflights16743015[0–1]. Both must pass
coefficient, candidate-density and full-prior log-profile checks before
CPU gate16743016 releases four persistent one-V100 workers16743017[0–3].
Reducer16743018 retains every numerical variant; no accepted estimate or
population conclusion yet. The unchanged design uses128 scientific rows
per cohort, A/B disjoint scene priors and separate selection normalization.
Runtime remains dominated by20.23m-atom centre scans for87 four-row groups.
The4-GPU ceiling is enforced by stage dependencies and array throttles.
Both preflights reached their full-prior centre scans and passed1m atoms
without error at the startup check. Scheduled status-only audit16743028 for
about two hours later; this records scheduler/progress state and is not an
agent wakeup or user notification. Saved repair_identity_audit.json verifies
all scientific settings,87 group input hashes, priors and normalizer input
hashes are unchanged from v2.
See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Isolate partial density-batch sensitivity before independent profiles

V3 preparation16742917 passed11 tests in0.24s and all input checks. The
bounded GPU replay16742918 reproduced exact coefficients and bitwise density
reordering on the first and middle full16,384-atom blocks. On the final13,365
atoms, coefficients still matched exactly, but5,777 density terms in the
fourth observation differed (maximum0.1183698); the first three rows were
bitwise identical. Saved the complete failed probe arrays. Its last density
batch is shorter, so reordering moves individual atoms into/out of that batch.
This isolates a batch-shape numerical effect; it does not yet explain the
six small candidate differences in v2, nor the historical shear failures.

The frozen v4 driver pads only the final atom block to16,384 using duplicate
atoms and discards those evaluations before summing or ranking. Four original
observation rows and4,096-atom density chunks remain unchanged. Added fixed
mixed-subset and duplicate-atom replays to test changes in the queried atom
set, not only permutations of an identical set. All probes finish and save
diagnostics before their combined gate; tolerances remain unchanged. Two
independent preflight groups can then run on two GPUs before the four-worker
broad comparison. CPU preparation16742975 and GPU replay16742976 are submitted;
no broad scans are submitted. Ruff and shell syntax passed.

## 2026-09-28 — Gate independent profiles on numerical replay; reuse fixed responses

The first full-centre scan in preflight16742809 completed, but candidate
reordering failed the unchanged1e-7 per-atom density replay:6 of65,536 terms,
maximum log-contribution difference3.69084e-6. The broad worker array was
held during a runtime review and all dependent workers/reducer subsequently
cancelled on preflight failure. No A/B likelihood profile has been accepted.
The old scheduled status-only job was cancelled as superseded.

Added experimental scripts/v39_group_response.py, retaining the original
response coefficients for one fixed four-observation group across shear
points. This stores about0.65GB per group and avoids repeating the same
emulator pair sums. It requires exact coefficient replay on reordered
candidate/tail sets and rejects changed observations. No production model,
batch shape, prior, observation, seed, grid or tolerance changed. Added a
focused row/atom/mask/cache-invalidation test; the likelihood driver now saves
replay diagnostics before failing and checks the union of near-modes across
all numerical variants rather than only the primary profile's near-mode.

Frozen v3 preserves v2 scientific settings and outputs. CPU preparation16742917
and bounded reorder diagnostic16742918 are submitted; broad scans are not
submitted pending this diagnostic. Ruff and shell syntax checks passed.
Preparation will run the existing ten tests plus the new memo test. The
reorder diagnostic compares original versus reused coefficients and ordered
versus permuted densities on three fixed atom blocks for each cohort.
See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Start scene-disjoint prior A/B likelihood-profile comparison

Owner clarified that continuation means the independent-population-realization
comparison and likelihood profiles proposed in the reflection. Discarded the
unsubmitted atom-thinning draft; no thinning experiment ran. Added
scripts/run_v39_independent_profiles.py. Use the original F8-ready prior's
scenes20000–20099 as A (10,116,902 atoms) and20100–20199 as B
(10,114,319 atoms). These are disjoint generated scenes from the same empirical
parent source, with the original inverse-inclusion masses separately
normalized. No extra bright atoms, model changes or fitted reweighting.

The first128 existing selected model draws whose generating atom belongs to
A are a conditional sample from A. Preserve their original four-row numerical
groups. The independent-input control is the first128 fresh unselected-parent
mock measurements, selected once. There are87 numerical groups in total;
only128 rows per cohort enter scientific sums. The A cohort has three
measured-mag<20 rows and the fresh-parent prefix has none; no identities were
changed after noting this limitation. Each cohort is evaluated against A and B.

Freeze a7x7 two-component grid, offsets±0.03 in0.01 steps about the original
full20k mock's measured mean, without truth-centred starts. Exact centre
scans supply per-prior top16384 candidates and the existing conditional-tail
proposal; use seeds8701/8702 and4k/8k/16k nested tail draws. Normalize A and B
separately using two independent weighted-prior samples and nested32k/64k
atom and32/64 CRN-QMC flow budgets. All normalizer and proposal variants are
retained. A small preflight compares both priors on the first original group
of each cohort against exhaustive sums at centre and g1±0.01;8k/16k must
agree within0.001 in centred per-object log numerator before broad grids run.
Coarse-grid peaks are diagnostic and require convergence checks and refinement.

Ruff, AST and shell syntax passed; preparation runs ten existing tail/reduction
tests. First attempt16742780 failed on import because the writer alias belongs
to the cohort driver; no science outputs were generated. Preserved v1 and
cancelled dependent preflight automatically. Corrected the alias in a frozen
v2 root. Preparation16742796 completed in33s with ten tests passed and source,
scene-disjointness, mass, cohort and padding checks passed. Preflight16742809
gates four one-V100 workers16742810[0–3], then reducer16742811. No other GPUs
were active. See [independent prior profiles](V39_POPULATION_AUDIT.md#independent-prior-profile-comparison).

## 2026-09-28 — Pause runs and reassess the inference failure mechanism

Owner requested reflection on same-prior mock recovery, actual and
matched-input curvature failures, and the unstable expanded-prior result.
No jobs were active and none were submitted. Reviewed generation code,
the four historical comparisons, parent-population audits, the completed
fresh-parent5130 result, exhaustive probes, and the mixture-curvature
derivation in MATH.md. Added a synthesis under
[reflection on the failure mechanism](V39_POPULATION_AUDIT.md#reflection-on-the-failure-mechanism).

The synthesis distinguishes finite-prior approximation, numerical atom
integration, selected-cohort conditioning, and local optimization. Same-prior
mocks contain their generating atom and are internal closure tests; their
iteration stability did not pass draw convergence. Matched-input mocks start
from already selected inputs, so are not ordinary parent-population closure.
The clean fresh-parent5130 run had positive information throughout its draw
ladder, preventing a blanket conclusion that off-prior mocks necessarily
fail. Exact local negative information does not establish a wrong likelihood
maximum; the original actual run never reached BFGS recentering.

Numerical integration errors are established. Finite-prior resolution and
strong mixture nonlinearity remain plausible explanations requiring separate
tests; broad population mismatch is not established. A proposed controlled
same-population comparison would generate from one finite prior and infer
using that prior versus an independently realized prior, with selection once
and converged likelihood profiles. This is a proposed discriminator, not a
newly authorized run or result. No code, model, prior, or acceptance change.

## 2026-09-28 — Exhaustive paired probes complete; negative information survives reweighting

At the owner's check, recovery worker16738603_0 completed in27m43s and
reducer16738604 in24s, both exit0. The scheduled status audit also completed;
no study jobs remain queued or running. The actual group's complete centre
replay has maximum error0 at the unchanged tolerance. All four groups pass
the direct prior-weight/zero-support checks. The reducer verified output
hashes, replayed36 source aggregates, and preserved unprobed rows bitwise.

Saved completion_audit.json in inference_v39_cohort_curvature_20260928_v2:
a light Python/NumPy audit verified pinned source receipts, all four exact
array hashes, and reproduced all32 reported per-object/prior score and
information results from the saved nine-node sums. Preparation had already
passed ten tests. Large centre-array hashes were checked by the scheduled
reducer, not reread on the login node. No implementation or model changes.

Actual row32's exhaustive minimum information eigenvalue is-248854.33 with
expanded weights and-253958.42 with corrected weights. Its repaired sampler
passes all12 comparisons against exhaustive moments (two priors, three
seeds,8k/16k); its baseline passes0/12. The three focal mocks also retain
negative directions with both weights, and their unchanged nonbright sampler
fails all36 comparisons. Sampling errors are real, but do not explain away
these local negative directions.

After replacing the15 scientific probed rows, all36 partial-cohort summaries
remain indefinite. At16k, corrected minimum eigenvalues span-359262 to
-296561 for actual1024 and-64507 to-45578 for mock836. Both uncorrected
expanded arms also remain indefinite. Corrected focal distinct-atom ESS
is4.57–8.11 and one atom contributes30.5–45.5% of each focal numerator.
Concentration makes finite-prior resolution a concern, not an established
cause of population bias. This is a targeted subset at one centre and h=.001;
it neither overturns the old full-mock positive-information result nor
completes the stopped full-cohort comparison. No accepted Newton estimate or
0.3% claim. The next bounded diagnostic would check smaller shear spacings
and local likelihood profiles for these objects; no new run was launched.
See [completed paired probes](V39_POPULATION_AUDIT.md#completed-paired-exact-curvature-probes).

## 2026-09-28 — Three exhaustive mock probes complete; repair actual-centre replay

At the owner's status check,16735501[1–3] completed exit0 in27m27s–27m45s.
Worker0 failed after10m41s: one of65,536 log contributions differed from the
saved actual centre by1.43686243e-7, exceeding the fixed1e-7 replay tolerance.
The dependent reducer never ran; cancelled obsolete16735502. Preserved the
failure and all completed outputs. The scheduled audit16735503 recorded
failure but did not notify an agent; this was discovered at the user check.

Verified completed mock exact-array hashes and saved partial_mock_review.json
in inference_v39_cohort_curvature_20260928_v1. Exhaustive minimum information
eigenvalues for expanded/corrected weights are respectively: mock151
-46145.90/-46150.64; mock734 -28283.90/-28283.91; mock853
-25551.01/-28115.16. Thus these three local negative directions persist
under exhaustive integration and both weightings. Sampling errors remain
substantial (individual16k information-entry errors up to18606), but do not
fully explain the negative curvature. Corrected distinct-atom ESS is6.44,
5.76,8.11. These are per-object, post hoc diagnostics, not a full-cohort result.

Inspection found a replay-input discrepancy: the saved centre-only scan used
the original full-precision centre, while the nine-node stencil rounds it to
14 decimal places (component differences about2e-15). Updated only the replay
path in scripts/probe_v39_cohort_curvature.py to use the original exact float
inputs, retaining the existing rounded inputs for all scientific stencil
sums. Before the full rerun, a focused check over the interval containing
the failure must reproduce rounded-input violations and pass original-input
replay at the unchanged1e-7 tolerance; otherwise it stops. This explanation
is pending that GPU check, not yet established.

New root inference_v39_cohort_curvature_20260928_v2 reuses the three completed
mock groups with their original hash receipts and protocol identities. Only
actual worker0 is rerun. Ruff/AST/shell syntax passed. Preparation16738602
gates one-GPU16738603_0, then reducer16738604; status-only16738605 is scheduled
in two hours. Scientific settings and gates are unchanged. No full-cohort
continuation or model change. See [replay recovery](V39_POPULATION_AUDIT.md#paired-probe-replay-recovery).

V2 preparation completed in27s with ten tests passed. GPU recovery check
confirmed the cause: over the interval containing the old failure, rounded
centre inputs reproduce exactly one1.4368624334792912e-7 violation, while
the original unrounded inputs reproduce every retained value bit for bit
(maximum absolute error0). Direct prior-weight replay also passed all nine
nodes. The actual worker then began its exhaustive scan with no failures.
Only the replay comparison changed; scientific stencil sums and the1e-7
tolerance remain unchanged. Allow about25–30minutes for the remaining
actual scan and reducer; completed mock results are reused.

## 2026-09-28 — Launch four exact curvature probes with paired prior weights

Owner requested continuation after clarifying that the old full-mock positive
information and new subset negatives are different comparisons. Added
scripts/probe_v39_cohort_curvature.py: use the existing likelihood loaders,
normalizations, finite-difference moments and subset reductions for actual
row32 and mock rows151,734,853. These are post hoc stress probes from the
completed subset decomposition, with original four-row density groups.
No new model, prior correction, seed or likelihood centre is introduced.

Sum all28,012,032 atoms at the same nine shear nodes for both expanded and
domain-corrected weights. Reuse each expanded density calculation through
the exact weight ratio; check this against direct corrected densities at
all nodes, including zero-weight atoms. Replay the existing actual group's
full centre array. Retain new mock centre arrays for possible later reuse.
Compare old baseline/bright-repaired moments against the exhaustive reference
at8k/16k for all three seeds, with unchanged numerical tolerances. Separately
replace only scientific rows of the probed groups in the existing subset
totals, exclude padding, verify all untouched moments bitwise and replay
the source aggregate summaries. This is partial diagnostic replacement,
not full-cohort closure or a0.3% calibration test.

Ruff, AST and shell syntax passed. Frozen root:
inference_v39_cohort_curvature_20260928_v1. CPU preparation16735500 runs
the ten existing exact-tail/reduction tests and checks source hashes and
the old actual centre array. Four one-V100 workers16735501[0–3] depend on
preparation; reducer16735502 follows success. Two-hour worker limits and
four-GPU ceiling; no other study GPU jobs were active. Initial expected
runtime30–45minutes from the earlier nine-node pilot, pending actual scan
throughput. Status-only audit16735503 is scheduled in two hours. Startup
monitoring is underway. See [paired exact probes](V39_POPULATION_AUDIT.md#paired-exact-curvature-probes).

Preparation completed exit0 in24s, with ten tests passed and the retained
actual centre hash verified. The groups contain15 scientific observations
and one padding row (mock732), excluded from aggregate replacement. Focal
measured magnitudes are19.4751(actual32),21.04614(mock151),22.18870(mock734),
20.13503(mock853). A lightweight reduction precheck exactly reproduced36
source aggregate score/information/eigenvalue comparisons with the original
padding masks. Active/frozen probe sources agree. Resource receipts confirm
one32GB V100 per worker; startup checks continue through the first scans.

All four workers passed direct expanded-to-corrected weight replay at all
nine nodes, including zero-support checks, and advanced beyond1,064,960
atoms each with no failures. The actual group's retained centre replay also
passes through the scanned prefix. The first65 blocks take about51s per
worker, suggesting roughly22minutes of scanning plus startup/reduction;
allow about25minutes remaining at this check. Leave the normally advancing
jobs and dependent reducer running; final exhaustive and aggregate results
remain pending.

## 2026-09-28 — Clarify positive full-mock result versus negative short diagnostic

Owner recalled that adding bright atoms made the mock information positive.
Rechecked the frozen expanded comparison and current subset results: this
is correct for the original full16,314 mocks at16k draws (minimum eigenvalue
+112804, unstable g1 proposal0.1473); full actual20k remained indefinite
(-995019). Added a comparison-boundary note to
[completed short diagnostic](V39_POPULATION_AUDIT.md#completed-short-cohort-diagnostic).
The new1024/836 test also changes numerical integration, and retains both
expanded and domain-corrected weights. Both repaired subset arms are negative
at16k; this cannot attribute the difference from the earlier full-mock result
to reweighting alone. Full-cohort controlled results remain unavailable
because that run was stopped. No scientific settings or jobs changed.

## 2026-09-28 — Short cohort comparison completes; negative information persists

Workers16735075[0–3] completed exit0 in33m30s–33m47s; reducer16735076
completed exit0 in20s. All eight centre scans and selected-density replay
checks passed. Verified protocol/combined-moment hashes, four worker receipts
and all small repair arrays; large centre arrays were not rehashed during
this completion audit. Confirmed fainter moments remain unchanged. Saved
completion_audit.json in inference_v39_cohort1024_recovery_20260928_v1,
refreshed status_latest.json and cancelled the now-unnecessary16735077 audit.

At16,384 draws, both expanded and domain-corrected bright-repaired arms
have indefinite summed information in all three seeds for actual1024,
actual same836 IDs and mock836. Corrected actual minimum eigenvalues are
[-315697,-359242,-296517]; corrected mock values are[-42441,-38947,-70409].
There is no accepted Newton estimate. Some lower-rung cases and unrepaired
mock seeds are positive; do not generalize the16k result to every rung or
select those positive cases as recovery evidence.

Corrected repaired bright per-object8k→16k changes are at most0.00901 in
score and8.85 in information entries. Along each seed's minimum-information
direction, the actual bright contribution is about-246k, with row32 a large
negative contributor; fainter rows add-50k to-113k. In mocks the bright sum
is positive(+34k to+35k), while fainter rows contribute-74k to-104k, with
rows151,734,853 prominent. Fainter sampling remains unstable. Reweighting
and bright-only repair do not resolve this fixed-centre diagnostic; this
does not establish the full likelihood shape or the original20k outcome.
Next useful bounded diagnostic: exact local likelihood checks for the
dominant bright actual row and fainter mock contributors. No new GPU jobs
or automatic full-cohort continuation were launched. See
[completed short diagnostic](V39_POPULATION_AUDIT.md#completed-short-cohort-diagnostic).

## 2026-09-28 — Replace full-cohort recovery with a short1024-identity diagnostic

Owner asked why the comparison was slow and requested smaller test jobs.
Stopped workers16733516, dependent reducer16733517 and status audit16734855.
Preserved293 completed baseline chunks (157 mock,136 actual), original
inputs/protocols and logs; no bright repairs had completed. The full run
will not resume automatically. Its266 full-prior centre scans dominated
the remaining estimate, about15hours on four GPUs.

Extended scripts/run_v39_cohort_recovery.py to take a fixed prefix, reuse
hash-verified baseline chunks after checking scientific settings against
the source protocol, and reduce the actual subset length. Default full-
cohort behavior remains available; frozen earlier drivers are unchanged.
New root inference_v39_cohort1024_recovery_20260928_v1 takes the first1024
original identity slots in each sample, preserving four-row density groups,
object RNG IDs and padding exclusions. Both priors, three seeds, all draw
rungs, centre, normalization and measured selection are unchanged. Repair
requires only eight groups (four scientific bright rows per sample).
This diagnoses curvature and integration stability; it cannot establish
0.3% bias or settle the original full-cohort failures.

Ruff, AST and shell syntax passed. Preparation16735074 checks source hashes,
runs the existing ten helper/reduction tests and copies the sixteen cached
baseline chunks. It gates four one-V100 workers16735075[0–3], followed by
reducer16735076. The prior GPU preflight is explicitly inherited; every new
repair still requires direct centre-density replay. Worker limits are2hours,
with an initial30–45minute runtime estimate plus queue time. Status-only
audit16735077 is scheduled two hours later. Startup validation is pending.
See [short cohort diagnostic](V39_POPULATION_AUDIT.md#short-original-cohort-diagnostic).

Preparation completed exit0 in25s; all ten tests passed. The prefix contains
1024 actual measurements and836 scientific mocks (188 flagged padding rows),
with four measured-bright rows in each. All sixteen baseline chunks were
copied after hash verification. All four32GB V100 workers started; resource
receipts confirm the same hardware as the source run. AST comparison confirms
repair, evaluate and load_pair are unchanged from its frozen driver, and
the new active/frozen driver hashes agree. All four workers then verified
their cached baseline chunks and reached their first full-prior centre scan
(mock groups8,9,44,145), with no failures. Startup includes about five minutes
of loading, verification and first density compilation; the30–45minute total
remains provisional pending sustained scan throughput. Leave the bounded
run, dependent reducer and status audit running.

## 2026-09-28 — Launch controlled original-cohort prior and bright-repair comparison

Owner requested continuation. Added scripts/run_v39_cohort_recovery.py,
reusing the frozen likelihood loaders, sampler, exact-tail helper and Newton
reduction. The [original-cohort protocol](V39_POPULATION_AUDIT.md#original-cohort-recovery-comparison)
keeps actual20k and matched16,314 scientific mocks at the same centre and
compares two priors × baseline/bright-repaired integration. Three seeds and
every draw rung remain fixed; no optimization or production promotion.

The baseline uses the original expanded-prior proposal for both targets,
preserving common draws/probabilities while target weights and normalization
change. Repair covers all135 actual/134 mock measured-mag<20 rows in266
original groups. Fainter rows remain in the estimates and numerically
approximate; their known integration errors and remaining draw drift must be
reported. This is a bounded partial repair, not a claim of reliable whole-
sample integration. One expanded centre scan serves both prior arms via
the exact prior-weight ratio; selected-density replay verifies that reuse.

Validation before submission: Ruff/AST/bash syntax passed. Preparation16733505
checks input hashes/padding and runs ten existing targeted tests. GPU
preflight16733506 gates four one-V100 workers16733507[0–3] and checks the
historical K1024 replay, corrected nine-node K16,384 sums and prior-weight
identity including zeroed added atoms. Reducer16733508 follows successful
workers; status-only audit16733509 is scheduled about two hours later.
Frozen root inference_v39_original_cohorts_recovery_20260928_v1 pins driver
and source receipts. Initial runtime estimate20–24hours,36hour limits,
four-GPU ceiling intact. Startup is being monitored before yielding.

Preparation16733505 passed all ten tests and source/padding checks in22s.
During startup review, caught a replay-only configuration mismatch: historical
primary arrays use min_ess32/max_fraction.5, whereas the newer parent defaults
are256/.25. Stopped preflight16733506 and cancelled pending16733507/16733508/
16733509 before scientific workers allocated. The corrected replay now calls
the frozen historical evaluator itself; scientific K16,384 settings and all
comparison thresholds are unchanged. Preserve v1 with a superseded receipt;
active root is inference_v39_original_cohorts_recovery_20260928_v2.
Ruff/AST passed again. New preparation16733514 gates preflight16733515,
four workers16733516[0–3], reducer16733517 and two-hour status16733518.

V2 preparation completed exit0 in12s, ten tests passed. GPU preflight passed:
both historical actual1958/mock4460 arrays replay bit for bit; corrected
nine-node K16,384 sums agree within1.05e-13/1.40e-13 in log space. The
weight-ratio identity and zero-weight added-atom checks pass, as do both
target evaluations with a common proposal. Peak allocated GPU memory18.07GB
on a32GB V100. Dependent worker startup is now being checked.

Preflight finished exit0 in5m04s. Four workers16733516[0–3] started on
th-cl-nv01 with one32GB V100 each. Followed startup until every worker
completed its first original four-row group for both priors and all three
seeds, with no failures. Work is balanced as78–79 baseline chunks and66–67
repair groups per worker. Active/frozen source hashes agree. Jobs now
progress normally; leave the dependent reducer and two-hour status audit
in place. Runtime estimate20–24hours awaits sustained-checkpoint refinement.

## 2026-09-28 — Intermediate-magnitude repair passes all96 numerical checks

Owner reported completion. Workers16733295[0–3] completed exit0 in26m16s–26m20s
and reducer16733296 in23s. Refreshed external status, verified four completion
and centre-replay receipts plus all small output hashes; the reducer had
also verified the four retained centre-array hashes. See the
[completed intermediate pilot](V39_POPULATION_AUDIT.md#completed-intermediate-magnitude-pilot).

Repaired integration passes24/24 focal checks and72/72 companion checks,
versus0/24 and24/72 for the baseline. Across all16 rows, repaired maximum
absolute score/information-entry errors are0.08432/1.29420, within unchanged
entrywise thresholds for every seed and both8k/16k. No favorable seed or
focal observation was substituted. All six draw rungs remain saved.

Exhaustive per-observation information is still indefinite on all four focal
rows. This is not evidence that the full-sample matrix must fail, and the
repair correctly preserves negative directions. Three focal posteriors have
distinct-atom ESS2.48–10.25; the fourth has ESS1212.46 and only28.59% minimum
candidate coverage, confirming that its tail cannot be discarded. Finite-prior
resolution and local optimization remain distinct from integration accuracy.

The original actual20k negative information and matched16,314 g1=0.1473
proposal are still open full-sample comparisons. This check changes no code,
models, weights, thresholds or production defaults, and launches no additional
GPU job. Next work returns to those original fixed cohorts with the validated
integration, keeping population correction separate from numerical repair.

## 2026-09-27 — Launch four intermediate-magnitude probes with existing integration methods

Owner authorized continuing while avoiding overengineering. Added only the
diagnostic driver scripts/probe_v39_intermediate.py, reusing the frozen parent
evaluator, exact-tail proposal, corrected-prior loader and moment reduction.
The [fixed four-probe protocol](V39_POPULATION_AUDIT.md#intermediate-magnitude-numerical-pilot)
compares baseline and repaired integration against nine-point exhaustive sums:
two fresh-mock stress probes (one per measured-mag20–22/22–24 bin) and two
fixed-seed actual20k controls. Original four-row groups and random identities
remain unchanged; all twelve incidental companions are also reported.

One full-prior scan per worker accumulates nine-node sums and retains centre
logs/top16,384 for reuse. No estimator, production setting, model, prior or
tolerance changes. Keep all three existing seeds and all draw rungs; apply
the previous gate to every focal row at both8k/16k. This does not replace
the unresolved full actual20k/matched16,314 comparison or establish0.3% bias.

Validation before submission: Ruff, AST and launcher bash syntax checks pass.
Frozen root inference_v39_intermediate_pilot_20260927_v1 pins source and input
receipts. Preparation16733294 runs ten existing targeted tests and input
checks; workers16733295[0–3] depend on its success, then reducer16733296.
Status-only audit16733297 is scheduled in about two hours. Queue was empty
before submission; four persistent one-V100 jobs in inter preserve the cap.
Startup and numerical replay checks are being monitored before yielding.

Preparation subsequently completed exit0 in22s, ten tests passed. Selected
rows are fresh4826/3602 (measured mags20.45964/22.57042) and actual11173/2510
(21.1134/22.3313). Four one32GB-V100 allocations started on th-cl-nv01;
active/frozen driver hashes match and resource receipts confirm one visible
GPU per worker. Both fresh baseline groups subsequently replayed all saved
arrays exactly across all three seeds, and actual baselines completed.
All four exhaustive scans reached1,064,960 of28,012,032 atoms in50.4–50.8s
of scan time, without failures. This is meaningful normal progress; estimated
30–40minutes remain including tail comparisons. Centre replay and final
numerical gates are pending. Leave jobs running with their dependent reducer
and two-hour status audit; no full-sample shear conclusion follows yet.

## 2026-09-27 — Fresh-parent inference completes; remaining draw drift is in nonbright rows

Owner requested a check. Workers16732143[0–3] completed exit0 in2h20m–3h12m23s
and reducer16732144 in57s. All1283 baseline groups and43 bright groups are
complete, with no failures. Refreshed the external status JSON and documented
the [completed first-pass result](V39_POPULATION_AUDIT.md#completed-fresh-parent-first-pass-diagnostic).
Verified aggregate moments/protocol/input hashes, four worker receipts and
43 bright-group receipts. The scheduled reducer had verified every consumed
moment-file hash and unchanged nonbright rows. Small saved-array reductions
independently reproduce the final proposals and attribute draw-count movement
within1e-14; output is completion_audit.json in the frozen run root.

On5,130 scientific mocks (44 measured-bright, two numerical copies excluded),
the three16k bright-repaired g1 proposals are0.016889±0.007046,
0.016785±0.006936 and0.017216±0.007600, versus injected0.02. These are one
fixed-centre Newton step, not converged estimates; seeds share observations.
All36 arm/seed/rung summed information matrices are positive, but no result
meets the small-step flag. Repaired8k→16k movements are+0.000340,+0.000421,
+0.000165: still above the0.000060 scale corresponding to0.3% of injected shear.

Exact additive attribution shows bright rows contribute at most2.80e-7 in
absolute net g1 movement; measured-mag>=20 rows cause the remaining drift.
Bright rows still account for88.8–89.3% of squared g1 sandwich influence,
leaving finite-prior resolution and local quadratic validity unresolved.
Next: numerical probes of intermediate-magnitude rows before a larger run
or recentering. No production code/model/prior changes or new GPU submissions.

## 2026-09-27 — Exact-tail pilot passes at K16k; launch fresh-parent first-pass diagnostic

Owner reported completion. Workers16731880[0–3] finished exit0 in17m35s–17m43s
and reducer16731881 in20s. K16,384 passes all24 predeclared checks across
four focal rows, three seeds and both8k/16k draw rungs. K4096 passes22/24,
K1024 passes6/24; neither passes the combined gate. Verified worker hashes
and centre replay receipts. The hardest actual probe has maximum score error
0.010354 and information-entry error20.6443; the latter is just within the
corresponding21.3091 tolerance. No thresholds or seeds changed, and no general
convergence or0.3% shear result is asserted. See the
[completed pilot and next protocol](V39_POPULATION_AUDIT.md#completed-exact-tail-pilot-and-fresh-parent-first-pass).

Added scripts/run_v39_parent_inference.py and froze root
inference_v39_parent5130_exacttail_20260927_v1. Reuse the verified corrected
prior and normalization at the already frozen actual-sample centre, independent
of these new mock results. All5,130 scientific mocks get a K16,384 Gaussian
baseline; replace only44 measured-mag<20 rows in43 original groups with the
accepted exact-tail integration. Keep three original seeds and every draw rung.
Two flagged numerical copies fill the final group and are excluded from every
estimate. Existing input IDs preserve the per-object RNG streams. This is one
fixed-centre Newton diagnostic, not a converged inference or model acceptance.

Validation: Ruff/AST/bash syntax pass. Initializer16732141 completed exit0 in16s,
including10 targeted proposal/reduction tests, all four large saved-centre
hashes, fresh-mock source hashes, measured-cut checks and no scientific-row
loss. GPU preflight16732142 verifies original four-row group replay and audits
the nearly seed-constant K1024 candidate curvature residual separately from
sampling noise. Canonical K16,384 replay must pass before the four persistent
one-GPU workers16732143[0–3] start; CPU reducer16732144 depends on them.
Initial runtime estimate3–4hours,6hour limits, with the four-GPU ceiling intact.
Status-only audit16732145 is scheduled for about two hours. No production model,
prior, numerical default or acceptance threshold changed.

Preflight16732142 subsequently passed in3m21s and all four workers started on
th-cl-nv01. Fresh four-row groups replay bit for bit. The same-atom direct1024
versus canonical16,384-prefix diagnostic confirms a separate density batch
effect: information11 differences+36.5221 for actual5359 and−41.3445 for
mock3192, from log-integral differences about2e-5. These explain much of the
seed-stable small-K curvature residual; the exact internal numerical operation
is not isolated. Canonical K16,384 replay passes, supporting the new run's
full4096-atom density blocks without rewriting any old result.

Worker startup was monitored through saved inference checkpoints. At3m52s
elapsed all four workers were progressing normally, with40 of1283 baseline
groups complete across the array (all three seeds per group), no failure
receipts, and one32GB V100 visible per worker. The bright work is distributed
as11/12/12/8 groups. The scheduled two-hour status audit remains active;
the3–4hour initial estimate is still appropriate pending bright scan progress.

## 2026-09-27 — Launch exact-centre conditional tail proposal pilot

Owner said to continue numerical repair. Added scripts/v39_exact_tail.py,
scripts/probe_v39_exact_tail.py and tests/test_v39_exact_tail.py. The
[conditional-tail protocol](V39_POPULATION_AUDIT.md#exact-centre-conditional-complement-pilot)
uses the same exact candidate sets and replaces Gaussian whole-prior tail
draws with90% exact centre-likelihood complement plus10% conditional scene
prior. It excludes exact candidates from draws and reports actual CDF-bin
probabilities. This preserves the numerator integral's expectation and full
positive-prior support; log/derivative accuracy still requires the existing
numerical gate. No production runtime or scientific settings changed.

Frozen root inference_v39_exacttail_pilot_20260927_v1 pins script/helper/test
hashes and both reference results. Keep four original groups, K1024/4096/16384,
seeds8701–8703, draw ladder512–16384, h0.001, prior and normalization unchanged.
All scores and information entries must pass at both8k/16k for every seed.
Retain expensive full-prior centre contributions on disk for future reuse;
verify their sum and selected atom densities against the preceding run.

Validation: Ruff and AST passed; CPU16731872 ran the new proposal tests plus
tests/test_catalogue_null.py,37 passed in22.11s, zero regression failures.
Checks include an exact expectation oracle with an atom whose centre
likelihood is zero but displaced likelihood is positive, excluded candidates,
extreme log scales, nested draws and per-object uniform streams. Launcher
bash syntax checks passed. The first array16731873/reducer16731874 were
cancelled while pending to correct a reference-root substitution; no worker
allocated and no scientific result was produced.

Corrected array16731880[0–3] depends on successful tests and uses four one-GPU
V100 workers in inter,45minute limits; CPU reducer16731881 follows it.
Status-only audit16731875 is scheduled in about two hours. Startup checks
confirmed one32GB V100 visible per worker, all four on th-cl-nv01. At4m05s
elapsed all workers had scored1,064,960 of28,012,032 atoms with no failures;
steady throughput was about30s per1.05million atoms after compilation.
Estimated15–20minutes remaining including proposal comparisons. Active and
frozen source hashes match. Next, compare every predeclared check with the
frozen exhaustive sums; no accepted shear result or model promotion is implied.

## 2026-09-27 — Whole-prior ranking completes; mock probes converge but actual curvature still fails

Owner requested status. Array16731606[0–3] completed exit0 in17m24s–17m28s
and reducer16731607 in19s. Verified all worker/reference NPZ hashes and
all four exact-centre replay receipts; refreshed the external status JSON.
Documented the [completed whole-prior pilot](V39_POPULATION_AUDIT.md#completed-whole-prior-ranking-pilot-on-september27).

At K16,384 both mock focal rows pass all six seed/rung checks. Actual1958
passes0/6 and actual5359 passes3/6; their scores pass but curvature does not
consistently meet the unchanged entrywise tolerance. Minimum nine-node
candidate coverage is99.629465%,99.803972%,99.915333%,99.996588% for the
two actual and two mock probes respectively. All three K arms still fail
the combined gate. No seed or result was omitted.

Added external tail_diagnostic.json using small saved-array reductions:
candidate-only versus exhaustive log-numerator finite differences isolate
the remaining tail contribution with normalization cancelling. Omitting the
tail at K16,384 changes actual1958's g1 score by+0.66258 and its11information
entry by+172.77, so coverage alone cannot justify truncation. No runtime code,
models, production prior, gate or defaults changed; no new GPU job submitted
at this status check. Next work is a controlled complement-accuracy repair
on these probes, followed by fresh unselected-parent mock inference after
numerical acceptance. A0.3% shear result remains unestablished.

## 2026-09-27 — Corrected prior verified; pool-limited ranking improves but fails integration gate

Owner requested status. All16730737workers finished exit0 in55m04s–55m07s,
reducer16730742 in26s, independent replay16730778 in1s, and scheduled audit
16730779 in2s. Verified worker NPZ hashes and receipts. All28normalization
shards reproduce old-weight masses exactly; directly reweighted exhaustive
numerators match the earlier component reconstruction within2.85e-14.
Documented the [completed pilot](V39_POPULATION_AUDIT.md#completed-domain-matched-pilot-on-september27).

Every arm fails the unchanged score/information gate on all four focal rows.
Exact ranking still materially improves the1024-atom stratum: minimum
stencil coverage for actual1958 rises from0.000459% to93.4727%, and mock4460
from0.0000239% to98.7186%. Their existing Gaussian ranker misses most local
mass, exposing the integral to noisy complement draws. However, the complete
broader pool itself captures only94.4472% for actual1958 and98.7884% for
mock4460. Saved restricted_pool_errors.json separates deterministic pool
errors from complement noise; increasing K within that pool cannot recover
the omitted mass. No gate, observations, seeds or reported results were changed.

Fresh one-draw generation16730768 also completed exit0 in15s:20,000unselected
input attempts,8,972usable,5,130retained including44measured-bright rows.
No top-up or generating-atom insertion. No inference result is yet accepted
on that cohort.

Continue the already authorized numerical repair with a small
[whole-prior ranking probe](V39_POPULATION_AUDIT.md#exact-whole-prior-ranking-follow-up).
Added scripts/probe_v39_full_rank.py; frozen root
inference_v39_fullrank_pilot_20260927_v1. At the same centre, evaluate all
atoms once and retain the highest exact contributions; compare K1024,4096,
16384 with unchanged full-support complement, seeds and both draw rungs.
Reuse the completed nine-node exhaustive references and normalization.
Centre sums and selected atom densities must replay within log-space1e-7.
Array16731606[0–3] and reducer16731607 stay within four one-GPU V100 workers;
30minute worker limits. Ruff and AST checks pass; existing numerical tests
are unchanged, with runtime replay checks added for this diagnostic. Startup
succeeded on four32GBV100s, one per worker; at6minutes elapsed roughly5.2million
atoms had been scored per worker without failure receipts. Estimated another
15–20minutes at observed throughput. Scheduled status-only audit16731657
for about two hours; it records results without promoting a sampler. No large
inference, model change or0.3% claim follows from the failed pool-limited gate.

## 2026-09-27 — Launch domain-matched bright pilot and prepare unselected-parent mocks

Owner authorized the recommended next steps. Added
scripts/run_v39_domain_matched.py and scripts/run_v39_parent_mock.py; the
[pilot protocol](V39_POPULATION_AUDIT.md#domain-matched-pilot-authorized-on-september27)
and [fresh input cohort](V39_POPULATION_AUDIT.md#fresh-unselected-parent-mock-preparation)
are documented with frozen external roots, seeds, gates and limitations.
No production models, priors, measurements, cuts or defaults changed.

Candidate initialization16730721 passed in28s: zero weight for848,650 added
atoms outside the generator domain,7,201,662 valid bright atoms sharing the
original bright mass, and every nonbright weight unchanged. Four original
groups retain focal actual rows1958/5359 and mock rows4460/3192; the latter
row per sample is selected by fixed seed2026092711, independently of inferred
shear. Normalization reuses cached per-atom C8 probabilities and replays all
64-QMC forward draws/node with identical shard seeds, reducing both old and
new weights to verify the baseline before comparing algorithms.

Four one-GPU V100 workers16730737[0–3] run in inter, CPU reducer16730742
depends on completion, and independent numerator replay16730778 follows it.
Compare Gaussian candidate ranking, exact centre ranking within the existing
131072-atom location pool, and exact ranking within its union with a whole-
catalogue Gaussian131072pool. Each retains K1024, the original tilted
complement, both8k/16krungs and seeds8701–8703. All28,012,032 aligned atoms
are summed at all nine shear stencil points for each four-row group. Exact
pool coverage and all score/information entries are checked against the
predeclared numerical gate; this is not a0.3% scientific acceptance test.

The first allocation16730727 was cancelled during startup because a two-node
nodelist requested both hosts per worker. Corrected launcher to nodes=1 and
one V100; cancelled its reducer16730729. No completed stage was used from
that allocation. Current workers have one GPU each, three on th-cl-nv01 and
one on th-cl-nv02. Their first normalization shards all replayed the old
weighted masses exactly, maximum difference0.0, and progressed to second
shards without failure receipts. Estimated1.5–2hours after startup;4hour
limits. Status-only audit16730779 is scheduled in about two hours and writes
status_latest.json; it does not notify the owner or promote results.

In parallel, CPU16730765 prepared20,000 uniformly sampled unselected inputs
from all100ConstGold cases in1m43s, without truth cuts or input measurement
selection. All identities match;9,982 target-role and10,018 other-role rows,
with four chance overlaps with the former selected20k. Their new forward
generation16730768 waits for successful preparation and the GPU array to
end, preserving the combined four-GPU ceiling. It performs one normal draw
and one measured selection, reporting retained count without top-up. No
generating input is inserted into the inference prior. Generation is useful
regardless of sampler-pilot outcome; subsequent inference remains pending.

Validation: Ruff/AST checks pass. Tests16730722 passed317/skipped3 without
artifact roots; rerun16730766 with configured roots passed319/skipped1,
zero regression or snapshot failures. Parent input construction validates
source hashes and zero unmatched rows; candidate weights and frozen execution
identities passed. No new shear result or resolved sampler failure is claimed
while the normalization/sampling/exhaustive stages run.

## 2026-09-27 — Deep audit separates bright prior domain, cohort conditioning and integration

Owner asked whether population mismatch requires reweighting, then requested
a deeper investigation. Added read-only scripts/audit_v39_population.py and
scripts/profile_v39_population.py; full evidence and interpretation are in
[the population audit](V39_POPULATION_AUDIT.md#question-and-scope), linked from
the matched-input comparison. No production prior, measurement sample, model,
cut, seed or optimizer setting changed.

CPU16730323 scanned69,956,785 unselected input rows across100cases plus both
weighted priors, and audited the saved per-observation curvature. It completed
exit0 in1m35s. The original prior's largest intrinsic-property marginal CDF
discrepancy is0.02382percentage points; no actual input lies outside the
generator morphology domain. The expanded prior assigns10.5418% of its
bright mass to848,650 added rows outside that domain. A histogram-only
counterfactual excludes that mass, preserves the bright total and nonbright
weights, and reduces the bright n CDF discrepancy from9.8729 to0.6158points.
This is generator-support matching, not a truth cut on selected measurements.

CPU16730457 additionally compared27,982,713 unselected target-role contexts
from80cases with F8/C8 neighbour features; exit0 in42s. Original-prior maximum
neighbour-feature marginal CDF difference is0.03609points. These descriptive
checks do not prove full joint equality or0.3% calibration. The matched-input
mock starts from an already selected cohort and applies a second selection;
documented its distinct conditional input distribution and normalization.

The135 measured-bright actual rows contribute-999244.60 along the final
minimum-information direction, versus-995018.53 for all20k. Influential rows
have sampled ESS near1 and large8k-to16k changes. Three GPU jobs16730364[0–2]
sum all28,012,032 atoms for two post-hoc influential probes at the shared
centre and g1±.001; original four-row groups and4096 density chunks retained.
CPU reducer16730370 depends on all workers. Original and domain-matched
numerators are reconstructed from separately accumulated support components;
the latter is explicitly not a selection-normalized inference result.
External root: inference_v39_population_audit_20260927_v1. Source snapshots,
protocols, hashes, coverage and output receipts are preserved.

Validation: system Ruff and AST syntax checks passed for both new scripts;
full tests16730371 passed319, skipped1, no regression/snapshot failures.
No new model training or large inference continuation was launched.

All three exact workers completed exit0 in13m42s–13m46s; reducer16730370
completed in9s. Exhaustive expanded g1 information is-158392.95 for actual
row1958 and+165073.89 for mock row4460, compared with their16k estimates
-476790.52 and-233919.29. The actual16k score also has the wrong sign.
Thus the sampler demonstrably misestimates influential rows under the same
fixed likelihood; population reweighting alone cannot explain the discrepancy.
Exhaustive distinct-atom ESS is5.608 and1.662, with largest atom fractions
39.57% and76.70%, so finite-support resolution remains a separate concern.
Outside-domain atoms carry13.284% and3.431% of each probe's likelihood mass.
The domain-matched numerator g1 information becomes-8834.75 and+170036.63;
this is not a normalized candidate inference or a full-sample curvature test.
CPU16730526 verified exact reconstruction of all old weights in2s.

Recommended next sequence is domain-matched candidate weights with fresh
normalization, broader exhaustive bright integration checks, independent
prior-resolution checks and an unselected-parent small mock closure. All
diagnostic jobs are complete; no scheduled continuation remains. No accepted
shear or0.3% result is claimed. Histograms, post-hoc two-row selection, finite
QMC normalization, h=.001 and floating-point density limitations are recorded.

## 2026-09-27 — Expanded prior completes; mock curvature positive but inference unstable

Owner reported completion. sacct confirms16729459[0–3] all exit0 in
1h39m41s–1h41m24s; status-only audit16729469 also completed. Verified all
eight5000-row worker receipts and SHA256 hashes, complete comparison.json,
and absence of failure receipts. No active jobs remain. Updated external
jobs.json/status_latest.json and documented the
[expanded-prior result](V39_MATCHED_INPUT_MOCK.md#completed-expanded-prior-result-on-september27).

At16384 draws, minimum information eigenvalues change from−1767373.8285
to112803.9799 for matched mocks,−2394269.7775 to−995216.0549 for actual
same IDs, and−2387169.1061 to−995018.5262 for actual20k. Positive mock
information is not a successful recovery: its g1 proposal changes from
0.01905090 at512 draws to0.18813134 at8192 and0.14730972 at16384. The
final two-component step norm is about0.14356, beyond the historical0.05
recentring guard; conditional sandwich errors are about(0.90423,0.27986).
Both actual samples become positive at4096/8192 but return to indefinite
information at16384. The summary flag same_problem_at_shared_center=false
describes only the final mock matrix, not accepted or converged inference.

No accepted shear, robust0.3% result or optimizer continuation follows.
Expanded support and its changed proposal materially affect the result,
but prior support, integration error and local curvature remain unresolved.
No scientific code/settings/output changed and no extra GPU run launched.
Next useful diagnostic remains bright likelihood profiles with stronger
independent integration checks, without choosing a favorable draw prefix.

## 2026-09-27 — Expanded-prior normalization and replay pass; mock evaluation progressing

Owner requested status. All four16729459 workers remain running at56minutes
elapsed, with no failure receipts. All28 normalization shards and assembly
completed. Expanded-prior batching preflight passed exact equality of all
ten moment/diagnostic arrays for60 mock and64 actual rows, including bright
groups, on the same32GB V100 hardware.

Twelve of20 mock evaluation chunks are saved (12000 of20000 evaluation
rows, including explicitly excluded padding); every worker has entered its
fourth1000-row chunk. Actual-measurement evaluation follows each worker's
mock block and has not started yet. No comparison result exists yet. Recent
1000-row chunks take about346–361seconds, suggesting approximately45–55
minutes remaining including actual evaluation and reduction. Updated external
jobs/status snapshots and the
[expanded-prior record](V39_MATCHED_INPUT_MOCK.md#expanded-prior-comparison-requested-september27).
No scientific settings or code changed; no partial shear result inferred.

## 2026-09-27 — Restore extra bright atoms for the fixed matched-input comparison

Owner requested the expanded prior after the matched-input curvature failure.
Added scripts/run_v39_expanded_prior.py and documented the
[controlled expanded-prior comparison](V39_MATCHED_INPUT_MOCK.md#expanded-prior-comparison-requested-september27).
Reuse exact saved16314 mocks and original actual20k, fixed centre, V3.9
models, original four-row groups and sampler seeds/settings. Restore all
7,780,811 extra atoms with existing expanded weights, preserving nonbright
weights and total bright probability. Prior support is28,012,032 atoms.
No generating truth atoms are inserted and no new mock observations drawn.

Preparation validates the cached V3.9 support, identical original atom
features/coordinates/floors, fixed proposal metric and bright-mass allocation.
Old proposal C8 base probabilities stay unchanged; new atoms use their same
evaluation centre. Full selection normalization is recomputed at the fixed
actual centre with64 CRN QMC draws per atom/node and original shard seeds.
Both mock and actual likelihoods are reevaluated under the expanded prior;
final reduction compares with saved original-prior moments and all six draw
counts, rejecting indefinite information without a repair or forced estimate.

Frozen root: inference_v39_constgold20k_matched_expanded_20260927_v1.
CPU initialization16729457 passed its audit in25.8s of Python execution.
Full tests16729474 and initialization gate four persistent V100 workers
16729459[0–3], max4 concurrent GPUs, on inter/th-cl-nv01. Workers normalize
seven shards each, synchronize assembly, pass exact proposal-batching replay
on both datasets including bright groups, then evaluate both5000-row blocks.
Worker0 reduces after all workers finish. Estimated1.5–2hours after GPU
startup;4hour limits. Login resource snapshot retained; heavy work scheduled.
Syntax and system Ruff checks pass (py31 itself has no Ruff module).
No production default, model, optimizer, cut or seed changed; this remains
a selected-input fixed-centre diagnostic, not population closure or a0.3%
acceptance test. Startup/progress verification follows below.

Initial tests16729456 stalled in Python startup on cip-cl-compute4; a
read-only process check showed cl_sync_io_wait and no pytest output. Moved
the identical suite to inter/th-cl-1024us01 as16729474, updated the GPU
afterok dependency, then cancelled16729456. The successful initialization
used this inter CPU node. Scheduled status-only audit16729469 for about two
hours later; it saves a snapshot without agent wakeup or user notification.

Startup verified: replacement tests16729474 pass319 tests with1 optional
skip in27.28s. All four16729459 workers are running and have completed their
first two normalization nodes (zero shear and the fixed actual centre), with
finite positive masses0.01288–0.01293 and no failures. The cached proposal
C8 reference is zero shear; added atoms use that same point. Corrected this
description in the active driver's comment; its frozen running copy remains
unchanged and obtains the correct point directly from the pinned cache.
Saved current jobs/status snapshot; subsequent inference and replay stages
remain automatically gated. Long-run monitoring yields after this progress.

## 2026-09-27 — Completed matched-input mocks reproduce the curvature failure

Owner reported completion. Verified sacct for16729155[0–3] and16729156:
all completed with exit0, four GPU workers in30m30s–30m45s and comparison
in20s. Saved comparison.json/status.json report complete, with no failure
receipts. Of20000 exact input identities,16314 model draws survived C8 and
the new measured cuts; all3686 numerical padding rows are excluded.

At the fixed actual expansion point and16384 draws, observed-information
eigenvalues are(-1767373.8285,2002216.2944) for the mocks,
(-2394269.7775,3423542.5435) for actual measurements of the same IDs,
and(-2387169.1061,3445424.7863) for original actual20k. Read-only JSON
inspection confirms all three matrices are indefinite at all six draw
counts512–16384. No accepted shear estimate or bias result is available.
The failure therefore does not require actual-image measurement mismatch;
finite-prior support, integration error and local curvature remain unresolved.
This selected-input diagnostic is not population closure or a0.3% result.

Updated doc/V39_MATCHED_INPUT_MOCK.md with
[completed evidence](V39_MATCHED_INPUT_MOCK.md#completed-comparison-on-september27)
and limitations, plus external jobs.json/status_latest.json. Cancelled the
redundant pending status-only audit16729189 after checking results directly.
No scientific code, settings, seeds or outputs changed; no additional tests
or GPU runs were needed. A small bright likelihood profile with independent
integration checks remains the proposed next diagnostic.

## 2026-09-27 — Matched-input forward mocks requested for the actual20k cohort

Owner requested mocks from exactly the galaxy inputs associated with the
actual20k measurements, then chose normal forward sampling with the retained
count reported. Added scripts/v39_matched_input.py,
scripts/run_v39_matched_inputs.py and tests/test_v39_matched_input.py.
The preparation joins original(case,input_index) keys, checks catalogue
hashes, reconstructs physical F8/C8 crowding from complete scenes, and builds
the retained disk-emulator pairs without new truth cuts. Matching features
are checked against frozen development contexts where available. Generation
uses one C8 Bernoulli per input and one F8 draw per usable row at(0.02,0),
then disk transport and independent new measured0.6arcsec/25.8 cuts. No
replacement or top-up; both unusable and cut-failing counts are retained.

Inference retains the original weighted20.23m prior and evaluates at the
failed actual run's fixed expansion point, reusing its verified population
normalization. It compares generated observations, actual observations of
identical retained IDs, and original actual20k moments across the complete
draw ladder. Original four-row density groups are preserved using explicitly
excluded numerical padding at nonretained IDs. Importance draws retain
original row IDs. The matched selected input cohort is not population
closure; no shear-bias acceptance or new optimizer is implied.

Frozen active root: inference_v39_constgold20k_matched_mock_20260927_v2.
Preparation16729151, full tests16729152, generation16729153, replay16729154,
four workers16729155[0–3], comparison16729156 are chained by afterok with
invalid-dependency cancellation. Max4 GPUs; no other GPU jobs running at
submission. Initialv1 preparation16729136 and downstream16729138–41 were
cancelled before generation to preserve original density groups following
selection; logs retained. Its15 focused tests and316 full-suite tests passed
(3 optional skips). v2 reruns tests including the padding invariant. Ruff
and syntax checks pass; login resource snapshot retained, heavy preparation
and inference scheduled. See the
[matched-input protocol](V39_MATCHED_INPUT_MOCK.md#controlled-curvature-comparison).

v2 startup:16 focused tests pass; full-suite16729152 passes319 tests with
1 optional skip in26.80s. Preparation16729151 completes all100 cases in
108.8s with zero missing identities and7980 cached target-role feature
replays; reconstructed intrinsic ellipticities match recorded components
to7.50e-16. Generation16729153 completes in10.2s:1402 unusable inputs,
18598 usable draws,2284 measured-cut failures,16314 retained mocks, no
replacement.134 mock-bright rows and134 actual-bright rows share the
retained ID set.3686 numerical padding rows are explicitly excluded.
Replay16729154 passed exact equality of all ten arrays on128 original-group
rows in233.6s of Python execution. All four inference workers16729155[0–3]
then started. Comparison16729156 depends on their success; unlike the
stopped optimizer it records indefinite curvature as a diagnostic outcome.
Scheduled status audit16729189 saves a snapshot about two hours later;
it does not wake an agent or notify the owner.
Production startup verified for all four workers: first4/1000 likelihood
rows emitted in each original5000-row block, no failures. Long-run polling
stops after this startup check; CPU comparison remains automatically chained.

The exact actual20k cohort has9979 structural target-role rows and10021
outside that role; V3.9 development used the target-role parent. This is
verified from original per-case generator counts, not inferred from shape
flags. Retained mock counts are8118 and8196 respectively. Saved
input_role_audit.json and documented this population distinction without
applying a new cut or claiming it explains the failure.

## 2026-09-27 — Actual20k evaluated completely; first solve rejected negative curvature

Owner requested status at09:39 CEST. Slurm and saved receipts show all four
16726749 workers stopped at01:37 CEST after52m15–24s. Every5000-row worker
output and all20 normalization shards completed successfully. The reducer
rejected total observed-information eigenvalues(-2387169.1061,3445424.7863)
at the actual raw-mean centre(0.00983605988,-0.00029219982). The other
workers exited because worker0 stopped. No first estimate or bright update
was produced; all input, likelihood moments and normalization are preserved.
The scheduled CPU audit16726751 correctly recorded the stopped state, but
did not notify/wake an agent, as already documented.

Added scripts/audit_v39_actual_first_pass.py for a read-only saved-moment
audit: receipt/coverage/max-draw checks, curvature by magnitude, nested draw
counts, normalization contribution and influential-row diagnostics. It does
not drop observations or repair the solver. Ruff passes. Initial audit job
16728958 failed on importing SBSI from the external script location; fixed
the launcher to use this run's pinned inference runtime, then submitted
CPU retry16728966. Updated external jobs.json and
[actual-run status](V39_ACTUAL20K_BFGS.md#status-on-september-27-stopped-at-the-first-solve).

Retry16728966 completed in28s. All input/output hashes, row coverage,
finite moments and primary/max-ladder equality checks pass. The first-pass
matrix is indefinite at all six draw counts512–16384. Excluding the
normalization contribution algebraically leaves minimum eigenvalue
-2360166.8824, versus-2387169.1061 including it. The135 measured-mag<20
rows contribute-2239340.4103 along the negative eigenvector (93.8% of net
negative curvature). The774 mag20–22 rows contribute-216479.1020; both
fainter bins contribute small positive totals. Bright zero-view likelihood
relative-error diagnostics: median0.09066,90th percentile0.68966; these
are not shear errors. The same-proposal ladder cannot distinguish true local
non-concavity from correlated numerical integration error. Preserved
first_pass_audit.json; no cuts, solver replacement or GPU restart applied.
Next diagnostic is a safeguarded likelihood profile plus targeted bright
sampling checks, rather than treating this as a converged bias result.

## 2026-09-27 — Actual ConstGold20k input and five-bright-update run submitted

Owner requested the same current inference procedure on actual measurements
instead of model-generated mocks. Reused the first20k of the existing
randomly ordered uniform500k ConstGold plus-leg sample, seed20260914,
cases40–139, g=(0.02,0), after measured radius>0.6arcsec/MAG_AUTO<25.8
and disk usability, without truth-property cuts. Parent input hashes and
matching audit are preserved; this is a new inference on actual measurements,
not reuse of old V3.6 inference features. Pinned V3.9 F8/C8/disk and restored
20.23m prior are unchanged; new7.78m bright extension remains excluded.

Added scripts/v39_image_input.py and tests/test_v39_image_input.py, rejecting
misaligned/duplicate IDs, truth-filtered parents and invalid/cut-boundary
measurements. Prefix order is preserved; measured interior log disk gap is
derived without clipping. Extended scripts/run_v39_bright_bfgs.py to establish
a fresh full20k pass at this sample's raw mean, then five fresh bright-only
BFGS updates from its own first estimate. No mock centre or saved moment
initializes the scientific chain. Initial/fixed-base hashes, full worker
coverage and original four-row density groups are checked.24-row retrieval
and proposal batches require exact numerical replay on actual observations;
that replay's cached reference centre is excluded from the science result.

External root: inference_v39_constgold20k_bright20_bfgs_20260927_v1.
Init16726746 passed38 focused tests in24.26s, verified20k rows with zero new
drops, all100 cases,135 bright rows and536 evaluated rows including401
padding. Full-suite16726747:316 passed,1 skipped in44.53s. Ruff and syntax
checks pass. GPU replay16726748 passed exact equality on all ten arrays for
248 actual observations in5m05s. All four persistent workers16726749[0–3]
then started and emitted their first pass00 normalization masses without
failures. The afterany dependency on completed mock16721029 plus afterok
stage dependencies preserved the four-GPU ceiling. CPU status audit16726751
saves a snapshot about two hours later; it is not an agent wakeup or owner
notification. Estimated duration about
three hours from GPU startup, subject to actual throughput. See the
[actual20k protocol and limits](V39_ACTUAL20K_BFGS.md#numerical-procedure).

The earlier mock five-update run completed: all16721029 workers exit0,
wall1h37m40s. Final g=(0.0205598964,0.0009044658), conditional row
SE=(0.0003832200,0.0003184539). Two consecutive small steps establish
iteration stability, but paired8k/16k changes remain(-0.0001270075,
-0.0000891862), SE=(0.0000858462,0.0000830116), failing draw stability.
Preserved all results and the frozen mock runner. No0.3% acceptance;
actual-case overlap, single positive shear, faint carry and conditional
uncertainty limitations are explicit. Updated doc/V39_BRIGHT20_BFGS.md,
doc/V39_ACTUAL20K_BFGS.md and the inference guide.

## 2026-09-27 — Bright20 BFGS fourth update is small; fifth iteration running

Owner requested status at00:24 CEST. Read all saved iteration receipts,
Slurm accounting and all four worker logs for
inference_v39_mock20k_bright20_bfgs_20260926_v1. Four of five requested bright
updates completed; workers16721029[0–3] remain healthy. Fifth iteration has
12/20 normalization shards complete. Recent fresh iterations take24minutes;
estimated remaining time10–15minutes, not a completion promise.

g1 sequence:0.02076256 ->0.02047071 ->0.02056567 ->0.02055958. Latest
conditional SE0.00037764; g2=0.00090159 ±0.00041802. The latest step
(-6.09e-6,-1.17e-5) satisfies the small-step test once, with two consecutive
passes required. Current relative g1 offset is+2.798% ±1.888pp. Paired
8k-to16k change remains-0.644% ±0.424pp and fails numerical draw stability.
No0.3% acceptance or bright-carry validation; no scientific settings or code
changed. Updated [intermediate results](V39_BRIGHT20_BFGS.md#progress-at-0024-cest-on-september-27)
and refreshed external status. Existing audit16721087 remains scheduled.

## 2026-09-26 — Five measured-mag20 BFGS updates from the first full pass

Owner requested five bright-only iterations at measured magnitude<20, then
explicitly corrected the start to the original first pass rather than the
later BFGS proposal. New continuation starts from Gaussian g=(0.02508452413,
0.00267601819), uses the original pass00 observed-information solver state,
and carries faint numerator moments from its fixed raw-mean evaluation
point(0.01374048162,-0.00264758006). It does not initialize from full-pass01's
BFGS proposal. All20000 observations and the original measurement cuts remain;
only142 measured-bright moments are recomputed. Four-row numerical groups
require560 evaluations with418 padding rows discarded from the replacement.

First bright iteration reuses exactly matching pass01 bright evaluations and
normalization, while every other row is carried from pass00. This avoids a
redundant GPU pass and yields(0.02076255641,0.00075839956). It counts as the
first of five requested updates. No later full-pass faint moments enter the
hybrid. The remaining four iterations refresh full-prior64-QMC normalization
at every centre; carry replaces normalization derivatives on every row.
Keep the original Gaussian-prefilter sampler, K1024/M16384, h0.001, seed8701
and model/prior identities so this isolates the BFGS/brightness change.

Added scripts/run_v39_bright_bfgs.py, scripts/v39_bright_math.py,
jobs/job_v39_bright_bfgs.sh and tests/test_v39_bright_math.py. Existing
historical BFGS algebra is reused, with strict secant/solver checks. New
analytic tests cover nonquadratic normalization and cross-component carry;
CPU init16721013 ran20 focused tests, all passed in15.30s, then verified
source hashes, coverage and matching saved centre and combined the cached
first iteration. Ruff, Bash syntax and whitespace checks pass. Resource
snapshot saved externally; heavy work remains in Slurm.

External root: inference_v39_mock20k_bright20_bfgs_20260926_v1. GPU replay
16721017 must exactly reproduce all10 retained arrays for560 original-group
rows before four persistent workers16721029[0–3] may start. Dependencies
and array throttle preserve4 GPUs total. Five updates are requested; no
early success declaration, truth-based stopping or0.3% acceptance. Carry at
threshold20 and finite-draw/normalizer errors remain unvalidated. Full
[protocol and limits](V39_BRIGHT20_BFGS.md#owner-instruction-and-starting-point).

The earlier ranking probe16720780 completed in9m35s. Both centres exactly
replayed the228 original rows; whole-prior ranking retained identical global
draw IDs and q agreed within5.43e-19. Median top-K overlap was0.885 on128
random rows and0.757 on100 stress rows. It did not consistently reduce weight
relative error or8k-to16k score sensitivity, so it is not substituted into
this continuation. This probe is not a population-bias result.
Replay16721017 started on th-cl-nv01 and emitted4/560 original-group rows.
Scheduled saved-status audit16721087 replaces cancelled old audit16720572;
it runs after two hours but does not wake the agent or notify the owner.
Replay16721017 subsequently passed all ten arrays exactly on560 rows in5m16s;
four workers16721029[0–3] are released from their dependency and awaiting
scheduler allocation.
Production startup followed through: all four V100 workers16721029[0–3]
started on th-cl-nv01 and emitted their first full-shard normalization masses
for bright iteration2, matching the earlier zero-shear masses. No failures.
Yield while the long normalization/inference chain proceeds; audit16721087
is scheduled, and further results must be read from the new root.

## 2026-09-26 — Correct historical recovery; stop unstable Newton chain and audit saved scores

Owner challenged Gaussian g1=0.02508452 ->0.01102622. Reviewed archived
cont.205,342,345,346,369–370 and read the later complete-likelihood100k
chain.json. The earlier recovery omitted material features: cont.370 used
whole-prior Gaussian top-K, normalization-aware measured-m<22 carry and BFGS
score-root updates; cont.205 used a likelihood-increase line search. Current
plain Newton with a131072-location prefilter reproduces neither procedure.
The historical100k final g1 is0.0199264172 ±0.0004137699 for injection0.02;
its models/cuts differ and it is not a0.3% certification.

Cancelled owned workers16715922[0–3] after2h24m52s rather than continue blind
Newton passes; preserved both complete passes and partial third normalization,
recording external intervention.json without changing result receipts. Added
scripts/audit_v39_recenter.py for hash-checked saved-moment decomposition,
normalization-aware carry and reuse of the historical BFGS implementation.
CPU job16720722 completed in24s. It exactly reproduces both saved estimates,
verifies alignment and final-ladder equality, and checks the all-row quadratic
carry invariant. Gaussian secant curvature is2.112 times its initial local
curvature;967 measured-bright rows explain91.92% of absolute projected score
departure from linear carry and86.62% of initial directional information.
The saved normalization term shifts the second estimate by only4.53e-5.

BFGS applied to the same full Gaussian scores proposes g1=0.01916557; using
normalization-aware bright/faint carry proposes0.01918260. These are proposed
centres, not a converged new chain or demonstrated accuracy. The paired
observed-Newton carry/full difference is0.00015457 ±0.00018492, insufficient
to accept the shortcut at0.3%. Distance-only full-score BFGS proposes0.01854994.

Added scripts/probe_v39_historical_ranking.py; one-V100 job16720780 compares
whole-prior top-K with original prefilter on128 random rows plus explicitly
post-hoc stress groups at both saved centres, requiring exact saved replay
of the original four-row groups. No model, prior, density batch or proposal
seed changes. GPU probe is pending validation, not a production-chain restart.
Ruff passes for both new scripts. Updated
[audit and limitations](V39_RECENTER_20K.md#historical-procedure-correction-and-saved-moment-audit)
and doc/INFERENCE.md with the missing historical procedure. Next: read paired
ranking diagnostics, then validate a safeguarded/BFGS bright-carry recovery
separately from sampler convergence; no truth-directed offsets or acceptance.
Probe startup verified on th-cl-nv01,228 total rows: original Gaussian replay
emitted4/228 progress. Existing one-shot audit16720572 will save updated job
and intervention status; it does not wake the agent or notify the owner.

## 2026-09-26 — Original-prior 20k first pass completed; recentering in progress

Owner requested status at 22:10 CEST. Read scheduler accounting, GPU replay
receipt, full pass_00/result.json and all four worker logs for the original-prior
recenter study. Initial normalization and assembly completed; preflight16715921
passed exact replay for both arms on136 rows with identical global draws and
20.41 GB peak allocated memory. Workers16715922[0–3] are running without
failure receipts. First pass completed20k; second pass completed12/20 paired
checkpoints and its fresh full-prior normalization.

First-pass Gaussian16k g1=0.0250845241 ±0.0016664645 conditional SE;
distance16k g1=0.0235238099 ±0.0014268882, for injection0.02. Relative offsets
are +25.42% ±8.33 pp and +17.62% ±7.13 pp. These are intermediate
one-step estimates; recentering has not yet established closure. Both arms
fail the declared paired8k-to16k stability screen at this centre. Updated
[progress and limitations](V39_RECENTER_20K.md#progress-at-2210-cest-on-september-26).
No scientific settings or running code changed; validation is receipt/log
inspection and documentation whitespace checking. Await subsequent complete
passes; do not compare this fresh restored-prior mock as paired with the old500k.

## 2026-09-26 — Launch20k recovery of the iterated pipeline with pre-extension prior

Owner authorized recovery of the successful historical inference setup on20k
mocks, then requested removing extra bright priors. Cancelled expanded-prior
preflight16715903 before production. Asked whether to remove only the new
7,780,811 bright atoms or also the earlier231,221 certainty-stratum additions;
proceeded after the optional clarification interval with removal of the new
extension. Restore the previous20,231,221-atom prior and its original weights,
not renormalized expanded-prior weights. Generate a fresh20k matched-prior
mock, seeds202609281/282/283. Models, physical likelihood, measured0.6arcsec/
25.8 cuts and absence of truth-property cuts are unchanged.

Recovered historical fixedpoint_chain2 receipt binds K1024,131072-candidate
Gaussian reranking, tilted-stratified delta0.1/temperature1, seed8701, h0.001,
full2D steps, float64 proxy scores, retained512–8192 ladder and successive
estimate centres. New study adds16384 as the predeclared primary and a paired
distance-only arm. Both arms use the Gaussian16k chain's centre. Up to4 full
passes; fresh full-prior64-QMC normalization and state-dependent probabilities
at every centre. Exact current density groups of4 replace historical128 to
respect the prior numerical replay finding. No hybrid linear carry is used.

Added scripts/run_v39_recenter.py, scripts/v39_recenter_math.py,
scripts/prepare_v39_reduced_prior.py, two jobs/job_v39_recenter*.sh launchers,
and tests/test_v39_recenter_math.py. Generalized the study driver to an explicit
prior shard count (default28 unchanged) and checked reuse of unsheared reduced
prior coordinates. Runtime/library/model files are reused by pinned hash;
new drivers/protocols are frozen externally. Existing study artifacts remain
immutable. New [protocol and limitations](V39_RECENTER_20K.md#paired-design-and-stopping).

Validation: Ruff, Python syntax and shell syntax checks pass. Full CPU tests
16715901 and16715915 both completed308 passed/1 skipped. New tests cover joint
quadratic recovery from a distant centre, paired covariance cancellation, and
rejection of indefinite information/unsafe steps. GPU preflight is required
before production: exact all-array replay and candidate/draw equality on136
rows, shared global draws across arms, and15% memory headroom.

Submitted dependency chain: restoration16715917[0–3] ->mock16715918
->initial normalization16715919[0–3] ->assembly16715920 ->preflight16715921
->four persistent inference workers16715922[0–3]. No more than4 GPUs at once.
Prior-restoration workers started on th-cl-1024us01; dependent stages pending.
No numerical inference result yet. Monitor first progress and stage transitions;
report iteration, sampling and normalization limits separately, without a0.3%
acceptance claim.

Startup follow-through: all restoration tasks completed successfully in122–125s;
20,231,221 rows and mass1.0000000000000029. Mock16715918 completed all20000
observations (67.8s driver time), with no invalid retained draws and2066
selected true-r>=26 objects. Raw observed mean centre is
(0.013740481623385736,-0.002647580061745958). Initial normalization16715919[0–3]
started on four V100s at th-cl-nv01; all four emitted their first full-shard
node masses without error. Saved-status audit16715934 is scheduled after two
hours; it writes an external snapshot, not a Codex wakeup or notification.

## 2026-09-26 — Recover historical mock closure and correct one-step interpretation

Owner recalled earlier unbiased mock results. Reviewed archived WORKLOG_FULL
cont.174–175,204,259,338,341–342,346,354 and the cross-worktree inference
record. The relevant positive evidence is finite-shear profile recovery and
Newton recentering: the older25000-object full chain changed g1 from0.022960
to0.01906560 then0.01958117 (truth0.02; final conditional SE0.00061195).
Read both saved chain result JSONs and independently checked the saved
five-pass20000-object hybrid chain, which settles at0.01953533 ±0.00071477.
The old8million-object uncut V1 shape-flow result was+0.034% ±0.252%; selected
claims in that entry were subsequently corrected for normalizer uncertainty.
The million-object cont.259 result was explicitly unconverged, and the
complete-likelihood cont.354 closure had only200 objects with69.53pp error.

Current500k V3.9 code and result receipt confirm a single Newton update from
(0.01303596,-0.00261937), not a recentered likelihood fit. The previous status
answer overemphasized atom draws without recovering this established
linearization failure. Draw sensitivity remains real, but it does not isolate
the cause of the endpoint error. Older profiles/chains omitted current
measured selection and disk response; their successful recovery cannot be
transferred as a V3.9 acceptance result. No historical finite-catalogue error
is added to current uncertainties.

Added [historical recovery and current diagnostic implications](INFERENCE.md#historical-mock-recovery-and-the-v39-one-step-limitation).
Next diagnostic should separate recentering, draw-budget, and normalization
effects on fixed existing mock rows, with fresh normalization stencils at new
centres, common random numbers, and stopping independent of injected truth.
Validation: stdlib JSON receipt reads, source inspection, and documentation
whitespace check; no scientific computation, new jobs, or archived restarts.

## 2026-09-26 — Complete500k model-mock inference; draw ladder remains unconverged

All500 partitions/500000 measured-selected observations are complete. Persistent
array16706357[0–3] completed successfully:262000 remaining observations in
6h39m49s wall time (September26 11:42:26–18:22:15 Europe/Berlin). Workers handled
66000/65000/66000/65000 rows; median1000-row checkpoint360.07s. Combiner16706358
completed successfully in52s at18:23:36, verifying original and amended execution
receipts, output hashes, row coverage, and scientific identities. Immediate
completion audit confirms exactly262 distinct optimized partitions and500 total
receipts, unchanged scientific identity and the pinned runtime amendment.
No study jobs remain queued/running. External jobs/status receipts updated.

At the fixed primary M16384: g1=0.020891396494 ±0.000437371819 and
 g2=0.001622446128 ±0.000220339621 (reported robust1SE). Injected shear(0.02,0).
Relative g1 error100*(g1/0.02−1)=+4.456982% ±2.186859 percentage points.
This is fixed-shear model-closure error, not an independently calibrated
multiplicative-slope measurement; see CONVENTIONS.md §7. Measured radius>0.6arcsec
and magnitude<25.8, no truth-property cuts; V3.9-like models and expanded weighted
prior, original fixed seeds, one Newton update and64-draw QMC normalization.

Full draw ladder (relative g1 error% ±1SE pp):
512:−35.380935±0.796048;1024:−31.042642±0.833733;
2048:−25.626496±0.922614;4096:−19.323900±1.089082;
8192:−9.035359±1.476396;16384:+4.456982±2.186859.
All information sums are positive definite, but the largest two rungs differ
by13.492342pp. g2 also remains visibly nonzero. Completion is computational
success, not scientific closure or a robust0.3% result. More observations did
not establish convergence at the retained draw budget. Diagnose finite-draw
score/curvature and the one-step/normalizer approximations before interpreting
another larger observation sample. No new experiment was launched on this
status request. Updated doc/V39_MODEL.md with the completed result and limits.

Source: external500k v2 `result.json`, persistent worker receipts,
`logs/v39_persistent_16706358_4294967294.out`, and sacct for16706357/16706358.
Validation here is receipt/accounting consistency; the already-successful
strict combiner was not rerun and no scientific outputs were modified.

## 2026-09-26 — Resume262k unfinished rows on four persistent V100 workers

Accepted validation16706278:128-row atom search,24-row tilted proposal blocks,
and original4-row likelihood groups. All10 arrays on1032 held-out/stress
observations replay **exactly** (zero maximum difference), as do all candidate
indices, global draw indices and probabilities. Peak allocated26,017,293,824
bytes=24.23GiB, reserved24.55GiB on31.73GiB V100s. End-to-end benchmark including
proposal preparation and uncached response evaluation:2.951926 observations/s.
Four such workers project approximately6.2h for262k remaining rows; use6–7h as
an initial conditional estimate, not a completion guarantee.

Production array16706357[0–3]%4 uses four32GB V100 allocations on inter, one GPU
per persistent job, a shared queue, and atomic1000-row checkpoints. Claims and
checkpoint publication are atomic; same-job Slurm requeue reclaims unpublished
work. Array starts after accepted replay16706278. Full tests16706292 completed
305 passed/1 skipped; Slurm had already aged that job out of dependency tracking,
so its successful accounting/log receipt was verified before submission.
Combiner16706358 waits afterok for all four workers, checks the pinned execution
amendment/validation on every new checkpoint, then invokes the original strict
500-partition coverage/hash/scientific-identity combiner.

Original238 completed partitions, models, mock, prior, all seeds, K1024/M16384,
selection normalizer and all six stencil-prefix moment ladders remain intact.
Original driver/protocol/runtime hashes were rechecked unchanged. No larger
density batch was accepted. External `batch_optimization/amendment.json` SHA
`a6c5b89be71d5377dc0b71defc27147ea54fcd22bfb03514a5a3a10e2a2e6df9`
pins the additional producer and exact-replay receipt. `jobs.json` now points to
the persistent array/combiner and records the original cancelled array separately.
Saved-status job16705939 is rescheduled for two hours after submission; it writes
a scheduler/data snapshot, not a Codex wakeup. Startup verified: all four jobs
are running and independently processing partitions238–241. Each reached176–180
observations without errors; warm density throughput is2.8–2.9 observations/s
per worker. Allowing proposal setup, approximately7h remains from this check.
The scientific run continues; final500k bias/convergence result is still pending.
Updated `doc/V39_MODEL.md` documents the accepted execution and rejected density
batch changes. Long-run polling stops after this successful startup check.

## 2026-09-26 — Proposal-only batching replays exactly; retain four-row densities

Job16706245 tested128-observation candidate searches and32-observation tilted
proposal blocks, with the original4-observation density groups. All10 stored
moment/diagnostic arrays on1032 observations replayed **exactly**, including both
boundary groups; all candidates, draw indices and probabilities also matched.
Observed throughput2.9028 observations/s includes proposal preparation and
uncached response evaluation. GPU peak allocated32,062,547,456 bytes (~29.86GiB)
left insufficient15% headroom, so that deployment was rejected despite exact
numerical replay. Preserved source snapshots and `proposal_validation_16706245`
receipts/arrays remain under the500k root.

The final candidate keeps128-observation search blocks but reduces tilted-CDF
blocks to24. Active helper/validator sources are frozen externally with a24
suffix to preserve the already-running32-block audit. Validation16706278 is in
progress. Current worker requires the exact replay receipt and15% allocated
memory headroom, preserves original1000-row checkpoints, and can reclaim its
own unpublished checkpoint after Slurm requeue. Other workers never steal a
live claim. Four V100 workers will share the262 unfinished partitions238–499.
AGENTS.md records the owner's four-GPU limit for this continuation and the
required four-row density grouping. Latest full-suite16706292:305 passed,
1 skipped; Ruff and shell syntax checks pass. Production switch awaits the
24-block validation receipt; no larger density batch is accepted.

## 2026-09-26 — Reject larger density batches; validate separate proposal batching

Controlled V100 replay16706111 reproduced the original four-row groups exactly,
but density batches8–192 changed float32 likelihood finite differences even
though candidates and full global draws/probabilities replayed exactly. Initial
comparison16706064 mixed V100 with A40 checkpoints;16706065 also regrouped the
extreme rows. These were confounded and are retained as failed diagnostics.
The controlled check uses original V100 checkpoints and intact four-row groups.

Independent1024 ordinary observations234000–235023 plus the8 original-group
rows around86174/112155 were checked in jobs16706174(batch32),16706207(128),
16706206(160),16706212(192). Before those runs, declared a maximum projected
shear shift including2SE of0.003 percentage points (1% of the0.3% bias target).
Using the238k completed information matrix to map paired moment differences,
all four batches gave an estimated g1-relative shift approximately+0.097959pp
with paired SE0.068034pp (upper bound0.234027pp); g2 normalized to0.02 also
failed. This is a finite-sample numerical sensitivity estimate, not a measured
change in the full500k answer. None is accepted. Exact rows, resulting arrays,
hashes and statistics are saved in `batch_optimization/validation_*.json/.npz`.

GPU distance matrices alone require26.71GiB at256 and53.43GiB at512, so
16706180/16706181 exceeded V10032GB memory;1000 probe16706196 was cancelled
once that limit was established. Warm128-row speed comparisons benefited from
reused response-cache entries and must not be advertised as production speedups.
The1032-row validations also warm their first chunk; actual new-checkpoint timing
is required before updating the production ETA.

New `v39_batched_proposal.py` precomputes candidate sets/draws in128-row search
blocks, preserves original global RNG IDs, then serves those exact arrays to
the unchanged four-row likelihood groups. Frees unused search/CDF allocations
between stages. `validate_v39_batched_proposal.py` checks all saved moments and
diagnostics against original V100 checkpoints and independently checks all
candidate/draw identities and probabilities. Job16706245 is in progress; four
persistent production jobs will start only after this stricter replay passes.
The persistent worker retains1000-row atomic checkpoints and a shared queue.
Ruff/shell checks pass; latest full-suite16706226:305 passed,1 skipped.

## 2026-09-26 — Benchmark larger batches and prepare four persistent500k workers

Owner authorized larger observation batches and longer jobs, then explicitly
requested four jobs, one GPU each. New scripts `benchmark_v39_batches.py` and
`run_v39_persistent.py`, plus `jobs/job_v39_persistent.sh`, reuse the immutable
500k v2 driver/runtime and scientific protocol. Persistent workers load once,
claim unfinished1000-row partitions atomically, and publish complete checkpoints
without overwriting completed work. Shared work allocation avoids assigning an
equal observation count to unequal GPUs. Execution amendments and benchmark
hashes accompany every new checkpoint; an amendment-aware combiner also invokes
the original strict coverage/hash/scientific-identity checks.

Benchmarks16706016(inter) and16706022(cip24GB) await original tasks230/234/235.
Both compare128 fixed observations including boundary rows86174/112155 against
original checkpoints and batch4, with exact candidate/draw/probability replay
and all moment/diagnostic arrays (rtol1e-9,atol1e-8). Larger batches must leave
15% GPU memory headroom; warm throughput determines the retained batch among
passing configurations. Resource receipts are recorded on the allocated GPUs.
Validation and production switch remain pending; no claimed speedup yet.

Original queued partitions238–499 and obsolete combiner16702089 were cancelled;
running original tasks continue. No completed partition was removed. Saved
`batch_optimization/migration_snapshot.json` records the262 pending partitions
and four original running tasks at migration. The old array throttle was reduced
to2 while making room for two benchmark GPUs; production will use exactly four
persistent workers. Ruff and shell syntax checks pass. Follow this entry with
measured results and replacement job IDs after replay checks finish.

## 2026-09-26 — Compare current500k runtime with historical full-A40 run

Owner recalled approximately12h on four GPUs. Confirmed historical array
16606146 via Slurm accounting:500000 observations,20 partitions of25000,
first start September20 14:36:22 and last finish September21 01:17:39,
**10h41m17s** elapsed. Four-GPU ceiling, full A40 allocations, object_chunk16,
K1024/M16384 and the same six prefix rungs. Historical final curvature check
failed, but all500k numerical evaluations completed; its runtime is valid
for execution-cost comparison. Source: production_lru_v1/part_00/result.json
and cross-worktree WORKLOG September20 LRU/production entry.

Current16702088 uses500 partitions of1000, object_chunk4, three inter V100s
and three cip virtual A40 workers. Historical prior24m versus current28.012m
adds17% atoms to proposal scans. Models, proposal ranking and precision path
also differ; this is not a controlled kernel benchmark. Completed current
inter tasks median9.53min/1000; cip28.6min/1000. Inter median load138.9s plus
first-four/compile35.2s: about3min of each9.5min task is startup. Historical
25000-row tasks amortized this cost; the current run unnecessarily repeats
it500 times. Smaller object batches are another plausible throughput cost,
but their isolated effect requires a matched replay benchmark. The earlier
24h estimate accurately extrapolated100k setup, not an intrinsic cost of500k.

Snapshot:228/500 partitions complete after9.48h, no failures; all audit gates
passed, including original boundary rows86174/112155. Current cumulative
throughput projects20.8h inference total and11.3h remaining, conditional on
unchanged capacity. Saved runtime_comparison_20260926.json under the500k v2
root. No production jobs, numerical settings or results changed. A future
speed test should compare larger batches on sufficient-memory GPUs and
persistent workers with exact replay before changing the active implementation.

## 2026-09-26 — Preserve disk radial precision and resume fixed-seed500k

Owner authorized continuing the failed500k run. Added `sbsi/disk_precision.py`:
record log(1−|e|²) analytically from unbounded flow coordinates before Cartesian
rounding; propagate it through Möbius transport and use it in inverse density.
This is deterministic precision metadata for the same shape, not a new random
measurement or selection cut. Origin uses cancellation-free series; transport
uses Cartesian log1p only well inside the disk and the analytic gap identity
near the edge. Metadata must agree with Cartesian values to floating-point
tolerance. No clipping, sample replacement, RNG search or parameter changes.

Physical-flow/bundle sampling optionally returns the gap with exactly unchanged
physical draws and random-state consumption. DiskCatalogueLikelihood accepts
the explicit metadata column, uses the same physical-density Jacobian and
original network precision, and retains the old path for ordinary catalogues
without it. New compiled density applies only when this metadata is supplied.
Study driver now records gap metadata, validates encoded shapes, checks
forward/inverse transport and replays the saved failed second batch exactly.
Added `scripts/audit_disk_precision_mock.py` to exercise true-atom conditional
density and the full nine-node M16384 sampler on every rounded-boundary row
plus four ordinary rows before the full inference array can run.

Tests cover120-digit Decimal transport/radial identities from radius0 to80,
forward-Jacobian density at rounded boundaries, old density in the interior,
identical pseudorandom/QMC draws and RNG state, and inconsistent-metadata
rejection. Full-suite16702066:305 passed,1 skipped in30.42s. GPU preflight
16702078 passed old/new and eager/compiled density comparisons in addition to
the existing two-shear checks. Ruff passes. All old experiment snapshots and
the failed catalogue are preserved. Separate external root
`inference_v39_mock500k_distance_20260926_v2` pins the precision amendment,
changed module hashes and unchanged model IDs/seeds. Mock16702083 is submitted;
remaining preparation/inference awaits its validity/replay check.

Mock16702083 completed successfully:500000 observations from2097152 candidate
atoms, same original500k seeds, no replacements. Entire failed second batch
replays identical atoms and physical draws. Both originally invalid retained
Cartesian shapes pass the gap encoding; all encoded observations valid.
53808 retained targets have true r>=26; forward/inverse shape error4.44e-16.
New centre(0.013035957659442396,−0.002619371804842549).

Queued chain: prep16702084[0–27]%6 → assemble16702085 → boundary sampler
audit16702086 → smoke16702087 → inference16702088[0–499]%6 → combine16702089.
All use afterok gates; no GPU stages overlap. Three cip and three inter
preparation workers started. Scheduled saved-status audit16702094 for+2h;
it writes a receipt, not a task notification. Precision audit includes rows
rounded outside either before or after transport, runs original object IDs,
and must produce finite full-stencil moments before releasing inference.
See [precision amendment](V39_MODEL.md#disk-boundary-precision-amendment-for500k).
Verified all six initial workers produced finite positive normalization masses;
four reached the new-centre node. Normal preparation progress established.
Final Ruff/whitespace checks pass. Full sampler audit and500k results pending.

## 2026-09-26 — 500k mock stopped by retained disk-boundary rounding

Preflight16702041 passed unchanged density and selection checks. Mock16702042
failed in1m44s after its first68217 accepted observations: the second batch
contains two measured-selected draws whose squared shape norms round to
1.0000000000000002. Both have finite shapes and positive radius/flux.
Global zero-based selected ordinals86174 and112155 correspond to prior atoms
23945110 and3185269. These are inside the requested500k prefix, unlike the
unused boundary draws in the completed20k/100k runs. No discarded/replaced
objects, clipping or seed changes; no downstream GPU jobs submitted.

The generation guard correctly rejects the non-invertible rounded values.
The output is not a completed500k catalogue and no500k shear result exists.
Saved second-batch draws and audit under the new root's `mock/`, with
`failure_diagnosis.json` recording exact indices, values and validation.
Updated external jobs.json status and the V3.9 study doc. Diagnosis was a
small saved-array inspection; all expensive generation ran on Slurm.
The mathematical radial-tanh map has open-disk support, but float64 output
rounding loses the tiny distance to the boundary; current density rejects
those stored values. Before continuing, generation/likelihood needs a
consistent representation retaining that information, tested for density
and transport agreement. Dropping these observations would alter selection;
changing seeds until500k passes is not acceptable. The unchanged inference
runtime estimate remains about24h once this prerequisite is resolved.

## 2026-09-26 — Start authorized500k distance-only follow-up

Owner requested500k and asked whether it should take approximately10hours.
Measured100k inference elapsed4h41m22s, plus34m56s from mock start to inference
start.79 inter tasks had median9.5min per1000;21 cip tasks median39.6min.
Total GPU time was26.01h. Same-mix scaling predicts23.45h inference plus setup,
so quote about24h subject to contention, not10h. Six workers at inter speed
would need about13.2h inference with unchanged1000-row partitions.

Create separate external root `inference_v39_mock500k_distance_20260926_v1`.
Driver/runtime are byte-identical to the completed100k run. New protocol pins
500000 observations, fresh scene/detection/flow seeds202609271/272/273,
same F8/C8/emulator, expanded weighted prior, measured0.6arcsec/25.8 cuts,
no truth cuts, distance-only K1024/M16384 and every prefix rung. New mean-shape
centre requires new normalization; immutable features and proposal moments
can be reused by verified provenance. This is not a nested sample-size test.
Retained invalid numerical shapes still abort; no seed search or replacement.

Submitted preflight16702041 then mock16702042 to inter V100s. Full inference
will use500 disjoint1000-row partitions, shared six-GPU cap across inter/cip,
and strict complete-population combination. AGENTS.md records the500k scope.
No scientific implementation or estimator changes; completed100k tests and
validation remain applicable, with a new identity/model preflight. Further
submission depends on successful generation; no500k result yet.

## 2026-09-26 — Verify completed100k distance-only study

Immediate owner status check: all28 preparation tasks, assembly16699856,
smoke16699857, all100 inference tasks16699858 and combine16699859 completed
successfully. No pending/running user jobs. The2h limits avoided a repeat
of the20k timeouts; slowest inference task completed in1h04m31s. Strict combine
verified all100 disjoint1000-row partitions and pinned artifact identities.

Primary M16384 result: g1=0.019502411533783973 ±0.0007965901983862846,
relative realization error−2.487942% ±3.982951pp (one robust SE);
g2=0.000339781553263256 ±0.00030211698731576545. At8192 draws the relative
error is−14.030730%; last doubling shifts it by11.542788pp. Unconverged,
not0.3% acceptance. Full ladder and interpretation recorded in
[V3.9 completed100k result](V39_MODEL.md#completed100k-distance-only-result).
This is a fresh realization with a different mean-shape centre, not a paired
20k-versus100k sample-size comparison. No models, numerics or defaults changed;
no additional GPU work launched. Source `result.json` retains full limitations.
Validation: scheduler completion/exit codes,100 completion manifests and strict
combine success; documentation whitespace check. Next diagnostic would compare
exact scores/curvatures and inspect high-influence observations on a fixed subset.

## 2026-09-25 — Launch fresh100k distance-only V3.9 model mock

Owner requested100k using the distance-only arm. New external root
`inference_v39_mock100k_distance_20260925_v1` pins a fresh realization at
injected g=(.02,0), with predeclared scene/detection/flow seeds
202609261/202609262/202609263. Same expanded weighted28,012,032-atom prior,
F8/C8/disk models, measured radius>0.6arcsec and magnitude<25.8, no truth cuts.
This is a new realization, not a nested reuse of the inspected20k sample.
Any numerically invalid retained shape aborts; no dropping, clipping or
seed search. Mean observed shape sets the new centre; normalizer is rebuilt.

Generalized `scripts/run_v39_sampling_study.py` and its launcher to accept
an external study root, protocol-specified arms and observation counts.
Both-arm defaults remain available; prior20k runtime/script snapshots are
untouched. New protocol explicitly pins old feature provenance and coordinate
execution hashes. Reuse verified features and zero-shear proposal moments;
recompute classifier/selection nodes and assembled normalization for the new
centre. Retain K1024/M16384, all512–16384 draw prefixes, full nine-node stencil,
proposal seed8701 and one Newton step.100 disjoint1000-row distance-only tasks
will combine with hash, identity and exact row-coverage checks. Same-draw
invariance is still checked on two probe observations in each task.

`AGENTS.md` records that six GPUs across cip/inter covers this follow-up.
Ruff, Python syntax and Bash syntax pass. Full-suite CPU job16699846 and GPU
preflight16699847 submitted; mock16699848 depends on both. Further stages
will preserve a combined six-GPU cap and longer job limits. Results pending;
more observations do not fix the20k finite-draw convergence failure by itself.

Full-suite16699846 passed294 tests,3 skipped (artifact environment not set),
in42s elapsed; preflight16699847 passed
both tested shears and selection boundaries. Dependency chain:
mock16699848 → prep16699855[0–27]%6 → assemble16699856 → smoke16699857 →
distance-only inference16699858[0–99]%6 → combine16699859. GPU preparation,
smoke and inference use cip/inter with low-memory GPUs excluded and2h limits;
all stages require predecessor success, so total concurrency stays at six.
One-shot saved-status audit16699868 scheduled for +2hours (not a task wakeup).

Mock16699848 completed in1m45s:100000 retained observations,524288 prior
candidates, centre(0.012079337150125146,−0.00029905259816080903).
All retained draws are finite, positive in auxiliary outputs and strictly
inside the disk; no replacement or truth cuts.10827 retained targets have
true r>=26. One unused draw after the100k stopping point rounded to the
boundary; it is recorded in the generation audit. Forward/inverse maximum
shape difference4.44e-16. Preparation16699855 started six tasks concurrently:
three cip A40 workers and three inter V100 workers; all passed coordinate
reuse hash checks and started new normalization. Inference remains gated.
Confirmed every initial worker completed its zero-shear normalization node,
with finite positive mass, and four completed the new-centre node too.
Normal preparation progress established; yield under the long-run policy.

## 2026-09-25 — Complete V3.9 20k comparison; do not adopt distance-only

Both recovery tasks16699600_17/_19 completed in16m52s/16m19s; strict
combine16697082 passed in27s, with20 complete partitions and20000 observations.
At fixed16384 random draws, Gaussian gives g1=0.02215861 ±0.00097257;
distance-only gives0.01864592 ±0.00080721 (robust one-SE). Against injected
0.02 these are +10.79% ±4.86pp and −6.77% ±4.04pp. Paired distance-minus-Gaussian
is −17.56 ±4.41pp. These are realization errors, not established calibration
biases. Distance-only moves from−16.24% at8192 draws to−6.77% at16384, failing
convergence. Gaussian's similar final g1 points do not prove numerical closure.
Neither supports0.3% accuracy or adoption of the simpler exact-set ranking.

Recorded full draw ladder and interpretation in
[V3.9 completed comparison](V39_MODEL.md#completed-20k-comparison).
No scientific code, defaults or model artifacts changed. Result.json preserves
all20-partition hash/identity checks and limitations (one mock/seed, one Newton
step,64-QMC normalizer). Next work would diagnose scores/curvatures using
exact sums on a fixed subset before another large run. Complete result saved;
no further GPU jobs submitted. Cancel obsolete delayed status audit16699607
and save its read-only snapshot immediately.

## 2026-09-25 — Recover two timed-out V3.9 inference partitions

Owner requested immediate status. Eighteen of20 partitions completed both
arms (18000 observations); cip tasks16697081_17 and_19 timed out at one hour
in the distance arm, last logged820/1000 and848/1000 rows. The afterok gate
correctly blocked combine16697082. Inter tasks completed in15–17minutes;
completed cip tasks required29–40minutes, with worse throughput in the last
wave. No combined result exists; partial-population estimates are not used.

Preserved incomplete partition directories under external
`timeouts_16697081/`. Submitted only17/19 as array16699600%2 on inter V100s,
32GiB host memory and a two-hour limit. Both arms rerun from the same pinned
code, observations and seeds; all18 completed partitions remain untouched.
Repointed combine16697082 to afterok:16699600 and verified scheduler acceptance.
External jobs.json records recovery; the read-only status script now includes
retry accounting, and audit16699607 is scheduled for +2hours. This audit
writes a receipt, not a task wakeup. Both retries started on th-cl-nv02
and reached INFERENCE_START after ~161seconds loading. The complete20k
result remains pending.

## 2026-09-25 — Enable cip and inter for V3.9 inference

Owner requested both partitions. Updated pending tasks16697081_[1-19] in
place to `Partition=cip,inter`, generic one-GPU allocation and32GiB host
memory, retaining the single array-wide six-worker cap. Excluded8GiB cip
GPU nodes and the older met visualization GPUs; eligible cip virtual nodes
have at least16GiB GPU memory. Smoke16697080 completed both arms and the
same-draw check with peak RSS10995672K (~10.5GiB), supporting32GiB host
allocation at unchanged object_chunk4. Task0 was already running on an
inter V100 and remains untouched. Assembly16697079 completed successfully.
No seeds, partitions of observations, models, numerical code or dependencies
changed. Hardware can differ between tasks; the paired arms run on the same
GPU within each task. Scheduler accepted and verified all pending updates.
External `jobs.json` records the resource amendment; `AGENTS.md` records the
owner's partition authorization. Three cip A40-16GiB tasks started alongside
the existing inter V100 task. All loaded their inputs; two cip workers have
completed their first four inference rows, and the inter worker has reached
164/1000 rows in its first arm. The remaining cip worker is still compiling
its first batch. No failures observed. Other tasks wait on availability/quota;
results and final validation remain pending. Existing status audit retained.

## 2026-09-25 — Increase V3.9 mock-study inference to six GPUs

Owner explicitly requested more GPUs for the current 20k model-generated
study. Updated live array16697081 with `scontrol update JobId=16697081
ArrayTaskThrottle=6` (previously2), retaining its `inter` V100 allocations
and afterok dependency on smoke16697080. The20 fixed observation partitions,
paired arms, models, seeds, protocol and pinned execution sources are unchanged.
Updated `AGENTS.md` with this scoped authorization and external `jobs.json`
with the scheduler amendment. No other SBSI GPU jobs were active or pending
outside this chain. All28 preparation tasks16697078 completed successfully;
CPU assembly16697079 is running, smoke and inference remain dependency-gated.
Scheduler accepted the throttle update; scientific results remain pending.
The existing one-shot status audit16697094 remains queued.

## 2026-09-25 — Start V3.9 model-generated 20k sampling comparison

Owner requested20k observations and confirmed model-generated V3.9 mocks.
New `sbsi/v39_inference.py` implements physical0.5/1/2arcsec crowding over all
nonself neighbours within10arcsec, and the shared F8/C8 shear-state cache.
`scripts/prepare_v39_study_features.py` rebuilds features for the28,012,032-atom
expanded prior, keeping every supplied bright row, original weights and donor
environments. Emulator pairs remain the separately pinned nearest20 geometry.
No truth cuts. `scripts/run_v39_sampling_study.py` generates20k retained mock
measurements at g=(.02,0), measured radius>0.6arcsec and magnitude<25.8, then
prepares new V3.9 proposal/classifier/selection caches and compares Gaussian
and distance-only exact sets with identical random proposals/draws. Both arms
use K1024/M16384, full nine-node stencil, common normalizer and mean-observed
initial centre. Retain all prefix rungs; primary comparison uses fixed M16384,
not a method-dependent adaptive stopping count. No production defaults change.

External root: `inference_v39_mock20k_distance_20260925_v1`. Protocol and
execution manifests pin sources, models, scripts and isolated runtimes.
Features use the preserved original runtime plus the new module; inference
uses a separate copy with the corrected required base-cache geometry arguments.
This constructor issue was caught by first full-suite job16696807; replacement
16696823 passes296 tests,1 skip. Added tests cover all-neighbour support,
retention of faint secondary neighbours, self removal, scene isolation, shear
invariance of spin0 features and wrong-model rejection. Ruff/compilation/Bash
checks pass. Feature arrays16696810 and16696822 run at two CPU workers each;
GPU stages will preserve the combined two-GPU limit. Numerical preflight and
subsequent mock/cache/inference stages pending; no results or0.3% claim yet.

All28 feature shards completed, total28,012,032 atoms and prior mass
1.0000000000000024. Initial density preflight16696887 correctly rejected one
test draw outside measured selection; corrected the preflight to test retained
draws.16696925 and final pinned-source16696960 pass at shears(.02,0) and
(-.015,.01), with127 selected rows each and maximum standardized log-density
differences3.43e-4 and1.57e-4 (within declared float32 tolerance). Strict cut
boundaries and unchanged spin0 features pass. Explicitly enable the fractional
location metric as well as fractional scatter flooring; they are distinct
runtime settings. Preserve earlier preflight script/execution snapshots.

Queued dependency chain: mock16696965 → preparation16696966[0–27]%2 →
assembly16696967 → paired8-row smoke16696968 → inference16696970[0–19]%2 →
combine16696972. Every downstream stage requires predecessor success; GPU
stages do not overlap and arrays permit at most two GPUs. The mock is running.
Study details are in [V3.9 model closure](V39_MODEL.md#20k-model-closure-sampling-study).

Mock16696965 stopped on an overly broad physical-validity assertion over the
entire generated batch; downstream stages were cancelled before execution.
Diagnostic16697019, then16697024, replayed the identical seeds: one of117897
detected draws rounds to the unit-disk boundary. It is selected object51871
out of68171 in the oversized batch, outside the requested first20000. No
nonfinite or nonpositive auxiliary draws occurred. Move validation to the
predeclared first20000 measured-selected observations and record all batch
rounding counts separately. Selection never uses the validity mask to replace
or redraw an object; any invalid retained draw still aborts. Seeds, population,
cuts and prefix remain unchanged. Partial support/mock files and the failed
execution snapshot are preserved under `failure_16696965`. Diagnostic16697012
was only a launcher failure from a node-local /tmp path; replacement used the
shared project path.

Replacement chain: preflight16697071 → mock16697074 → prepare16697078[0–27]%2
→ assemble16697079 → smoke16697080 → inference16697081[0–19]%2 → combine16697082.
This retains the two-GPU cap and all afterok gates. Results pending.

Replacement preflight16697071 passed. Mock16697074 completed in1m40s with
exactly20000 selected, strictly interior, finite positive observations, no
replacement draws and no truth cuts. Initial mean-shape centre is
(0.015691824930896287,0.0017382505470351646), injected shear(.02,0).
The unused batch boundary draw is recorded explicitly; every retained draw
passes forward/inverse transport checks. New V3.9 preparation16697078 is queued
next. Full inference estimates are not available yet.

Confirmed preparation startup on two V100s: shards0/1 completed proposal
coordinates and their first two selection-normalization nodes, with finite
positive masses (~0.01288/0.01291 per shard). The mock includes2127 selected
targets with true r>=26; forward/inverse maximum shape difference4.44e-16.
Fresh caches and both inference arms remain pending completion. Scheduled
one-shot Slurm status audit16697094 for about two hours later; it writes
`status_latest.json`. No task-wakeup/automation tool is exposed in this
session, so this is a saved scheduler audit, not a promised task notification.
Final Ruff and whitespace checks pass. Yielding after normal GPU progress,
per the long-run monitoring policy; no inference default or acceptance claim.

## 2026-09-25 — Compare simpler atom-selection scores

Owner requested a simpler, more principled sampling score. Added
`scripts/compare_simple_atom_scores.py` and its Slurm launcher. GPU16696646
compares the existing floored Gaussian with nearest-location distance, a
prior/usability-weighted shared-scale kernel, and the Gaussian using raw
flow scatter. All use the existing 131072 location prefilter and fixed
metric; compare K1024/M16384 and K8192/M8192 on the three existing exact-sum
observations, both priors, and all seeds8701–8716. Only deterministic
selection changes: full-support sampling probabilities and draws stay fixed.
Also calculate exact repeated-draw variance from the full finite-prior sum,
since sixteen seeds can miss rare large importance weights. No production
change or shear-bias claim. Ruff, Python compilation and Bash syntax pass.
Results pending under `inference_simple_scores_20260925_v1`.

Startup16696646 exposed a historical artifact-layout difference: baseline
draws are stored in the original atom-map NPZ, not the extension comparison
directory. Corrected the diagnostic reader and resubmitted as16696664 to
`inference_simple_scores_20260925_v2`; the failed source snapshot/log remain
preserved. No model evaluation or scientific settings changed.

GPU16696664 completed in1m46s, peak RSS11.57GiB. All six baseline exact sets,
seed8701 draw lists, full log sums, normalized probabilities and sixteen
reference estimates replay successfully; input/runtime hashes and frozen
metric checks pass. [Report and complete metrics](figures/inference_simple_scores_20260925/README.md#results)
published with source snapshots; linked from the maintained bright-prior doc.
Nearest-location at K1024 reduces expanded numerator RMSE from
84.897%/149.155%/87.967% to2.163%/7.006%/3.421%. Exact theoretical relative
SD remains3.717%/20.124%/8.084%. Raw-scatter Gaussian improves the original
prior more, but is less reliable on expanded support. Shared-kernel weighting
adds little in these cases. At K8192 nearest-location has slightly worse
empirical original-prior RMSE than current ranking; report preserves all arms.
Only the exact-set rule changed. These selected cases justify representative
testing of the simpler retrieval rule, not default promotion or a0.3% claim.
Ruff/Bash/compilation checks pass. No pending jobs remain.

## 2026-09-25 — Trace posterior-important atoms missed by the original sampler

Owner asked which atoms were missed in the annotated original-prior maps and
why. Added `scripts/diagnose_missed_atoms.py`, GPU16696539. Scope is fixed to
the20.23m original prior, observations142230/3563/409188, K1024/M16384 and
seed8701. Replays exact lists/draws and proposal probabilities, ranks important
missed atoms through the actual float32 location prefilter and float64 proxy
reranker, and reports their physical properties, raw/floored flow scatter,
expected draw counts and miss probabilities. No model or numerical settings
change. Ruff passes; job submitted, results pending.

GPU16696539 completed; exact-list/draw/probability/mass replay and independent
re-ranking checks all pass. [Published diagnosis](figures/inference_missed_atoms_20260925/README.md#dominant-missed-atoms):
row3563 atom8865748 has90.0723% posterior mass, location rank13, Gaussian
rank1259, and61.0343% zero-hit probability in16384 draws. Row409188
atom6984342 has69.2913% mass, location rank9, Gaussian rank1257, and58.1699%
zero-hit probability. Both passed the prefilter but fell below K1024 and
were not drawn. Their e1 proposal scatter is broadened47–54fold and flux
scatter19–23fold relative to raw flow scatter. Different Sérsic profiles;
shared characteristic is bright, narrow actual likelihood under a broad
proposal. Other environment realizations of the same intrinsic properties
were sampled, explaining the smaller reached fractions. Original row142230
dominant atom is protected by Gaussian rank234 despite zero random hits.

Saved full per-atom evidence and source under the published report directory.
No settings change. Reserving nearest-location candidates before re-ranking
is a targeted next hypothesis; these examples alone do not establish a
population-wide fix. No pending jobs remain.

## 2026-09-25 — Test 8192 exact plus 8192 sampled atoms

Owner requested the8k+8k split after the bright-prior expansion comparison.
Generalized `scripts/bright_prior_extension.py compare` with explicit exact
and sampled budgets and output directory; the scheduler wrapper forwards
arguments. Preserve the original1024+16384 defaults and source snapshot.
GPU16696513 runs K8192/M8192 on both original and expanded priors, the same
three observations and all16 seeds8701–8716. Model/likelihood, prefilter131072,
metric, floors and prior weights stay fixed. Existing exact centre sums are
reused; no shear fit or default promotion.

Added `scripts/report_inference_budget.py`; dependent CPU16696518 compares
both budgets, independently replays both priors' saved draws, verifies exact
sets are nested, and checks full log sums. Report also measures exact posterior
mass and dominant-atom capture. Ruff and Bash syntax pass. GPU startup is
normal; results pending. External results: `inference_bright_extension_20260925_v1/comparison_8k8k`.

Completed GPU16696513 in1m56s and CPU16696518 in14s. All six independent
log-sum/draw replays and nested-exact-set checks pass; inspected the PNG.
[Published results](figures/inference_8k8k_20260925/README.md#results) show that
K8192/M8192 resolves these three centre-integration cases on the original prior:
relative numerator RMSE is0.00930%,0.04321%,0.00491% over16 seeds, with
99.982%,99.887%,99.978% of exact posterior mass summed deterministically.
These are RMSEs, not maximum errors or shear bias.

For the expanded prior, corresponding RMSE is1.861%,267.712%,87.365%, versus
84.897%,149.155%,87.967% at K1024/M16384. Row142230 recovers its dominant atom
and97.319% exact mass, but rows3563/409188 still miss their dominant atoms
and sum only0.656%/1.634% exactly. Larger K helps but does not resolve ranking
on expanded support; halving the sampled remainder can increase tail variance.
No default change and no full shear fit. Both priors, original results and
all downloaded rows remain preserved. Ruff/Bash checks pass; no jobs remain.

## 2026-09-25 — Download and evaluate the supplied full bright catalogue

Owner supplied FS2_bright and authorized adding all rows to the existing prior.
CPU16696367 downloaded the389.8MB feather and README next to the actual
FS2_25876 source, checked file hashes, and audited7,780,811 unique physical
galaxy IDs. Overlapping151,828 source identities have different assigned
morphologies;848,650 rows are outside the old generator domain. Retain all
new rows and explicitly report the changed prior. Details and frozen protocol:
[bright extension](BRIGHT_PRIOR_EXTENSION.md#weights-neighbours-and-identities).

Added `scripts/bright_prior_extension.py` and scheduler wrapper
`jobs/job_bright_prior_extension.sh`: independent complete neighbour contexts,
canonical target/pair rescaling checks, fixed bright mass, centre-only exact
likelihood comparison on the original three rows, and16 fixed seeds for each
prior. Uses the previously verified external runtime; baseline caches and
default models are untouched. Construction CPU16696388 submitted. Ruff and
Bash syntax checks pass. GPU stages and results pending; no shear-bias or
improvement claim. All raw/download/audit receipts are external under
`inference_bright_extension_20260925_v1`. CPU16696388 completed in2m37s
(peak2.17GiB): all28,012,032 combined atoms represented, new pair rescaling
checks passed. GPU array16696392 started with two V100 workers and an
eight-shard/two-worker throttle. Comparison16696395 depends on its success.

Startup16696392 caught malformed NPY headers: NumPy scalar shape dimensions
were serialized as `np.int64(...)`. Cancelled remaining workers and dependent
comparison. Corrected dimensions to Python integers and added file-reopen
checks; preserved the failed construction separately. Rebuild16696407 passed
in2m19s. Replacement GPU array16696409 has two workers making flow-coordinate
progress; comparison16696411 and CPU report16696424 are dependent stages.
Added `scripts/report_bright_prior_extension.py` for the paired16-seed plot,
RMSE against each exact numerator and posterior-atom ESS. Ruff checks pass.

Owner clarified that the parent distribution is intended to be the same.
Corrected the overly strong interpretation of different individual morphology
assignments: these alone demonstrate no distribution shift. Added descriptive
source-distribution and magnitude-conditioned checks in
`scripts/audit_bright_distribution.py`, CPU16696435. Its matched-generator-domain
audit is distinct from retaining all rows in the primary experiment.
Final CPU16696444 also preserves original float32 boundary comparisons and
matches the saved new-domain mask exactly. Raw magnitude/radius/n/q maximum
CDF differences are0.197/0.685/0.294/0.300pp; median n is2.198459 in both.
Old/new generator exclusions are11.2206%/10.9070%, predominantly n=6.
These descriptive checks support additional realizations of the intended
same parent; they do not prove equality of the complete joint distribution.

All eight GPU shards completed in approximately2m26s–2m38s each. First
comparison16696411 and report16696424 completed; independent log-sum and draw
replay checks passed, but source review caught an implicit proposal metric
refit: `ProposalCoordinateTable.__post_init__` recomputes fractional-coordinate
quartiles even when its public centre/scale fields are supplied. This violated
the prospectively fixed-metric comparison. Preserved those results externally
as `comparison_implicit_metric_refit` / `report_implicit_metric_refit` and marked
the initial reported sampler outcomes provisional. Freeze the derived metric
too and assert exact standardized-coordinate equality on every10000th baseline
atom. Corrected comparison16696478 and report16696481 reuse all valid density
evaluations; no model or target likelihood changed.

Corrected comparison16696478 completed in1m30s; report16696481 completed in9s.
All three independent log-sum/draw-replay checks pass, the original seed8701
baseline is reproduced, and the fixed metric identity assertion passes.
Final [results](BRIGHT_PRIOR_EXTENSION.md#results-and-decision): relative
centre-numerator RMSE over16 seeds changes0.000311→0.848967,
0.927664→1.491547 and1.013688→0.879673. Posterior atom ESS improves modestly
1.15→1.64,1.22→4.73,1.74→2.66, but integration does not reliably improve.
For row142230 an atom carrying77.62% of expanded posterior mass is displaced
from the exact list and missed by seed8701. Outside-generator-domain rows
carry only0.0381%,2.5173%,1.3497% posterior mass; not the main failure.

Published PNG/PDF, summaries, independent validation, distribution audit and
source snapshots under `doc/figures/bright_prior_20260925/`; inspected PNG.
Saved an explicit experimental weighted-union manifest externally as
`combined_prior.json`. Retain all new rows and the unchanged total bright
mass, but keep the baseline default: no representative inference improvement
or shear-bias claim. Next work should improve candidate ranking and establish
numerical convergence before a full fit. Removed only this task's malformed
temporary arrays, retaining failure receipts. Ruff/Bash/whitespace checks
pass; all jobs complete and no new jobs remain.

## 2026-09-25 — Remake sampled-atom maps with the latest tested inference proposal

Owner requested the earlier scatter plots using the current inference
algorithm. Fixed the original three diagnostic observation rows142230/3563/409188
before evaluation. Added `scripts/plot_current_inference_atoms.py` and
`jobs/job_current_inference_atom_maps.sh`. Uses the 20,231,221-atom weighted
bright-enriched prior, median scatter floor, fractional flux metric, K1024,
prefilter131072, M16384, seed8701 and existing initial shear centre from
job16631250. These are V3.6-like model artifacts, not V3.9 inference.

Snapshot commit773295d into the external `inference_atom_maps_20260925_v1/runtime`
directory: every implementation hash matches the saved reference result.
The live proposal-comparison branch only differs in its later truth-cut store
extension; its sampler is unchanged. No live worktree or prepared cache is
modified. The old plotting script chose its exact stratum by top mixture
probability; the production driver uses location-prefilter candidates.
The new plot calls the actual production candidate and draw methods instead.

Each row gets a full-prior centre-node density scan, top64 posterior atoms,
actual reached mass and estimated/exact numerator ratio, plus measured-space
and intrinsic-truth scatter panels. Nonpositive proposal flux means are
reported and omitted from the logarithmic axis. Saved log terms and draw IDs
permit independent checks. A40 job16696306 was cancelled while pending due
to unavailable resources; replacement Slurm16696310 started on th-cl-nv01
(one V100-SXM2-32GB, four CPUs,96GiB,30minutes). Resource-check and Matplotlib
skills applied; scheduler allocations bound usage rather than whole-node
resource-detector recommendations.

Completed [figures and results](figures/inference_atoms_20260925/README.md#results):
full-prior posterior mass reached is99.9247%,9.4684%,30.6568% for
rows142230/3563/409188. Estimated/exact likelihood ratios are0.999554,
0.376566,0.561436. The latter two rows miss dominant atoms; many distinct
sampled atoms do not guarantee useful coverage. These are centre-density
diagnostics, not new shear estimates.

Validation: GPU16696310 completed in9m50s. Rendering16696335 completed in16s;
independent CPU verification16696321 completed in6s, checking output hashes,
full-prior normalization, reached mass, likelihood ratios and sample/exact
membership. Row3563 reproduces the saved production count of14443 distinct
complement atoms. All checks pass. Inspected all three final PNGs; corrected
overlapping title text and painted highest-mass atoms last so minor atoms
cannot obscure them. PNG/PDF, plot-data NPZ, reports, resource/provenance
receipts and both calculation/rendering source snapshots are preserved in
`doc/figures/inference_atoms_20260925/`. Full log terms remain external.
Python syntax/Ruff, launcher Bash syntax and documentation whitespace/link
checks pass. Updated the repository map. No training, default change or new
shear estimate. These are three previously inspected failure cases at one
centre, not independent or population-level validation. No jobs remain and
no further experiments are scheduled.

## 2026-09-25 — Recover later inference tests from other worktrees

Owner requested a cross-branch check after the recap missed later inference
tests. Inspected all eight registered worktrees, branch histories, relevant
working-tree changes, and external result receipts. The newer numerical work
is on `worktree-proposal-coord-compare` (also inherited by `worktree-nocut-retrain`),
not the current `dev` checkout. Added the
[cross-worktree results](INFERENCE.md#later-cross-worktree-inference-tests)
without merging code or changing defaults/models/caches.

That branch declares v1.3-infer: K1024/16384 tilted-stratified inference with
median proposal-scatter floors and fractional flux distance. The saved disk
experiments remain labelled custom, based on v1.2-infer-16k, with original
V3.6 hashes. A weighted 20,231,221-atom bright-enriched prior gave positive
information on two separate 10k windows, jobs16631250/16631862. Pooled
g1=0.03768952 with robust SE0.01724200; injected g1=0.02. Earlier branch
entries confused the Newton step with final shear; the later entry and this
independent re-reduction include the starting centre. The r<27/10,237,320-atom
comparison (16637857/16637858) fails pooled positive definiteness.

Validation: read four result.json receipts, verified all four moment-file
SHA-256 values, and independently reduced the small saved score/information
arrays with single-threaded NumPy. Reproduced both windows, both pooled
information eigenvalues, and robust covariance from per-object influence
residuals. These checks re-read existing outputs; no new likelihood evaluation
or scheduler job. Documentation whitespace/link checks pass. The separate
engineering-review branch adds config binding and tests; sampler-night's
latest committed preflight is August30, not a newer disk inference result.

Limitations: two windows from one catalogue/one prior seed, heavy importance
weights and large robust errors; no robust 0.3% inference claim. V3.9 models
were not used. No branch was merged and no experiment restarted. Next steps:
none scheduled; preserve branch provenance in future pipeline recaps.

## 2026-09-25 — Freeze V3.9-like and close development

Owner requested wrap-up, the V3.9-like name, and thorough repository cleanup.
Pinned the existing F8 flow, primary seed20260913 C8 classifier and unchanged
physical-moment disk emulator in `configs/models_v3_9_like.json`. Artifact
SHA-256 values were verified against the external files. Added the named
preset through a shared disk-model manifest loader in `sbsi/models.py`;
V3.5/V3.6 identities and configured likelihood/estimator remain unchanged.

[The model guide](V39_MODEL.md#development-evidence) records measured-only
per-leg radius >0.6 arcsec/magnitude <25.8 cuts, no truth-property evaluation
cuts, the target-role parent, +0.020770% ±0.265152 percentage-point development
bias, seed replicate, failed broader classifier checks and absence of
independent acceptance. Copied original comparison/protocol/training/selection
receipts into `doc/evidence/v3_9_like/` and pinned their hashes in the manifest.
No training, recalibration, favorable seed selection or emulator reduction.

Moved 37 completed experiment scripts, 38 fixed-run launchers, 15 experimental
test files and five investigation documents out of the active tree, preserving
all bytes under `archive/research-2026-09-25/`. Its hash manifest also records
66 snapshots of pre-existing dirty working-tree files. Retained reusable
preparation/inference/audit code and core regression tests. Renamed model-preset
tests to `tests/test_model_presets.py`; added V3.9 identity/evidence checks and
an optional real-artifact prediction test. Removed unused circularized-feature
training helpers and the historical LRU-patch cache-identity exception: cache
reuse now requires exact scientific and implementation identity. Old prepared
caches must be rebuilt; the disk preparation driver is still V3.6-specific.

Updated README, model guide, API, conventions, inference/environment guides,
AGENTS and [repository map](REPOSITORY.md#september-25-cleanup). Added the
reusable one-CPU Slurm test launcher `jobs/job_tests.sh`. Scientific figures,
large artifacts, datasets, external research checkouts and result files are
preserved. No git reset/clean or commit was performed.

Validation: all four external model/metadata hashes match; all 161 archive
entries pass byte-count and SHA-256 checks; active Python syntax, config JSON,
local imports and maintained-document file links pass. Changed-file Ruff
and `git diff --check` pass. Final full suite, including external V3.9 model
loading and joint prediction: **294 passed, 1 skipped** in Slurm16696170 using
py31 (52.98s); the system Python run16696166 also passed294/skip1 (36.50s).
[Verification receipt](evidence/v3_9_like/cleanup_validation.json) pins source
hashes and preserved test logs.

Initial jobs16696148/16696162 exposed two test portability problems: NumPy
boolean indexing of a Torch tensor and a small random XGBoost fixture that
did not necessarily split on both conditioned outputs. Fixed the test to use
an explicit Torch boolean mask and a balanced factorial fixture; precision,
physical-prediction and exact-pooling assertions remain intact. No production
model calculations were changed for those repairs. The rerun revealed stale
shared bytecode; generated bytecode caches were removed. The reusable test
launcher now uses fresh job-local bytecode and `SBSI_TEST_PYTHON` so an inherited
system `PYTHON` cannot silently override the project interpreter.

Next steps: none scheduled. Development is closed. A future V3.9 inference
integration or independent calibration study requires a new owner request;
model naming alone does not establish robust 0.3% accuracy.

## Earlier work logs

- [Full log before this cleanup](../archive/research-2026-09-25/snapshots/doc/WORKLOG.md)
- [September20 snapshot](../archive/research-2026-09-20/doc/WORKLOG.md)
- [September14 snapshot](../archive/research-2026-09-14/doc/WORKLOG_FULL.md)
