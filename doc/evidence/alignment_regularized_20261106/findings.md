# Matched response-regularization continuations

## Question and experiment

Whether the Cartesian likelihood-only performance survives a response penalty
is unknown. [Protocol](rule.md#fixed-comparison) defines five fixed continuations
from seed611: NLL control, single-cut and nine-cut at weights0.001/0.01.
All share64 updates, AdamW lr1e-5, NLL batches, row/latent streams and fixed
endpoints. Full2x2 response errors are averaged per cut/component, with two
independent pair banks and smooth measured selection. Training targets use
all160 training cases. No coherent-simulation target fitting or model selection.

## Preparation and scheduling

CPU preparation17788090 failed because the cache allocated eight raw context
columns while preprocessing returns16 features. Fixed width inference from the
actual preprocessor; preserved partial outputs as prepared_failed_17788090;
replacement17788350 uses the identical data/transforms and targets. No training
weights changed in the failed attempt. Resource audit records~14TB available
on the output filesystem; arrays/targets are prepared on CPU nodes.

Training17788119 has five tasks, max three GPUs. All tasks completed after
preparation and after the previously active broad-scan tasks exited. The broad
scan was subsequently cancelled after cross-device CUDA RNG replay failures;
release17788490 was cancelled as obsolete. Exact-grid baseline17787232 waited
for training and portable-sampling verification. Training report17788251
completed, checking common baselines without selecting a winner.

Constant-shear evaluation: ten pilot tasks17788252 (one case, both axes, five
endpoints), then remaining390 tasks17788286; six GPUs maximum. Pilot/full reports
17788287/17788351. The evaluator requires each trained checkpoint's completion
SHA and exact measured replay against the original nine-cut baseline. Original
C8 and physical-moment emulator remain fixed. Two-hour bounded check17788449.

## Interpretation

A reduction in the flow-only training response loss does not guarantee a
reduction in coherent joint bias: the training catalogue perturbs the target
while the coherent test shears neighbours too. Report local nine-cut behavior,
main-cut m1/m2, density cost and paired differences against both the starting
checkpoint and NLL continuation. All existing40 scenes remain development data;
case SE conditions on fixed models. The limited64-step experiment does not
establish behavior at every strength or at converged regularized optima.


## Preparation and gradient verification complete

Corrected preparation17788350 completed0 in2m53s:46,833,127 NLL rows and
23,279,984 response pairs in160 training cases;11,703,585 rows and5,817,708 pairs
in40 validation cases. Exact nine-cut training targets are preserved. Models
run on A40-16GB with24GB host memory. Control and regularized workers verify
replay/direct gradients at~1e-7 relative error, finite losses and exact reload.

Broad-scan resource prerequisites were removed after those jobs exited. The
broad scan itself was stopped after cross-device CUDA RNG replay failures.
Portable evaluation was separately verified on A40/RTX: identical latent hashes
and physical outputs. Baseline/endpoint constant-shear comparisons all use this
new shared realization; previous values are not pasted into the new comparison.

## Completed fixed-step continuations

All five tasks in17788119 completed with exit0, as did report17788251. Every
arm passed direct-autograd versus replay gradients and exact checkpoint reload.
The [training report](training_report.json) preserves all fixed-step64 endpoints.
Each validation response score below uses the same nine-cut probe, including
for arms trained on only one cut. Lower scores are better; these are normalized
proxy losses, not multiplicative calibration biases or significance estimates.

| Continuation | Validation NLL change (nat/row) | Nine-cut validation response score |
|---|---:|---:|
| Original checkpoint | 0 | 10.2099 |
| NLL-only control | +0.018848 | 11.4363 |
| One cut, weight0.001 | +0.000880 | 3.7259 |
| Nine cuts, weight0.001 | +0.000704 | 3.8198 |
| One cut, weight0.01 | +0.002966 | 2.4406 |
| Nine cuts, weight0.01 | +0.002764 | 2.0632 |

Both penalties improve the fixed standalone response probe at modest density
cost. All64 updates clipped the combined gradient in every arm, including the
NLL control; this is the declared norm5 rule, with histories retained. These
results establish neither convergence nor preservation of coherent joint bias.
No endpoint is selected or rejected from this table.

The portable baseline's axis2 pilot17787232_1 completed exit0 in5m13s; axis1 is
progressing normally on a slower A40-8Q. All40-case, both-axis endpoint evaluations
remain queued behind the matched baseline. The original baseline must be
remeasured with the same portable latent banks before paired bias comparisons.

## Interim constant-shear comparison: 19 common cases

The original baseline completed all40 cases on both axes; all ten endpoint
pilots passed, and the remaining endpoint array is progressing without failed
tasks. At the status check194 of400 endpoint evaluations had completed. CPU
reducer17803908 completed exit0 in21s. Its optional interim phase selects cases
by the presence of completed output/receipt pairs for every arm and both axes,
before reading numerical values. Pilot and full40-case reduction behavior is
unchanged. Python AST and bash syntax checks passed.

The [interim report](interim_report.json) freezes19 common cases. At radius>3
pixels and magnitude<25, predicted usability and generated per-leg selection:

| Continuation | m1 (%) | m2 (%) |
|---|---:|---:|
| Original checkpoint | -0.2763 +/-0.4436 | -0.2416 +/-0.3482 |
| NLL-only control | -0.5086 +/-0.4422 | -0.7635 +/-0.3457 |
| One cut, weight0.001 | +0.8735 +/-0.4495 | +0.2755 +/-0.3494 |
| Nine cuts, weight0.001 | +0.8779 +/-0.4497 | +0.3929 +/-0.3498 |
| One cut, weight0.01 | +1.0817 +/-0.4495 | +0.2331 +/-0.3499 |
| Nine cuts, weight0.01 | +1.2642 +/-0.4499 | +0.3342 +/-0.3499 |

Definitions follow [Conventions](../../CONVENTIONS.md#7-multiplicative-bias):
m=100(Rsim/Rmodel-1), with fixed-model whole-case bootstrap standard errors in
percentage points. These19-case baseline values must not be directly compared
to a different40-case subset. All arms use the same portable latent realization.

The m1 shifts relative to the matched original are+1.1498 +/-0.0067pp,
+1.1542 +/-0.0069pp,+1.3580 +/-0.0080pp and+1.5405 +/-0.0083pp for the four
regularized arms in table order. These small paired uncertainties describe the
model change on shared scenes, not absolute calibration precision. The effect
persists with true usability: corresponding m1 values are+1.0776,+1.0827,
+1.2886,+1.4721%, compared with-0.0716% for the original. Generated selection
is retained in both flag modes.

Nine-cut RMS m1 changes from0.2730% to0.9620%,0.9609%,1.1408%,1.3164%; m2 RMS
changes from0.2147% to0.6135%,0.7152%,0.5438%,0.6487%. Thus these bounded
continuations improve the standalone proxy but currently worsen joint local
calibration. This does not establish that all regularization recipes fail.
All five fixed endpoints and all40 planned cases remain in the experiment;
no stopping or model selection is based on the interim results. Final
reducer17788351 remains queued; bounded follow-up check17803918 in two hours.

## Owner-requested pause of further coherent evaluations

After the localization follow-up began, the owner requested pausing earlier
evaluations. Pending17788286 tasks221-399 are held (JobHeldUser); running217-220
may finish. Existing results and the fixed planned40-case protocol are retained.
The final reducer stays dependency-pending. Automatic restore17804373 was
cancelled; these held tasks require an owner-requested resumption. The separate
alignment and same-image neighbour-response diagnostics are not paused.
