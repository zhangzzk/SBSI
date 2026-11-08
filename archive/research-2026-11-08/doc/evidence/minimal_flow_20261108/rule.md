# Minimal latest-flow experiment

## Scope correction: stopped

The owner clarified that simplification concerns the training-stage sequence,
not network depth. Unfinished jobs17833512–17833515 were cancelled; partial
training checkpoints, numerical checks and source pins are preserved. No
performance-retention conclusion was produced. Continue with the
[training-stage audit](../training_stage_audit_20261108/rule.md#corrected-question).

## Owner request and scope

On 2026-11-08 the owner asks to apply Occam's razor: trim the latest model
while retaining its performance. The frozen teacher is the latest weighted
photometry-first flow, SHA256
43673e5db42e90d976d45dc7de1e54682adb0fd18bea74c6602e8dd4630c8cbd.
Its central m1/m2 are +0.55576/+0.21011%, and two-component nine-cut RMS is
0.63457%, on the 40 existing scenes with new paired-seed measurements.

Keep the classifier, output-conditioned disk-response emulator, physical
support transforms, standardizers and population definitions fixed. This is
flow compression, not the archived F8 emulator-simplification study. Do not
change production presets. No new images or training-data shape measurements.

## Predeclared first reductions

Three students, all initialized from the same teacher:

| Arm | Photometry/shape coupling depth | Extra neighbour score |
|---|---|---|
| no_score | 5/5 | Removed |
| shallow | 3/3 | Retained |
| shallow_no_score | 3/3 | Removed |

Depth reduction retains the first three alternating affine couplings in each
factor. This removes 40% of coupling networks, not a guaranteed 40% reduction
in total pipeline runtime. The no-score arm removes the appended standardized
reference-emulator coefficient from every coupling, including its learned
input weights; initialization equals setting this score to its training mean.
It still retains the original crowding inputs. At deployment it uses the base
preprocessor and no extra reference-score computation. The emulator needed
for secondary response remains. The comparison evaluator computes scores for
the teacher/score-retaining arm only and shares that setup for efficiency.

The joint photometry/conditional-shape factorization remains in this first
stage. Failed performance of the older joint checkpoint does not prove that
all joint architectures fail, but motivates starting with controlled removals
from the successful latest checkpoint.

## Training contract

Use teacher distillation on the original 46,833,127 training contexts only.
No validation or constant-shear measurement supplies gradients. The target is
the teacher's standardized unbounded four-output transport for common latent
draws, plus its finite-difference transport response for both shear axes.
This avoids a new chain of measured-target calibration losses and preserves
the teacher's inherited limitations rather than correcting them empirically.

Each update samples 1024 training rows, four PCG64 antithetic latent draws per
row, accumulated in 256-row chunks. Use the same rows/streams for all students.
Teacher and student share each latent draw across the base context and four
contexts with true ellipticity sheared by ±0.02 along each axis. Apply the
same analytic Mobius map as the coherent evaluator and retain other inputs.
The extra score, when present, stays fixed across shear legs, as in the teacher.

Loss is mean squared standardized transport error at the base context plus
0.1 times mean squared finite-difference transport-response error, averaging
both axes and all four outputs; derivative denominator 0.04. These are training
loss terms, not empirical corrections to predicted response. Fixed 4096 AdamW
updates, learning rate 1e-4, no weight decay, norm-5 clipping. All remaining
coupling weights train; physical transforms and preprocessing remain fixed.
Seeds and exact formulas are pinned in manifest/train.py. Save restart model,
optimizer and history every 128 steps, and report the fixed final endpoint.
Never choose an endpoint from calibration outcomes.

Report transport and response discrepancy and real-data NLL on 32768 fixed
validation rows. These diagnose fidelity; matching teacher transport alone
does not establish selected shear-calibration equivalence. Compression failure
at this endpoint does not prove that smaller models are impossible.

## Performance contract

Use all nine joint cuts radius>2.8/3.0/3.2 pixels crossed with magnitude<24.9/
25.0/25.1, both shear components. Keep the latest checkpoint as reference.
Use independent per-leg generated photometry/predicted usability selection,
64 portable antithetic draws, 1024-parent evaluation blocks, unchanged
classifier and full secondary-response composition. Reuse the completed paired-
seed measured reference. Define m_i=100*(Rmeasured_ii/Rmodel_ii−1), with matrix
rows=shape components, columns=shear axes. Pool scenes before separately
normalizing legs; no true-property population cuts or pasted corrections.
[Response conventions](../../CONVENTIONS.md#6-response-estimators).

Provisional noninferiority margins fixed before outcomes: at most +0.1
percentage point increase in absolute bias in any of the 18 cut/component
values, and at most +0.05 percentage point increase in their RMS. Report
signed changes too. Use paired whole-scene jackknife errors and 10000 paired
bootstrap draws. The screen uses a simultaneous one-sided 95% bootstrap
upper band across the 18 absolute-bias differences and a one-sided 95% upper
bound on RMS change. These are provisional engineering tolerances, not a
user-specified scientific acceptance standard.

First evaluate four existing pilot scenes 1000/1040/1080/1120, both axes,
all cuts. Recompute teacher predictions in the same evaluator and replay its
saved sums. The four-scene result is only a screen: no replacement or
performance-retention claim until a candidate completes the same full
40-scene comparison. Select the simplest passing candidate only after that
comparison. Examine real-data density and numerical fidelity alongside bias.
If these first reductions fail, retain the teacher and inspect where fidelity
was lost before adjusting capacity/training. This does not establish global
minimality, independent acceptance or training-seed robustness.

## Numerical gates and execution

A short GPU pilot checks all three initial architectures: forward/inverse and
Jacobian consistency, physical support, shear-map replay against NumPy, exact
zero-score equivalence for the untruncated arm, save/load identity and gradient
agreement between direct and uneven-chunk accumulation. All training jobs
depend on this gate. Each student then has one persistent GPU allocation;
completed checkpoints gate coherent evaluation.

One coherent case/axis pilot must replay the teacher's saved normalized sums
within 1e-9 and selected mass to roundoff; pooled/separate emulator predictions
must agree within 2e-14. Three packed workers then cover the remaining seven
case/axis tasks. Final report independently scalar-reduces point and leave-one-
scene-out response matrices before reporting paired comparisons.

At most three simultaneous GPUs/12 CPUs for this first chain, within the
study's authorized six GPUs/500 CPUs. Numerical batches do not change with
job packing. Training/checkpoints use the existing installed frozen runtime;
no archive imports or production source modifications. New scripts/source
hashes, model identities, scheduler receipts and failures are preserved.
Broad array17788286 remains held; dependent17788351 is untouched. The owner's
automated follow-up loop remains disabled. Monitor startup and first meaningful
progress, then yield normally for long jobs.
