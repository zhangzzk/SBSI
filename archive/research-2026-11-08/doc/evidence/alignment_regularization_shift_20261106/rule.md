# Localizing the regularization shift

## Questions and fixed evidence

Why do the one/nine-cut continuations improve standalone response probes while
moving coherent calibration away from the starting Cartesian flow? Does the
improved GalSBI flow recover the original measured–intrinsic alignment relation,
and does response regularization damage that recovery?

Freeze the 19 common coherent cases from the existing interim report for exact
component attribution. Use every arm, both shear axes, all target cuts and both
usability modes. No endpoint selection, retraining or changed simulation.
Keep the ongoing 40-case constant-shear comparison intact.

Owner update: the broad comparison is now paused. Preserve its outputs and
hold17788286; focused diagnostics below continue, with at most six GPUs total.

## Coherent response decomposition

Use saved per-leg mass, flow shape and emulator shape-shift sums. Their responses
add exactly using each arm's own selected denominator. Report paired changes
in flow and emulator contributions, which are changes in the joint estimator,
not standalone component errors. Split by intrinsic size with the same total
denominator; contributions must sum to the total.

For each size bin, symmetrically decompose changes in selected fraction times
conditional mean into a fraction term and a conditional-mean term. Also split
the total ratio change into numerator and denominator terms. These are exact
descriptive identities, not causal interventions that isolate selection from
shape learning. Use 10,000 shared whole-case bootstrap replicates.

## Standalone alignment and response

Re-evaluate the original improved Cartesian checkpoint and all five fixed
continuations on the same 40 original flow-validation cases. Two banks of 16
antithetic draws use portable PCG64 CPU latents, identical between arms and
shear legs. Recompute the original seven hard-cut response matrices, size/crowding
contributions, and zero-shear alignment profiles at magnitude25 and25.8.
Use own output selection and normalization. Validate measured sums against
the preserved original Cartesian evaluation. Compare the existing FS2 NLL-only
alignment profile as an independent-population reference, not a paired model
change on the GalSBI catalogue. Pin all checkpoint hashes.

Additionally split the finite selected-shape change exactly:
w1*e1-w0*e0 = (w1+w0)/2*(e1-e0) + (e1+e0)/2*(w1-w0).
The first term holds the average selection weight; the second contains changes
in selection membership and its normalization correction. Sum identities are
checked for every block. This is a finite-pair algebraic diagnostic, not a new
causal simulation or a substitute for coherent joint response.

## Resources and limits

Run CPU reductions through Slurm. Temporarily throttle coherent evaluation to
four GPUs and allow two standalone diagnostics only after one of the five
previously running coherent workers exits. Restore six after the diagnostic
array exits. The combined six-GPU cap remains in force. All40 standalone cases
and coherent scenes are development data. Case uncertainties condition on
fixed model weights; alignment is not itself a full response-matrix test.

## Reuse existing four-state image experiment

Evaluate the starting Cartesian checkpoint and the weak nine-cut continuation
on cases1000/1040/1080/1120 from the earlier four-state image experiment. The
nine-cut arm represents the owner's local-grid objective; its smaller tested
weight is fixed here without scanning mixed-state outcomes. No new images or
measurement changes. The original ++/-- states must replay the new portable
coherent sums, and measured sums must replay the preserved four-state cache.

Keep groups, +/-0.02 shears, original classifier/emulator, 64 portable draws,
cuts, own/mixed velocities, and per-state actual/common usability unchanged.
Compute own-group and opposite-group contrasts with both global and within-group
normalizations. Own includes same-group neighbours; do not identify it as a
pure target response or double the opposite-group term. On common usability,
within-group opposite contrasts must have zero flow contribution. A case1000
two-arm replay pilot gates the other three cases. Run at most two GPUs after
standalone diagnostics, retaining four for the coherent array; restore six
only after the mixed-state array exits. Four-case uncertainties remain limited.

### Numerical replay amendment after pilot

Original Cartesian case1000 passed strict historical-cache replay. Weak-nine
case1000 on A40 failed the raw-sum tolerance by0.00016172 relative to the prior
V100 cache, despite identical checkpoint/context/portable latent identifiers.
Do not change or offset any prediction. Preserve the failed job/log. A V100
device comparison was queued but the scheduler predicted a multi-day wait, so
replace that pending check with a direct coherent evaluation on the same GPU
inside every four-state worker. At every block, sum of split-neighbour responses
must match a separately evaluated all-neighbour response at1e-12. Its aggregate
coherent moments must match the four-state moments at the original2e-7 absolute,
2e-10 relative tolerance. Measured replay retains the original tolerance.

Separately record all differences from the historical hardware cache. Before
rerun, require selected-mass fractional differences<1e-7, per-leg normalized
shape/transport mean differences<1e-7, and component/total response differences
<5e-6. These bounds are in physical units; the maximum allowed response change
is about0.0006 calibration percentage points, far below the observed~1pp shift.
This is an explicit numerical-validation amendment, not a correction or a claim
that the hardware cause has already been proven. Fail if either the strict
same-device replay or bounded historical-cache comparison fails. Report the
actual observed differences, not just that they are below the bound.

The retry passed strict same-device checks but exceeded the cache mass-fraction
bound (1.92e-7 versus1e-7 at mag25). Retain that failure and do not widen the
bound again. Re-run each task on the GPU family of its historical coherent
cache: A40 tasks0/2/3/4, V100 tasks1/5/6, RTX task7. Since the owner has held
the broad array and all its active workers have finished, the remaining focused
tasks may use2 A40+2 V100+1 RTX workers after the case1000 pilot. This is at most
five GPUs, below the existing six-GPU study ceiling. No held work is resumed.

## Deep neighbour diagnostic fixed before results (2026-11-06)

Use all four existing scenes and unchanged measured catalogues. Extend the CPU
report with the previously saved exact common-usability shape/selection edge
sums. Recompute all case resamples jointly with new model sums; do not combine
SEs independently or import older bootstrap error bars. Under within-group
normalization the model opposite-group selection and flow terms must vanish.
Report the measured selection term and the remaining shape-response residual
separately for all four cuts and both target groups.

Then evaluate the unchanged, hash-pinned GalSBI-dev emulator using measured
shape, radius and flux on the common-all-four usable cohort. This is a
conditional diagnostic, not a deployable prediction or a correction. For each
opposite-group edge, hold target own shear fixed. At each measured endpoint,
compute A/B pair coefficients with that endpoint's measured radius/flux, invert
the model's total disk transport, then transport to the other endpoint's shear.
Keep coefficients and radius/flux fixed during this predicted change. Average
the forward and reverse endpoint predictions with the exact measured average
selection weights. Retain both endpoint predictions to expose anchor dependence.
Compare their shape numerator to the exact measured shape numerator and report
the separate measured selection term. Also record the infinitesimal disk-gain
approximation as a numerical/finite-transport diagnostic, not ground truth.

Use fixed intrinsic-size bins0/.15/.25/.4/.7/infinity for descriptive residual
contributions; no truth cut changes the target population. Report all unmatched
catalogue rows from the existing loader. Validate measured sums against the
old four-state cache, forward/inverse transport, finite outputs, and exact
shape-plus-selection closure. One GPU worker at a time for this new diagnostic,
alongside at most five existing workers. Pin emulator identity, source hashes,
case/bootstrap seeds, and never adjust predictions to close a response budget.

### Conditional-shape follow-up after measured-input pilot

The case1000 measured-anchor shape error is-0.000153, compared with+0.005598
using original flow draws. To distinguish measured photometry from measured
shape, evaluate both fixed flows' conditional p(shape|measured radius,flux,x)
on the same common cohort and same endpoint-average selection weights. Use
64 portable antithetic draws with seed202611061800+10000019*case+1000003*block,
blocks of2048 targets. Pair latents across endpoints and both checkpoints.
Use the same measured photometry for emulator coefficients, then transport each
conditional base-shape draw into both neighbour-shear states. Retain both
endpoint predictions and the local disk-gain approximation. Replay measured
sums against the measured-anchor output and verify the conditional sampler
reconstructs joint draws at their generated photometry. This comparison changes
shape distributions only relative to the measured-anchor diagnostic; relative
to the original generated-selection estimator it also changes selection and
photometry. Do not claim a unique causal split of those latter two effects.
Run one GPU after the measured-anchor array; case1000 gates the other cases.
