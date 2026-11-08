# Matched constant-scene shear decomposition, 2026-11-05

## Owner request and decision

Owner requested a deep autonomous investigation after pointing out correctly
that a shared population should preserve standalone-flow error cancellation.
The preceding truth-cell maps locate an error but do not explain the excess
joint error. This study tests the original joint model on the same scenes,
shear magnitude, measurement pipeline and selection definition in every contrast.
No training, retuning, offsets, or restarted single-cut experiment.

## Four-state image experiment

Use the four existing constant scenes1000,1040,1080,1120. Split every original
input catalogue into fixed independent Bernoulli groups A/B using seed
202611050001+case in original catalogue order. Preserve galaxy IDs, positions,
intrinsic properties, noise and observing conditions. Original states ++/--
apply g1=+/-0.02 to every galaxy. Render new states +- and -+ with the groups
receiving opposite signs; g2 remains zero. The pipeline receives the original
case ID and noise seed1000*case. Render, detect, crossmatch and measure all
objects exactly as before: centred65-pixel stamps, same pixel scale and PSF.
Original data are read-only; new files live in a separate study root.

For group A, define own-group contrast as
`(F_A(++) - F_A(-+) + F_A(+-) - F_A(--))/(4h)` and opposite-group contrast as
`(F_A(++) - F_A(+-) + F_A(-+) - F_A(--))/(4h)`.
Swap own/opposite for group B. Their sum is exactly each group's coherent
response. Here F may be either a within-group selected mean (physical group
response) or the group numerator divided by the whole-scene selected denominator
(an exact additive decomposition of the overall reported response). Save both;
do not conflate their normalization. The opposite-group contrast holds the
target's own shear fixed. Own-group also shears same-group neighbours, so do
not call it pure self-response or empirically double an opposite-group result.

Use actual per-leg usability and independent own-leg radius>3px/MAG_AUTO<25
selection for simulation. The model keeps generated outputs and their own cuts;
report predicted usability and actual per-state usability separately. Also save
uncut, magnitude-only and radius-only results. Fixed truth-cell contributions
and flag/selection migrations are diagnostics with exact totals.

On the common-all-four usable cohort only, additionally split every within-group
selected-mean contrast by the exact finite-difference product/ratio identity:
`delta(mean) = [sum(Sbar*delta(e)) + sum(ebar*delta(S)) - meanbar*delta(sum(S))]/mean(sum(S))`.
The first term is the shape contribution and the last two form selection.
This requires valid shapes at both endpoints; do not extend it to missing
measurements by zero filling. For an opposite-group contrast on this fixed
cohort, model flow and generated selection are unchanged, so any model response
comes from emulator transport. Save model flow/emulator pieces separately.

## Model replay and closure

Use pinned older16-cut GalSBI-dev flow/C8/emulator. Original64 latent draws,
1024-parent blocks, seed7301+10000019*case+1000003*block. At each own shear sign,
sum emulator coefficients separately over A and B neighbours; apply the signed
sum through the original Mobius transport. Every original ++/-- count and
response numerator must reproduce the saved constant-grid outputs. No measured
selection mask replaces generated selection. Predictions can be cached before
the new images complete. Final reductions await validated new catalogues.

## Pipeline controls and uncertainty

Re-render original ++ state for case1000 in a separate control root. Require
pixel and detection/input-catalogue replay before interpreting mixed states.
Shape fitting draws a fresh unseeded RandomState per object in the original
code; shape catalogues therefore cannot be assumed bit-identical. Audit that
optimizer variability separately on existing images, including the effect on
usability and response. Do not mistake optimiser noise for a systematic.
Use whole-case paired bootstrap for final contrasts and explicitly distinguish
four-case sampling limitations, latent integration and fit randomness.

The optimizer audit samples8000 input IDs per case uniformly without replacement
from the union of original +/- detections passing radius>3px/MAG_AUTO<25,
using seed202611050100+case. Selection does not condition on old fitted shapes.
Re-fit both original images twice, changing only RandomState to an explicit
SeedSequence([20261105,case,input_id,leg,replicate]) seed in a copied function.
Streams are independent between legs and replicas to represent the original
independent fitter starts. Preserve original results and fit failures. Report
repeat changes on this matched sample, not a new full-catalogue response.

Follow-up triggered by large repeat scatter: uniform512 IDs from the case1000
audit sample (seed202611050201), both legs, replay each recorded seed0 twice
and seed1 once. Require agreement with the cached seeded result to1e-10 and
record optimizer success, likelihood, evaluation count and fitted T. Separately
inspect the16 largest vector changes from this replay as explicitly selected
stress examples, recording T, centroid and PSF parameters. These are diagnostic
re-fits with unchanged settings, not a positivity cut or a calibration fix.

## Resources and autonomous execution

New simulations use CPU Slurm jobs (eight mixed case/leg tasks, bounded to eight
concurrent allocations) and an independent one-case render control. At most two
GPUs total for model diagnostics. Preserve prerequisites and gate final reports
on validation. Monitor startup and first meaningful progress. Long image jobs
normally take hours: submit dependent checks/reductions and a one-shot status
check about two hours later, rather than polling continuously.
