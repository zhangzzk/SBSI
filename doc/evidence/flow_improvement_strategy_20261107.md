# Improving the GalSBI joint measurement model

Later on2026-11-07, the proposed probability-loss trial and36-scene replication
completed. The [updated review](response_strategy_review_20261107/findings.md#decision-after-reviewing-the-complete-chain)
supersedes the experiment sequence below: joint nine-cut RMS improves modestly,
the component difference remains, and an exact-current-checkpoint measured-input
test is now the next diagnostic. The original proposal is preserved below.

## Status and recommendation

2026-11-07: analysis and proposed experiment sequence, not a launched training
protocol. The owner asked how to improve the model after reviewing the input
substitutions. No jobs, schedules, model identities or scientific settings change.
The broad constant-shear evaluation remains paused.

First test a more direct conditional selection objective on the existing
photometry factor, with a matched continuation and the shape factor fixed for
this intervention. Evaluate the selected shape response, not just selection
counts or a static neighbour score. In parallel conceptually, prepare a precise
test of the missing neighbour-shear selection channel before investing in a
large feature/training change. Freezing shape parameters is an experimental
control, not a claim that the shape model is correct or a permanent architecture
requirement. A failure of this candidate would not justify increasing its loss
weight until the constant-shear answer looks right.

## What the evidence does and does not establish

The [matched full-response substitution](joint_measured_photometry_20261106/findings.md#full-forty-scene-result)
shifts bias downward by about 1.0--1.15 percentage points; it does not uniformly
improve absolute bias. Original Cartesian NLL has measured-input m1/m2 of
-1.2025%/-1.4070% on forty scenes. Original weak-nine has -0.0026%/-0.7331%.
Thus the measured-input test does not establish that everything outside the
photometry generator is accurate. These original checkpoints are distinct from
the later neighbour-conditioned frozen-shape continuations.

The [four-way neighbour test](photometry_selection_factorial_20261106/findings.md#evidence-and-subsequent-localization)
finds errors +0.005217, +0.005391, +0.004562, -0.001339 for
generated/generated, generated/measured, measured/generated and
measured/measured photometry/membership. These are restricted shape-response
units, not coherent m. The large interaction supports examining conditional
associations, rather than repairing only marginal photometry distributions.

The [additional scene-information diagnostic](selection_information_20261106/findings.md#four-held-out-scenes)
supports richer conditioning for selection. However, adding that scalar to the
flow improved selection association only modestly; the existing response
regularizer worsened it again on both training and validation. See the
[own-data audit](neighbour_flow_selection_20261106/findings.md#main-finding).
This leaves optimization/objective limitations as well as missing features.

The [frozen-shape experiment](frozen_shape_neighbour_20261107/findings.md#completed-frozen-shape-comparison-on-both-axes)
already rules out treating a better static selected-neighbour score as sufficient
evidence of better calibration. It does not rule out all photometry objectives.

The [physical neighbour replay](neighbour_crossing_response_20261107/findings.md#giving-the-model-the-actual-entrants-and-exits)
demonstrates an omitted changing-membership channel but cannot quantify its
contribution to percent-scale coherent bias. The [fit-repeat trace](primary_neighbour_replay_20261107/findings.md#why-two-optimizer-seeds-disagree)
also shows that enlarging a physical attribution experiment must account for
fitter randomness as well as the number of crossing objects. Neither explains
the mean bias by itself.

## The quantity training and validation must ultimately protect

For a cut c and a specified physical shear intervention g, let

    D_c(g) = E[U(g) S_c(g)]
    N_c(g) = E[U(g) S_c(g) e_final(g)]
    mean_e_c(g) = N_c(g) / D_c(g).

U is usable measurement, S is the size/magnitude cut, and e_final includes the
actual model composition with the emulator. The expectation is over the parent
population and measurement distribution, with the prescribed sampling weights.
On a discrete model, these are classifier-probability-weighted joint draw
moments; do not factor the shape and cut expectations into separate products.
An unusable object contributes zero mass and numerator; no shape needs to be
invented for it. Unmatched records are explicitly reported, not silently
interpreted as non-detections.

The exact antithetic target is

    R_c = [N_c(+h)/D_c(+h) - N_c(-h)/D_c(-h)] / (2h).

The derivative identity R = (N' - mean_e D')/D explains why matching counts,
flux slopes, or a selected static coefficient alone is insufficient. It is
not a replacement for the actual finite-difference estimator. Internal
cancellations remain allowed under the owner's current target-cut policy;
the component moments are diagnostics, not demands that every bin be unbiased.

The current [nine-cut regularizer](alignment_regularized_20261106/rule.md#targets-checkpoints-and-diagnostics)
uses a flow-only forward response on matched usable pairs. The additional
[neighbour penalty](neighbour_moment_guard_20261106/rule.md#additional-objective)
matches population averages of passing mass and a static coefficient. Neither
alone is the full U/selection/transport response above. Both were legitimate
diagnostics, but their improvement is not a guarantee about that target.

## First candidate: conditional cut probabilities from the flow

Use the existing neighbour-conditioned model and original training split.
For each usable training row with context x and each of the nine cuts, derive
q_c(x) = P_flow(pass cut c | x,U=1) from the same joint photometry distribution
used at inference. The target y_c is that row's measured cut indicator. Add a
per-row proper probability score such as (q_c(x)-y_c)^2 to photometry NLL.
This supervises which contexts pass, rather than only two population averages.
One observed indicator is a noisy draw, not the exact probability for its row.
It is not appropriate to force every training probability to zero or one.

This differs from the previous loss: that squared the population-mean error
and its B-weighted counterpart. Per-row scoring penalizes conditional mistakes
that can cancel in those averages. Retain the scalar neighbour feature already
tested; do not simultaneously change features, shape optimizer and objective.
Preserve the same reference response term in both candidate and matched-control
arms, so the comparison isolates the new conditional objective.

If using Monte Carlo, two independent latent banks on the SAME row give
(q_A-y)(q_B-y), whose expectation is the desired square. Using independent rows
here instead would return to a population-mean objective. A differentiable soft
cut must use the same soft measurement target and be described as a surrogate;
check exact hard cuts independently. Validate gradient estimates, integration
noise and softness sensitivity before training; do not silently differentiate
a hard indicator or ignore the finite-draw bias of a nonlinear probability loss.

Initially optimize photometry only; keep the conditional shape density fixed
at fixed inputs, and keep classifier/emulator unchanged. Retain likelihood
and response readouts on disjoint training-validation scenes. Do not use
constant-shear bias to choose a weight, endpoint, seed or number of steps.
This is a candidate experiment, not evidence that it will fix m.

## Protect shapes by measuring behavior, not only parameter equality

The current factorization is p(radius,flux|x,U) times
p(shape|radius,flux,x,U). Changing the first factor changes the selected shape
distribution even when the second factor's weights are identical.

Report conditional-shape likelihood at measured photometry, the selected shape
numerator and selected mean on each leg, and their paired primary-shear
contrasts. Preserve both axes and all nine cuts. Use actual usability only as a
diagnostic alongside the complete predicted-usability result. A frozen density
check certifies the intervention, not preserved shear calibration.

If training explicitly constrains selected-shape moments, targets must come
from compatible measured TRAINING populations and physical interventions. Use
the complete model appropriate to those data. Existing matched-usable primary
pairs can supervise a conditional primary statistic, but cannot supply missing
neighbour-only or classifier-transition labels. Do not reuse held-out coherent
results as targets, call the old flow a truth target, or assign a measured
response residual to the emulator without identification.

## Second candidate: supply neighbour-shear information where it is missing

The present scalar crowding/reference-coefficient inputs do not change under
neighbour-only shear. They cannot produce that intervention's change in
photometry or usability, regardless of optimization quality.

If a precision test shows this channel matters at the desired calibration
scale, candidate inputs are neighbour ellipticity components, relative
positions, flux ratios, sizes and profiles, summarized with explicit rotation
rules or a small permutation-invariant neighbour encoder. Test the smallest
adequate summary against the existing scalar context. Inputs must be available
from latent scene properties during inference, not from measured validation
outcomes or a known true shear label. The classifier needs compatible context
and all-parent usable/unusable training labels; the photometry flow remains
conditional on usability.

Crucially, adding neighbour information to the shape branch may duplicate
response already represented by the disk emulator. Before extending that
branch, specify which reference distribution the flow represents and which
remaining response the emulator supplies. Even neighbour-dependent photometry
can alter conditional shapes indirectly. First test features in the
photometry/classifier paths with the current shape/emulator fixed, then verify
full transport under primary-only, neighbour-only and coherent interventions.
New training labels must vary those interventions separately if we want to
identify their roles. A richer input alone is not sufficient supervision.

## Sequence, precision and decisions

1. Prepare the conditional-probability loss and verify its numerical estimator
   on training-only data. Freeze a bounded matched-control experiment before
   coherent evaluation. The immediate work can reuse existing training data;
   a large new image campaign is not a prerequisite.
2. Assess whether conditional cut prediction improves on disjoint validation
   scenes without degrading selected-shape behavior. Treat this as a diagnostic
   gate, then perform the declared coherent development comparison, preserving
   actual/predicted usability, both axes and all nine cuts. A score improvement
   with worse full bias rejects this proposed remedy at the tested setting.
3. For the missing physical channel, set an uncertainty target in full-response
   units before allocating images. As a planning scale, with R approximately
   0.9, 0.3% in m corresponds to about 0.0027 in response. This is not a promised
   achievable SE or a new acceptance rule. Existing crossing uncertainty near
   0.06 is far too large for attribution at that scale. Do not extrapolate a
   sample budget by a single inverse-square formula: crossing frequency,
   influential shapes, spatial variation and optimizer noise all matter.
4. Use a predeclared boundary-enriched probability sample plus a nonzero-probability
   random complement, including nondetections, for any enlarged physical test.
   Include both axes and retain weights. Fit repeats estimate the original
   randomized measurement's expectation; selecting a better-looking fit,
   imposing a new positive-size cut, or dropping difficult objects changes the
   measurement definition. Keep fitting work in BlendEMU.
5. Any coherent training data must have separate scene identities and provenance
   from validation and final acceptance. The forty reused scenes remain
   development evidence. Final success concerns the nine target cuts and both
   axes on fresh scenes, with adequate precision. Distant cuts and subpopulation
   cancellations remain diagnostics under the current owner policy.

## External methodological context

[Sheldon and Huff (2017)](https://arxiv.org/abs/1702.02601) explicitly distinguish
shape response and selection response. [Sheldon et al. (2020)](https://arxiv.org/abs/1911.02505)
show that shear-dependent detection matters in blended-image metacalibration.
These support checking the complete selected measurement process; their
numerical biases are not estimates for SBSI and do not establish our cause.

## Validation and execution limits

This memo is grounded in the linked completed receipts, the moment-loss source,
the factorized-flow implementation, and the physical replay input contract.
No new numerical result is claimed. Links and formulas were checked against
those sources; no tests or compute allocations are required for this proposal.
Any subsequent experiment retains the current six-GPU/500-CPU ceilings,
packed-job preference, per-case seeds and startup gates. Automated follow-ups
remain off unless the owner requests them again.
