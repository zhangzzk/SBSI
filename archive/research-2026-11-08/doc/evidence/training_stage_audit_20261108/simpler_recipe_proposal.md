# One-pass factor training with a measured-photometry primary-response constraint

## Recommendation and status

Proposal after reviewing the completed experiments and executed training code,
not a launched experiment or demonstrated improvement. Keep the useful primary
response constraint, but train it on conditional shapes at measured TRAINING
photometry/membership. Train the photometry factor on its own conditional NLL,
with independent optimizer/clipping state. Both factors train throughout one
fixed run. Do not distill the latest model, introduce another sequence of
photometry repairs, or use constant-shear simulations as training labels.

The central hypothesis is that response fitting should not be allowed to
purchase a better primary-response score by distorting size/flux probabilities.
This removes a verified gradient path, not a proven sole cause of the error.
NLL-only photometry was already imperfect, so removing that path is not by
itself a claim that the entire conditional distribution will be corrected.

## Evidence supporting this particular intervention

- Measured radius/flux AND membership largely close the restricted neighbour
  response discrepancy. At fixed radius/flux within cells, generated selection
  still weights different scene configurations. Thus conditional association,
  rather than marginal histograms alone, matters.
- The own-data audit found adding response regularization worsened validation
  photometry NLL by0.005461nat/row and increased the selected-neighbour
  discrepancy0.002520→0.003148. Existing NLL-only photometry was not perfect:
  the initial discrepancy was0.003117. This is an observed training tradeoff,
  not a proof of the unique optimization cause.
- Factor swaps assigned roughly94–96% of the regularization-induced central
  response shift to the conditional-shape factor. This supports preserving its
  useful response training while testing removal of photometry distortion.
- Direct code inspection confirms the current guard draws generated size/flux,
  computes soft selection from those draws and backpropagates through both
  factors. The frozen-shape trials retained this response gradient into
  photometry. Original joint training also clipped both factors together;
  all64 updates clipped in the cited factor-swap experiment. Clipping is a
  possible indirect coupling, not an established dominant cause.
- Probability and static-neighbour moment losses already improved their own
  scores without reliably improving calibration. The proposed experiment adds
  neither of those losses.

Sources: [fixed-cell localization](../photometry_fixed_cell_20261106/findings.md#four-scene-fixed-photometry-result),
[own-data tradeoff](../neighbour_flow_selection_20261106/findings.md#which-objects-and-likelihood-factor),
[factor swaps](../neighbour_factor_swap_20261107/findings.md#final-conclusion-both-shear-axes),
[current primary target](../response_strategy_review_20261107/findings.md#training-target-coverage-audit),
[probability loss context dependence](../photometry_learning_information_20261107/findings.md#relation-to-the-earlier-comparisons).
Executed code checked: `alignment_regularized_20261106/train.py`,
`conditional_cut_probability_20261107/train.py`, and installed
`sbsi/flow_guard_response.py` (GuardResponsePopulation.contributions).

## One fixed recipe

Let t=(measured radius,measured flux), e=measured ellipticity, x=truth/context.
Retain the factorization p_theta(t|x,U=1) p_phi(e|t,x,U=1).

Photometry objective:

    L_photo(theta) = -mean log p_theta(t_obs | x,U=1).

Conditional-shape objective:

    L_shape(phi) = -mean log p_phi(e_obs | t_obs,x,U=1)
                   + lambda L_primary_measured_photo(phi).

Use measured training photometry as the condition in both shape terms. The NLL
already does so; the substantive change is in the response term. No generated
photometry or probability masks enter that term. There is no gradient from it
into theta, and independent clipping prevents the shape gradient from changing
the effective photometry update. All remaining shape and photometry weights
train throughout; this is not the previous frozen-shape intervention.

Keep the same available inputs in candidate/control from initialization. For
this test retain the existing useful neighbour-reference scalar from the start,
rather than introducing it in a separate continuation. Keep physical maps,
classifier and disk emulator unchanged. This is not an additional neighbour-
shear-varying feature or a change to the response decomposition.

## Compatible training response and target

Use original matched-usable zero/forward PRIMARY perturbation pairs and their
existing regression design. For row i, leg l, cut c, compute the measured soft
weight w_obs(i,l,c). Generate conditional shapes only at that leg's measured
photometry and context, obtaining their conditional mean e_bar_phi(i,l).

    psi_phi(i,l,c) = w_obs(i,l,c)
                    * (e_bar_phi(i,l) - measured_baseline_mean(c))
                    / measured_baseline_mass(c).

Apply the original design to psi_phi(i,forward,c)-psi_phi(i,zero,c). Construct
its target with observed shapes in the same formula, same eligible pairs,
same cuts, same weights and normalization. The exact existing measured target
can only be reused after algebra/numerical replay verifies equivalence.
Do not substitute a coherent antithetic target or silently change the sample.

With a correct conditional-shape model this estimates the observed conditional
primary statistic averaged over observed photometry. It does not directly
constrain photometry-generated selection response, classifier transitions, or
neighbour-only response. Those remain genuine validation questions. It is not
a stop-gradient trick on generated photometry: that would still let shape
weights compensate for an incorrect generated population.

Use shared shape random numbers across paired legs and independent replicas
for the squared mean-response loss as in the established unbiased estimator.
Check finite differences and analytical/chunked gradient agreement. Require
exactly absent photometry gradients from the response term, supported physical
outputs, correct measured cut masks and identical target replay. Reuse the
validated conditional-shape sampling algebra, not its evaluation data.

## Decisive comparison without a new sequence of repair stages

Before a full run, a numerical pilot verifies the conditional target/gradient
contract and estimator variance on training data only. This is an implementation
check, not a new production training stage.

Then compare two fixed single-run recipes from the same initialization, same
inputs, training budget and row/latent streams:

1. Joint generated-photometry response training, the existing gradient route.
2. The measured-photometry conditional-shape guard with separate photo updates.

Report independent clipping as part of the intervention; do not attribute a
positive outcome uniquely to one of its two mechanisms. A later ablation can
separate them only if it is necessary. Freeze budget/endpoints and loss scaling
before coherent evaluation; do not infer convergence from another64-step trial.
Retain latest trained checkpoint as a practical reference, not as a training
teacher. A selected validation NLL improvement cannot by itself establish
calibration improvement.

On original disjoint training-validation scenes, check both factors' NLL,
primary response at measured photometry, generated size/flux distributions,
joint passing probabilities, selected shape numerators and selected emulator
coefficient as a function of detailed scene score. Report mass and numerator
separately. Check across a predeclared continuous range using hard cuts,
including thresholds between the nine training cuts. Do not turn every
subpopulation into a zero-bias requirement; internal cancellations remain
allowed. Condition-dependent likelihood/rank/probability checks must not be
replaced by a single global histogram, count or static moment.

Constant-shear scenes evaluate the normal generated pipeline only after the
training choices are frozen. Use both axes, all target cuts, common random
numbers and the same new measured reference. The reused40 remain development
checks, not an independent acceptance set. No coherent result chooses a loss
weight, epoch or training label. Generalization beyond the tested cut range
remains unproven.

## What could fail and what that would mean

The original NLL-only flow already underfit conditional photometry; keeping its
loss uncontaminated may recover the damage from regularization without solving
that older error. If photometry likelihood/conditional diagnostics do not improve,
this proposal has not solved the issue. Next distinguish density-model capacity,
optimization and missing conditioning using training/validation data; do not
respond automatically with another passing-probability or static-moment loss.
The diagnostic regressors improved a local anomaly but were not globally better,
so they are not ready-made replacement densities or exact teacher labels.

Correct measured-photometry shape response does not guarantee correct response
on inaccurate generated photometry. A static score does not restore missing
neighbour-only selection sensitivity. The measured-input oracle retained radius
dependence. These are limitations to test, not reasons to reinterpret a failed
normal generated-pipeline result as success. No jobs, weights or code changed
in this proposal.
