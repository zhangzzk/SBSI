# Constant-shear evaluation of factorized GalSBI flows

## Purpose and predeclared scope

Test whether the [improved alignment relation](../alignment_factorized_20261105/findings.md#alignment-and-likelihood)
reduces the original joint constant-shear bias. Both NLL-selected flows are
frozen; original emulator and C8 remain pinned. Same40 existing development
scenes as the earlier grid evaluation, no new simulations or fitting.
See [protocol](rule.md#purpose-and-fixed-choices) and [manifest](manifest.json).

## Completed40-case comparison

All80 candidate evaluations17768588 and pilot/full reducers17768589/17768590
completed with exit0. Retained case1000 replay17768579 reproduces measured sums
exactly and model sums within2.62e-10. Every candidate case passes measured
replay and exact measured-selection count checks. See [completed receipt](completed_receipt.json).
Owner-authorized expansion used six GPUs across cip/inter with unchanged science
settings; no duplicate cases. Results below supersede the four-case pilot.

### Joint multiplicative bias

m=100(R_sim11/R_model11−1), percent; uncertainties are1sigma percentage points
from10000 paired whole-case bootstraps, conditional on these fitted models.
Predicted usability and generated per-leg selection:

| Selection | Retained GalSBI | Factorized Cartesian | Factorized intrinsic frame |
|---|---:|---:|---:|
| Uncut | +0.083±0.341 | −1.009±0.337 | +0.484±0.341 |
| mag<25 | −0.020±0.351 | −0.576±0.347 | +0.639±0.352 |
| mag<25.8 | +0.028±0.294 | −1.156±0.289 | +0.621±0.293 |
| radius>3,mag<25 | +2.429±0.310 | **−0.274±0.299** | +0.437±0.304 |
| radius>3,mag<25.8 | +2.610±0.278 | **−0.216±0.269** | +0.877±0.275 |
| radius>=3.75,mag<25.8 | +0.936±0.364 | +2.346±0.371 | +2.597±0.370 |

Radius is in pixels,0.2arcsec/pixel. With actual usability but generated selection,
Cartesian gives−0.067±0.301% at radius>3/mag25 and+0.049±0.271% at radius>3/mag25.8;
frame gives+0.642±0.306% and+1.145±0.276%, respectively. At radius>=3.75/mag25.8,
actual-usability biases are+2.820±0.374% Cartesian,+3.064±0.372% frame.

The new conditional factorization substantially improves the original main-cut
bias with fixed images, emulator and classifier. It is not a general calibration
solution: tighter-radius and marginal cuts expose remaining errors. Better
alignment/NLL alone does not rank final joint calibration: frame has the best
likelihood/alignment but Cartesian performs better on the primary constant cut.
Only one training seed and existing development scenes have been tested.

### Absolute response and component change

At radius>3/mag25, simulation R11=0.907137759. Model contributions share the
same selection denominators and add exactly:

| Model | Flow term | Emulator shift term | Simulation−model |
|---|---:|---:|---:|
| Retained | 0.712862546 | 0.172765214 | +0.021509998 |
| Cartesian | 0.734986975 | 0.174642473 | −0.002491689 |
| Frame | 0.729407788 | 0.173780462 | +0.003949508 |

Cartesian minus retained increases model R11 by0.024001687: flow+0.022124428
and emulator+0.001877259. Thus92.2% of this model change is in the flow term.
This partitions the change, not the original physical bias into independently
measurable components. The paired m change is−2.702709±0.013706 percentage
points; its small uncertainty reflects common simulation noise cancellation,
not total calibration precision or training-seed uncertainty. Main-cut Cartesian
leave-one-case-out m ranges−0.368% to−0.155%. Selected count ratios are close to
unity (Cartesian~0.997), so near agreement of counts alone is not a response test.

### Meaning of measured selection

Actual usability+measured selection applied to unconditional generated outputs
gives Cartesian m=+3.113±0.316% (radius>3/mag25),+4.705±0.286% (mag25.8),
and+5.107±0.351% (radius>=3.75/mag25.8). Frame gives+3.978±0.318%,
+5.697±0.288%,+5.827±0.356%. These are interventions, not a fully corrected
prediction: multiplying an independent generated shape by actual membership
removes the shape–photometry correlation. It estimates S(y_obs)E[e|x], whereas
the selected conditional prediction needs S(y_obs)E[e|radius_obs,flux_obs,x].
Large residuals in this mode do not by themselves identify a selection failure.

Next: the [predeclared conditional-photometry check](../alignment_conditioned_photometry_20261105/rule.md#predeclared-scope)
uses measured photometry inside the trained shape factor on the40 standalone
validation cases. This distinguishes photometric mixing from conditional-shape
underfit before further architecture changes. It is not a constant-joint causal
budget, and no empirical response offset is introduced.

The [matched constant-shear controls](../alignment_matched_constant_20261105/rule.md#question-and-predeclared-scope)
add the earlier full-dimensional flows trained with the same schedule/budget.
They are needed to separate architecture effects from other training changes.

![Completed constant-shear comparison](comparison.png)

## Estimator and flags

Original64 antithetic draws,1024-parent blocks, seed7301+10000019*case+
1000003*block; common latent streams across models/legs. Original Mobius
emulator transport using each generated draw's photometry. Simulation uses
its own per-leg measured selection and usability. Definitions:

- Predicted usability + generated selection: the predictive joint model.
- Actual usability + generated selection: replaces detection/usable flags only.
- Actual usability + measured selection: replaces the final membership mask;
  generated shapes and generated-output emulator inputs remain. Diagnostic only.

Cuts include radius>3/mag25, radius>3/mag25.8, radius>=3.75/mag25.8 and marginal
controls. DeltaR=simulation−model; m=100(R_sim/R_model−1). Flow and emulator
response terms share denominators and add exactly; neither is a standalone
error against a separately measured component. Only g1 is simulated here,
so this checks R11 and R21, not the full2x2 response matrix.

## Checks and uncertainty

Retained replay gates all candidates. Every candidate case must reproduce the
saved simulation sums; true-final-selection model masses must equal measured
selected counts. Parent/C8 IDs and truth rows agree; unmatched measurement
entries follow the frozen measured-leg routine and are reported per case.
Use installed F8 runtime, explicit helper imports and custom factorized loader;
no archive launcher or Python path injection. Preset/checkpoint/source hashes
and original parent/flag cache hashes are saved in per-case receipts.

Small reducer test passes known3% m andDeltaR11=.024 under unequal leg counts,
additive flow/emulator responses and exact-zero paired identity. Sources parse
and shell/JSON checks pass. Whole-case10000 paired bootstraps preserve covariance
across models. Report per-case results and leave-one-case-out ranges as well;
four-case interim estimates are weak and full40 still condition on fitted
models and measured catalogues. Development comparison, not independent acceptance.

All large outputs are at
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_factorized_constant_20261105/`.
No production code, retained identities, emulator/classifier training or images
changed. No replacement model is accepted; conditional-shape diagnostics continue.

## Existing g1 scenes: second-component null checks

Owner asks whether the other response component on these same scenes already
tests recovered g2~0. CPU17772085 computes both odd R21 and the even e2 offset,
with the same10000 paired whole-case bootstrap weights. At r>3/mag25 with
predicted usability and generated selection:

| Model | DeltaR21 | Delta even e2 |
|---|---:|---:|
| Retained | −0.001994±0.001946 | +0.000372±0.000119 |
| Cartesian | −0.000218±0.001936 | +0.000203±0.000120 |
| Frame | −0.003430±0.001934 | +0.000045±0.000115 |

Errors are simulation minus model. Even e2=(mean_e2_plus+mean_e2_minus)/2;
this is a finite-shear additive diagnostic, not an inferred g2. R21 comes from
the difference of those means divided by0.04. Thus existing g1 scenes constrain
cross-component response and even offset, but not R22, whose term vanishes at
g2=0. The Cartesian even residual is about1.7sigma under fixed-model case errors.
See [null-component results](null_component.json). New g2 scenes independently
probe R12/R22; neither check substitutes for the other.
