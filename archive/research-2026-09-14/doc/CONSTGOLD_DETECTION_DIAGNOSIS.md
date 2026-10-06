# Detection diagnosis for the circularized lambda10 plus trial9 model

**Completed:** Slurm16457017 and16457064 exited0:0, all provenance,
original-result reproduction, bootstrap accounting and localization checks
passed. The complete response failure involves incorrect population weighting
and a poorly represented detection-boundary population, in addition to an
independently measured detector-response error. The controls do not support
attributing the entire failure to the detector's shear derivative.

## Scope and fixed inputs — 2026-09-11

Owner requested a deep investigation of the complete-detection result after
the new lambda10 epoch154 flow plus ordinary-MSE trial9 emulator gave m
+0.2338% (pure shape), -0.7160% (fixed detected parent with measured cuts),
and -2.1112% (complete detection and selection). No training or calibration
correction is part of this diagnosis. The same transition-aware detector
SHA9966cf...455, flow SHA9c5bb0...a6f and emulator SHA972358...e46d remain fixed.
The recent emulator provenance correction is retained; no newer detector
replacement was found in the recent worklogs/artifact roots.

Use cases40–89, strict true magnitude18–25.8 and semi-major Re0.37–1.5arcsec,
the existing measured cuts, and central g1 +/-0.02. Follow
[response definitions](CONVENTIONS.md#6-response-estimators) and
[m convention](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention).
External root is the completed `constgold50_trial9_v1/detection_diagnosis_v1`
within `physical_circularized_grid_free_shape_radius_lambda10_v2`.

## Full-population evidence from saved sums

`aggregate.py` independently pools every leg's saved numerators and
denominators and jointly bootstraps50cases10,000times with seed20260902.
This retains covariance when subtracting response branches. Case errors
remain conditional on the fitted models and the original QMC seed.

| Comparison | Measured R11 | Predicted R11 | Prediction minus measured |
| --- | ---: | ---: | ---: |
| Fixed both-usable/detected, no measured cut | 0.96002785 | 0.95778853 | -0.00223933 +/-0.00214300 |
| Independent detection, no measured cut | 0.95029592 | 0.95994608 | +0.00965016 +/-0.00216040 |
| Fixed both-usable/detected, own measured cuts | 0.61118255 | 0.61559013 | +0.00440758 +/-0.00183209 |
| Independent detection and measured cuts | 0.60537833 | 0.61843461 | +0.01305629 +/-0.00190702 |

Enabling independent detection/population weighting without measured cuts
changes measured R11 by -0.00973194 +/-0.00038265 and predicted R11 by
+0.00215755 +/-0.00006446. Their paired difference is
+0.01188949 +/-0.00038180. With measured cuts the corresponding changes are
-0.00580422 +/-0.00037730 and +0.00284448 +/-0.00007309, leaving
+0.00864870 +/-0.00037972. These differences are very well determined, but
the branch comparison changes the parent population as well as applying a
detector; it is not alone an isolated causal detector-error estimate.

Exact ordered accounting gives the complete residual as the pure-shape
residual -0.00223933 plus the no-cut detection/population increment error
+0.01188949 plus measured-selection-after-detection error +0.00340612.
The last term has case SE0.00289369. This is a decomposition of these
observed comparisons, not a proof of three independent physical causes.

Observed raw detection counts are4,354,851 /4,354,749, compared with
classifier expected counts4,364,140.398 /4,363,776.783. Total counts differ
by only about0.21%. Usable counts are4,354,645 /4,354,546;206/203 invalid
shapes are excluded and no crossmatched rows lack shape records. The
fixed both-usable parent has4,338,426 identities, so the independent valid
legs add16,219/16,120 objects. Their small frequency does not imply a small
shape-weighted response effect.

The saved true-shape test bypasses both flow shape predictions and R_blend.
Actual/model detection-weighted true-shape R11 is0.99307701/0.99513158,
residual+0.00205457 +/-0.00025467; R21 residual is-0.00101881
+/-0.00028735. Thus the classifier/population model has an independently
detectable shape-weighting error. This does not measure its contribution to
the measured-shape result directly. With measured selection, true-shape
residual instead is-0.01330092 +/-0.00136610, demonstrating that the
measured-shape total cannot be attributed to the classifier alone.

## Targeted controls and estimator

The new Slurm diagnostic scores the detector and observed populations on all
50 cases. It evaluates the full flow on the ten systematically spaced cases
40,45,50,55,60,65,70,75,80,85, with two independent QMC seeds7301/7302,
64 draws and CRN between legs and interventions. The first seed must reproduce
the production per-case uncut/selected, weighted/fixed-parent sums before
interpretation. No case is selected for an extreme residual.

For each leg retain per-object conditional mean shape, selected shape
numerator and cut probability. Reweight those same moments using:

- Original leg-specific detector probabilities p+/p-.
- Their per-object midpoint, held fixed in both legs.
- Observed per-leg usable detection flags (an oracle diagnostic).
- Their midpoint, held fixed in both legs.
- The fixed both-usable parent, reproducing the original control.

All50cases additionally compare true sheared and fixed intrinsic shapes under
predicted/observed detection weights and their midpoints. This bypasses flow
Monte Carlo and permits independent detector-only localization. Prespecified
true magnitude, semi-major size, intrinsic ellipticity magnitude and midpoint
probability bins retain complete coverage and use global denominators so
bin contributions sum to the total intrinsic-shape response error.

For moments u+/u- and cut probabilities q+/q-, let A=sum(w*u), B=sum(w*q)
in each leg. The exact central response decomposes into a weight-change term
`[sum(delta_w * u_bar) - mean_e_bar * sum(delta_w * q_bar)] / (0.04 * B_bar)`
and a conditional-change term using w_bar, delta_u and delta_q. Here bars
mean the two-leg arithmetic average, including mean_e_bar=(A+/B+ + A-/B-)/2.
Their sum is algebraically exact, including changing denominators. Synthetic
checks exercise nonconstant weights and cut probabilities; frozen weights
must give zero weight-change term. Midpoint/oracle interventions are
diagnostics, not proposed calibrated models or physical noise trajectories.

## Completed detector-only localization — all50cases

Hold the intrinsic ellipticity fixed between legs, so the response comes only
from changing detection weights. Actual R11 is-0.00694282 +/-0.00025357,
predicted-0.00490118 +/-0.00001139; paired residual+0.00204165
+/-0.00025280,8.08 case SE. The predicted suppression is29.4% too weak.
This bypasses both the flow and blend emulator. R21 residual is-0.00096673
+/-0.00028627. Raw both-detected count4,338,782, plus-only16,069,
minus-only15,967; usable-mask counts differ because invalid shapes are excluded.

Conditional detection fractions expose errors hidden by the small net count
offset. These bins use intrinsic ellipticity magnitude, not measured cuts:

| Intrinsic magnitude | Actual detection fraction | Predicted fraction |
| --- | ---: | ---: |
| 0–0.2 | 0.910339 | 0.926477 |
| 0.2–0.4 | 0.899384 | 0.906874 |
| 0.4–0.6 | 0.892594 | 0.886394 |
| 0.6–0.8 | 0.873234 | 0.847934 |
| 0.8–1 | 0.856948 | 0.785740 |

Thus round objects receive excess weight and elongated objects insufficient
weight, despite total predicted detections being only about0.21%high.
These are conditional calibration checks, not causal effects of ellipticity.

Global-denominator attribution of the fixed-intrinsic R11 error places90.6%
at true magnitude24–25.8,66.7% at semi-major Re1–1.5arcsec, and90.5% among
objects with a representative neighbour. These overlapping marginal groups
must not be added together or described as one measured joint subgroup.
The added Re0.37–0.5 range contributes only+0.00008852 +/-0.00013760;
the error is not concentrated at the newly lowered size boundary.
Neighbour/no-neighbour contributions are+0.00184727 +/-0.00023454 and
+0.00019438 +/-0.00009354. They do not establish that omitted neighbours
cause the discrepancy.

## Completed flow controls — ten systematic cases

All ten planned cases40,45,...85 completed,971,519 truth objects. Each used
64QMC draws, two independent seeds, and CRN between legs and interventions.
The following table uses seed7301 and each control's own measured population.
All uncertainties are paired-case bootstrap SE, not an integration-error bar.

| Cut | Weights | Model R11 | Measured R11 | Residual +/- case SE |
| --- | --- | ---: | ---: | ---: |
| None | Predicted per-leg probabilities | 0.96023044 | 0.94875105 | +0.01147939 +/-0.00472046 |
| None | Observed usable-detection masks | 0.95565164 | 0.94875105 | +0.00690059 +/-0.00480334 |
| None | Fixed both-usable parent | 0.95795799 | 0.95782536 | +0.00013263 +/-0.00493969 |
| All measured cuts | Predicted per-leg probabilities | 0.61818884 | 0.59654879 | +0.02164006 +/-0.00431587 |
| All measured cuts | Observed usable-detection masks | 0.61344645 | 0.59654879 | +0.01689766 +/-0.00423012 |
| All measured cuts | Fixed both-usable parent | 0.61536886 | 0.60284098 | +0.01252788 +/-0.00432634 |

This ten-case subset's measured selected response differs from the full50-case
mean. Its ordinary-model m=-3.50056% +/-0.69692 and oracle m=-2.75455%
+/-0.68857 must not replace the full50-case m=-2.11118%. The control contrasts,
using exactly the same subset and random draws, are the useful diagnostic.
The uncut oracle residual alone is1.44 case SE, while the selected oracle
residual is3.99 SE. Conditioning on observed masks does not eliminate the
selected measurement-composition error.

Replacing predicted probabilities with observed masks reduces model R11 by
0.00457881 +/-0.00022702 without cuts and0.00474239 +/-0.00039968 with cuts.
The midpoint decomposition is more informative:

| Predicted-minus-oracle component | No cuts | All measured cuts |
| --- | ---: | ---: |
| Fixed midpoint population composition | +0.00457956 +/-0.00015079 | +0.00464860 +/-0.00014525 |
| Difference of moving-weight effects | -0.00000076 +/-0.00019784 | +0.00009379 +/-0.00032156 |

The detector's own uncut moving-weight term using model shapes is-0.00186536,
versus-0.00186460 with observed masks. With cuts the exact corresponding
terms are-0.00130015/-0.00139348. Thus the predicted-versus-oracle gap is
dominated by different population composition in this control, not by a
resolved discrepancy of these model-shape-weighted derivative terms. This
does not contradict the all50case true-shape detector error: the weighting
function is different and model conditional-shape errors can hide it.

### The single-leg detection boundary

The subset contains867,905 usable both-leg identities and3,274/3,195 usable
plus-only/minus-only objects. The latter are about0.37% of each usable leg.
For these single-leg objects, the observed mean e1 is-0.02998394 in the
plus-only set and+0.02943017 in the minus-only set. Model conditional means
are instead+0.00153523 and-0.01205514, respectively, including the same fixed
trial9 blend correction.

Using observed detection masks and the actual global per-leg denominators,
the single-leg population contributes+0.00676530 +/-0.00099999 to the uncut
model-minus-data response. Both-usable objects contribute+0.00013529
+/-0.00492090. They sum exactly to the oracle residual. These are different
sets of objects in the two legs, so their mean difference is not a matched
within-galaxy derivative. Joint shared-noise detection selects unusual measured
shapes at the boundary. A model conditioned on a single leg is not required
to reproduce every joint-mask subgroup separately; nevertheless the composed
aggregate must account for their contribution, including any compensating
error elsewhere.

An equivalent paired comparison avoids mistaking the noisy fixed-parent
residual for a new finding: opening the detected population changes measured
R11 by-0.00907431, but changes the oracle-weighted flow by only-0.00230635.
The gap is+0.00676795 +/-0.00100441. With measured cuts the corresponding
gap is+0.00436979 +/-0.00082650. These population increments remain wrong
even after replacing predicted detection probabilities by observed masks.

### Sampling sensitivity and scope

Seed7302 shifts ordinary-model R11 by+0.00004261 without cuts and+0.00046716
with cuts. Predicted-minus-oracle contrasts change only-0.00000428 and
-0.00004050, respectively. The single-leg uncut residual contribution becomes
+0.00677891 +/-0.00100049. Thus the main contrasts reproduce across the two
seeds. Two seeds do not establish absolute QMC convergence for hard cuts or
provide a robust integration-error estimate.

These experiments distinguish a directly detected classifier error, a
population-weighting problem, and inadequate aggregate treatment of boundary
measurements. They do not uniquely identify missing scene variables, the
semi-major condition, the transition surrogate, flow regularization, or the
fixed blend correction as the sole cause. No model was fitted or promoted.

## What the detector was trained to predict

The pinned detector is the transition-aware lambda1 epoch19 checkpoint from
WORKLOG cont.239, not a recently retrained detector. The newer trial9 model
replaced the blend emulator only. The detector has17 inputs describing the
primary and one impact-a1 neighbour within3arcsec, including pre-shear
semi-major sizes and sheared ellipticities. It has not received the flow's
circularized-size change. The same state ambiguity is therefore a plausible
detector limitation; this experiment does not establish its causal contribution.

Under the circularized rendering contract, area is proportional to a0^2*q0
and the invariant circularized radius is a0*sqrt(q0). Across unsheared
galaxies, increasing intrinsic ellipticity at fixed semi-major a0 and flux
shrinks area and raises surface brightness. Applied area-preserving shear
changes ellipticity without that area change. Supplying only a0 and sheared
ellipticity loses the distinction because q0 is not separately supplied.
The ordinary detector's historically positive fixed-intrinsic detection
response, despite a negative measured response, is compatible with this
mechanism. A circularized detector comparison would be required to establish
how much it explains; no such retraining is performed here. Historical image
renderer provenance has the same qualification as the circularized-flow trial.

Archived source at git `f3343bb:sbsi/detection_classifier.py`, lines55–87,
and `f3343bb:scripts/train_detection_classifier_transition_pair.py`,
lines348–367, implements marginal BCE plus lambda1 times the mean conditional
BCE `BCEWithLogits(z_g-z_0, y_g)` on eligible discordant pairs. Response was
not supervised. Training used cases0–24, tuning25–29, test30–39, and the broader
domain18<r<28,0.1<Re<1.5. ConstGold cont.241 had already found a significant
response residual in that broader domain. Its sign cannot be transferred to
the present narrower domain without recomputing the conditional population.
The later100-case audit in cont.394 also retained the identical detector SHA
and found a sheared-intrinsic detection-response residual+0.00352462
+/-0.000276 on its registered support. Its apparent joint-selection agreement
was explicitly described as containing cancellation. This corroborates that
the detector issue predates the present lambda10 flow; it is not evidence of
a later corrected detector checkpoint. Supports and estimands differ, so
these historical numbers are not substituted for the present50-case test.

There is a specific statistical limitation of the transition objective.
Conditional on paired covariates, let marginal detection probabilities be
p0,p1 and the joint both-detected probability be q. The actual probability
that leg1 detects, given exactly one detection, is
`(p1-q)/(p0+p1-2*q)`. The loss's `sigmoid(logit(p1)-logit(p0))` equals this
under independent Bernoulli legs, q=p0*p1; it is not an identity for arbitrary
paired labels. The simulations share noise. A synthetic valid joint with
p0=0.8,p1=0.801,q=0.799 gives actual direction probability2/3 but the marginal
logit difference gives0.501566. The independence identity and these values
were checked algebraically and numerically. Thus accurate marginal
probabilities can conflict with this auxiliary target under correlated
pairing. This identifies a limitation of its probability interpretation,
not proof that it causes the present response residual. Conditional-direction
classification alone also does not enforce the frequency and shape weighting
of transitions required by shear calibration.

The flow NLL population consists of independently usable individual rows;
only the response loss requires usable pairs. Do not describe its NLL as
trained exclusively on both-detected objects. Nevertheless, a flow accurate
on a fixed both-detected comparison need not be accurate for objects entering
or leaving detection, or for the detection-conditioned selected moments.

## Statistical meaning of the composition

For scene conditions x, let p(x,g) be detection probability, q(x,g) the
conditional measured-cut probability among usable detections, and u(x,g)
the conditional mean of `e * I(pass)` there. The predicted selected mean is
`sum_x p(x,g)*u(x,g) / sum_x p(x,g)*q(x,g)`. Counts constrain the denominator;
shear response also depends on changes of the shape-weighted numerator and
their cancellation. Accurate mean detection fractions alone do not test this.

The factorization `P(D=1|x,g) P(y|D=1,x,g)` is mathematically valid: it does
not require detection and measured shape to be independent. Its measurement
factor must describe the correct detection-conditioned population. Using
incomplete scene conditions, optimizing mainly marginal row density, imposing
a paired-survivor response objective, or transferring to a differently weighted
population can leave the response moments wrong. These are candidate
limitations, not claims individually identified by the present experiment.

Observed-mask controls bypass the predicted detector but still use model
conditional moments, including the fixed blend correction and model measured
cuts. A residual there cannot be repaired just by supplying the observed
detection identities. The detector-probability versus observed-mask difference
also changes which contexts contribute; it is not exclusively an error in
the detector's shear derivative. Midpoint controls distinguish that population
composition effect from the effect of leg-dependent weights within this
diagnostic, without declaring either intervention a deployable model.

## Provenance and validation

All original model, source and lookup hashes are reused;250 raw-catalogue
file stats are pinned before and after the new job. New manifest SHA256 is
`64aedc3645e17b1bc677b82d18ddc5e0b398ef332a4a06e9d353845313023325`.
Worker and analysis scripts are frozen before launch. The login work is
limited to small saved-sum analysis, syntax and provenance checks; raw scene
and model evaluation use Slurm. The resource skill records no login GPU,
330.83GB available memory and6.7TB free storage. The scientific-critical-
thinking skill guides the distinction between direct evidence and causal
interpretation. Wake-ups remain disabled by owner instruction.

Frozen import/source/input checks and synthetic exact-decomposition checks
passed from `/tmp`. The initial repository-cwd invocation correctly rejected
importing the live checkout; the job explicitly changes to `/tmp` before
using the frozen environment. Submitted Slurm16457017 at21:25:12 Berlin,
started21:25:36 on th-cl-nv01, V100/12CPU/96GB/90min cap. Cases40 and45
completed both QMC seeds and reproduced the original seed7301 production
sums, with empty stderr. The completed controls are reported above.

Added a compact-sum `review.py` for paired control differences, own-parent
comparisons, two-seed sensitivity, and independent decomposition checks.
A dependent CPU reduction `localize.py` uses already saved arrays to attribute
the uncut oracle residual to both-usable and single-leg-usable objects and
the intrinsic detector error to presence/absence of a representative neighbour.
These descriptive groups do not identify causal mechanisms. Submitted
Slurm16457064 at21:36:00 Berlin, afterok16457017,2CPU/12GB/10min cap.
Python/Bash syntax passes and its code hashes are checked before/after.

Final review:16457017 completed21:48:38 Berlin after23:02;16457064 started
21:49:10 on met-cl-vis01 and completed21:49:17 after7seconds. Both exited0:0
with empty stderr. The compact independent review ran in0.65seconds using
one BLAS thread on small saved sums. All original-result reproduction,
source/input/model/stat checks, exact decomposition identities, fifty-case
coverage and dependent result hashes passed. Saved `completion_review.json`.
Results SHA256 is`f11d48d3ffcef093e9eaa1346fdac81a29aae8d6ee3c8b3096aedc06c5b1c26b`;
review SHA`a4c34509b07fb0dcda184215fa5d216e935e56a9ac8fa8a3c3ce4a7541945539`;
localization SHA`7d065ced8da280b9e6f8c8097fde6da48aa8a0bbb537308a128031e5d4e1e3a1`.
No additional jobs, model changes, training, offsets, promotion or wake-ups.
