# Joint flow and emulator calibration investigation

## Goal and retained population

Owner request, 2026-09-17: reach approximately 0.3% joint
`m = R_measured / (R_flow + R_blend) - 1`, retaining a good joint density and
removing the need to average multiple independently trained flows. A single
checkpoint is the proposed deployable model; independent training replicas
may still be needed to assess robustness. No post-hoc response correction,
cut tuning, or ConstGold-based checkpoint selection is permitted.

The working population retains the latest strict `r_input_p < 26` boundary.
Report both the valid-measurement population and declared measured selections.
The owner's 2026-09-18 clarification fixes the primary acceptance population
at **measured FLUX_RADIUS >0.6 arcsec (3 pixels), MAG_AUTO<25.8**.
The owner's 2026-09-19 clarification gives cross-components lower priority.
Current work prioritizes diagonal joint m, density and frozen-model
reproducibility; the cross-component diagnostics remain visible caveats.
Apply measured cuts independently in each leg. Uncut and >=0.75 arcsec
results remain diagnostics, not substitutes for this population. Do not
silently substitute the historical V3 truth-size domain. The current frozen
common-descent single flow with disk transport gives actual-U joint
**m=-0.135051 +/-0.265483%** on the80 main scenes and
**+0.477637 +/-0.183980%** on the40 independent scenes. These oracle-U
diagnostics include realized detection/measurement selection but do not
establish predictive detection calibration. The current smooth-crowding
single classifier gives predicted-U joint **-0.199979 +/-0.263649%** main80,
**+0.402474 +/-0.188664%** independent40, **+0.347168 +/-0.817220%** on g1
and **-0.443430 +/-0.794924%** on g2 for the four additional existing scenes.
The full-vector audit also finds unresolved cross-component residuals on those
four scenes; see [response-vector validation](#smooth-classifier-response-vector-validation).
All errors here are case
bootstrap standard errors in percentage points. See
[detection scope](#practical-acceptance-and-detection-scope) and
[completed smooth classifier results](#completed-smooth-classifier-and-reproducibility-repeats).

The normalized primary selected-density score improves slightly relative to
the initial single flow when conditioning on actual U; the smooth classifier
also improves the predictive-U selected score by about0.033nat/object versus
the historical classifier on every endpoint. Neither a near-zero point nor a relative
density gain proves the requested end state. Main and independent endpoints
differ in both scenes and shear step (0.02 versus0.05); do not treat their
difference as an isolated generalization or finite-step effect. The historical
results below describe their respective earlier checkpoints, not the current
candidate. No deployment configuration is promoted by this investigation.

## Historical comparison

The frozen V3 record is `m=+0.348825%` on 5,642,349 joined rows, using sixteen
V2.2 flows and the narrow-domain emulator. Its truth support includes
`18<r<25.8` and `0.5<Re<1.5`; this differs materially from the current full
true-mag-limited support. Original-E/V3.1 subsequently gave
`+0.23433 +/- 0.15040%` with four flows on those same rows. Neither number can
be transferred to the current population. See the archived worklog entries
"2026-08-17a" and "2026-08-24 cont.223" and
[historical population conventions](../archive/research-2026-09-14/doc/CONVENTIONS_FULL.md#3-building-the-population--order-matters).

## Mechanism summary and V3 scope

V3 established a small joint bias for its own narrow truth support and
multi-flow model. Its reported result does not identify whether ensemble
averaging, the restricted population, or cancellation supplied the agreement.
The present full-parent, measured>0.6-arcsec experiment changes all of those
conditions. A numerical comparison of the two headline biases therefore
cannot isolate the effect of removing the flow ensemble. The current
current candidate is evaluated as one flow throughout; its six diagnostic
classifier arms are comparisons, not six flows averaged into a model.

The investigation identified distinct mechanisms in the newer pipeline:

* Some apparent disagreements compared different populations or response
  definitions: a fixed observed-g0 anchor, per-leg selection, a forward
  matched-usable response, and a coherent antithetic response are different
  estimands. The current evaluation fixes the generated parent before
  detection and reselects each measured/generated leg independently.
  See [response estimands](#distinguish-the-response-estimands).
* The historical shared-row guard objective includes an unintended row
  variance penalty. Corrected independent-row sampling and full validation
  coverage address that estimator defect; their existence alone is not proof
  of calibration. See [guard audit](#completed-audit-results).
* Additive blend transport can move shapes outside the usable disk, with a
  disproportionate response contribution from those draws. The current
  disk-preserving map uses a coefficient trained against physical moments
  and an inverse-map density with its Jacobian. It does not clip or rescale
  the measured response. See [support check](#density-identity-and-physical-support-check)
  and [physical-moment fit](#physical-moment-transport-fit-and-independent-checks).
* Detection/usability was the largest remaining discrepancy after the
  single-flow/disk-emulator combination approached the target using observed
  U. Replacing observed U with the historical classifier shifted joint bias
  by about-1pp on both large samples, predominantly through blend-response
  weighting. State-consistent geometry and smooth crowding improve the
  classifier's probability loss and reduce that discrepancy to about0.07pp.
  Neither feature recipe nor checkpoint selection fits a joint response
  offset. See [detection scope](#practical-acceptance-and-detection-scope) and
  [smooth results](#completed-smooth-classifier-and-reproducibility-repeats).

The final classifier changes hold the flow and emulator fixed and improve
joint bias by about+0.94pp versus historical predicted U on both main and
independent scenes. Their selected parent/output log score also improves
on every endpoint. That paired evidence identifies a repeatable correction
to the newer pipeline; it does not retrospectively establish why V3's
different model and population happened to agree. Density evidence remains
explicitly scoped: the single-flow continuation has a small uncut NLL cost
within its declared budget, improves the properly normalized primary selected
score, and still has modest output-moment discrepancies. Sampling and
classifier-training repeats test the current candidate's robustness without
choosing a favorable seed or averaging models into deployment.

The completed flow-refinement repeat also changes primary m by only about
0.0015–0.0019pp on the two large samples, and primary selected NLL by less
than1e-5nat/object on every endpoint/leg. See
[completed flow repeat](#completed-flow-refinement-response-and-density-repeat).

## Distinguish the response estimands

The current full-domain guard constrains a *per-leg reselected* mean through
the influence function `w(y) [e-mu_0] / D_0`. Its training labels are matched
usable zero/forward pairs at random shear direction and magnitude 0.05.
Several subsequent hard-cut diagnostics instead select on the g0 draw and
carry the selected identities to the sheared leg without recutting. The latter
depends on the cross-leg coupling, which a per-leg density does not identify.
A failed fixed-anchor diagnostic is not by itself a test of the guard target.

Even the latest "uncut" ConstGold panels require an observed usable g0 anchor;
they are not an evaluation of an unconditional truth-parent population. All
future comparisons must name this support explicitly. The configured
likelihood's per-leg selected normalization and a fixed-g0 experiment remain
separate scientific contracts; see
[population order](CONVENTIONS.md#3-population-and-selection-order).

## Experiment 1: guard objective and validation audit

For latent-averaged row response `a_i(theta)`, fixed full-population target `t`,
and response scale `s`, the intended scalar penalty is
`0.5 * [(E_i a_i - t)/s]^2`. The historical two-replica implementation uses
the same `N` sampled rows and independent latent draws in each replica. Its
expected penalty is instead

```text
0.5 * [(E_i a_i - t)/s]^2 + Var_i(a_i) / (2 N s^2).
```

The additional term is parameter-dependent and can shrink real response
variation. Use independent row samples as well as independent latent groups
to remove it; retain common random numbers between shear legs *within* each
replica. An exact two-object unit test checks both expected loss and replayed
gradient at a correct population mean. Historical mode remains available to
reproduce existing artifacts; the new behavior is explicit opt-in.

There is a separate coverage issue: checkpoint selection compares model
predictions on 200k/400k fixed pairs with measured targets fitted on all 3.9M
validation pairs. It does not include the difference between the subset's
context distribution and the full distribution in its uncertainty estimate.

`scripts/audit_guard_population_score.py` evaluates every validation pair for
the wide and sharp guard checkpoints. Two independent latent replicas on the
same rows estimate the between-row covariance and hence the extra penalty.
It reports full versus historical-subset scores, subset mean shifts, full
2x2 responses, and paired case-bootstrap uncertainties. It uses each model's
original guard widths/scales and saved validation subset; this diagnoses the
two recipes individually and is not a width-controlled model comparison.

This audit is a mechanism test, not an acceptance result: it excludes the
emulator, uses a forward shear amplitude of 0.05, and revisits cases used for
checkpoint selection. The existing combined ConstGold result is not used as
a fit target. If the objective defect is consequential, the next controlled
continuation will use independent row replicas and complete validation
coverage, retaining NLL and the single-flow architecture.

### Completed audit results

Both audits finished on all 3,908,555 validation pairs. On the wide recipe the
estimated shared-row penalty is 0.02711, compared with full-population loss
0.01326. On the sharp recipe it is 0.04743 versus 0.05363. The unintended
penalty is therefore consequential on the scale of the optimization objective;
whether removing it improves scientific calibration remains to be tested.

The per-leg soft joint-cut self-response biases are `+0.709 +/- 0.398%` (wide)
and `+0.737 +/- 0.411%` (sharp), with case-bootstrap errors. The corresponding
uncut self-response biases are `+0.070 +/- 0.423%` and `+0.189 +/- 0.423%`.
These are half-shear, matched-usable, forward influence responses, not the
combined ConstGold result. Original validation subsets also shift the wide
model joint response by +0.001495 and sharp by +0.000595 relative to full
coverage. Latent integration error is not included in those case errors.

## Experiment 2: controlled population-objective continuation

Start both arms from the same wide-guard checkpoint. Preserve the joint
four-output density, truth-only conditions, training/validation case split,
and NLL. Both arms use wide 0.05-pixel/0.05-mag guards, response scale 0.01,
eight response steps of 65,536 pairs and eight latent draws per epoch,
learning rate 1e-5, and twelve epochs. Both select against NLL plus guard loss
on **every** validation pair, with two independent groups of four draws and
the same fixed streams. The initial checkpoint participates as epoch zero.

The sole between-arm difference is whether the two response replicas draw
the same rows (control) or independent rows (corrected). This is an A/B test of
an objective, not an ensemble of deployable flows. The settings differ from
the old sharp-refinement recipe; historical-versus-new changes cannot be
attributed solely to row independence, but the two concurrent arms can.

Freeze the selected checkpoints before the 100-case ConstGold diagnostic.
Evaluate baseline, shared-row control, and independent-row model with the
same existing emulator lookups, classifier ensemble, 64 antithetic draws,
latent seed7301, and 10,000 paired case-bootstrap replicates (seed20260914).
Report both the full valid-g0 cohort and the original fixed-g0 magnitude/radius
cut, retaining true magnitude <26 in both. Property bins are diagnostic only.
No ConstGold number chooses a checkpoint. Compare full-validation density NLL
alongside the response change. The uncut branch also extrapolates the existing
fixed-g0-trained emulator beyond its training cohort, so it cannot by itself
validate that component's full-domain accuracy.

### Remaining population-matching question

NLL uses independently usable rows in each leg, whereas the response target
currently uses the intersection of both usable legs. On the forty validation
cases, 12,133 g0-only rows and 11,378 g05-only rows are excluded by that
intersection (about 0.3% of each valid leg). The omitted fraction alone does
not measure response bias, but it is comparable to the target precision and
can be concentrated near measurement failures. A subsequent population audit
must distinguish true-parent, per-leg usability and both-leg usability before
attributing the remaining ConstGold discrepancy to flow response or emulator.
Neither the current guard audit nor the A/B test resolves this issue.

The trainer is `scripts/refine_full_domain_guard_flow.py` with explicit
`--population-objective` options; artifacts live under
`$DATA_DIR/sbsi_caches/population_guard_ab_20260917_v1/`. Its `protocol.json`
records exact inputs, recipe, source hashes, population, and response
statistics for each arm. Historical behavior remains the default.

Both arms completed all twelve epochs without beating epoch zero.
Their selected models are still the original baseline. Exact checkpoint
comparison confirms equality of every state tensor, preprocessor, target
transform and architecture; only metadata differs. Every response update was
gradient-clipped and post-guard density NLL degraded substantially. Removing
the objective's row-variance term therefore does not by itself make this
alternating optimization recipe effective. Redundant ConstGold evaluations
were cancelled after this identity check; the already-progressing baseline
uncut evaluation remains. No new model has been promoted.

## Experiment 3: remove observed-g0 conditioning from the joint diagnostic

Construct the complete target-role truth parent on existing cases40--119,
using the full-parent classifier preparation for its truth/context values and
the original ConstGold truth catalogue for the strict r<26 boundary. No
measurement or U0 flag selects these parent rows. Keep U0 and fixed-g0 science
membership only as diagnostic masks; do not invent measurements for the newly
retained objects. The existing flow and emulator are unchanged.

The completed parent contains 8,876,218 objects. Of these, 1,035,108 (11.662%)
lack a valid g0 flow measurement and were therefore absent from the previous
"uncut" diagnostic. The usable-g0 and fixed-cut subsets contain 7,841,110 and
6,472,963 objects. Three g0 rows outside the target-role parent (one each in
cases105,106,118) are dropped and reported, rather than extending the parent.

The initial preparation stopped at case105 when this role-boundary mismatch
first appeared. Its partial outputs remain under `parent/`; the corrected,
complete product is `parent_v2/`. Truth-magnitude selection uses original
catalogue precision, avoiding dependence on rounded float32 model contexts.

Evaluate the fixed wide-guard baseline on three cohorts: the truth parent,
its usable-g0 subset, and its fixed-cut-g0 subset. Within each cohort, report
no measured cut, per-leg MAG_AUTO<25.8 with FLUX_RADIUS>0.60 arcsec, and the
configured per-leg FLUX_RADIUS>=0.75 arcsec with the same magnitude cut.
Apply each predicate independently to observed and joint generated outputs;
integrate selected shape and blend-shift numerators before normalizing.
Retain matched-U, actual per-leg-U and classifier-weighted branches, explicit
missing/invalid counts, model flow/blend components, and paired case statistics.
Only the truth-parent per-leg branches avoid observed-g0 conditioning; matched-U
still conditions on both observed shear legs.

This is a controlled population test, not model training or cut optimization.
The emulator remains the existing fixed-g0-trained model, extrapolated on
newly included objects; that limitation is explicit. The current flow's U
contract permits finite zero ellipticity, while the classifier preparation
excludes exact zero-shape sentinels, so their counts are reported rather than
silently changing the flow population. A nonzero count requires auditing.

Scripts: `prepare_constgold_truth_parent.py`, the existing exact-key blending
lookup builder with explicit population provenance, and
`evaluate_constgold_truth_parent.py`. Products are under
`$DATA_DIR/sbsi_caches/constgold_truth_parent_20260917_v1/`.
First run a two-case GPU smoke check, then the frozen eighty-case comparison.
Scheduler dependencies keep this pipeline within the combined two-GPU limit.

### Completed truth-parent results

The eighty-case evaluation completed in 9m15s with empty stderr. No usable
exact-zero shape rows occur in either leg across the full run. The wide-guard
single flow plus fixed emulator gives:

| Truth-parent measured selection | Actual per-leg U: m | Classifier-weighted U: m |
| --- | ---: | ---: |
| No measured cuts | -3.451 +/- 0.266% | -4.272 +/- 0.263% |
| MAG_AUTO<25.8, radius>0.60 arcsec | -1.128 +/- 0.263% | -1.885 +/- 0.260% |
| MAG_AUTO<25.8, radius>=0.75 arcsec | +1.098 +/- 0.220% | +0.329 +/- 0.221% |

Uncertainties are common case-bootstrap standard deviations (10,000 draws,
seed20260914); flow sampling uses 64 antithetic draws with seed7301 and the
registered case/batch offsets. These are ConstGold +/-0.02 R11 ratios on
cases40--119, with strict original-precision true r<26 and no measured-|e| cut.

The near-0.3% configured-cut headline is not a demonstrated joint calibration:
the actual-U branch is still +1.10%, and other retained samples fail. In that
cut the flow response is effectively unchanged by replacing actual flags with
classifier weights (0.6961067 versus 0.6961231), while the blending contribution
rises from 0.1872262 to 0.1939847. The total measured response is 0.8930318.
Thus the favorable headline comes mostly from changed blending weights, not
an independently established improvement in either response component. No
checkpoint or classifier is selected on this result.

## Experiment 4: neighbour-induced measured-selection response

The current external emulator moves ellipticity but leaves radius and flux
unchanged. On the existing primary-unsheared/neighbour-sheared instrument,
that composition cannot move objects across a radius/flux cut through the
external shift alone. Test the missing effect directly using the 500 existing
cases400--899 in both original and complement anchor strata. No flow,
emulator fitting, or new image simulation enters this measurement.

On identities with usable measurements in both shear legs, define
`b=(e_plus+e_minus)/2` and `d=(e_plus-e_minus)/(2h)`. For independently selected
sets S_plus and S_minus, the following decomposition is exact at finite h:

```text
R_total = [mean(e_plus|S_plus) - mean(e_minus|S_minus)] / (2h)
R_within = [mean(d|S_plus) + mean(d|S_minus)] / 2
R_boundary = [mean(b|S_plus) - mean(b|S_minus)] / (2h)
R_total = R_within + R_boundary.
```

Use h=0.02 and the same uncut/0.60/0.75-arcsec predicates as Experiment 3.
Pool the two strata with their original inverse-sampling weights within each
case, then average cases equally and bootstrap cases (10,000 resamples,
seed20260917). The uncut boundary term must be exactly zero. The selected
boundary term diagnoses neighbour-induced output selection, but its value
cannot be transferred directly to ConstGold: the V2 anchor truth parent and
common-usability support differ. Preserve this distinction even if the term
has the sign needed to reduce the ConstGold residual.

`scripts/audit_neighbour_selection_response.py` performs this decomposition;
products are under `$DATA_DIR/sbsi_caches/neighbour_selection_audit_20260917_v1/`.

### Completed neighbour-cut audit

All 500 cases completed with exact decomposition closure and empty stderr.
For R11, the boundary term is `+0.0000114 +/- 0.0002993` at the 0.60-arcsec
joint cut and `+0.0014602 +/- 0.0008586` at the 0.75-arcsec joint cut. The
uncut boundary is identically zero. Both selected-cut 95% case-bootstrap
intervals include zero; the 0.75-arcsec term is about 1.7 standard errors from
zero. Total measured blending responses are respectively 0.1375230 (uncut),
0.1416114 (0.60 arcsec) and 0.1801601 (0.75 arcsec), each on this scene parent.

This does not support neighbour-induced radius/flux selection as the dominant
explanation of the percent-level ConstGold discrepancy. It leaves a possible
smaller contribution at 0.75 arcsec, but no response correction is applied and
no numerical transfer across populations is made. Priority remains measuring
the emulator's response amplitude on correctly matched domains and improving
the density-preserving optimization of the flow response objective.

## Next optimization decision

The independent-row loss is unbiased over row sampling, but the attempted
alternating recipe still produces large clipped gradients and degrades NLL.
A next candidate should reduce response-integral noise *before* taking a
guard step and evaluate density alongside it. One concrete route is a frozen
reference-flow control variate: estimate its full-population mean once, then
integrate `response(current)-response(reference)` with common draws, adding
back that reference integral inside the training estimator. At initialization
the sampled difference is exactly zero. This changes integration variance,
not the final model's predicted response, and the reference integration error
must be measured explicitly. It would still deploy one checkpoint, not a flow
ensemble. The optional reference estimator is now implemented in
`sampled_guard_response_backward`, with two independent integral replicas and
independent sampled row groups. Exact enumeration tests recover the original
population loss and gradient both at and away from the reference. Parameter
sharing with the reference is rejected; only the current flow receives a
gradient. Default training behavior is unchanged.

`scripts/audit_guard_control_variate.py` measures four independent reference
integrals over all 15,630,658 training pairs, each with four antithetic draws.
It then compares eight identically seeded minibatch gradients from the
ordinary estimator and two separately integrated reference banks. Record
minibatch variance, reference integration error, and between-bank gradient
agreement before choosing another optimization run. This audit makes no
parameter update and does not itself establish density or calibration gains.

### Completed variance audit and density-gated refinement

The full training-population audit completed20m07s with empty stderr. Across
eight identical minibatch streams, ordinary independent-row gradients had
RMS noise46.415, compared with1.967 and1.899 for the two reference banks.
Relative to each arm's mean gradient, those are98.6%,5.75% and5.58%.
The two independently integrated mean gradient directions have cosine0.9999523
and Euclidean distance0.3725 at mean norms34.1946 and34.0249. The largest
reference-integration standard error among guard/matrix components is
4.565e-5 for the four-group mean. These are integration diagnostics, not
uncertainties on scientific calibration.

`scripts/refine_guard_control_variate_flow.py` therefore tests six small
gradient-descent steps, starting at learning rate1e-5 and backtracking by
factors of two up to six attempts. A step must reduce the training response
score and keep training NLL within0.001nat per row of the original checkpoint.
Validation selects response improvements only within the same density budget,
with the original model included as step0; the final selected model must also
pass the density gate on all validation rows. This replaces the previous
alternating Adam updates with explicit bounded steps. No NLL tolerance is
interpreted as a calibration acceptance criterion.

The refinement uses one frozen bank, independent gradient/minibatch streams,
all validation response pairs, and no ConstGold data. Its subsequent joint
truth-parent diagnostic is skipped if step0 wins; otherwise it uses the same
80cases and streams as Experiment3 with the unchanged emulator/classifier.

### Completed refinement and case uncertainty

The six-step control-variate refinement completed and retained step 0. Each
step lowered training response loss and satisfied the training density budget;
validation guard loss increased from 0.0448341 to about 0.079. Selected full
validation NLL remains exactly -4.76270675245. The dependent joint evaluation
therefore skipped the unchanged model. Variance reduction solved a numerical
problem but did not establish an update that generalizes.

The additional full-training case audit uses 15,630,658 pairs, two independent
four-draw replicas, the same original flow and soft cuts. The self-response
trace biases are -0.88983 +/- 0.21148% uncut and -0.20992 +/- 0.18321% for
radius>3px and magnitude<25.8; validation gives +0.07043 +/- 0.42293% and
+0.70907 +/- 0.39759%, respectively. These are self-only matched-pair
influence responses, not the joint ConstGold result. Componentwise differences
between fitting and validation residuals are approximately 0.35--2.11 standard
errors for the displayed global/radius/joint cuts. The response target's
scene sampling uncertainty matters even after latent gradient noise is reduced.
This does not establish absence of systematic flow error. These case errors
hold baseline mass/mean-shape nuisance estimates fixed and exclude training
and integration uncertainty.

Sources: `guard_control_variate_20260917_v1/refinement/completed.json`,
`training_case_audit.json`, and
`guard_population_audit_20260917_v1/audit_0.json`. The audit CLI now accepts
`--split train` and an optional reference subset; its default remains validation.

## Experiment 5: native emulator amplitude and composition

`scripts/audit_neighbour_emulator_amplitude.py` compares the frozen fixed-g0
emulator with the exact same common-usable identities and per-leg selections
used in Experiment 4. Keep the full scene when forming the finite-k pair list,
then zero contributions from unsheared secondary anchors. Compare the native
per-unit-neighbour-shear scalar response with the primary-shape proxy used by
the likelihood, using the analytic finite +/-0.02 intrinsic-shape map.
No emulator parameters, output scale, or target are fitted.

Use the same three output cuts, inverse-sampling stratum weights, 500 cases,
and case bootstrap as Experiment 4. Report absolute paired residuals against
both total and within-pair measured R11; the boundary contribution is already
measured separately. This distinguishes response amplitude from cut motion
without subtracting an estimated flow response. Results live under
`$DATA_DIR/sbsi_caches/neighbour_amplitude_audit_20260917_v1/`.
The V2 anchor parent still differs from the full ConstGold parent, so these
residuals cannot be pasted into ConstGold as corrections.

### Completed amplitude audit: selected emulator deficit

All 500 cases completed on 3,441,882 common-usable anchor rows, with empty
stderr and exact selected-response decomposition closure. No predicted pair
had an unsheared secondary anchor, so masking those contributions does not
change this particular instrument. Native R11 results are:

| Per-leg measured cut | Measured response | Emulator response | Paired measured-minus-emulator |
| --- | ---: | ---: | ---: |
| Uncut | 0.1375230 | 0.1365491 | +0.0009740 +/- 0.0029603 |
| MAG_AUTO<25.8, radius>0.60 arcsec | 0.1416114 | 0.1383253 | +0.0032861 +/- 0.0030763 |
| MAG_AUTO<25.8, radius>=0.75 arcsec | 0.1801601 | 0.1657806 | +0.0143795 +/- 0.0022995 |

Errors use 10,000 paired case-bootstrap resamples, seed20260917, equal-weight
cases400--899 after inverse-sampling stratum pooling. The 0.75-arcsec deficit
has 95% interval[0.0098266,0.0188950] and is about6.25 standard errors from
zero. Removing the measured boundary term still leaves
`+0.0129193 +/- 0.0021387`, about6.04 standard errors. Neither the uncut nor
the 0.60-arcsec discrepancy is established at95% confidence.

The primary-shape proxy changes the 0.75-arcsec prediction by only-0.0000327;
it cannot explain the observed deficit. The evidence instead identifies
selected-population emulator amplitude as a priority. Possible sources include
transport from the fixed-g0 training selection and unmodelled correlation of
response with the measured-output selection at fixed truth features. This
audit does not distinguish those mechanisms or justify rescaling. A next
emulator experiment must use matched populations and independent scene
validation; its training belongs in BlendEMU. Flow optimization should not be
used to absorb this independently measured blending error.

## Experiment 6: full-domain output-conditioned response emulator

The selected response is an expectation of `S(y) B`, so a truth-only estimate
of `E[B|x]` need not reproduce it when response and measured radius/flux remain
correlated at fixed x. Test `E[B|x_pair,radius_0,flux_0]` as a conditional
response model. When composed with a flow, these output values must come from
the same generated joint draw. They are not extra observed conditions of the
flow density. At g=0 the response shift vanishes, leaving that density intact;
finite-shear density and likelihood normalization would still need validation.

The training implementation lives in BlendEMU:
`scripts/prepare_output_conditioned_response.py` and
`scripts/train_output_conditioned_response.py`. Read existing forward0.2
response labels, join the primary-role baseline g0 measurement by exact
identity, reject any baseline shape disagreement, retain strict true r<26,
and apply no measured science cut. Both label shapes must be finite and inside
the ellipticity disk; baseline magnitude must be finite and radius positive.
Missing/unusable baseline rows are dropped and counted. Retain the original
seven rescaled truth/pair features; the conditional arm adds log measured
radius in pixels and measured magnitude (equivalent to log flux at zeropoint30).

Fit two arms on exactly the same cases40--199 and rows: truth-only control
and output-conditioned candidate. Use ordinary squared error, the frozen
trial9 hyperparameters,218 trees,seed20260816, and no response weighting or
rescaling. Cases0--39 are case-disjoint development data and select no tree
count or hyperparameters. Prepared label standardization uses fitting rows
only. The chunked QuantileDMatrix implementation follows the installed
XGBoost3.0.5 [iterator API](https://xgboost.readthedocs.io/en/release_3.0.0/python/examples/quantile_data_iterator.html).
Both models are experimental path-only artifacts, outside configured model
loading; neither is selected using ConstGold bias.

SBSI's `sbsi/output_conditioned_response.py` verifies artifact hashes, feature
order and physical target units. The native500-case audit compares both arms
on the same independently reselected scenes as Experiment5. Those scenes lack
a g0 leg, so the conditional input uses the symmetric mean of plus/minus
physical radius and flux as an explicit finite-h proxy. This is a development
test, not acceptance: fresh zero-shear scene outputs and coherent multi-axis
validation are needed if the approach succeeds. Large forward0.2 training
labels, both-leg label usability, and the anchor-parent mismatch also remain
limitations. No numerical deficit from Experiment5 is used as a correction.

Training products live under
`$DATA_DIR/blendemu_runs/output_conditioned_response_20260917_v1/`;
native validation uses
`$DATA_DIR/sbsi_caches/output_conditioned_response_20260917_v1/`.

### Completed training and exact joint-draw prediction

CPU training completed both frozen 218-tree arms. On the 32,202,670 development
pairs, the selected radius>=3.75px/magnitude<25.8 scalar label mean is
0.01147197 per pair; predictions are 0.01078535 for the truth-only control and
0.01140462 for the output-conditioned arm. This supports the proposed
conditional correlation mechanism on the development instrument, but is not
a native small-shear or joint calibration result. The 500 native cases are
evaluated separately with the declared photometry proxy.

The experimental loader can sum pair predictions for each joint flow draw.
It pools draws only when they occupy exactly the same output-feature tree
cells for that primary. Predictions are piecewise constant there, so this is
an exact computational reuse, with no interpolation or response adjustment.
Tests compare with exhaustive per-draw predictions, including isolated
primaries and prediction chunk boundaries. It is now an explicit optional
candidate in the truth-parent diagnostic, using hash-verified frozen pair
features. Their original predictions must exactly reproduce the baseline
emulator lookup before candidate evaluation. The configured likelihood is
unchanged. The feature interface requires positive finite radius/flux, with
an explicit exact tree-limit path for float32 softplus underflow to zero in
generated draws; physical outputs and cuts remain unchanged. A test checks
that this limit equals predictions at arbitrarily small positive values.

### Completed independent native validation

Both 218-tree arms were evaluated on all 500 native cases, using the same
3,441,882 common-usable anchor rows and fixed inverse-stratum weights as
Experiment 5. At radius>=0.75 arcsec and magnitude<25.8, the original emulator
predicts 0.16578059, the truth-only control 0.16375420, and the conditioned
candidate 0.17026532, versus measured total R11=0.18016009. Conditioning
recovers part of the discrepancy, but leaves total residual
+0.00989477 +/- 0.00229968 and within-pair residual
+0.00843455 +/- 0.00213917. This is not a successful calibration model.

At radius>0.60 arcsec, the conditioned prediction is 0.13838043 versus
measured 0.14161140 (residual +0.00323097 +/- 0.00307323). Uncut, it predicts
0.13467474 versus 0.13752302 (residual +0.00284828 +/- 0.00295789).
These are native neighbour-response numbers, not joint multiplicative biases.
The midpoint-photometry proxy and fixed-g0 training population limitations
remain. Source: `output_conditioned_response_20260917_v1/result.json`.

The joint A/B keeps the single original flow, classifiers, complete truth
parent, 64-draw common streams and all declared cuts unchanged. Each candidate
predicts its response using the same generated radius/flux draw used in
selection. The composition retains the original intrinsic-shape shear
increment and shape-only shift, isolating the emulator change. Both arms and
all branches are reported; ConstGold does not choose a model or tuning value.

## Experiment 7: directional information in scalar response labels

The scalar emulator predicts the response parallel to each random secondary
shear, averaged over its direction. This identifies an orientation-averaged
response, whereas the native coherent-g1 experiment measures R11. To test
whether a scalar response discards a relevant component difference, reconstruct
the measured response vector d from the stored parallel/cross responses and
the sky-cos-sin half-angle. For a uniform direction u, the tensor label
`2*d_a*u_b` averages to R_ab in the linear regime. Its trace/2 is exactly the
original scalar label; a synthetic full-matrix test checks this identity and
the recovered off-diagonal responses.

`scripts/audit_response_label_tensor.py` reads and hash-checks the frozen
prepared parts, then reports fitting and development cases separately for
the same three output cuts. It reports sums per represented primary, angular
second moments and case-bootstrap errors; primaries without any prepared pair
are excluded and explicitly outside this estimand. This is a diagnostic of
existing forward0.2, both-leg-usable labels, not a small-shear calibration
measurement. The summed labels also contain other-neighbour instrument noise.
No component correction or tensor model is fitted by this audit.

### Completed directional-label audit

At radius>=3.75px/magnitude<25.8, R11 minus the scalar trace response is
+0.00033616 +/- 0.00085898 on the 160 fitting cases, and
+0.00045716 +/- 0.00145315 on the 40 development cases. The other cuts also
show no significant diagonal difference; direction second moments are close
to the isotropic half-identity matrix. These training labels do not support
an axis-anisotropy explanation of the native selected deficit. This does not
validate a scalar model on every population. Source:
`output_conditioned_response_20260917_v1/label_tensor.json`.

## Experiment 8: matched forward-amplitude audit

The existing random-neighbour training scene family has measured primary
outputs at 0, 0.02, 0.05 and 0.2 for all 200 cases. Compare forward response
labels at the three nonzero steps on exactly the same prepared pairs, cuts
on the baseline outputs, and primaries usable at every step. Verify that all
non-shear intrinsic catalogue fields match exactly and shear vectors scale
by the declared amplitude. Reconstruct the original 0.2 labels and require
agreement before comparing smaller steps. Report all dropped pair rows and
case-bootstrap uncertainty of paired amplitude differences, with fit and
development cases separate.

This tests the large-step training-label limitation without fitting a response
offset or allocating new rendering. It measures random-direction, fixed-g0-cut
responses on the common-usable population; it is not the native coherent-shear
or joint inference acceptance estimand. Source implementation:
`scripts/audit_response_label_amplitude.py`; output root:
`output_conditioned_response_20260917_v1/`.

### Completed amplitude audit

**Interpretation limitation:** subsequent remeasurement found a leg-dependent
marginal mismatch and historical centering alternatives. These original-leg
differences are not a clean physical step-size test until all legs have a
matched measurement definition; see the
[centering investigation](#measurement-centering-investigation).

The 200-case scan passed exact intrinsic-scene and scaled-shear checks.
At the >=0.75arcsec cut, the fitting scalar responses are 0.11425372,
0.09890155 and 0.09833789 for steps 0.02, 0.05 and 0.2, respectively.
The paired 0.02-minus-0.2 difference is +0.01591583 +/- 0.00721672; the
0.05 difference is only +0.00056366 +/- 0.00285458. Development cases give
-0.01075815 +/- 0.01261160 and +0.00309400 +/- 0.00628646. Smaller-step
labels are noisy, and these split results do not support replacing the
training target by a fitted amplitude correction. Source:
`output_conditioned_response_20260917_v1/label_amplitude.json`.

## Experiment 9: paired optimizer-stream measurement audit

BlendEMU's existing ngmix measurement creates an unseeded `RandomState` for
each fit. Its optimization guesses are therefore not coupled across shear
legs, even when the intrinsic scenes match. This is a possible source of the
large small-step response variance, not an established explanation of bias.

The measurement function now accepts an optional explicit RNG while retaining
the historical default. A separate BlendEMU audit remeasures 512 deterministically
sampled, old-all-leg-usable true-r<26 primary objects in each of cases 0 and 1
on the existing 0/0.02/0.2 images. It uses two independent optimizer replicas,
sharing each object's stream across shear legs, and tests identical-seed
reproducibility on an actual stamp. It estimates optimizer-induced response
variance with common and independent streams, reports all failed fits, and
stores new shapes separately. No source measurement catalogue is overwritten.
Image-noise stream equivalence is not established by this test. Implementation:
BlendEMU `scripts/audit_measurement_seed_response.py` and optional `rng` in
`blendemu/shape.py`.

### Completed optimizer probe and expanded pilot

Both 512-object probes completed with all 1,024 sampled objects usable in
all replicas and no fit exceptions. At h=0.02, common/independent optimizer
variance ratios for (e1,e2) are (0.04468,0.00563) in case0 and
(0.00835,0.00715) in case1: approximately 95--99% variance reduction. At
h=0.2 the ratios are 0.13--0.28. These estimates isolate optimizer noise
using two repeated fits per image; they do not measure total response-label
variance or calibrate a model. Identical-seed replay passed on actual stamps.

The expanded pilot remeasures complete true-r<26 primary populations in cases
0,1,40,41 on existing 0/0.02/0.2 images. It retains each leg's detection
positions and does not condition measurement work on old fitted shapes.
Eight CPU workers per case use the registered object/case seed identically
across amplitudes. Outputs and fit-status arrays are separate from original
catalogues under `blendemu_runs/common_optimizer_primary_pilot_20260917_v1/`.
Only ngmix columns for the declared primary population are remeasured; other
columns remain original. The pilot records missing detections, invalid
positions, fit failures and nonzero fit flags.

Dependent validation checks exact replay of the earlier sampled fits,
compares pair-label variance and forward-amplitude responses on common rows,
and records all exclusions. Coupling optimizer streams preserves their
marginal sampling rule, but can change a both-leg-usable subset; this
population effect must be checked before using new labels for training.
Neither the probe nor the four-scene pilot is calibration acceptance.

### Per-leg marginal gate before training-label replacement

The expanded audit also compares the original and remeasured per-leg
distributions on the complete declared truth parent. Each arm uses its own
usable objects. It records selected mass, both directions of fit-success
disagreement, mean shapes and second shape moments, with paired object
influence errors. The old and new radius/magnitude columns must agree exactly.
Restricting this check to common-success objects could hide a distribution
change, so common-row restrictions apply only to the separate label-variance
comparison. The paired object errors condition on the four fixed scenes;
they do not establish independent-scene density equivalence. Tests verify
different success populations and exact cancellation for identical data.

The four geometric-centre pilot jobs and audit16556497 completed. The earlier
512-object probes replay exactly in cases0 and1. On common prepared pairs,
h=0.02 label variance is 5.7--6.9% of the original for uncut/>0.60arcsec
and 12--14% at >=0.75arcsec. However, the marginal gate failed: the measurement
change cannot be treated as a variance-only replacement. Source:
`output_conditioned_response_20260917_v1/common_optimizer_pilot.json`.

### Measurement centering investigation

Across cases0,1,40,41, geometric-centre remeasurement of the original g=0 and
g=0.2 primary images increases mean e2 by 0.00195--0.00352 (3.3--5.9 paired
object standard errors) and second shape moments by roughly 0.00084--0.00149
(4--7 errors). The g=0.02 comparisons largely agree within optimizer noise.
These errors condition on the four scenes. Original and new photometry agree
exactly, and the marginal check retains each arm's own usability.

Historical BlendEMU commit `46107b8` contains an explicit subpixel Jacobian
centre in `shape.make_obs`/`scripts/run_shape.py`; the current checkout used
the geometric stamp centre. Shape file dates also differ between shear legs:
case0 primary g=0/0.2 products are dated July15, while g=0.02 is dated June25.
Secondary g=0 is July15 and g=0.05 is May27. Dates and source history motivate
a centering hypothesis; they do not by themselves identify the convention
used to produce each file. In particular, the current flow's secondary
0/0.05 training pair also needs a provenance check.

The archived work log's July15 cont.39 entry explicitly records remeasurement
of all 280 ConstGold shape catalogues and 200+200 main-family g=0 primary and
secondary catalogues with subpixel centering. Its cont.40 entry records that
g=0.05 products were stale and describes a subsequent primary-only rebuild.
The August antithetic-self investigation also explicitly preserves geometric
centering for the old +0.02 secondary products. These historical records
establish that mixed measurement conventions existed; empirical rescaling
proposals in those old entries are not part of the current workflow.

Current provenance audit16556676 completed22s: all 391,435 flow-domain shape
targets in cases0/1, at g=0 and g=0.05, exactly match the corresponding raw
secondary shape catalogues (maximum difference0, unmatched0). Thus the old
catalogues are directly relevant to the present flow, rather than merely
historical artifacts. Source:
`output_conditioned_response_20260917_v1/flow_measurement_provenance.json`.

An explicit optional `gal_cen` argument restores the historical comparison
without changing the current default. BlendEMU's pilot now accepts
`--centering geometric|subpixel`, with geometric still the default. Job16556618
remeasures case0 primaries at 0/0.02/0.2 with subpixel centering, identical
images, positions and object optimizer seeds. Dependent audit16556648 compares
with both original measurements and the completed geometric pilot, checking
source identities and reporting paired first/second moments and usability.
No full label replacement or retraining follows until this mismatch is
understood. Each pilot preserves its measurement source in `code_snapshot/`.
Sources: `blendemu_runs/common_optimizer_{primary,subpixel}_pilot_20260917_v1/`;
the pending result is `output_conditioned_response_20260917_v1/centering_pilot.json`.

A complementary fingerprint samples 4,096 previously usable true-r<26 objects
per catalogue, fitting both centres with two optimizer replicas. The seven
CPU tasks16556668 cover secondary 0/0.05 in cases0/1, ConstGold case40 +0.02,
and native case400 +/-0.02. They compare pointwise squared discrepancies
from the old catalogues and report common-fit exclusions and optimizer
variability. These sampled common-success comparisons can identify a likely
historical measurement recipe, but do not replace the full-parent marginal
gate and cannot rule out additional historical algorithm changes. Source:
BlendEMU `scripts/audit_measurement_centering.py`; output files
`output_conditioned_response_20260917_v1/centering_fingerprint_0.json` through
`centering_fingerprint_6.json`.

The first two fingerprint tasks completed, but their full-sample squared-error
differences between centres are smaller than optimizer variability. A fixed
stratified summary (true r<24 and measured magnitude<24, alongside the full
sample) is scheduled as16556685 to distinguish the centering signal in more
stable bright fits. These strata are used only for measurement provenance;
they do not change any calibration population or select a model.

All seven fingerprint jobs and their summary subsequently completed. Every
squared-error difference is within1.32 paired object errors, including the
bright strata. This small-sample metric is inconclusive because optimizer
variability dominates. Native case400's shape file is dated August12, so the
current checkout's default alone cannot establish its historical centering.
The larger per-leg marginal pilot supplies the discriminatory measurement
evidence; the fingerprint does not establish a native-family mismatch.

The full case0 subpixel pilot16556618 and audit16556648 subsequently completed
in20m26/17s. All first/second shape-moment discrepancies from the old g=0 and
g=0.2 catalogues are within1.86 paired object errors across the declared cuts.
At g=0.02 the subpixel change instead moves the uncut mean e2 by -0.002431
(4.03 errors) and the two second moments by -0.001146/-0.001387
(5.45/6.51 errors). Coupled geometric-versus-subpixel fits reproduce the
direction of these shifts in every leg. Together with the historical record,
this identifies a centering mismatch in the current random-neighbour source
family. A single scene's marginal agreement is not a density-equivalence
certificate.

### Matched-centering secondary training pilot

Separate BlendEMU script `remeasure_common_optimizer_targets.py` generalizes
the validated measurement driver to explicit structural role and amplitude
lists. Its defaults do not change production measurement behavior. Pilot
array16556705 remeasures full true-r<26 secondary populations in cases0 and40
at g=0/0.05, with explicit subpixel centering and common object seeds. This
uses two fitting scenes under the flow's actual seed501 grouped split. It retains original images,
detection positions and photometry, records all failures, and writes under
`blendemu_runs/common_optimizer_secondary_pilot_20260917_v1/`, including a
source snapshot. Both jobs passed4,096 rows with empty stderr.

Dependent audit16556719 checks per-leg marginal distributions with independent
usability, exact unchanged photometry and parent identity, and the 2x2 forward
random-direction self-response tensor on common usable objects under the
three fixed-g0 cuts. It also compares label variance. Its g=0 check tests
compatibility with the existing density data; g=0.05 intentionally changes
the centering convention. This is preparation for harmonized training, not
a response correction, a ConstGold fit or calibration acceptance. Full-data
remeasurement and retraining have not yet been launched.

The emulator's cases0--39 development split differs from the flow's shuffled
case split. Checking the actual flow manifest shows both0 and40 are fitting
cases; case41 is held out. The pilot is extended to case41 before drawing
training-versus-validation conclusions, with the downstream audit waiting
for all three cases. The established full-data split will be preserved.

### Harmonized flow-domain preparation

`scripts/prepare_harmonized_flow_domain.py` reconstructs the true-r<26
secondary parent from raw truth, joins the existing truth-only crowd features,
and applies validity independently to each new measurement leg. Updating only
the old valid-row cache would omit newly successful fits and is not used.
The builder checks invariant truth, expected shear vectors, source hashes,
unchanged radius/flux and compatible conditions on shared old/new rows. It
reports old-only, new-only and unmatched rows, and stores full-parent usable
flags separately. The baseline flow's global case split is inherited exactly.

Pilot preparation16556746 waits for the three-case measurement audit16556737.
Dependent density audit16556752 evaluates the original single checkpoint on
both original and harmonized targets, in its unchanged standardized-target
measure. Full per-leg NLL uses each arm's own valid population; paired NLL
differences use shared rows with paired object errors. This is a diagnostic
of the measurement change, not an improvement to a trained model. Outputs:
`output_conditioned_response_20260917_v1/harmonized_pilot_domain/` and
`harmonized_pilot_density.json`.

### Completed secondary pilot and full rebuild

All three measurement cases and the dependent audit/preparation/density stages
completed. New g=0 moments agree with the old ones within1.85 paired object
errors. At g=0.05, uncut e2 decreases by0.00328--0.00398 (5.5--6.8 errors)
and second shape moments also decrease. This confirms that the original
secondary0/0.05 pair used inconsistent centering. Common-seed label variance
is0.295--0.318 of the old value uncut/at0.60arcsec and0.351--0.371 at0.75arcsec.
Individual-case response differences remain noisy and are not calibration
estimates or a fitted offset.

The frozen flow's paired g=0 NLL changes are -0.001386 +/- 0.002539,
+0.002055 +/- 0.002535 and +0.001880 +/- 0.002491nat per row in cases0,40,41.
At g=0.05 the changes are -0.016075 +/- 0.002596,
-0.010895 +/- 0.002549 and -0.011908 +/- 0.002676. These measure compatibility
of the new data with the same model, not an improvement in trained density.
Preparation retains33 newly successful leg rows and drops15 old-only rows
across the three cases, with unchanged shared-row photometry and contexts
agreeing within the declared floating-point tolerance.

Full measurement array16556786 now processes the remaining197 cases with
eight concurrent eight-CPU jobs; validated cases0/40/41 are linked into the
new root. The entire BlendEMU Python package and measurement driver are
snapshotted, with package versions and source hashes, before submission.
The snapshot matches the pilot code. It uses no GPU. Dependent full-domain
preparation16556795 will preserve the existing200-case split. Outputs:
`blendemu_runs/harmonized_secondary_20260917_v1/` and
`sbsi_caches/harmonized_secondary_20260917_v1/domain/`. No training or
calibration acceptance is implied by launching the measurement rebuild.

## Joint candidate execution

The CPU joint smoke was superseded after its first case because tree prediction
dominated runtime. A GPU prediction check on 171,608 frozen feature rows found
maximum standardized prediction difference 2.98e-8 from CPU. Ordinary DMatrix
GPU prediction works on the local virtual A40 and avoids the earlier quantile
training failure. The GPU joint smoke finished both cases in 2m08s with empty
stderr, compared with about fourteen minutes for the first CPU case. This is
an execution change, not retraining or a change in response composition.
The full 80-case A/B runs both frozen arms with the original single flow and
64 common draws, followed by paired comparisons with the existing baseline.
The CPU loader remains the default; the launcher explicitly requests CUDA.

### Completed truth-only joint control

The truth-only arm finished all80 cases. Its flow checkpoint, every per-case
flow numerator and every selected model denominator are identical to baseline;
only the emulator contribution changes. With actual usable flags, full-parent
joint m is -2.69186 +/- 0.26761% uncut, -0.59811 +/- 0.26415% at >0.60arcsec,
and +1.56819 +/- 0.22039% at >=0.75arcsec. Changes from the original emulator
are respectively +0.75889 +/- 0.00340, +0.52970 +/- 0.00270, and
+0.47020 +/- 0.00249 percentage points under paired case resampling.

Using modeled usability, the corresponding biases are -3.57799%, -1.41221%,
and +0.74409%. Thus the wider-population truth-only training control improves
some cuts and worsens another; it is not a calibration solution. No arm is
selected on these results. The output-conditioned arm remains a separate
frozen comparison. Sources: `output_conditioned_response_20260917_v1/`
`joint_truth_only.json` and `joint_truth_only_comparison.json`.

### Completed output-conditioned joint comparison

The conditional arm16556472_1 completed all80 cases in1h04m01. Its flow
checkpoint matches baseline; all162 inspected flow-numerator/denominator
arrays across cohorts, cuts and usability arms match exactly (maximum
difference0). Thus the paired changes isolate the emulator composition.

On the full true-r<26 parent with actual usable flags, m is
-1.96343 +/- 0.26988% uncut, -0.62183 +/- 0.26426% at >0.60arcsec, and
+0.71947 +/- 0.21885% at >=0.75arcsec. The respective changes are
+1.48732 +/- 0.00557, +0.50598 +/- 0.00332, and
-0.37852 +/- 0.00316 percentage points under paired case resampling.
The model improves these three comparisons, but none establishes the0.3%
goal. With modeled usability the biases are -3.06097%, -1.62785%, and
-0.27191%. The last number cannot substitute for the actual-U calibration
check or resolve the physical-support failure below. These use the same
80 scenes,64 paired draws and antithetic +/-0.02 R11 convention as the
baseline. No candidate is promoted. Sources:
`output_conditioned_response_20260917_v1/joint_output_conditioned.json` and
`joint_output_conditioned_comparison.json`.

### Density identity and physical-support check

For fixed truth atom x and shear g, write v=(radius,flux),
a=e_truth(g)-e_truth(0), and b(v,x) for the summed conditional pair response.
The experimental shape translation and its inverse are

```text
T(e,v)     = (e + b(v,x)*a, v)
T^{-1}(e,v)= (e - b(v,x)*a, v).
```

This map preserves volume: for each v it is a translation in e. Consequently
the translated density is exactly `p_flow(e-b(v,x)*a,v | x(g),U)` on its
translated support, with no Jacobian correction. The argument uses conditional
translation and remains valid at the tree model's discontinuous split
boundaries. At g=0 the density is identically the original single-flow density.
Because the declared science cuts depend only on v, their normalization mass
is also unchanged by this translation for a fixed atom and shear.

This identity proves normalization over the translated shape support; it
does not prove calibration or preservation of the original |e|<1 usable
domain. Both fixed and conditional blend translations can potentially move
draws outside that disk. A separate 64-draw two-case diagnostic compares
flow-only, original scalar and conditional translations, retaining the same
draws and cuts. It records existing and newly invalid shape mass without
clipping or renormalizing. It is a conditional-flow sampling check, not a
classifier-weighted population bound. The configured likelihood remains
unchanged. Implementation: `scripts/audit_conditional_response_support.py`.

The two-case audit completed as job16556609 after a report-writing path-type
failure was repaired. At >=0.75arcsec, the conditional translation places
0.1004--0.1095% of selected draws outside the open disk; the original scalar
translation gives a similar 0.0992--0.1074%. Flow-only violations are
0.00085--0.00124%, including radial-tanh floating-point saturation.

Small invalid mass does not imply a small response contribution. Using the
original selected denominators, the outside-disk draws contribute R11=0.039095
to the conditional model's >=0.75arcsec response and 0.038118 to the scalar
model's response (flow-only -0.00001186). The conditional contributions are
0.037169 uncut and 0.029896 at >0.60arcsec. These are two-case conditional-flow
diagnostics, without classifier weighting; they are not full-population bias
estimates and must not be subtracted as a correction. Physical clipping or
renormalization would change the density and estimator. A support-preserving
response composition requires its own matched training target and exact
density accounting before promotion. Source:
`output_conditioned_response_20260917_v1/conditional_support.json`.

### Experimental disk-preserving response transport

`sbsi/disk_response_transport.py` implements an experimental alternative for
the support problem. In complex shape notation it uses
`T(e,t)=(e+t)/(1+conj(t)*e)`, with the shift t obtained by applying radial tanh
to a two-component velocity. For a fixed interior t, the inverse is T(e,-t),
and the shape log-Jacobian is
`2*log(1-|t|^2)-2*log(|1+conj(t)*e|^2)`.
The map preserves the disk mathematically and has an exact density. If
velocity depends on radius/flux but not shape, the full four-output Jacobian
is triangular and has this same determinant. At zero velocity the flow
density is unchanged. Numerical boundary rounding remains explicit; the
implementation never clips shapes.

Existing physical R_blend numbers cannot simply be inserted as the velocity.
At zero velocity the physical response matrix is

```text
L(e) = [[1-e1^2+e2^2, -2*e1*e2],
        [-2*e1*e2, 1+e1^2-e2^2]].
```

For a unit shear direction u, the gain s=u^T L(e)u is positive for interior
shapes. A possible scalar transport fit uses the measured parallel response z
and conditional moment objective `E[0.5*s*b(x)^2-z*b(x)]`; its gradient is
`s*b-z` and its Hessian s. This fits physical response moments rather than
averaging inverse-transformed labels. It still requires compatible baseline
measurement distributions, finite-amplitude validation and tensor checks.
The fitting experiment below uses this objective. The configured likelihood
has not been changed.
Tests check inversion, density Jacobians, physical support, zero identity,
the local response derivative and the joint Jacobian with auxiliary-output
dependence. This is a tested mathematical component, not a calibration result.

### Physical-moment transport fit and independent checks

The 2026-09-18 experiment fits a new nine-input transport coefficient using
the same 161,012,399 primary-role g=0/0.2 response pairs, grouped fit cases
40--199 and development cases0--39, trial9 tree settings and218 trees.
It rejoins the original baseline shapes with exact source receipts and hashes.
The numerically positive gain is evaluated as
`s=(1-|e|^2)*|u|^2+2*(e dot u_perp)^2`; no gain or shape is clipped.
The loss gradient and Hessian are `s*f-(z-s*mean)/std` and s for the
standardized coefficient `b=mean+std*f`. A one-tree test verifies recovery
of physical conditional means, including a nonzero mean/scale, rather than
the generally incorrect average of z/s. The training intercept comes only
from the fit population. Neither development nor ConstGold selects a tree count.

BlendEMU owns `prepare_disk_response_moment.py` and
`train_disk_response_moment.py`. Preparation16556819 completed in5m16s;
training16556820 completed14m44s on32 CPUs, MaxRSS11.14GiB. Gain preparation retains all128,809,729 fit
and32,202,670 development pairs. The model root is
`$DATA_DIR/blendemu_runs/disk_response_moment_20260918_v1/model`.
The artifact format and loader explicitly distinguish disk velocity from
physical additive response, and reject cross-loading the two conventions.
The single218-tree artifact hash is
`194c1d11b67d6fd52039060270663e8e1cd610b478b9e9a66e34271701761fe7`.
Development pair-mean physical prediction/label are0.00813300/0.00817961
uncut,0.00915566/0.00926279 at>0.60arcsec and0.01140667/0.01147197
at>=0.75arcsec. These are pair-weighted response-label diagnostics, not
catalogue calibration estimates or independent pair uncertainties.
`transported_physical_log_prob` evaluates the inverse map and physical
four-output density Jacobian. Independent quadrature over the observed disk
checks normalization to1e-12 relative tolerance and preservation of the
auxiliary marginal; zero shear is exactly the base density and invalid
physical outputs have zero density. This helper is experimental and does
not include usability or catalogue selection.

SBSI's experimental joint evaluator composes the unchanged single flow with
`T(e,radial_tanh(sum_pair_b*external_shear))`, evaluating b on each draw's
radius/flux. It records strict boundary failures and excess beyond1e-12
separately; the latter is a report threshold, not a repair. The exact shape
Jacobian is checked for finite values. The flow contribution and selected
denominator retain the same draws and cuts. GPU smoke16556840 checks cases
40/41 after training. CPU native smoke16556841 checks independent cases
400/401, comparing both the finite transport and its linear derivative to
measured R11/R21. Density smoke16556926 follows the GPU smoke on the same
one-GPU dependency chain. `audit_disk_candidate_density.py` evaluates the
actual physical log density at observed g=0,+/-0.02 outputs, with model and
measurement hashes, per-case/leg/cut paired NLL changes and case bootstrap
errors. It verifies exact g=0 density identity. Its baseline is the unchanged
self-response flow, and the cut normalization cancels in paired differences
because the transport leaves radius/flux fixed. Full runs wait for those
implementation checks; density results do not select a checkpoint.
Native smoke16556841 completed successfully. Its finite-minus-linear R11
change is approximately-1.6e-5 uncut and-2.2e-5 at>=0.75arcsec, with no
numerical failure. This is not evidence for physical calibration from two
scenes. Full native array16556956 covers cases400--899 in20 blocks with
eight concurrent four-CPU tasks; summary16556957 requires all blocks.

The native audit uses the midpoint of the available +/- measured outputs;
there is no actual g=0 leg, and an O(h^2) proxy error remains. Training
g=0/0.2 centering is compatible, but its forward finite-amplitude label is
still only an approximation to the zero-shear derivative. Tensor closure,
finite-amplitude transfer, full-parent actual-U calibration, density quality,
and harmonized self-response retraining remain acceptance requirements.
The experiment does not reuse an old coefficient as a disk velocity or
apply an empirical correction.

### Completed physical-moment transport validation

All500 native cases and summary16556957 completed. The disk model predicts
R11=0.134817715 uncut,0.138591904 at>0.60arcsec and0.170402653
at>=0.75arcsec. At the latter cut the measured response is0.180160094;
the paired total deficit is **0.009757441 +/- 0.002299357**, consisting of
within-selection deficit0.008297211 +/- 0.002138802 and measured boundary
response0.001460229 +/- 0.000858585. Finite transport differs from its
linear derivative by-0.000020498 at this cut. Predicted R21 is-0.000051626
versus measured-0.001948300 +/- 0.002377138. This preserves the previously
observed selected neighbour-response deficit; physical support alone does
not solve calibration.

Joint smoke16556840 completed3m51s. All108 compared flow-numerator and
denominator arrays exactly match the baseline for the same two cases.
No generated shape exceeds squared norm1 by more than1e-12; the largest
excess is1.11e-15, with explicit boundary-rounding counts retained. Actual-U
full-parent m at>=0.75arcsec is+0.69803 +/- 0.49201% on only two cases,
which is not an acceptance result.

Density smoke16556926 failed because ConstGold has no separate g=0 image
directory. The audit now loads the original g0 targets identified by the
truth-parent's exact source hash and verifies membership. These are reused
flow-domain measurements, not independent density validation. Immediate
retry16556983 still executed the old source on the compute node; its traceback
and unchanged bytecode timestamp are consistent with filesystem cache delay.
The successful retry16556987 ran an immutable hash-named source copy under
the experiment's `code/` directory. It verifies exact zero-shear identity;
at>=0.75arcsec paired physical NLL changes are+0.0002473 atg=+0.02 and
+0.0003008 atg=-0.02, on two cases. No nonfinite scored density was discarded.

Full observed-density16556993 completed4m09s on all80 cases. Exact g=0
density identity holds throughout. Atg=+0.02, paired NLL increases are
0.00004143 +/- 0.00001968 uncut,0.00004479 +/- 0.00002016 at>0.60arcsec,
and0.00009112 +/- 0.00002588 at>=0.75arcsec. Atg=-0.02 they are
0.00015418 +/- 0.00002364,0.00016627 +/- 0.00002559 and
0.00022934 +/- 0.00002885 respectively. These small, in several cases
statistically resolved regressions are below the earlier0.001nat density
budget, but are not a density-quality improvement. Normalization and support
do not establish conditional distribution accuracy.

Joint-response16556994 completed all 80 cases in 2h14m23s. Actual-U joint m is
-1.963760 +/- 0.269947% uncut, **-0.623079 +/- 0.264152% at >0.60 arcsec**,
and +0.741934 +/- 0.218576% at >=0.75 arcsec. Modeled-U values are
-3.052185%, -1.619293%, and -0.236583%, respectively; they cannot substitute
for actual-U calibration. The primary 0.60-arcsec result improves by 0.504730
percentage points over the original emulator, but does not establish the
0.3% target. At this same cut, independent native response residual is
0.003019496 +/- 0.003075559, so the clear native deficit at 0.75 arcsec should
not be transferred to the primary population. Harmonized-domain density and
reference checks now run serially after repaired full-domain preparation.
The configured likelihood and model remain unchanged. Full SBSI validation
after the transport/density additions: **414 passed,2 skipped** in29.08s,
with the existing NumPy warning; the later audit path fix passes Ruff.

### Matched-centering primary amplitude pilot after transport validation

The native deficit survives the physical transport, while the old primary
g=0.02 amplitude comparison is confounded by centering. Array16557008
therefore extends the successful subpixel case0 measurement experiment to
cases1/40/41, covering emulator development and fit cases, atg=0/0.02/0.2.
It uses the already frozen measurement driver/package from the secondary
rebuild, fixed images/detection positions, identical per-object optimizer
seeds across amplitudes, and the full structural primary true-r<26 parent.
It runs three eight-CPU tasks and writes only new outputs under
`$DATA_DIR/blendemu_runs/primary_amplitude_pilot_20260918_v1/`.

Dependent CPU audit16557012 compares independent per-leg marginals, fit
success, paired physical response labels and their variance. It also
compares the same-seed geometric pilot, separating centering effects from
optimizer draws. No model is fitted to these three cases, and no full
primary rebuild has been submitted. This establishes whether compatible
small-amplitude training labels can address the remaining native response
deficit without using ConstGold as a fitted correction.

### Completed fresh primary amplitude comparison

All four subpixel primary cases (0, 1, 40, 41) are complete. Marginal audit
16557012 finds g=0 and g=0.2 differences within 2.83 paired object errors,
while g=0.02 differences reach 6.30--6.47 errors, confirming the historical
centering mismatch. This old-accepted-pair audit diagnoses optimizer variance;
it does not define a new training population.

BlendEMU `prepare_harmonized_response_case.py` instead rebuilds each amplitude's
pairs from its own new valid measurement legs, retaining structural neighbours,
strict primary true-r<26, baseline photometry, and no measured science cut or
old accepted-row gate. Optional secondary identities and bounded worker count
were added to `blendemu.response.retrieve_response` without changing defaults.
Preparation16557030/16557033 and SBSI audit16557060 completed. The latter joins
exact primary-secondary identities, checks source provenance and shear direction,
uses raw-precision photometry for cuts, and clusters uncertainty by primary
before comparing amplitudes. Jobs16557053/16557054 were cancelled after a
launcher syntax defect and replaced by validated16557059/16557060.

At the primary >0.60-arcsec cut, common-pair h=0.02 minus h=0.2 response
changes are +0.027722, +0.021795, -0.008253, and -0.001133 for cases0/1/40/41.
The equal-case mean is +0.010033, with case standard error 0.008710 and
conditional primary-cluster standard error 0.012660. These four scenes do
not establish a finite-amplitude shift. Each scene also reports 3,779--4,164
small-amplitude-only pairs and 241--373 large-amplitude-only pairs separately.
Despite optimizer noise reduction, small-amplitude labels retain about twenty
times the large-amplitude variance. No pilot-trained emulator or full primary
rebuild has been launched. Result: `disk_response_moment_20260918_v1/`
`fresh_amplitude_pilot.json`.

### Full-domain generated-role boundary repair

All 200 secondary measurement cases completed. Initial preparation16556795
failed because half the rendered row count is not the generated primary/secondary
role boundary when rendering omits objects. The preparer now applies the role
and true-r cut to the exact hash-verified generated catalogue, asserts equality
with the full measured parent, then joins rendered identities and reports any
unrendered rows dropped. Existing shear, photometry, context and source checks
remain. A regression test includes omitted rendered rows and rejects an
incorrect measured parent. The failed empty output is preserved as
`domain_failed16556795`; retry16561334 has passed its first full-case outputs.
Density16556887 now depends on that retry, and reference16556888 follows density.

### Full harmonized-domain checks before further flow optimization

After all 200 secondary cases and repaired domain preparation16561334 complete,
GPU density audit16556887 compares original and harmonized targets under
the same frozen single flow, reporting separate leg/case populations and
paired NLL changes. Reference audit16556888 follows it on one GPU and
recomputes the four training-population response integrals and independent
gradient-bank agreement against the corrected labels. It uses the existing
guard bank and seeds, and performs no optimization. Both stages require
successful predecessors, so together they occupy at most one GPU alongside
the separate joint candidate evaluation. They do not reuse the reference
integrals from the old mixed-centering domain. Outputs live under
`$DATA_DIR/sbsi_caches/harmonized_secondary_20260917_v1/`.

### Completed full harmonized-domain preparation

Retry16561334 completed in 13m05s with empty stderr. The generated secondary
boundary is identity349784 in every case. Six generated-parent identities lack
rendered truth and are explicitly dropped, leaving 22,192,875 parent rows in
each leg. Independent validity retains 19,601,281 g0 and 19,597,477 g=0.05 rows.
Relative to the old valid caches, g0 retains869 newly successful rows and drops
893 old-only rows; g=0.05 retains884 new rows and drops907 old-only rows.
All detection-to-shape joins are complete. The original160/40 case split,
shared-row photometry, truth contexts and shear checks pass. Density audit
16556887 has started successfully on this complete domain.

### Completed full harmonized density audit

Density16556887 completed all200 cases in1m52s with empty stderr. On the40
held-out scenes, new-minus-old paired NLL under the unchanged flow is
+0.0004301 +/- 0.0004487 nat per row at g=0 and
-0.0142782 +/- 0.0005131 at g=0.05. On160 training scenes the changes are
-0.0001055 +/- 0.0002054 and -0.0136377 +/- 0.0005523, respectively.
Errors come from10,000 case-bootstrap replicates (seed20260918), with shared-row
weighting and separate split/leg resampling. This supports unchanged g0 density
compatibility and a reproducible centering effect at g=0.05; it is a measurement
comparison under one model, not a learned-density gain. Exact counts, bootstrap
intervals and source hash are in `full_density_summary.json` beside the full
per-case `full_density.json`. Independent-reference16556888 follows this audit.

### Preregistered corrected-label refinement

After successful full density and independent-reference audits, repeat the
existing six-step `refine_guard_control_variate_flow.py` experiment on the
harmonized domain. The sole intended change from the previous experiment is
the measurement targets and their independently recomputed reference means.
The flow checkpoint, case split, guard bank (including 3-pixel thresholds),
independent-row objective, random streams, six-rate backtracking schedule,
0.001-nat density budget, and full-validation response selection remain fixed.
Reference provenance must match and independent mean gradient directions must
have cosine >=0.99. Epoch zero remains eligible; a nonzero step must improve
validation response within the density budget, then pass full validation NLL.
No ConstGold quantity enters training or checkpoint selection.

Launcher: `jobs/job_refine_harmonized_guard_flow.sh`, serialized after reference
with one GPU. This experiment tests whether corrected labels make the existing
small-update recipe useful; it is not a complete density retraining. If epoch
zero wins again, report that outcome and diagnose the train/validation response
direction before changing optimization. A changed selected checkpoint still
requires frozen actual-U joint evaluation on the primary measured >0.60-arcsec,
MAG_AUTO<25.8 population and physical-density validation before acceptance.

### Corrected training-response target and frozen reference

Four complete integration groups on15,630,719 corrected training pairs are
available. At the soft radius>3-pixel/MAG_AUTO<25.8 guard, measured R11 changes
from0.536498040 to0.525975189 and R22 from0.538172486 to0.526499356.
The unchanged flow predicts0.537989240 and0.538940206 on the new pair population;
reference-integration errors are0.000041864 and0.000018369. The corresponding
old predictions were0.538004725 and0.538932464. Thus measurement targets move
substantially while predicted responses barely change under the small membership
update. Off-diagonal target terms also shrink toward zero. The full16-cut
comparison with source hashes is `target_comparison.json`.

These are matched-usable, forward0.05, soft-cut training influence responses,
not the hard-cut joint ConstGold calibration or an offset to transfer to it.
Quoted integration errors exclude measurement and scene uncertainty. Reference audit16556888 subsequently completed in10m25s. Independent mean
gradient directions have cosine0.999999523 (distance0.295983 against norm~197).
RMS gradient noise falls from48.7901 to4.2286 with the frozen reference, an11.54x
reduction; relative noise falls from25.22% to2.15%. The compatibility and
cosine gates pass. Refinement16561508 has started on these corrected labels.

### Completed corrected-label refinement and density validation

Refinement16561508 selected step2, then stopped when every allowed step3 rate
failed the training density gate. Held-out guard loss improves from0.498035248
to0.356833213. Full held-out NLL changes from-4.769640411 to-4.768710490:
+0.000929921 nat per row, passing the fixed0.001 budget. The resulting single
flow is `harmonized_secondary_20260917_v1/refinement/selected.pt`, SHA256
`fade8aebfb4825344b8e9e36dfc07fce5357649340995670f1bb284df7068bd4`.
Its architecture is unchanged and80 of99 state tensors differ from baseline.
No ConstGold score selected this checkpoint.

Density smoke16561633 and joint smoke16561635 completed. The two-case actual-U
primary joint m changes from+0.714696% to+1.157847%; this small development
subset is a smoke test, not a calibration verdict. Full density16561637
completed80 cases. At the primary cut, candidate-minus-baseline transported
NLL is +0.000137175 +/- 0.000011514 nat atg0,
+0.000150296 +/- 0.000011973 atg=+0.02, and
+0.000120071 +/- 0.000011074 atg=-0.02. These are small resolved regressions
in restricted unnormalized physical density, not improvements or selected-density
normalization claims. Full joint16561643 has started with the frozen candidate;
its outcome remains pending. No candidate is promoted.

### Frozen corrected-flow evaluation protocol

`audit_disk_candidate_density.py` now accepts an explicit candidate flow and
optional reference flow. With a reference, both models use the same learned
disk transport and the exact same observed outputs; each flow retains its own
condition preprocessing and physical-target Jacobian. Paired differences are
candidate transported NLL minus reference transported NLL. The original
transport-versus-self-flow mode remains the default. At zero shear the
transport is still checked against its own base density exactly, even when
the two flows differ. Every scored density must be finite.

The cut summaries are restricted, unnormalized physical-density diagnostics.
Changing a flow can change selected mass, so these differences do not claim
normalized selected-density equivalence. Model acceptance still needs joint
shape/radius/flux checks and the full usable-domain validation density gate.

`job_evaluate_harmonized_candidate.sh` first validates the comparison on two
cases using the identical flow on both sides, requiring exactly zero paired
NLL difference and error. Subsequent candidate modes require the refinement's
full density gate and selected-checkpoint hash. They skip redundant evaluation
if step zero wins. Two-case density and joint smoke checks precede the full
80-case comparisons; joint composition, classifier, parent, cuts, random draws
and paired-case bootstrap are unchanged. The paired joint baseline is the
completed disk-transport candidate, isolating the flow update. The primary
reported result remains actual-U, measured radius >0.60 arcsec, MAG_AUTO<25.8.

### Independent latent-stream precision check

The80-case disk baseline uses64 antithetic draws and latent seed7301. Its case
bootstrap does not separately estimate integration error. Repeat the identical
frozen flow, emulator, classifier,80 scenes,64 draws,1024-object batches and
+/-0.02 CRN extraction with seed7302. Case/block seeds remain distinct between
streams. Neither stream is selected for a more favorable calibration result.
This repeats numerical integration of one flow, not an ensemble of trained
flows. It runs on the spare GPU alongside the serialized corrected-flow chain,
so the combined maximum remains two GPUs.

`compare_truth_parent_response.py --independent-latent-repeat` permits distinct
latent seeds only after checking exact model/evaluator/data provenance, response
composition and pair provenance. It checks that measured sufficient statistics
are identical, then reports per-cut changes in m with paired-case bootstrap
errors. The ordinary model-comparison mode continues to require common random
streams. Two streams diagnose repeatability; they do not establish finite-draw
bias or training-seed robustness. A material repeat difference would require
more integration before interpreting a near-0.3% result. Primary reporting is
the actual-U measured >0.60-arcsec population; no checkpoint selection uses this
check. Launcher: `jobs/job_audit_joint_latent_repeat.sh`.

### Fresh acceptance data inventory

A2026-09-18 directory inventory of the main +/-0.02 store finds140 ConstGold cases0--139 and
200 main training-family cases0--199. Every case identifier in that main store lies in
the flow-development scene range; 33 are in the flow validation split and
107 in its fitting split. The current80-case joint audit contains
18 validation and62 fitting scene identifiers. This does not make its
observed shear labels training targets, but it prevents calling these scenes
fresh independent acceptance. Cases120--139 have also been inspected in the
historical likelihood investigation.

If the corrected candidate passes frozen development checks, fresh multi-axis
acceptance must use new scene/image generation and a frozen measurement recipe
in BlendEMU, preserving the primary measured >0.60-arcsec population. The old
0.75-arcsec promotion criterion in the historical V3.5-like documentation is
now explicitly separated from this owner's active0.60-arcsec objective. No
configured likelihood cut, simulated dataset or model has been changed by this
inventory. The generic simulation launcher is not sufficient provenance for a
new acceptance set: its catalogue seeds, shear-axis construction, centering,
optimizer streams and measurement version must be frozen explicitly.

### Existing independent g1 endpoint recovered from the archive

The archived2026-09-13 cont.633 inventory identifies a separate existing root,
`lsst_sims_fs2_25876_constant_g005_c140-239`, beyond the main +/-0.02 store above.
All100 cases have both generated/truth/shape/crossmatch products (200 files of
each type). Cases200--239 lie outside the current flow/emulator/classifier
case range. They have been used in historical V2.2 diagnostics and therefore
are an existing generalization endpoint, not untouched multi-axis acceptance.

New truth-only audit16562378 verifies all200 current training-scene position
fingerprints and the40 prospective scenes: each is distinct, its two shear legs
share exactly the same invariant truth, and rendered shear is exactlyg1=+/-0.05,
g2=0. The declared secondary-role true-r<26 parent contains4,437,749 generated
rows; one unrendered identity is reported for dropping. No measured response
values are opened. Source hashes and reports are in `existing_g005_truth.json`.
Three focused tests check ordering/shear invariance, new positions and invalid
scene identities; the full SBSI suite passes422 tests with2 skips.

The old shape files predate the documented main-ConstGold centering rebuild.
BlendEMU `remeasure_common_optimizer_targets.py` now explicitly supports a
constant-shear family with a matched nonzero +/- pair, full-scene shape filenames,
and generated-plus-leg role membership. It validates identical generated truth
and fixed g1/g2. Its default half-shear behavior is retained. The same frozen
measurement package used for corrected training is copied alongside the new
hashed driver under `independent_g005_subpixel_pilot_20260918_v1/code_snapshot`.
Pilot16562406 remeasures cases200/201 on existing images with subpixel centers,
common optimizer seeds,8CPU workers each, strict true-r<26 secondary role and
independent leg detection. It uses no GPU or new rendering; no full40-case
rebuild or response evaluation is submitted yet. All14 BlendEMU tests pass.

### Independent scene overlap and anchor-free parent preparation

Scene-overlap inventory16562638 completed in1m17s. Actual generated-scene
fingerprints confirm cases140--199 match their training counterparts exactly;
only200--239 are disjoint. This strengthens the case-number inventory and
rules out treating all100 existing g=0.05 cases as independent validation.
Its first submission referenced an already completed job that Slurm had removed
from dependency tracking; after confirming that job's successful accounting,
the inventory was submitted without that stale dependency.

New `prepare_independent_truth_parent.py` selects structural secondary membership
and strict true-r<26 from the generated catalogue before joining rendered IDs.
It builds intrinsic shape/size and all-neighbour crowding conditions from raw
truth, then independently checks the three crowding columns against the retained
historical cache. No measured output, U0 flag or g0 selection is required or
invented. The manifest explicitly permits only the `truth_parent` cohort.
Two tests cover the generated-role boundary, strict floating-point magnitude
cut, dropped rows, and invariance to the applied catalogue shear.

Pilot16562654 completed34s on cases200/201, retaining110,640/111,263 parents.
Maximum crowding differences from the independent cache are2.36e-7/2.37e-7,
consistent with the stored float32 contexts. Full40-case preparation16562670
is submitted under `harmonized_secondary_20260917_v1/independent_parent/`.
The current running evaluator is left unchanged; its g0-diagnostic assumptions
must be made optional before these parents can be evaluated. No independent
response job is yet submitted.

`audit_independent_measurement_pilot.py` and dependent16562659 check old versus
remeasured per-leg marginals on each arm's own usable population, exact unchanged
photometry, source hashes and fit-array identity. Paired-object errors condition
on each pilot scene and are not calibration acceptance. They do not fit a
response correction. The full SBSI suite passes424 tests with2 skips.

### Completed independent-parent and measurement compatibility pilots

Both measurement pilots completed (15m22s and13m34s), followed by audit16562659
in36s. At the primary0.60-arcsec cut, the largest absolute paired-object z-score
among five moments is1.152/1.121 for case200 at+/-0.05 and1.038/1.589 for case201.
Across every declared cut the maximum is2.086. Exact non-shape columns and
fit-array identities pass; per-leg membership changes are explicitly reported.
This detects no centering mismatch in these two scenes, unlike the clear
training g=0.05 mismatch. It is not a distribution-equivalence certificate.
Retain the original independent catalogues for the prospective generalization
comparison; no full40-case remeasurement is justified or launched from this
pilot. Neither response values nor model biases selected that decision.

Full parent16562670 completed all40 cases. It retains4,437,748 rendered target
parents, reports one unrendered generated identity dropped, and matches the
historical crowding cache to maximum2.3842e-7 after float32 storage. No g0 masks
exist in these artifacts. CPU array16562810 prepares the four10-case neighbour
pair/lookup blocks with two concurrent tasks, using the same unchanged truth
pairing and legacy lookup verification recipe as the main comparison. These
are inputs for the frozen conditional disk model, not a new emulator fit.
It reads no measured response. The independent evaluation itself remains
unsubmitted until explicit truth-parent-only evaluator support is available
and the current source-hashed runs have finished.

Array16562810 subsequently completed all four blocks in98--125 seconds each,
with zero exit codes and empty error logs. The40 case artifacts contain
71,357,269 neighbour pairs for all4,437,748 parents. Each saved pair metadata
hash matches its successful builder log; the evaluator will verify full
artifact hashes when loading. `independent_pair_preparation.json` records
completion. No independent measured response or new model fit was evaluated.

### Staged truth-parent-only evaluation

`scripts/truth_parent_cohorts.py` validates requested cohort availability and
loads only requested masks. Endpoints without g0 must explicitly request
`truth_parent`; missing observed masks cannot be fabricated or silently replaced.
Boolean masks must align with parent identities, and the fixed-cut mask must
be a subset of usable-g0 when both are requested. Eleven new cases plus four
existing response tests pass. The staged evaluator preserves all27 legacy
statistic branches and produces only9 branches for a truth-parent-only request.
Full CPU suite16562981 passes435 tests with2 skips in45.20s; this covers the new
helper but does not replace validation after applying the staged integration.

The evaluator patch, source hashes and validation record live under
`harmonized_secondary_20260917_v1/evaluator_staging/`. It is deliberately not
applied to `evaluate_constgold_truth_parent.py` while16561643/16561899 are running,
because both record that file's hash at completion. Independent evaluation
now uses that frozen copy directly, as described below. Applying it to the
main source remains a separate step after the current runs finish.

`jobs/job_evaluate_independent_joint.sh` prepares baseline/candidate arms and
smoke/full scopes. It requires the staged evaluator and helper hashes, frozen
flow hashes, completed pair preparation, and a matching smoke for each full
run. Both arms use64 draws, seed7301, antithetic g1+/-0.05, the original independent
measurements, and only the truth-parent cohort. Both frozen arms will be
evaluated; the main ConstGold bias does not choose between them. The dependency
chains below have been submitted. This existing g1 endpoint cannot establish fresh
multi-axis acceptance.

### Prospective absolute joint-distribution diagnostics

Relative NLL gates and transported-density normalization do not establish
absolute agreement with measured joint outputs. The staged evaluator therefore
adds optional `--predictive-diagnostics`, enabled in the independent launcher.
It reuses the same generated draws, after emulator transport, without modifying
the response calculation. Each model draw carries its parent's actual per-leg
usability divided by the draw count. Measured output cuts are applied separately
to observed and generated values; no observed photometry selects model draws.

`scripts/joint_predictive_diagnostics.py` records a fixed8x8x8x10 histogram of
e1/e2/radius/MAG_AUTO, retaining overflow bins, plus selection fractions, means
and covariances in e1/e2/log(radius)/MAG_AUTO for all declared cuts. The bin
boundaries precede inspection of independent responses. Paired case-bootstrap
errors describe moment and selection discrepancies. Histogram total variation
is descriptive and includes empirical sampling noise; it is not a fitted
acceptance threshold. Disk-violating model mass is counted rather than dropped.
These coarse summaries cannot prove density equivalence and do not validate
the classifier's modeled usability probabilities.

Ten focused tests pass, including unequal joint dependence with identical
marginals, weighted draw integration, empty usable batches and invalid shape
accounting. A synthetic CPU run through the complete staged main passes with
missing-g0 parents and exactly unchanged response summaries when diagnostics
are enabled. Full CPU suite16563124 passes445 tests with2 skips in46.79s.
No real independent response was evaluated. The original running
evaluator remains unchanged; both its cohort and predictive changes are staged
together in the frozen standalone evaluator.

### Independent comparison queued from frozen evaluator source

The standalone evaluator passed a synthetic full-main check and imports through
`runpy` from the project working directory. The launcher accepts an explicit
`SBSI_EVALUATOR_PATH` and requires its hash to match the validated frozen copy;
the source need not replace the file used by current runs. Its helper hashes
are checked before execution and recorded in output. Freeze that copy and both
new helpers through completion of the independent full runs.

Two one-GPU dependency chains preserve the combined limit:

- Latent repeat16561899 -> baseline smoke16563242 -> baseline full16563244.
- Corrected-flow evaluation16561643 -> candidate smoke16563243 -> candidate
  full16563245.

CPU comparison16563246 requires both full runs to succeed. A full arm also
checks that its smoke has matching flow/evaluator/helper identities, seed,
draws, shear, cohorts and predictive diagnostics. At submission all four new
GPU jobs are pending dependencies, while only the two existing jobs run.
Scheduled snapshot16563249 replaces16562630 at the same13:21 Berlin time and
includes the new job IDs and result files. Each smoke and dependent full stage
still requires startup/log inspection when it begins. Independent response
values have not yet been evaluated or used to select a model.

### Discrete shape-symmetry density audit

The current guard trainer has no explicit rotation/reflection augmentation or
equivariance constraint. The recorded simulation noise catalogues specify a
circular PSF. This motivates testing simultaneous sign actions on intrinsic
and measured e1/e2, retaining radius, flux and scalar crowding conditions.
The four actions represent square-pixel rotation/reflection actions on spin-2
shapes. Circular PSF alone does not establish that the complete measurement
pipeline or conditional scene distribution respects this symmetry.

CPU audit16563273 completed in98s. It evaluates exact log probabilities for
both frozen flows on4096 fixed sampled rows from each of the40 corrected
held-out g0 scenes (163,840 rows per flow), without latent sampling or examining
independent ConstGold responses. Data and model hashes are recorded in
`harmonized_secondary_20260917_v1/shape_symmetry.json`.

For the baseline, the normalized arithmetic average over four transformed
densities reduces NLL by0.01374021 +/-0.00056932 nat per object; for the corrected
flow the reduction is0.01373573 +/-0.00056815. Both improve on all40 scenes.
Errors are paired scene-bootstrap errors on this fixed row sample. Mean
absolute log-density differences for the three nonidentity actions are about
0.121,0.141,0.175 nat in both flows, despite near-zero signed mean differences.
The small corrected-label refinement leaves this directional variability
almost unchanged.

The normalized orbit average is a diagnostic mixture, not a proposed
deployment model or a new flow ensemble. The result motivates a controlled
single-flow symmetry-augmentation or symmetry-regularization experiment with
held-out density and corrected guard targets. First check the relevant
measurement symmetries; retain the frozen candidates for the queued independent
comparison. No symmetry correction is applied to their results, and this
audit does not establish the source of V3's ensemble benefit or a shear-bias
improvement. Four focused tests pass; full CPU suite16563274 passes449 tests
with2 skips in42.11s.

### Native measurement symmetry pilot

BlendEMU's new `audit_measurement_shape_symmetry.py` applies identity, column
reflection, transpose and90-degree pixel rotation to existing g0 stamps and
PSFs, transforming the subpixel centre exactly and mapping fitted shapes back
to the original frame. Noise-border MAD is exactly invariant under each action;
the existing shape and centroid priors are isotropic. Two optimizer streams
are retained because identical seeds are not rotated versions of the same
initialization. The pipeline's geometric fallback for clipped stamps is retained.

CPU array16563404 completed cases4/96 in55/49 seconds, sampling1024 generated
secondary truth-r<26 parents per scene before checking detections. Missing
coordinates112/122 are recorded. Identity stream0 replays the harmonized
measurements to1e-12 (including missing results). One transformed fit raises
KeyError and is retained as a failure; otherwise recorded fits succeed.

SBSI summary16563411 compares each action's own usable marginal, without
common-fit conditioning. At the primary0.6-arcsec cut the largest absolute
standardized discrepancy across five moments/actions/streams is2.371 in case4
and1.505 in case96. The identity-stream control reaches1.841/1.347. Mean-shape
differences have paired-object errors of roughly0.005--0.008: this small pilot
does not constrain symmetry at a calibration-level tolerance. It detects no
clear transformation-specific failure, but does not establish equivalence.
Detection, photometry and the scene prior are not rerun or certified here.

The next justified test is a controlled single-flow training augmentation,
evaluated on untransformed validation density and corrected response targets;
do not impose a claimed exact pipeline symmetry or alter the queued frozen
comparison. No augmented model has yet been trained. Full suites pass16
BlendEMU tests (11.28s) and451 SBSI tests with2 skips (50.99s).

### Controlled single-flow augmentation continuation

`refine_flow_shape_augmentation.py` compares ordinary likelihood continuation
against uniform independent shape-sign augmentation, starting both from the
frozen corrected step2 checkpoint. It transforms physical input and output
shapes through the frozen affine standardizers; scalar features, radius and
flux are unchanged. This is an experimental regularizer, not an assertion of
exact measurement symmetry. Each output is one ordinary flow checkpoint.

The preregistered protocol is stored under
`flow_shape_augmentation_20260918_v1/experiment_protocol.json`: four epochs,
4million independently sampled training rows per epoch, batch8192, fresh AdamW
at1e-5 with zero weight decay and gradient clipping at5. Both arms share row
and optimizer settings. The full corrected160/40 split is unchanged. Every
prepared case hash is verified before loading.

Selection minimizes full, untransformed validation NLL subject to validation
guard loss no greater than the initial checkpoint's loss; epoch zero competes.
The guard uses the same16 cuts, corrected labels, four draws, two independent
groups, batch2048 and seed758000000001 as the preceding corrected-flow
refinement. Full validation density is evaluated for each response-eligible
epoch and rechecked after reloading the selected checkpoint. No ConstGold or
independent-scene response enters training or selection. Training gates only
establish eligibility for further joint validation, not production acceptance.

CPU smokes16563490/16563491 complete76/72s. Both reproduce identical initial
scores and retain epoch zero when the smoke update fails the density gate.
These use two cases per split and a small response subset, and are explicitly
ineligible for joint validation. Full suite16563492 passes454 tests with2 skips
in59.05s. Full control16563497 depends on independent baseline16563244; full
augmentation16563498 depends on independent candidate16563245. The two GPU
chains therefore remain capped at two simultaneous allocations. Snapshot
16563501 replaces16563249 at the same13:21 Berlin time and includes these jobs.
Freeze the trainer source through both queued runs; inspect each at startup.

### Completed corrected-flow joint response and latent repeat

Full corrected-flow16561643 completed83m41s on all80 truth-parent scenes.
At the primary measured0.6-arcsec cut, actual-U m changes from
-0.6230785 +/-0.2641523% to **-0.1686949 +/-0.2653391%**. The paired change is
+0.4543836 +/-0.0021160 percentage points. Measured R11 is0.6850254024;
corrected flow and blend contributions are0.5274019559 and0.1587810020,
summing to0.6861829579. Most of the change is in the flow response; the
unchanged conditional blend model also changes slightly when integrated over
the changed joint draws. No empirical correction or flow ensemble is used.

Uncut m is-1.5578398 +/-0.2710735%; the0.75-arcsec diagnostic is
+1.0453818 +/-0.2195068%. Neither replaces the primary cut. The primary
95% interval[-0.6992,+0.3349]% is too wide to establish a0.3% bound, and these
are reused development scenes. The earlier small NLL regressions remain;
the good-density requirement is not proven by this response result.

Frozen-baseline latent repeat16561899 completed101m31s with64 draws and seed7302.
Its primary m is-0.6293408 +/-0.2633879%, differing from seed7301 by
-0.0062623 +/-0.0055841 percentage points (paired95% interval[-0.01733,+0.00459]).
The comparison verifies identical model/data/evaluator provenance and exactly
identical measured sufficient statistics. Both streams remain reported; no
favorable seed is selected. This limits the observed repeat scatter, but two
streams do not bound all finite-draw bias or training uncertainty.

### Independent diagnostic underflow repair and retry

Baseline smoke16563242 failed after29s because the new log-moment diagnostic
required strictly positive generated photometry. The physical flow deliberately
exposes softplus underflow to zero rather than flooring outputs. The response
evaluator had accepted these finite draws; the added diagnostic must account
for their mass without changing those responses.

Predictive revision2 retains all finite draws in selection mass, adds an
explicit invalid-photometry histogram bin to normalization and total variation,
and reports zero-radius, zero-flux and negative-photometry fractions separately.
Log-coordinate moments have their own positive-photometry denominator, with
the undefined fraction reported for each cut. They are explicitly conditional
moments; no floor or silently renormalized histogram hides missing mass.
Nonfinite generated values still fail. Model weights and response calculations
are unchanged. The failed diagnostic source/manifest is preserved under
`evaluator_staging/failed_smoke_v1/`.

Both original evaluations finished before the tested evaluator copy was
integrated into the active script. The original source is preserved as
`evaluator_staging/completed_original_evaluator.py`. Twelve diagnostic tests
and two full-main synthetic tests pass, including zero flux and exact equality
of response estimates with diagnostics enabled/disabled. Full CPU16563585
passes458 tests with2 skips in45.65s.

Baseline retry16563587 and candidate smoke16563243 now use revision2.
Slurm would not release the failed-dependency baseline full job16563244 from
its administrative hold, so it was replaced by16563591 and then cancelled.
Comparison16563246 and control training16563497 were redirected to16563591;
candidate full16563245 and augmented training16563498 retain their chain.
Both smoke jobs pass their first meaningful draw progress with empty error
logs. Both smokes subsequently completed176/184s. Invalid-photometry mass is
0.80--0.88 parts per million across the two arms and shear legs, entirely zero
photometry rather than negative draws. The primary measured cut has no
undefined log moments. Mass at or outside the floating-point shape boundary
is4.8--7.0ppm and remains explicit. These two-case diagnostics are not density
acceptance. Both full40-case jobs have started and passed first meaningful
progress with empty error logs. Snapshot16563602 replaces16563501 about two
hours after this startup check; full results and subsequent training startup
remain to inspect.

### Fresh multi-axis rendering pilot

The current BlendEMU `generate_catalog_realization` calls its historical
half-scene random-direction mode `constant`; this is not the whole-scene
constant shear needed for acceptance. The old detached BlendEMU source has
whole-scene generation and a successful nine-leg smoke receipt, but its
measurement path predates the subpixel/common-optimizer correction.

New BlendEMU scripts `prepare_joint_multiaxis_pilot.py` and
`audit_joint_multiaxis_pilot.py` use current catalogue/configuration helpers.
They draw one intrinsic scene, then explicitly assign every object the same
sky shear: g0, g1=+/-0.02,+/-0.04, and g2=+/-0.02,+/-0.04. Intrinsic columns
are bitwise identical between legs. The pilot uses case20000, scene seed20123,
noise seed2026291801, 0.01 square degree, the existing FS2 mag<29 scene prior,
and no galaxy-count cap. The explicit noise seed is shared across legs; the
audit verifies common noise metadata, not isolated noise pixels. The direct
simulation subprocesses check return codes rather than assuming output exists.

This is a pipeline pilot with no candidate model input and no calibration
claim. Later measurement must use the corrected subpixel/common-optimizer
convention. The prospective analysis population remains generated secondary
role, strict true r<26, actual per-leg U, measured FLUX_RADIUS>3 pixels
(0.6 arcsec), and MAG_AUTO<25.8. No shape fitting is performed by these scripts.
The independent render audit checks applied shear, truth identity and intrinsic
agreement, reports unrendered parents, and rejects ambiguous crossmatches.

CPU16563738 passed25 tests but failed48s serializing a NumPy float32 density
into JSON, before rendering. Its generated catalogues are preserved under
`$DATA_DIR/blendemu_runs/joint_multiaxis_pilot_20260918_v1/failed_manifest_16563738`.
The scalar is now explicitly converted to a Python float. Retry16563748
passes26 BlendEMU tests in13.25s; audit16563763 depends on successful rendering.
Ruff and launcher syntax checks pass. Output and receipts are under the same
root's `images/`; results and measurement follow-through remain pending.

Added `measure_joint_multiaxis_pilot.py` and dependent CPU16563789. It reads the
audited per-leg SExtractor/CrossMatch tables directly, aligns by explicit
detection identity (not row number), and fits the complete generated secondary
true-r<26 parent with the existing corrected measurement worker. All nine legs
share its per-object optimizer seed; the source's subpixel centering is reused
without modifying it. Missing detections/positions, flags, fit exceptions and
unusable shapes remain explicit. Unmeasured detections retain invalid shape
sentinels. Rendering16563748 has passed six of nine legs with6,996 generated
objects and no galaxy-count cap. Measurement tests and completion remain pending.

All pilot stages subsequently completed: rendering16563748 in5m01s,
audit16563763 in9s, and measurement16563789 in80s. All nine legs contain6,996
rendered objects and the same1,075 generated true-r<26 secondary parents,
with no unrendered parents. Per-leg missing detections range114--118; usable
fits range957--961. No fit exceptions or nonzero fit flags occurred. The
measurement stage passes all27 BlendEMU tests in10.32s. This completes the
operational pilot, not the calibration goal. Next steps are a frozen fresh
full-scene protocol and generalizing the joint evaluator to both components
after the current independent evaluations finish; no pilot bias is used for
candidate selection. `execution.json`, `images/render_validation.json`, and
`images/measurement_manifest.json` preserve completion receipts.

### Explicit-axis joint evaluator staging

The current evaluator hard-codes g1 in intrinsic-shape shearing, disk transport,
and the diagonal used for `m`. New `scripts/joint_response_axes.py` handles
both axes explicitly and retains response vectors in sky e1,e2 order. Its
g1 contexts and summaries are bitwise identical to the existing functions;
g2 context shearing is checked against the complex reduced-shear formula.
The bias uses R11 for g1 and R22 for g2, without relabeling output components.

The evaluator copy at
`$DATA_DIR/sbsi_caches/joint_multiaxis_evaluator_20260918_v1/evaluate_constgold_truth_parent.py`
adds `--axis`, `--device`, and an explicit `--operational-single-case` switch.
That switch records undefined uncertainty as JSON null and refuses the
bootstrap predictive summary. A single small scene is never bootstrapped
as if it established calibration precision. The active/staged independent
g1 sources remain untouched during jobs16563591/16563245.

Full-main synthetic tests exercise both sheared contexts and the actual disk
transport on each axis, comparing with a closed-form response. Initial
tests16563831 found a nonrecursive NaN serialization error in the new
single-case output; the axis helper now recursively preserves undefined
uncertainty as null. Full CPU16563850 passes466 tests,2 skips and the existing
warning in35.25s. Ruff and launcher syntax checks pass.

Fresh pilot parent preparation16563852 completes with1,075 parents, no
unrendered drops, and16,977 neighbour pairs. It uses only generated/rendered
truth and preserves the strict true-r<26 structural-role population; g0
measurements do not define membership. CPU array16563859 runs the frozen
corrected single flow and disk emulator on both axes at h=0.02/0.04 with64
draws, seed7301. These are operational checks only. The two existing GPU
evaluation/training chains remain unchanged. Full fresh-scene acceptance,
componentwise uncertainties and joint-density acceptance remain outstanding.

All four CPU smokes complete in34--48s. Their source/flow identities, finite
two-component responses, flow+blend decomposition and null uncertainty fields
pass the receipt checks in `smoke_validation.json`. On this single small scene,
primary actual-U m is+0.795%,+0.038% for g1 at h=0.02/0.04 and-12.539%,-2.059%
for g2. These values have no case-based uncertainty estimate and show why this
pilot cannot certify calibration or justify choosing the more favorable step.
Both amplitudes and both axes remain recorded; no model or selection is changed.

### Finite scene-prior symmetry and residual decomposition

CPU16563921 audits the processed FS2 catalogue in the current `py31` runtime:
69,967,938 rows after the loader's mag<29/morphology cuts. Subsequent exact
reference replay shows this is not identical to the historical training prior;
see [runtime replay](#historical-runtime-replay-and-full-scene-restart). It applies
no new data cut and does not randomize position angles. Number-weighted mean
intrinsic shape is(-7.99064e-5,-3.85251e-6); total variation from the four-sign
orbit average in36 fixed angular bins is0.000199838. Flux-weighted mean shape
is(-0.000455437,-0.000325882), with angular TV0.0158859. Rare bright magnitude/
size strata have larger finite-prior anisotropy. The histogram and means are
exact descriptions of this finite sampling catalogue, not iid-row significance
tests or a bound on induced calibration error.

This supports the controlled augmentation experiment but prevents an exact
scene-symmetry claim. Neighbour orientations are not among the scalar crowding
conditions, so approximate marginal symmetry alone cannot prove conditional
likelihood invariance. Retain the untransformed validation density and response
gates; no empirical mean-shape subtraction or prior-angle modification follows.
`scene_prior_shape_symmetry.json` contains full magnitude/size stratification,
number/flux weights and source hashes. Two tests pass; the audit completes54s.

New `compare_joint_predictive_results.py` decomposes generated-minus-observed
joint moments into the mean of the +/-h residuals and their difference/(2h).
The former retains common shape/photometry offsets that can cancel in a
response estimate. It also reports fixed-bin marginal probability errors and
paired candidate-minus-baseline changes using common scene-bootstrap weights.
Comparability checks require identical observed sufficient statistics and
population/model-input/diagnostic conventions. These are descriptive density
checks, not an acceptance threshold or a checkpoint-selection rule.

CPU16563974 passes two targeted tests and completes the two-case diagnostic
smoke in19s. Full analysis16563973 waits on both40-case independent evaluations.
Full-suite validation16563977 completes with470 tests passing,2 skips and
the existing warning in35.33s. No running evaluator or queued
augmentation source was edited.

### Fresh full-area development and precision planning

Using the statistical-power skill, a prospective precision sensitivity calculation
uses the completed80-case primary g1 SE0.2653391 percentage points. Under
independent same-size scenes with unchanged case variance, SE scales as
0.2653391*sqrt(80/N). Reaching0.15pp or0.10pp would require about251 or564
scenes. At320 scenes, the estimated two-axis simultaneous95% half-width is
0.2974pp using a Bonferroni normal approximation. These are planning estimates,
not a measured g2 variance or an extrapolation across shear amplitude/sky area.
Strict confidence containment within +/-0.3% is a stronger optional claim than
the owner's approximate point-bias target; it is not silently imposed as a new
requirement. Sensitivity to stipulated true biases0/0.1/0.2pp and case-SD
factors0.75/1/1.5 is recorded externally, not computed as observed power at the
development estimate.

Before a large fresh acceptance launch, freeze a four-scene full-area
development/variance pilot: cases20010--20013, all with g0 and g1/g2 +/-0.02;
case20010 additionally gets +/-0.04 on both axes. Generated position area is
1 square degree and the simulator uses `one_tile`, matching the original
training/ConstGold geometry. Scene seeds are case+123; independent per-case
noise seeds are explicit and common across each scene's legs. Measurements
reuse the subpixel/common-optimizer code and retain the declared secondary-role
true-r<26 parent, actual per-leg U and measured0.6-arcsec primary cut.

The current loader estimates699,680 generated galaxies per square degree,
versus699,568 in the original training catalogue. Full-scene generation must
therefore first replay the original reference scene's intrinsic properties,
identities and positions exactly using its recorded seed123, then retain its
count. A mismatch aborts before rendering; no unverified prior change is accepted.
The small completed pilot's source files are preserved under its
`source_completed_small_pilot/` directory before adding this explicit full-scene
scope. The small-area default remains bounded to0.01 square degree.

CPU array16564117 is submitted with four cases, throttle2,8CPU/48G per case,
and source hashes frozen in
`$DATA_DIR/blendemu_runs/joint_fullscene_development_20260918_v1/protocol.json`.
The first two tasks pass29 BlendEMU tests and start loading the source prior.
Four scenes cannot certify calibration and will remain development data.
The precision receipt is
`$DATA_DIR/sbsi_caches/fresh_joint_validation_planning_20260918_v1/precision_sensitivity.json`.
Prior-replay, rendering and measurement completion remain to inspect.

### Completed independent response and density evidence

Baseline16563591 completes51m32s and corrected candidate16563245 completes52m14s
on all40 independently generated existing scenes200--239 (4,437,748 parents).
The primary actual-U result is-0.0110521 +/-0.1827114% for baseline and
+0.4465029 +/-0.1835285% for candidate. Candidate95% case interval is
[+0.0966257,+0.8189638]%; baseline[-0.3588798,+0.3594574]%.
Paired comparison16563246 completes40s: candidate-minus-baseline shift is
+0.4575550 +/-0.0011960pp, consistent with the+0.4543836pp development shift.
The original refinement effect repeats; target-level calibration is not established.

The measured primary R11 is0.6892386871. Baseline flow+blend is
0.5310053424+0.1583095286=0.6893148710; candidate is
0.5280347671+0.1581401293=0.6861748963. The endpoint is historical g1 +/-0.05,
whereas main development is +/-0.02, and it is not fresh multi-axis acceptance.
Do not choose a checkpoint by the more favorable endpoint.

Density analysis16563973 completes51s. Candidate primary shear-even
generated-minus-observed means in(e1,e2,log radius,MAG_AUTO) are
(-0.0008144,+0.0013531,-0.0007088,+0.0063454), with paired scene SEs
(0.0001699,0.0001861,0.0001753,0.0002319). Baseline offsets are similar:
(-0.0007648,+0.0013628,-0.0006996,+0.0059456). Candidate overpredicts the
primary selection fraction by0.003894--0.003974 (absolute probability),
versus baseline0.003380--0.003458. Coarse joint-histogram TV is0.03653--0.03732
for candidate and0.03646--0.03719 for baseline; these retain finite-sample noise
and are not density acceptance thresholds. Floating-point shape-boundary mass
is about6.4--6.7ppm and remains explicit.

The shared shape/photometry offsets persist over40 scenes and are not removed
by matching pooled response. The already frozen control/augmentation jobs
16563497/16563498 now start, reproduce baseline NLL-4.768710490 and guard
0.356833213, and pass substantial first-epoch training progress with empty
error logs. They continue to use training/validation gates, not this endpoint.

### Historical runtime replay and full-scene restart

The original full-scene array16564117 failed before rendering because the
reference-prior guard caught a real runtime difference. Audit16564191 replays
all699,568 reference objects exactly using historical `sims1` with Python3.9.20,
NumPy1.26.4 and pandas2.2.3. Its processed sampling pool has69,956,780 rows
and float32 angles. Current `py31` produces69,967,938 rows and float64 angles
from the same source/loader. This isolates a runtime compatibility requirement,
not the precise lower-level cause. No morphology cut, angle adjustment or
response correction is introduced to force agreement.

BlendEMU's full-scene launcher now uses `sims1` for preparation/rendering with
its library directory, and `py31` for tests/auditing/corrected measurements.
Failed outputs/protocol are preserved under `failed_py31_prior_16564117`.
Replacement CPU array16564237 has two running tasks and two pending, throttle2.
Both initial tasks pass29 tests, exact prior replay and catalogue preparation,
then enter full-image rendering. Four cases remain development/variance data.
Sources listed in its protocol must remain frozen through queued tasks.

Matched-runtime symmetry audit16564238 completes65s. On the actual historical
69,956,780-row finite prior, number-weighted mean shape is
(-7.899727e-5,-4.781791e-7), angular36-bin TV0.000199909. Flux-weighted mean
is(-0.000442909,-0.000314221), TV0.015884716. The qualitative conclusion
survives: near marginal symmetry does not establish exact conditional symmetry.
The authoritative training-prior artifact is
`harmonized_secondary_20260917_v1/scene_prior_shape_symmetry_training_matched.json`;
the earlier current-runtime audit remains preserved. Its standalone audit now
records runtime versions and avoids importing a PyTorch training helper.

### Completed continuation and frozen joint validation

Both four-epoch single-flow continuations finish: control16563497 in7m03s,
augmented16563498 in6m52s. Untransformed held-out density plus the original
response guard select control epoch1 and augmented epoch2, independently of
ConstGold. Full validation NLL changes from-4.768710490 to-4.774971864 and
-4.775263274 respectively. Guard loss changes from0.356833213 to0.116155758
and0.183126290. Augmentation's density advantage over the matched control
is small; none of these training metrics establishes joint calibration.

Frozen checkpoints under `flow_shape_augmentation_20260918_v1/` are
`control/selected.pt` (SHA1829b31cfeca33ab368c153d2d7a0b11e369484bcdbfdede385df40eed2de874)
and `augmented/selected.pt`
(SHAa88af411e3d7875d51e42c63c3741c10dde3a92f081334427beb335a9633f4e3).
New `job_evaluate_shape_continuation.sh` verifies training eligibility,
checkpoint/protocol/evaluator/helper hashes and matching smoke results.
Each arm uses one flow, the same disk emulator and classifiers, actual per-leg
U,64 antithetic draws with seed7301, strict true-r<26 parent and measured
0.6-arcsec/MAG_AUTO<25.8 primary selection. Fixed predictive diagnostics are
retained along with response; no post-hoc cut or checkpoint selection follows.

The two GPU chains are control main smoke16564321 -> main80 full16564323 ->
independent smoke16564326 -> independent40 full16564328, and augmented
16564322 ->16564324 ->16564327 ->16564329. Main endpoint uses h0.02;
existing independent endpoint uses h0.05. Each stage requires `afterok` on
its own preceding stage, preserving two GPUs total. Both smokes complete (control3m10s, augmented3m01s) with empty error logs;
full jobs16564323/16564324 pass the matching-smoke guard and begin substantial
first-case progress. On the two smoke scenes only, primary m is+1.792206%
control and+2.449707% augmented, versus+1.157847% for the initial corrected
flow on the same scenes. Measured R is identical0.6931936502; model total R
falls from0.6852593956 to0.6809889235/0.6766184816. This is an early sign of
training-guard/joint-estimator disagreement, not a population calibration
estimate or grounds to select a favorable checkpoint. Full comparisons remain
necessary to quantify the shift.

CPU comparisons16564444/16564445 wait for both arms at each endpoint and
retain paired response and shear-even/odd density residuals; the independent
comparison also evaluates each arm against the initial corrected flow.
`joint_validation_protocol.json` records all source/model identities and jobs.
One-shot snapshot16564479 is scheduled in approximately two hours for the
fresh-scene array and all validation stages, replacing16564263. It captures
logs/results, not an agent wakeup. No checkpoint is promoted.

### Corrected-label per-leg usability and exact-cut response audit

The completed continuation motivates testing the estimand, rather than merely
increasing the guard weight. New `audit_harmonized_usability_response.py` uses
all40 corrected validation cases and the stored full-parent usability records.
It independently accumulates random-direction forward influence-response
moments for both-usable versus actual per-leg usable populations, with the
existing0.05-pixel/0.05-mag soft guard and exact measured radius>3px,
MAG_AUTO<25.8 hard cut. Missing measurements contribute no selected mass or
shape; they are never imputed. Both legs share generated identities/shear,
all prepared label hashes are checked, and usability-source hashes are saved.

CPU16565769 completes55s including three tests. Counts are4,437,772 parents,
3,908,561 both usable,12,123 zero-only,11,364 sheared-only and505,724 neither.
The paired soft primary response exactly reproduces the training validation
target to3e-13, providing an end-to-end convention check:

| Label population/cut | R11 | R22 |
| --- | ---: | ---: |
| Both usable, soft | 0.524216997 | 0.525090983 |
| Per-leg usable, soft | 0.519200472 | 0.520394329 |
| Both usable, hard | 0.522046093 | 0.523589278 |
| Per-leg usable, hard | 0.517063712 | 0.518953758 |

At the hard primary cut, per-leg minus paired response is
-0.004982381 +/-0.000381229 in R11 and-0.004635520 +/-0.000443611 in R22.
Within the per-leg population, hard minus soft is
-0.002136759 +/-0.000465895 in R11 and-0.001440571 +/-0.000516283 in R22.
Errors are paired40-scene bootstrap standard errors,10,000 replicas,
seed20260914. These resolved shifts show that a small unmatched fraction or
softening width is insufficient grounds to equate the guard and selected
response. They do not determine how much of the *model residual* changes:
the model-side population/cut integral must change consistently.

Do not replace the matched-population target with this per-leg target while
retaining matched model contributions. That would mix estimands and could
double-count usability effects. A next model audit must use full-parent
contexts with each observed leg's usability and matching normalization,
then separate its integration error from label case uncertainty. Random
forward h0.05 still differs from coherent antithetic h0.02; none of these
numbers is an empirical joint-response correction or calibration acceptance.

The selected control's training primary diagonal predictions are
(0.531033146,0.518648873); augmented's are(0.525930978,0.512560126), against
the paired-soft target above. The aggregate16-cut/four-component guard loss
therefore does not guarantee improvement in each response component.
Preserve this distinction when assessing the pending joint evaluations.

Artifact: `flow_shape_augmentation_20260918_v1/validation_usability_response.json`.
Tests verify exact agreement with the established paired estimator, additive
case moments, retained one-leg-only mass, and rejection of nonfinite usable
measurements. Full-suite16565880 passes473 tests,2 skips and one existing
warning in45.00s. This audit adds no GPU load and changes no model/evaluator.

### Frozen model audit with per-leg usability and normalization controls

`scripts/audit_flow_per_leg_response.py` closes the model-side gap in the
label audit. It integrates the initial corrected, NLL-control and augmented
single flows on identical full-parent contexts and observed per-leg U. Each
frozen model is evaluated on the same rows and antithetic latent draws across
both random-shear training legs, soft/hard cuts and paired/per-leg populations.
Two independent latent groups distinguish integration variability from paired
scene-bootstrap errors. There is no emulator or classifier fitting in this
diagnostic; it is the forward0.05 self-response experiment.

For each population/cut it records both the model's own baseline selected
mass/mean and the measured-reference influence convention used by the old
guard. Their predictions remain separately named. Replacing the normalization
is diagnostic only; neither is used as a fitted correction to deployed density.
Zero generated photometry caused by floating-point underflow remains in the
uncut population and contributes no hard joint-cut mass. The measured side
uses exactly the label audit's sufficient statistics and case definitions.

CPU smoke16566462 completes35s, including three passing tests, on the first512
parents of each of two validation scenes. All three flows and four population/
cut combinations produce finite response tensors and two integration-group
records. This small prefix is an operational check, not a scientific response
estimate. Tests verify exact deterministic agreement with the label estimator
under one-leg missingness, explicit normalization differences, and zero-flux
handling without clipping or dropping uncut mass.

Full GPU16566470 initially followed control independent-full16564328. It now
follows control main-full16564323, with the control independent smoke16564326
following this audit. The completed CPU smoke remains verified by the launcher.
This prioritizes the population question without changing scientific settings
or the two-chain combined GPU limit. It covers all
40 validation scenes with two groups of eight draws per parent and2048-row
blocks. The launcher pins all model hashes and requires matching smoke source,
helper and domain identities. Freeze those audit sources until completion.
`model_per_leg_protocol.json` records identities and dependencies. Scheduled
snapshot16566473 replaces16564479 at the same approximate check time,15:56,
and includes the new audit logs. Full scientific results are pending.
Full CPU suite16566471 passes476 tests,2 skips and the existing warning
in37.36s.

### Fresh full-scene analysis chain and exact pilot replay

New `prepare_fresh_joint_development.py` assembles the four full-area scenes
only after successful render and corrected-measurement receipts. It verifies
all truth, crossmatch and measured-output hashes, both coherent shear axes,
scene/noise-seed independence, and the full-area original-prior replay scope.
The generated secondary role and strict true-r<26 cut precede any measurement
selection. Parents missing from any rendered leg are dropped explicitly and
listed by leg; g0 measurements never define parent membership. The source
catalogues remain unchanged behind a symlink layout for the existing evaluator.

Preparation smoke16566975 completes39s including three tests. Its1,075 parent
identities and contexts exactly reproduce the prior small-pilot preparation,
with no rendered-parent drops;16,977 structural neighbour pairs are rebuilt.
CPU evaluation array16567097 completes both axes in46/59s using the frozen
corrected flow, disk emulator,64 draws and seed7301. Every measured/model
sufficient statistic exactly matches the original staged small-pilot output
(maximum absolute difference0). This validates the new assembly/layout; the
small pilot remains operational only and has no case uncertainty.

Full preparation16567098 waits for all four tasks of render/measure array
16564237 and the successful preparation smoke. Six-task GPU array16567110
then evaluates each of the three frozen single flows on both axes, h0.02,
all four cases and the fixed predictive-density diagnostics. Each task also
retains h0.04 on case20010 with undefined case uncertainty, allowing a
within-scene amplitude diagnostic. It does not select a favorable amplitude,
cut or checkpoint. Its dependencies include preparation, both-axis smoke,
model audit16566470 and both chain endpoints16564328/16564329; throttle2 therefore preserves
the combined two-GPU limit even after the existing work completes.

The first full g0 image has completed rendering in about17 minutes; additional
legs and measurements remain in progress. The chain is prospective and no
full-area response result is available yet. Four development scenes cannot
certify the requested calibration. Source hashes/job dependencies live in
`fresh_fullscene_joint_20260918_v1/protocol.json`; freeze the new preparation,
axis helper and pair-builder sources through their queued stages. Monitor
snapshot16567122 supersedes16566473 at15:56 and includes this analysis chain.
CPU suite16567100 passes479 tests,2 skips and the existing warning in46.55s.

### Explicit-axis paired comparisons and amplitude reporting

The historical paired-comparison helper assumed R11 when bootstrapping m.
It is now explicit-axis aware: legacy records default to g1, g2 uses R22,
and different axes or vector orders are rejected. Deliberate single-case
operational outputs cannot acquire bootstrap confidence intervals. Model/data
pairing checks and the g1 numerical calculation remain unchanged. This change
is required for the fresh g2 comparisons; it changes no running evaluator.

CPU validation16567253 completes65s. The full suite passes482 tests,2 skips
and one existing warning in44.18s, including a g2 example with a different
cross response. Replaying the completed40-case independent g1 comparison
reproduces every old comparison field exactly, including primary delta
+0.457555015pp and SE0.001196030pp. The pre-change source and replay output
are preserved under `fresh_fullscene_joint_20260918_v1/comparison_validation/`.

Queued CPU16567262 depends on the complete fresh GPU array16567110 and this
validation. It compares initial/control/augmented pairs separately on each
axis, retaining both response and predictive-density residuals. It also
extracts case20010 from the h0.02 run and compares its point responses with
h0.04 under matching flow/emulator/parent/latent conventions. That one-scene
amplitude report has null case uncertainty and does not choose a step or
extrapolate an accepted zero-shear response. The comparison launcher passes
bash and embedded-Python syntax validation. Scheduled snapshot16567266
replaces16567122 at15:56 and includes comparison logs. Fresh results are pending.

### Why matched usability can contradict an exact per-leg density

The paired training guard has an additional structural issue beyond the size
of the omitted population. Let B=U0*Uh. A correct marginal flow supplies
E[f(Yj)|x,Uj=1]. The measured paired target instead involves
E[f(Yj)|x,B=1]. These are equal only under an additional conditional
independence condition involving the other leg's usability. A per-leg density
alone does not identify the cross-leg measurement/usability coupling.
In contrast, the identity
E[Uj*f(Yj)|x]=P(Uj=1|x)*E[f(Yj)|x,Uj=1]
justifies the per-leg weighting used by the new model audit.

An exact32-state oracle example now tests this distinction. Assign gamma
uniformly among +/-h on each axis, independently of two symmetric shape-noise
signs Z and a sign W. Set Y0=a*Z, Yh=-a*Z+R*gamma, U0=(W=+1), and
Uh=(sign(Z_axis)*sign(gamma_axis)=-W). For each gamma, every one of the four
shape atoms occurs equally often in each individually usable leg. Thus the
oracle conditional density is exact, with per-leg means0 and R*gamma.
But B selects Z_axis opposite to gamma, so the measured paired response is
R+2a/h while the marginal-density prediction remains R.
With h0.05,a0.01,R0.5, the full per-leg measured and model tensors are0.5I,
yet the paired measured tensor is0.9I against the exact model's0.5I.

This is an illustrative coupling chosen for exact algebra, not a measured
model of the image noise or an estimate of the real guard bias. It disproves
the general implication that zero paired-guard residual is necessary for a
correct per-leg density. The test checks the entire selected discrete output
distribution, both response axes and both retained hard cuts; it does not
merely compare means manufactured independently of a joint distribution.
CPU16567387 passes all four per-leg audit tests in15.83s (job25s), with
empty stderr. No model, training target or live source was changed.

A consistent candidate training statistic would retain all truth-parent
contexts and use each leg's own U in both observed and generated influence
contributions. With fixed measured baseline mass D0 and mean mu0,
v(y)=w(y)*(e-mu0)/D0, compare
Uh*E_qh[v(Y)]-U0*E_q0[v(Y)]
to Uh*v(Yh)-U0*v(Y0), using the same random-direction shear design.
Keep CRN between generated legs and independent row/latent replicas in the
population quadratic, as established by the earlier objective audit.
The actual magnitude of the paired-versus-per-leg *model residual* remains
to be measured by16566470 before setting a new experiment. No real-data
response offset follows from this counterexample.

### Experimental per-leg guard adapter and exact reference validation

New `scripts/per_leg_guard_response.py` implements the consistent influence
statistic described above. It is a separate experimental adapter and leaves
the legacy guard, evaluators and checkpoints unchanged. Every truth parent
has two contexts, including parents unusable in either or both legs. The
model contribution multiplies each leg by its own U before differencing;
measured targets use exactly the same population, soft guards and shear design.
Missing measurements are never imputed, and nonfinite usable measurements are
rejected. The adapter reuses the established antithetic CRN integrator and
independent-row gradient-replay implementation.

Four tests check reduction to the old estimator when all rows are usable,
recovery of the exact population-consistency counterexample, an analytic
response-loss derivative, gradient replay against direct autograd with two
independent row groups, and missing-measurement handling. CPU16567454 completes
2m33s. The full suite passes487 tests,2 skips and the existing warning in64.06s.
Its real40-case validation bank reproduces all16 legacy paired targets and
both available per-leg audit targets within2e-12. Primary per-leg soft target
is(R11,R12,R21,R22)=(0.519200472,0.002117063,0.000595998,0.520394329),
with baseline selected mass0.725567595 of the full truth parent.

CPU16567533 completes the corresponding160-case training target bank in4m20s. This is label preparation, not training or checkpoint
selection. The adapter/target reference does not yet replace a configured
objective; random-forward finite-step effects, normalization differences,
joint emulator accuracy and density quality remain to be evaluated.

To obtain the model-side evidence earlier, pending GPU16566470 now follows
main80 control16564323, then control independent smoke/full16564326/16564328
follow the audit. Augmented chain is unchanged. Fresh array16567110 explicitly
waits for both independent endpoints in addition to preparation/audit, retaining
the combined two-GPU limit. `squeue` verifies the new dependency graph;
completed smoke identities remain checked from their artifacts. Scheduling
changes are recorded in `execution_update_20260918_1437.json`, superseding
only older scheduling fields, not frozen scientific protocols.

### Complete per-leg target banks and physical-flow gradient check

Training target preparation16567533 completes4m20s with empty stderr. It
covers17,755,103 truth parents across160 cases:15,630,719 both usable,
49,878 zero-only and46,833 sheared-only. At the primary soft cut,
per-leg(R11,R12,R21,R22)=(0.520741494,-0.000545721,0.000084963,0.521013953),
versus paired(0.525975189,-0.000297976,0.000016275,0.526499356).
The diagonal changes(-0.005233695,-0.005485402) have the same sign and scale
as validation; this comparison alone supplies no train-versus-validation
significance claim. Training and validation have disjoint160/40-case splits,
identical domain/source hashes and the same16-cut bank. Receipt
`per_leg_guard_target_banks.json` pins both completed label artifacts.

New `smoke_per_leg_guard_flow.py` additionally checks the actual frozen
corrected physical flow, rather than only analytic models. It uses2,048
randomly chosen truth parents from one validation case, two independent
128-row groups, four draws and CRN. Replayed gradients agree with direct
autograd across the physical model at relative L2 error9.52e-8, below the
predeclared mixed-precision tolerance1e-6. The two scalar scores agree and
all participating gradients are finite. CPU16567735 completes22s with empty
stderr; Ruff passes. No optimizer step is taken, no checkpoint is written,
and the input checkpoint hash remains unchanged.

This operational subset does not estimate population response or prescribe
an update size. The completed full-bank preparation and gradient check make
a future consistent-guard experiment executable; scientific recipe/model
selection remains contingent on the full model audit and density constraints.
The artifact is `per_leg_physical_gradient_smoke.json` beside the target banks.

### Coherent-endpoint physical-density comparison for the continuations

Extended `audit_disk_candidate_density.py` with explicit CPU/CUDA selection
and an existing-leg list. Defaults preserve the previous0,+/-0.02 behavior.
The new continuation experiment uses only actual coherent +/-0.02 observations,
excluding the historical g0 training targets. It compares each frozen flow
against the initial corrected flow under the same physical disk transport,
parents and observations. Nonfinite observed density still fails rather than
silently dropping objects. Restricted cut scores remain unnormalized; they
must be interpreted with the independent selection-fraction diagnostics.

CPU array16567996 completes both two-scene smokes in2m42s/3m10s, empty errors.
Measurement hashes and retained row counts exactly match the prior density
reference. The initial reference's CPU versus previous GPU NLL differs by at
most4.386e-7 nat. On these two scenes, control-minus-initial uncut physical NLL
is-0.00608245 at+0.02 and-0.00638636 at-0.02; augmented is-0.00649053 and
-0.00651406. Primary0.6-arcsec restricted differences are respectively
-0.00719617/-0.00741799 and-0.00814888/-0.00825470. These are preliminary
smoke results, not full-density acceptance. Their improvement coexists with
the upward joint-bias shift in the response smoke; density and response
cannot substitute for one another.

Full GPU array16568039 waits for both existing response chains, CPU smokes
and suite16568033 (487 passed,2 skips, one existing warning in44.92s).
It evaluates all80 cases with two tasks and throttle2. Fresh array16567110
now additionally waits for this density array, preventing overlap beyond
the combined two-GPU limit. Existing full response and per-leg model audits
are unchanged. `transported_density_protocol.json` freezes the recipe and
source hashes. Snapshot16568103 replaces16567266 at15:56 and captures the
new density logs/results. Full density results remain pending.

### Exact decomposition of guard normalization effects

The per-leg population fix does not by itself equate a measured-reference
influence score with the model's own selected-mean response. For the identical
finite-step shear design, let `Dq/Dd` be the ratio of model to measured baseline
selected mass, `muq-mud` their selected mean-shape difference, and `kq` the
model's selection-mass response divided by the measured baseline mass. Then

```text
R_reference - R_own = (Dq/Dd - 1) R_own + (muq - mud) outer kq.
```

This is an exact algebraic identity for the recorded forward influence
estimator, including its finite shear step; it assumes neither infinitesimal
shear nor zero mean shape. It identifies two distinct contributions to a
normalization discrepancy. Agreement with a measured-reference guard can
coexist with an incorrect model-normalized response if selected mass or
centering differs. Density and selection diagnostics therefore remain necessary
when deciding whether the new per-leg guard improves the scientific model.

`scripts/decompose_guard_normalization.py` applies this identity to the frozen
model audit's additive moments, averaging latent-group moments before taking
ratios and using the same paired case bootstrap for every term. It rejects
different parent/shear designs and checks reconstruction in every bootstrap
replicate, for both shape axes, cuts, populations and models. The output is
diagnostic only: it changes no prediction, density or model parameter and is
not a correction to the coherent joint estimator. Operational smoke subsets
cannot establish the sizes of these terms in the full population.

CPU smoke16568502 completes28s; six tests pass in14.05s and all twelve
model/population/cut combinations reconstruct to at most2.23e-16. The first
attempt16568456 rejected floating-point reduction differences between
scene-wise and block-wise shear-design sums. The corrected check retains exact
row counts and permits relative matrix-norm roundoff of1e-12; a regression test
rejects an actual1e-7 design change. Full CPU decomposition16568526 follows
model audit16566470 and the successful smoke. It requires no additional GPU.
Full-population terms remain pending; the operational sample supplies no
calibration or normalization-effect estimate.

### Per-leg frozen-reference integration and gradient stability

The old variance-reduction bank integrated both-usable pairs and cannot be
reused with the corrected per-leg guard. Added
`scripts/audit_per_leg_guard_gradient.py` to prepare four independent reference
integrals on every training parent and compare ordinary independent-row
gradients with gradients using two disjoint reference banks. The starting
model is fixed to the initial corrected single flow, not chosen from ConstGold
response results. No optimizer update or checkpoint selection occurs.

The loader verifies aligned truth identities and source hashes, retains each
leg's own U, and checks the full160-case target against the previously frozen
training bank. Full integration uses four groups of four draws over all
17,755,103 parents. Gradient comparisons use eight replicas of65,536 parents
per independent group and eight draws, with CRN across shear legs and reference
differences. Reference uncertainty remains explicit; reference means are
model integrals and are never empirical corrections to deployed responses.

CPU smoke16568742 completes29s on1,024 randomly selected parents from the first
two training cases. All three gradient estimators and four reference groups
finish with finite gradients and unchanged checkpoint hash. The operational
bank-gradient cosine is0.99483; this small reused subset does not establish
full-population gradient stability. Ruff and bash syntax pass.

GPU16568798 follows model audit16566470; the control independent smoke16564326
now follows16568798, and its full40-case job16564328 remains downstream.
The augmented chain is unchanged, preserving two GPUs total. The existing
15:56 one-shot snapshot16568103 also captures this audit and the CPU
normalization decomposition. `execution_update_20260918_1511.json` records
these scheduling changes. Full gradient stability and the model/normalization
audits must be inspected before choosing a training recipe. No training run
is launched from smoke evidence alone.

### Density-constrained per-leg refinement runner

Prepared `scripts/refine_per_leg_guard_flow.py` without launching a full
optimization. It reuses the established independent-row gradient replay and
backtracking recipe with the new full-parent per-leg population. The full
runner requires compatible target/reference banks, disjoint training and
validation cases, and gradient-direction cosine at least0.99 between the two
independently integrated reference banks. This numerical requirement is not
a scientific calibration criterion.

The prospective recipe retains six gradient-descent steps, six halving trials
from learning rate1e-5, and a0.001nat-per-row baseline-relative density budget.
Each accepted training step must improve the training response objective.
Checkpoint selection uses per-leg validation response and checks **all**
validation likelihood rows before saving a candidate, in addition to the
subset check. Step0 always competes and remains the output if nothing qualifies.
The selected checkpoint is reloaded and its NLL/guard are replayed before the
completion receipt. No ConstGold target or result is read. Density and joint
response validation outside this training objective remain necessary.

CPU smoke16568936 completes27s. Thirteen targeted tests pass in6.63s, including
rejection of a checkpoint whose subset NLL passes but full NLL fails. The smoke
uses1,024 guard parents and8,192 likelihood rows per split, with no population
claim. All six trial steps reduce its training guard but violate the density
budget; even the smallest changes training subset NLL by+0.0010833nat. The
runner therefore restores step0, reproducing both validation NLL and guard
exactly and marking the artifact ineligible for joint validation. This tests
the rejection path and does not imply that the full-population optimizer will
reject every step. Full training remains pending the scientific and gradient
audit results.

Full suite16569008 passes495 tests,2 skips and the existing warning in42.64s;
Ruff and launcher bash syntax also pass. Scientific GPU schedules are unchanged.

### Paired main-sample comparisons against the initial corrected flow

The continuation comparison originally covered control versus augmentation
on the main sample, with initial-flow comparisons only on the independent
sample. Added both main-sample initial-flow comparisons to the pending CPU
job16564444. The initial80-case audit includes three reporting cohorts;
`scripts/project_truth_parent_audit.py` retains its existing nine truth-parent
branches and associated cohort counts. All original cases, measured/model
sufficient statistics and numerical summaries are unchanged. The derivative
artifact records its source hash and the projection operation explicitly.

CPU preparation16569419 completes39s: nine targeted tests pass in13.81s,
all80 cases remain, and a self-comparison returns exactly zero paired changes,
errors and confidence intervals. The comparison job additionally depends on
this successful preparation and uses the existing strict paired-bootstrap
helper. Its outputs will be `control/main_vs_initial_response.json` and
`augmented/main_vs_initial_response.json`. No additional model integration or
GPU allocation is introduced. The initial main audit has no predictive-moment
diagnostics, so no such density comparison is manufactured; the separate
physical-density job still compares each continuation with the initial flow.

### Completed 80-scene continuation response and density diagnostics

Both frozen continuation evaluations completed: control16564323 in1h42m46s
and augmentation16564324 in1h42m14s. The primary result uses all8,876,218
truth parents in cases40--119, actual per-leg U, antithetic coherent g1 at
h0.02,64 draws and seed7301, measured radius>0.6arcsec and MAG_AUTO<25.8.
The same disk emulator and classifiers are used throughout. Bias is
`m=R_measured/R_model-1` under the conventions already defined above.

| Single flow | Joint m (%) | Case SE (percentage points) | R_flow | R_blend |
| --- | ---: | ---: | ---: | ---: |
| Initial corrected | -0.168695 | 0.265339 | 0.527401956 | 0.158781002 |
| NLL control | +0.480712 | 0.266823 | 0.523130397 | 0.158617762 |
| Shape augmented | +1.126205 | 0.268783 | 0.518042710 | 0.159353817 |

Measured R11 is0.685025402 for all three. Paired changes from the initial flow
are+0.649407+/-0.003353pp for control and+1.294900+/-0.004612pp for augmentation.
Their95% paired case intervals are[0.642685,0.655889] and[1.285814,1.303803]pp.
Augmentation minus control is+0.645493+/-0.003537pp. These small paired errors
describe changes conditional on the frozen training/integration, not absolute
calibration uncertainty. The absolute95% case intervals are[-0.053972,0.985161]%
for control and[0.588554,1.635113]% for augmentation. The latter excludes the
target band on this development sample; control does not establish0.3% either.
Neither is promoted, and the initial model's cross-sample limitations remain.

The training-validation NLL and old paired guard both improved, while the
coherent joint response moved away from its starting point. Most of the change
comes from R_flow; the unchanged emulator's numerical contribution also changes
because it is evaluated with the different generated measurement distributions.
This is evidence that those training gates alone do not ensure joint calibration.

The completed paired predictive comparison16564444 also retains systematic
selection/density residuals. At the primary cut the shear-even selected-fraction
residual is+0.0041963+/-0.0001104 for control and+0.0042686+/-0.0001110 for
augmentation. The even MAG_AUTO mean residual is+0.0042139+/-0.0001941mag and
+0.0041365+/-0.0001939mag respectively. Mean log-radius residual changes from
0.0000212+/-0.0001379 to0.0004986+/-0.0001377. These are the existing predictive
diagnostic definitions, not a new density acceptance threshold. Full transported
physical-NLL comparisons remain queued.

The initial-flow comparisons required separate CPU job16569482 (34s): Slurm
captures batch scripts at submission, so editing the queued16564444 wrapper
did not add its new commands. A dependency submission referencing purged
completed job handles was rejected; the standalone job was submitted after
their completion and artifacts were verified. Strict cohort/input/measurement
comparability passes for both actual80-case comparisons. No response statistics
were recomputed or altered by the cohort projection.

Model audit16566470 has started and is integrating full validation cases.
Augmented independent smoke16564327 completed3m00s and its full40-case job
16564329 is progressing. The gradient audit16568798 now requests the same
physical A40 class on cip/inter, retaining its existing dependencies and one
GPU allocation. Updated one-shot snapshot16569483 replaces16568103 at15:56;
resubmission ensures its added audit coverage is captured by Slurm. The two-GPU
limit is preserved, and scientific settings are unchanged.

### Completed per-leg model and normalization audits

Full model audit16566470 completes6m12s on all40 corrected validation cases,
with two independent groups of eight draws and no emulator. Primary soft
per-leg measured R11/R22 is0.519200472/0.520394329. Using each model's own
selected normalization gives:

| Single flow | R11 | R22 | R11 model-minus-measured | R22 model-minus-measured |
| --- | ---: | ---: | ---: | ---: |
| Initial corrected | 0.529329898 | 0.530432957 | +0.010129426 | +0.010038627 |
| NLL control | 0.525110983 | 0.512858749 | +0.005910511 | -0.007535580 |
| Shape augmented | 0.520053647 | 0.506827899 | +0.000853176 | -0.013566430 |

Paired case standard errors on these residuals are approximately0.00158 per
diagonal component; the two-group integration estimates are0.000044--0.000066.
The hard-cut initial residuals are also positive,+0.0106950/+0.0097146, with
case errors0.0016891/0.0016565. Thus the initial self-response discrepancy is
not explained by the soft threshold alone. Both continuations substantially
change the relation between R11 and R22; the old aggregate guard score concealed
the primary R22 undershoot. Fresh coherent g2 validation remains pending.

Under the measured-reference convention used by the new training adapter,
the initial soft per-leg residual is+0.0128743/+0.0127900. Its old soft paired
residual is+0.0111366/+0.0113812. Correcting the usability population therefore
does not eliminate the real model discrepancy and must not be treated as a
numerical offset. The new reference-gradient audit is integrating this
population-consistent training objective before any full refinement is chosen.

CPU normalization decomposition16568526 completes20s and reconstructs every
bootstrap result within4.08e-16. For the initial soft per-leg primary cut,
model selected mass exceeds measured mass by0.003761404, or0.518409% relatively.
The reference-minus-own response is0.002744828/0.002751330. Its selected-mass
terms are0.002744091/0.002749810, with centering terms only0.000000737/0.000001520.
Control and augmentation have similar relative mass errors,0.523778% and
0.531029%, and similarly dominated normalization differences. This quantifies
a density/selection error on a scale relevant to the desired calibration;
no response is rescaled to remove it.

The near-zero main-sample initial joint bias coexists with a positive self-only
residual in this separate experiment. Error cancellation between flow and
emulator is therefore a hypothesis requiring matched measurement, not an
identified emulator correction: these populations and forward/antithetic shear
experiments differ. Inspection of BlendEMU's current physical-moment preparation
confirms that its native forward h0.2 labels require a usable g0 output and
valid shapes in both response legs. That training population also differs from
full-parent per-leg likelihood selection. The existing native500-case emulator
audit uses common-usable anchors and a midpoint proxy, so its residual cannot
be pasted into the coherent truth-parent response.

Next: inspect the full training reference stability, then use the prepared
density-constrained per-leg refinement as a controlled flow experiment if it
qualifies. Any joint improvement must be demonstrated with the emulator and
both coherent shear axes; the standalone guard does not establish that result.

### Selection-probability constraints and first full per-leg refinement

Full reference audit16568798 completes13m52s on all17,755,103 training parents.
The two independent reference-bank mean gradients have cosine0.99999875 and
distance0.38035 at norms192.417/192.626. Conditional minibatch RMS gradient
noise falls from39.435 (ordinary independent rows) to4.681/4.663 with the two
reference banks, or from22.91% to2.43%/2.42% relative to each mean gradient.
At the primary soft cut the reference R11/R22 is0.532174159/0.533293292 versus
training targets0.520741494/0.521013953. Four-group integration SE is
0.0000270/0.0000372. This establishes numerical update-direction stability,
not calibration or absence of training-label uncertainty.

The measured normalization discrepancy motivates an additional checkpoint
constraint before full refinement. A small average NLL budget alone does not
bound the error in a selected probability, and a measured-reference response
score can improve through a change in selected mass. The runner now requires
that **neither** the soft primary-cut nor the exact hard primary-cut baseline
mass error increase in absolute value relative to the initial checkpoint.
Both masses use each parent's own baseline U and divide by all truth parents,
not just usable or both-usable rows. The hard cut remains strictly radius>3px
and MAG_AUTO<25.8. These are validation checks, not output normalizations,
new fitted offsets or a replacement for subsequent joint validation.

`scripts/guard_selection_mass.py` integrates both probabilities with common
random numbers across candidate checkpoints. Full hard targets come from the
existing40-case label audit, with case/source alignment checked against the
soft-target bank; the operational subset uses its actual corresponding rows.
The selected model's likelihood, response and probability metrics are replayed
after reloading. The original pre-constraint smoke source is preserved alongside
its artifacts, with its original SHA verified.

New CPU smoke16572361 completes40s:16 targeted tests pass in13.59s, including
own-leg usability, the all-parent denominator, strict radius-boundary exclusion
and rejection of selection degradation despite improved NLL/response. It again
retains step0 because every trial exceeds the likelihood budget. Full CPU
suite16572827 passes500 tests,2 skips and the existing warning in48.08s;
Ruff and bash syntax pass. No population gain follows from the small smoke.

Queued full six-step experiment16572851 starts from the initial corrected
single flow after augmented independent full16564329 and the test suite.
Control independent full16564328 occupies the other chain. The two-task density
array16568039 additionally waits for16572851, preventing overlap beyond two
GPUs; fresh validation remains downstream of that array and its CPU preparation.
Snapshot16572869 is scheduled for17:56. External
`per_leg_refinement_protocol.json` pins the recipe, source hashes, tests and
dependencies. Full training/checkpoint outcomes are pending, and no model is
promoted based on the guard alone.

### Downstream validation of the per-leg refinement

The frozen downstream runner `scripts/validate_per_leg_candidate.py` checks the
completed full refinement, source/checkpoint identities, NLL budget, improved
validation response loss, and both primary selection-probability constraints.
Step zero produces an explicit skipped status, without a fabricated evaluation.
A nonzero selected step receives two-case startup checks followed by the same
80-scene main and 40-scene independent joint evaluations, unchanged emulator,
classifier ensemble, seed7301,64draws, measured radius>0.6arcsec and actual per-leg
U. Paired comparisons use the initial corrected flow. Main also receives
transported physical-density comparisons on the existing +/-0.02 endpoints;
cut-restricted scores remain unnormalized when selected probabilities differ.
The original main reference lacks predictive diagnostics, so no unsupported
main predictive comparison against it is synthesized.

Array16573305 runs those two endpoint chains after density array16568039.
Original fresh array16567110 now additionally waits for16573305; subsequent
array16573308 evaluates the selected per-leg flow on all four fresh scenes in
both axes, with the same one-scene h0.04 amplitude diagnostic. This serializes
all two-task GPU arrays and preserves the owner's combined two-GPU limit.
These jobs diagnose one training-selected model; joint results do not select a
checkpoint. Neither training eligibility nor a completed diagnostic implies
scientific acceptance. External `per_leg_downstream_protocol.json` pins21 files;
`per_leg_downstream_execution.json` records scheduling separately. The new
one-shot snapshot16573309 is scheduled17:56CEST and supersedes16572869.

CPU test job16573294 passes9 targeted tests in0.10s: probability signs, finite
NLL/response gates, explicit step-zero behavior and unchanged endpoint settings.
Ruff and bash syntax checks pass. GPU chains remain dependent/pending.

Fresh case20011 completed all five rendered/measured legs:111741 truth parents,
699568 rendered objects per leg. The32 recorded fitting exceptions were audited
read-only by CPU job16573300 (1m10s). Every failure reproduces ngmix flag1 with
missing `g`; all corresponding stored shapes are nonfinite. These are unusable
measurements under the existing convention, not valid shapes to impute. No
catalogues or fitting sources were changed. The detailed receipt is
`case20011/fit_exception_audit.json` in the full-scene run root. Other fresh
scenes and scientific validation remain pending.

### Matched random-sign response decomposition: estimator oracle

Existing random-direction forward0.05 self-response and coherent antithetic
joint-response audits cannot identify a common emulator residual. To address
this, `scripts/matched_response_probe.py` implements a prospective matched-scene
diagnostic, with a four-test exhaustive linear oracle. This is not yet a
rendered experiment or a calibration result.

For one fixed sky shear axis, let every generated source receive an independent
Rademacher sign s_j, including sources outside the target truth parent. Render
antithetic shears +/-h*s_j with identical intrinsic scene, noise and optimizer
streams. For target i define b_i=U_i*C_i at each leg, including newly entering
and leaving detections. The primary C is measured radius>3px and MAG_AUTO<25.8.
Let D0=sum_i b_i(0), mu0=sum_i b_i(0)e_i(0)/D0 and
v_i(g)=b_i(g)*(e_i(g)-mu0)/D0. Invalid shapes contribute zero only when their
own-leg b is zero; the parent is never intersected on usable measurements.

The vector estimate sum_i s_i*[v_i(h*s)-v_i(-h*s)]/(2h) targets the sum of
self derivatives. For a linearized scene response J_ij=dv_i/dg_j, independent
signs satisfy E[s_i*s_j]=delta_ij, so the mean is sum_i J_ii even though all
source objects affect each target. The parent is a subset of generated sources;
its own signs must be joined by generated identity. Coherent baseline-influence
response is sum_ij J_ij, so subtracting the probe self response estimates the
neighbour contribution. The underlying diagonal identity is a rectangular,
selected-row application of
[Hutchinson's stochastic trace construction](https://www.cs.tau.ac.il/~stoledo/Bib/Pubs/randomized-trace.pdf).
The selection-aware application and finite-step protocol here are our derivation.

The -mu0*delta_b term is required: omitting it loses selection normalization.
The helper reports exact endpoint-normalized coherent response separately from
the baseline-influence diagnostic. They agree at the derivative limit, not
identically at finite h. Antithetic O(h^2) error arguments require sufficient
smoothness of the population expectation; fixed-scene hard detections/cuts are
not assumed differentiable. Matched amplitude checks remain necessary. Exactly
balanced signs are prohibited: their off-diagonal moment is -1/(N-1). Opposite
legs of one probe are not two independent probes. Probe variability is reported
conditional on a scene and must not be presented as scene-based calibration SE.

CPU oracle16573403 passes4 tests in0.11s. Enumerating all16 signs for four
sources and two selected targets recovers known self and neighbour derivatives,
including nonzero selection derivatives, to1e-14. The oracle also checks
entering/leaving rows and the difference from finite-step endpoint ratios.
A bounded image pilot with pinned inputs is still needed before allocating a
full matched experiment. No current full-scene sources, images or models were
modified by this estimator work, and no scientific claim relies on its oracle.

### Per-leg startup provenance repair and independent augmentation result

Full refinement16572851 failed its startup identity check after1s, before
loading training data. The smoke receipt's selection-mass helper hash was
64fe29ee...: PyTorch `torch/utils/_contextlib.py`, because `torch.no_grad`
wraps the function and its `__code__.co_filename` names that wrapper. The
actual helper remained f18f7774... throughout. No scientific source drift or
numerical correction occurred. `refine_per_leg_guard_flow.py` now hashes the
helper module's `__file__`, preserving both the function implementation and
all training settings. Old receipts remain intact.

New smoke16573410 (39s) retains step0 and reproduces the earlier baseline NLL
-4.792804238672385 and guard31.58440176222965. Its receipt now matches all three
checked helper/source hashes. Full suite16573411:511 passed,4 skipped,
1 existing warning in58.09s. Replacement full refinement16573416 passes the
startup identity check and is loading the complete training population.

Immutable v2 refinement/downstream protocols replace the old execution
references. Density16568039 now waits for16564328 and16573416; replacement
existing-validation array16573419 follows density, original fresh16567110
waits additionally for16573419, and replacement per-leg fresh16573420 follows
16567110. Cancelled16573305/16573308; the launcher now takes an explicit
protocol path so each submitted script names its immutable protocol.
Snapshot16573422 at17:56CEST replaces16573309. Current control evaluation and
refinement occupy two GPUs; dependent two-task arrays stay serialized.

Augmented independent evaluation16564329 completes48m43s. On cases200--239,
h0.05, actual per-leg U, measured radius>0.6arcsec, joint m is
+1.7481591877% with case SE0.1863765011 percentage points and95% interval
[1.3939082311,2.1269608223]%. R_measured=0.6892386871,
R_flow=0.5186857054 and R_blend=0.1587110088. This independent-scene result
also excludes the +/-0.3% band for the augmented continuation; it is not a
new selection criterion for refinement. Control independent and formal paired
comparisons remain pending. No model is promoted.

### Bounded random-sign image pilot with zero-shear replay

The estimator oracle is now connected to an image-pipeline pilot in
`$DATA_DIR/blendemu_runs/random_sign_probe_pilot_20260918_v1/`. BlendEMU's new
`prepare_random_sign_probe_pilot.py` reuses the unchanged6996 generated objects
and noise seed of operational scene20000 (0.01deg2), instead of redrawing a
catalogue. Two independent source-sign vectors, seeds202609181701/202609181702,
are applied to every source at+/-0.02 along each axis. Both signs of each probe
share the same intrinsic scene/noise/optimizer streams. The generated secondary
true-r<26 target parent is joined by identity; signs are not balanced or selected
using measurements. Nine legs include an exact zero-shear replay control.

The pilot's prior is explicitly the earlier py31-generated small scene, which
is unsuitable for scientific acceptance of the historical full-area population.
It checks machinery only. Generator and rendered-truth checks require matching
intrinsics and correct per-source shear. Image/PSF arrays at zero shear must
match the original exactly; SBSI's `analyze_random_sign_probe_pilot.py` also
requires identical zero-shear fitted shapes and primary selection. It then
reports baseline-influence and endpoint-ratio coherent responses separately,
self/neighbor probe estimates, and conditional probe SE, with no scene-based CI
or model-acceptance claim. Own-leg usability and measured radius>0.6arcsec apply.

First startup16573641 passed2 tests but stopped before image preparation because
the historical outer measurement driver has a different hash. Both core fitting
files are unchanged. The retry records this driver difference explicitly and
requires exact pixel/shape replay to establish compatibility; it does not waive
any core fitter hash or replay check. Immutable protocol_v2.json pins the current
sources. CPU-only16573694 (4CPU,16GiB,30min limit) passes2 setup tests in4.79s and
has started. Existing full-scene sources and scientific GPU jobs are unchanged.
The completed image-pipeline and estimator output remain pending.

The ongoing full per-leg refinement has accepted two training steps, but neither
is selected: guard0.4569601 falls to0.4088917 then0.3871733, while soft/hard
primary probability residuals increase from[0.00373962,0.00396100] to
[0.00388815,0.00411313] and[0.00395917,0.00418726]. Full NLL changes remain below
0.001nat/row. These are intermediate results, not a completed model or reason to
relax the probability gates. If this persists, the training direction must
account for selection-probability error as well as response, rather than relying
on rejection at validation alone.

### Completed continuation density, independent response, and per-leg rejection

Control independent16564328 completes44m12s, and paired comparisons16564445
complete1m02s. Primary measured radius>0.6arcsec, actual-U joint m is
+1.1012578159 +/-0.1850564713 percentage points for control,95% interval
[0.7494327933,1.4770910952]%. Augmentation is+1.7481591877 +/-0.1863765011%,
versus initial corrected+0.4465028859 +/-0.1835284665%. Both continuations'
independent intervals exclude the +/-0.3% band. Paired changes against initial
are+0.6547549300 +/-0.0028984681pp (control) and
+1.3016563018 +/-0.0039637929pp (augmented). Augmentation-control is
+0.6469013718 +/-0.0024325600pp. These paired errors condition on the fitted
models and latent integration; they are not absolute calibration uncertainties.

Both transported-density tasks16568039_0/1 complete5m10s. On the80-scene main
population, uncut proper conditional physical-NLL changes against the initial
corrected flow with the SAME emulator are:

| Arm | h=+0.02 delta NLL | h=-0.02 delta NLL |
| --- | --- | --- |
| Control | -0.0061342442 +/-0.0000474996 | -0.0063992648 +/-0.0000524462 |
| Augmented | -0.0065417279 +/-0.0000745144 | -0.0065849724 +/-0.0000693032 |

Units are nat/object, errors case-bootstrap SE. Both improve the density score
while worsening joint response in both existing samples. Cut-restricted scores
are also recorded but remain unnormalized and are not conditional likelihood
comparisons when selection probability differs. Density improvement alone does
not establish the required calibration or absolute joint-density adequacy.

Full per-leg refinement16573416 completes11m15s and retains step0, replaying
NLL-4.7687104899, guard0.4569600927, and probability residuals exactly. Both
accepted training steps fail probability selection; every third-step learning
rate fails the train-NLL gate. The smallest third-step rate3.125e-7 yields
train NLL-4.7662397334 versus baseline-4.7672941567 (change0.0010544233, beyond
0.001 budget). No constraint was relaxed. Existing validation16573419_0/1
correctly emits explicit skipped receipts after1s each; no redundant joint
numbers are manufactured for the unchanged model.

### Completed image replay and joint gradient feasibility audit

Random-sign pilot16573694 completes6m42s. All nine legs have correct generated
identities/shear, matched noise metadata and unchanged intrinsics. Zero-shear
image and PSF pixels, fitted shapes, and primary selection reproduce exactly.
The1075 truth parents include790 selected baseline objects. Two recorded fitting
KeyErrors remain unusable; no values are imputed. The outer-driver difference
was recorded and did not alter the replay.

Primary self diagonal estimates areR11=0.57278667 (conditional probe SE0.02179874)
and R22=0.48052597 (SE0.01346568); corresponding neighbour differences are
0.14306455/0.14027790. These are only one small scene and two probes. Coherent
baseline-influence diagonal is0.71585121/0.62080387, while exact endpoint ratios
are0.71784150/0.62295558: finite-step normalization differences are visible and
must not be silently identified. No model comparison, case uncertainty,
full-area-prior claim or calibration acceptance follows from this pilot.

New `selection_probability_gradient.py` differentiates the training own-U0 soft
primary probability over all parents. `audit_joint_guard_probability_gradient.py`
compares that signed-error gradient with the response-loss gradient, using two
frozen response-integration banks. It diagnoses a common descent direction from
the sum of unit gradients, normalized to the response-gradient norm. Its dot
product with each gradient is negative unless the normalized gradients oppose
exactly. This is a choice of Euclidean parameter geometry, not a density
constraint, optimizer update or output correction. Hard-cut probability remains
a separate validation gate. Independent row/latent half-splits assess direction
stability; agreement across reference banks sharing one mass gradient alone
would not establish that stability.

Smoke16573818 completes45s:3 targeted gradient/oracle tests pass13.89s, source
parameters are unchanged, and1024-parent physical integration completes. Earlier
test jobs16573787/16573793 failed because the synthetic cut fixture lacked its
required name; the fixture was repaired and its stale compiled cache removed.
Smoke response/probability gradient cosine is-0.25379/-0.24659; response-only
descent increases probability error, while the common direction has both slopes
negative. Shared-mass bank-direction cosine0.99620 versus independent-half
cosine0.84746 demonstrates why this small smoke is not sufficient evidence for
training. Full read-only audit16573829 uses all17,755,103 training parents for
the mass reference, then four65,536-row gradient replicas at8draws. It changes
no weights and consumes no ConstGold labels. Fresh array16567110 now waits for
this audit and CPU preparation16567098, preserving the combined two-GPU limit.
Snapshot16573832 at17:56CEST supersedes16573422. Full gradient results and fresh
joint scientific validation remain pending.

### Full-scene matched probe and common-descent refinement

Read-only gradient audit16573829 completes4m35s. On17,755,103 training parents,
soft primary probability residual is+0.0034511544. Response-loss and signed
probability-error gradients have cosine-0.182369/-0.182309 in the two response
reference banks. Response-only descent has probability slope+123.224/+123.318;
the common direction has probability slope-432.024/-432.512 and response-loss
slope-23686.552/-23739.266. These slopes describe parameter-space first-order
changes, not calibrated physical responses. The bank direction cosine is
0.99999917; independent halves agree at0.99943071. This supports a bounded
update experiment, not a claim that finite steps satisfy the constraints.

New `refine_joint_guard_probability_flow.py` retains the original full-NLL and
soft/hard primary probability validation gates, including step0. It recomputes
a common training descent direction each step. Training mass is estimated from
the frozen full model integral plus a sampled current-minus-reference model
difference with common random numbers; no measured response is pasted into the
prediction. Backtracking requires both training response loss and soft
probability error to improve, within the original train-NLL budget. Validation
still checks exact hard measured radius>0.6arcsec as well as the soft surrogate.
The parameter norm equals response-only descent; no predictions are rescaled.

CPU smoke16574698 completes41s:8 targeted tests pass, a training step at6.25e-7
is accepted but validation rejects it, and step0 is reloaded/replayed. Full
suite16574895:514 passed,4 skipped,1 existing warning in61.67s. GPU full run
16574987 follows the suite and has passed its smoke-identity check, loading the
complete population. Six steps and learning rates1e-5/2**i,i=0..5 are fixed;
NLL budget remains0.001nat/row. No ConstGold/fresh measurements enter selection.
Fresh16567110 now waits for this and CPU preparation16567098; only one GPU is
used by the new training run and later two-task arrays remain serialized.

The successful image pilot is extended explicitly to full-area source case20011,
the first completed fresh scene chosen before any model response comparison.
`prepare_random_sign_probe_pilot.py` now requires an explicit full-scene flag,
an original-prior replay receipt, area1deg2 and historical Python3.9/NumPy1.26.4
for this mode. It copies699568 unchanged generated objects, records runtime,
uses independent case-specific signs on every source, and retains exact g0
image/PSF/shape/selection replay. It does not draw a new prior. The prior small
pilot's preparation/analysis/launcher sources are preserved under its
`source_v2/` with verified original hashes before extension.

CPU job16574297 (8CPU,48GiB,8h limit,no GPU) passes3 setup tests6.32s, prepares
nine legs and starts rendering. The receipt confirms sims1 Python3.9.20,
NumPy1.26.4,pandas2.2.3. It will measure111741 truth parents under the unchanged
core fitter and report both axes with two independent probes. The root is
`$DATA_DIR/blendemu_runs/random_sign_probe_fullscene_20260918_v1/`. One scene
and two probes remain a development diagnostic, not calibration acceptance.
A model comparison must use the same probe shears, masks and normalization:
the old coherent `model_flow_response` includes actual-U changes induced by
neighbours and is not automatically the same quantity as a diagonal image
Jacobian. Simple subtraction across these conventions would be misleading.

External refinement execution protocol pins sources and scheduling. Snapshot
16575356 at17:56CEST includes both the new training and full probe, superseding
16573832. Full training selection, full probe results and fresh scientific joint
validation are still pending; no new checkpoint is promoted.

### Selected common-descent checkpoint and fixed joint validation

Full run16574987 completes10m35s and selects step2, SHA256
`5434cfafb94ecb169b8b6a012752191bb191b82a787b196de8ed6dac2f1501a5`.
Validation response loss falls0.4569600927 to0.3180713609. Primary soft/hard
probability residuals fall from[0.0037396151,0.0039609966] to
[0.0011381840,0.0012846537]. Full validation NLL is-4.7680338901 versus
-4.7687104899: cost0.0006765998nat/row, within the fixed0.001 budget. Step1
had slightly better NLL but larger response loss; the prespecified selector
chooses minimum response loss among eligible checkpoints. No joint-bias result
enters this selection. All third-step trials fail the training NLL gate, so
training stops without changing the constraint.

New `validate_joint_probability_candidate.py` reuses the frozen endpoint command
construction, probability gates and paired-comparison helpers, changing only the
declared candidate root and training-source identity checks. Array16575454 runs
main80 and independent40, each after its two-case startup check. Main also runs
physical-density comparison against the initial corrected flow with the same
emulator. Both smokes finish and the full evaluations start successfully, with
empty error logs. Their two-case bias estimates are operational checks and do
not establish calibration; the final selected candidate remains experimental.

Original three-flow fresh array16567110 waits for preparation16567098 and
16575454. Old unchanged per-leg candidate16573420 follows that array and will
skip step0; new common-descent fresh array16575483 follows16573420, avoiding a
four-GPU overlap between candidate arrays. It evaluates both axes and the
existing one-scene h0.04 amplitude diagnostic. All scientific sampling settings
remain unchanged. `joint_probability_downstream_protocol.json` pins sources,
models and settings; a separate execution receipt records job IDs and the
final checkpoint hash. The updated snapshot16575587 at17:56CEST supersedes
16575356. Full probe16574297 has completed its first g0 replay rendering leg;
remaining legs, matched comparisons and scientific calibration are pending.

### Common-descent normalization audit and full-area zero control

The selected step2 primary soft response, under the guard's measured-reference
normalization, has diagonal [0.52982523, 0.53109164] against target
[0.51920047, 0.52039433]. Both residuals remain positive. Reduced selection-mass
error contributes to improved guard loss; converting to own-model normalization
also requires the centering term, so a mass-ratio rescaling would be insufficient.
Read-only job 16576138 reruns the frozen 40-case auditor for the initial and
selected flows with common random numbers, both hard/soft cuts and both
normalizations. This is a diagnostic, not checkpoint reselection. The shared
estimator, population manifest and both checkpoint identities are checked at
startup. Output: `model_per_leg_common_descent.json` under the augmentation root.

The audit waits for 16575454_1. Original fresh array 16567110 additionally waits
for this audit, alongside preparation 16567098 and validation array 16575454.
The subsequent old-candidate skip and new-candidate fresh arrays keep their
existing serial ordering. Replacement snapshot 16576327 at 17:56 CEST includes
the audit and supersedes cancelled snapshot 16575587.

Full-area random-sign scene 20011 passes early zero-image replay in CPU job
16575858 (9 seconds): all raw 18181 by 18181 image pixels and 48 by 48 PSF pixels
are identical to the original. Background-subtracted images are absent on both
sides and are not tested. The receipt is `early_zero_replay.json` under the
full-scene probe root. Remaining shear legs and fitted-catalogue/selection replay
are still required before interpreting matched response estimates. No catalogue
is modified and no calibration claim follows from this image control.

Full image job16574297 later completes in3h51m12s. The fitted zero-shear
shapes and selection replay exactly, in addition to the earlier raw image/PSF
check. All111741 parents and9 signed/replay legs are measured;59 caught KeyError
fit exceptions remain recorded as unusable rather than repaired shapes.
The primary measured self-response diagonals are0.50477409 (g1) and0.52847498
(g2), with conditional two-probe SE0.00472584/0.01133164; neighbour diagonals
are0.16840191/0.15920159. These errors condition on one scene and are not
case uncertainty or m. Full model comparison16577323_0/1 starts after the
classifier job, verifies all13 input legs, and shows meaningful progress for
both declared latent seeds. Wait for the matched model-minus-image residuals
before attributing physical errors from these image-only numbers.

### Model-side matched shear-field contract

`scripts/matched_model_probe.py` supplies the model primitives for the same
source-sign experiment. Given the recorded all-generated-source signs s_j,
the primary's intrinsic flow context receives shear +/-h*s_i along the chosen
axis. At each joint radius/flux draw, its neighbour velocity is
`+/-h * sum_j b_ij(radius, flux)*s_j` along that axis. The existing disk map then
applies radial tanh to this velocity. The secondary sign must enter each pair
term before summation; multiplying the coherent pair sum by the primary sign
would describe a different applied field. This is a declared extension of the
existing pair-sum model to the simulated shear field, not an empirically fitted
response adjustment.

The source lookup includes nonparent and unusable sources and rejects missing
IDs; no missing neighbour silently receives zero shear. Exact output-cell
pooling is reused because the source sign is independent of the draw. Isolated
primaries retain zero neighbour velocity. Five new tests cover these contracts;
all-positive signs reproduce the existing coherent contexts and pair sums
exactly. CPU job 16577170 passes all 18 targeted helper/estimator/transport tests
in 9.59 seconds. No active evaluator is changed by this helper.

The integrated model/image comparison remains to be implemented once signed
measurements are available. It must reuse the identical truth-parent population,
actual per-leg usability and primary measured radius >0.6 arcsec / MAG_AUTO <25.8
selection convention. Model cuts apply to the same joint generated draw used
for the response coefficient. Use common latent draws across antithetic contexts
and report both own-model and measured-reference normalization explicitly.
The selection-centering term belongs in the influence difference. A coherent
flow component using neighbour-dependent observed U is not automatically the
physical self-response trace; only the matched signed experiment identifies
that comparison. One scene and two probes provide conditional probe uncertainty,
not a calibration acceptance interval.

### Integrated matched model/image comparison

`scripts/evaluate_matched_model_probe.py` joins the recorded all-source signs,
truth-parent contexts, frozen response pairs and measured signed legs. It
requires the exact zero-image and fitted-selection replay receipts and verifies
its observed sufficient-statistic aggregation against the existing image-only
probe result before loading a model. The 13 evaluated legs are zero shear,
four coherent endpoints, and eight signed endpoints (two probes, two axes,
two orientations). Model and observed rows use identical actual per-leg U;
each applies the strict primary cut to its own measured/generated outputs.
Classifier probabilities are not part of this diagnostic.

Common latent draws are reused across all legs and model arms. Selected mass,
flow shape numerator and transport shape numerator are integrated per parent.
The model's own zero-leg mean/mass and the measured zero-leg mean/mass define
two explicitly reported normalization branches. The selection-centering term
is retained. Algebraic flow-plus-transport components sum exactly, but they are
not relabeled as physical self and neighbour responses: those come from the
matched sign-weighted derivatives. Model-minus-observed probe differences give
paired conditional probe SE. The exact coherent endpoint ratio is kept separate
from the baseline-influence response at finite h.

The first small-scene startup, job 16577296, passes the full suite (522 passed,
4 skipped, one existing warning; 39.78 seconds), then stops before model draws
because the preserved historical small-pilot diagnostic lacks the later `scope`
field. Explicit compatibility now requires its known small-pilot format and
area <=0.01 square degrees. Full-area mode separately requires the original-prior
receipt and full-area scope; no missing field silently becomes full validation.
Job 16577302 retries on CPU. Its 12 existing targeted tests pass in 7.12 seconds;
a stale compiled test omitted the new compatibility test, so that cache was
removed and the explicit new test passes separately in job 16577308 (14.43
seconds). Both scope branches therefore have direct verification.

The full-area launcher `jobs/job_matched_model_probe_full.sh` uses fixed latent
seeds 7301 and 7302 as integration replicas, each evaluating initial and selected
step2 flows separately. They are not a flow ensemble. It requires a frozen
source/model protocol and successful small-scene result. Submission follows the
pilot's completion, and must depend on both full image completion and the
existing fresh candidate array so the combined two-GPU limit is preserved.
Even a completed full-area comparison is a one-scene/two-probe development
diagnostic, not the multi-scene calibration acceptance test.

Pilot 16577302 completes in 2m32s. For the initial flow, own-normalized model
self responses are [0.54503059, 0.55156481] on the two diagonal axes, versus
measured [0.57278667, 0.48052597]. Paired self-response errors are
[-0.02775608, +0.07103884], conditional probe SE [0.02771129, 0.00613188].
Model neighbour responses are [0.16358951, 0.15471366], with residuals
[+0.02052496, +0.01443577]. Selected step2 self residuals are
[-0.02983462, +0.07212599] and neighbour residuals [+0.02041627, +0.01471502].
These are raw response differences, not multiplicative bias. The small pilot
has a different prior runtime, only 1,075 parents, and only two sign probes;
the paired conditional SE omits scene and finite-shear uncertainty. All results
are retained and neither checkpoint is selected from this diagnostic.

Full array 16577323 is queued with two tasks, one per fixed latent seed. It
depends on full image job 16574297, preparation 16567098 and completion of
fresh candidate array 16575483. The frozen protocol
`matched_model_probe_full_protocol.json` includes all evaluator/helper, model,
launcher and successful smoke-result hashes. Its execution receipt records
the dependency chain. Snapshot 16577325 at 17:56 CEST includes these jobs and
supersedes cancelled snapshot 16576327. Full scientific results are pending.

### Completed common-descent joint and normalization evidence

Array 16575454 completes main80 in 1h57m23s (including transported density) and
independent40 in 52m41s. Both use the frozen selected step2 single flow, fixed
disk emulator, actual per-leg U, generated secondary-role true-r<26 parent,
strict measured radius >0.6 arcsec and MAG_AUTO <25.8. Main uses h=0.02;
independent uses h=0.05, so their difference cannot isolate shear amplitude
from scene differences. Conventions remain those of
[population order](CONVENTIONS.md#3-population-and-selection-order) and
[response definitions](CONVENTIONS.md#6-response-estimators).

| Endpoint | Joint m (%) | Case SE (percentage points) | 95% case interval (%) | Paired change from initial (pp) |
|---|---:|---:|---|---:|
| Main80 | -0.1350513 | 0.2654831 | [-0.6649399, 0.3693254] | +0.0336436 +/-0.0033818 |
| Independent40 | +0.4776374 | 0.1839796 | [0.1268353, 0.8512339] | +0.0311345 +/-0.0021902 |

Main response decomposition is measured 0.68502540, model flow 0.52717728,
model transport 0.15877451, total 0.68595179. Independent is measured
0.68923869, flow 0.52782777, transport 0.15813450, total 0.68596227. The small
main point estimate does not demonstrate the target across validation sets.
The paired errors quantify model changes on shared scenes, not absolute
calibration uncertainty. No result is pooled across amplitudes or used to
reselect the checkpoint.

Full transported physical density worsens against the initial corrected flow
by +0.000764050 +/-0.000039634 nat/object at +0.02 and
+0.000759293 +/-0.000037069 at -0.02, uncut. At the primary cut the corresponding
raw NLL differences are +0.00314135 and +0.00312024; these cut-restricted
scores are not normalized selected-density comparisons when pass mass changes.
On independent40, primary shear-even selection-fraction residual improves
0.00393413 to 0.00092028. The mean MAG_AUTO residual improves 0.00634545 to
0.00403823 and log-radius residual -0.00070884 to +0.00023961. Neither these
partial distribution improvements nor the bounded uncut NLL cost establishes
the requested joint density/calibration quality.

Read-only self-response audit 16576138 completes in 3m42s on the same 40-case
harmonized validation population as the prior audit. Soft primary measured
response diagonals are [0.51920047, 0.52039433]. Initial and selected
own-normalized diagonals are [0.52932990, 0.53043296] and
[0.52899557, 0.53052232]. Thus the changes are [-0.000334324, +0.000089359],
leaving both residuals near +0.01. Under measured-reference normalization the
changes are [-0.002233357, -0.001813029]. The change in the normalization gap
accounts for [-0.001899033, -0.001902388], explaining most of the apparent
primary response improvement. For hard cuts, own-normalized changes are
[-0.000254880, +0.000275794], with gap changes [-0.001929934, -0.001931856].
The second diagonal therefore worsens slightly under own normalization in
both cases. This conclusion uses matched CRN evaluations, not a guessed
rescaling from probability ratios.

A subsequent training objective should evaluate the response of the model's
own selected mean separately from the pass probability. With full-parent
expectations, define D=E[U0*w0], A=E[U0*w0*e0],
B=E[(Ug*wg-U0*w0)*d], and H=E[(Ug*wg*eg-U0*w0*e0) outer d], where d is the
same fixed regression design used by the per-leg target. The own-normalized
response is `(H - (A/D) outer B)/D`. Differentiation must include D and A;
holding them at the observed constants returns the previous proxy objective.
Multiplying every modeled selection weight by a common factor leaves this
response invariant, furnishing a required analytic gradient check. This is
a proposed training-estimator change, not a post-fit correction or an already
validated refinement. It does not resolve emulator error on its own.

The earlier compatible four-scene primary amplitude pilot remains inconclusive:
h=0.02 minus h=0.2 response is +0.010033 with case SE 0.008710. Its result cannot
be applied as a fitted correction; see
[completed amplitude comparison](#completed-fresh-primary-amplitude-comparison).
Matched signed-image evidence remains necessary to attribute joint residuals.

All four fresh scenes are now complete. Preparation 16567098 completes in
1m35s after three tests pass in 16.21s. Cases 20010--20013 retain respectively
110699, 111741, 110894 and 111070 truth parents, with no unmatched-parent drops.
Initial-flow fresh GPU tasks 16567110_0/1 pass startup and begin both axes;
remaining comparison tasks and full signed-image job 16574297 are pending or
running. Snapshot 16578601 is scheduled for 21:50 CEST. Exact hash-linked
completed evidence is `common_descent_completed_evidence.json` under the
augmentation root. No model is promoted and the 0.3% goal remains open.

### Own-normalized gradient audit

`scripts/own_normalized_guard.py` implements the nine raw moments D, A, B, H
defined above and computes `(H-(A/D) outer B)/D`. `loss_cotangent` differentiates
the squared response loss through both model baseline moments before a
memory-bounded replay differentiates the raw row contributions. The caller
must integrate the reference moments at the current parameter values. This
is a finite-integration plug-in ratio, not an unbiased ratio estimator; the
code makes no optimizer update and adds no empirical scaling to predictions.

Three tests verify the probability-scaling invariance in both value and
gradient, explicitly demonstrate that the observed-normalization proxy lacks
that invariance, compare batched replay to full autograd with entering/leaving
usability, and reject invalid baseline mass. CPU job 16579484 passes in
12.70s. The real-model smoke 16579565 uses the existing deterministic two-case
training subset (1024 parents), four raw-moment integration groups and two
gradient replicas. It completes in 54s, including three tests in 21.39s,
and verifies model parameters/checkpoint unchanged.

The smoke common-descent directions have independent-half cosine 0.661089;
this is insufficient for an update. Full job 16579583 therefore audits all
160 training cases with four independent full-reference groups (four draws
each), two moment-reference banks, and four 65536-row gradient replicas with
eight draws. It reports ratio integration variability, response and signed
primary-probability gradients, and independent-half direction stability. A
future update still requires cosine >=0.99, sufficiently accurate integration,
the unchanged NLL/soft/hard probability gates, and joint validation. No
ConstGold calibration result enters the gradient objective or sampling.

The full audit waits for matched-model array 16577323; no extra GPU overlaps
the existing two-task arrays. The external frozen protocol pins helpers,
training population, target bank, initial flow, launcher and successful smoke.
Full suite job 16579569 passes 526 tests, skips four, and retains one existing
warning in 44.94s. Snapshot 16579587 at 21:50 CEST supersedes 16578601.

Scheduling revision at 20:00 CEST: while the signed image render remains live,
audit 16579583 is moved directly after fresh candidate array 16575483, and
matched-model array 16577323 additionally waits for the audit. This uses the
expected GPU-idle rendering interval. It supersedes the original ordering
above; frozen scientific protocols are unchanged. A separate schedule-revision
receipt records both DAGs, and the combined two-GPU bound is preserved.
The first matched-array update is rejected because completed preparation ID
16567098 has left the live controller. Its successful completion is reconfirmed
with `sacct`; the final verified live dependencies are 16575483, 16574297 and
16579583. A separate confirmed-schedule receipt records this recovery.

Fresh initial-flow validation meanwhile completes on cases 20010--20013:
primary m is +0.3786005 +/-0.8180457% along g1 and
-0.3899166 +/-0.7235736% along g2. Corresponding 95% case intervals are
[-1.0996510, 2.0155801]% and [-2.0198711, 0.6477915]%. The one-case h=0.04
diagnostic gives -0.2904418% / +0.9197301%, with case uncertainty explicitly
undefined; it is not a multi-case amplitude estimate. The control and
augmentation arms and selected candidate remain in the frozen evaluation
chain. Four fresh scenes cannot establish the 0.3% goal.

The full own-normalized audit16579583 subsequently completes in11m01s.
The primary training own-normalized diagonal is [0.52963942,0.53074874]
against [0.52074149,0.52101395], with four-group integration SE
[0.00005234,0.00001487]. The primary soft-mass residual is +0.00345053.
The independent-half common-direction cosine is0.99941660, passing the
prespecified0.99 numerical-stability threshold. Response/probability gradients
have cosine about-0.5685, so response-only descent worsens probability mass;
a common descent direction exists. This is a read-only training-gradient
result, not evidence that an update improves coherent joint calibration.
No update is launched while the matched physical response and usability/blend
coupling diagnostics remain unresolved.

### Practical acceptance and detection scope

Owner guidance on 2026-09-18: performance around 0.3% is acceptable if it is
reproducible rather than lucky or cherry-picked. This does not require every
noisy point estimate or confidence interval to fall inside +/-0.3%. It does
require retaining the declared model, population, cut, axes, seeds and
validation comparisons, and investigating apparent cancellation. Do not
select an estimator branch or checkpoint because its measured joint bias
happens to lie closer to zero.

The primary `actual_usable_flags` diagnostic includes detection and measurement
selection as realized in each image leg. Its generated truth-parent population
is fixed before detection; a missing crossmatch or unusable measurement gives
that leg zero U, rather than restricting both legs to a shared detected set.
Measured/generated output cuts then act separately on the corresponding joint
outputs. Thus the observed effect of detection is present, but using actual U
does not demonstrate that the classifier predicts that effect correctly. See
[D, U and S definitions](CONVENTIONS.md#5-usability-detection-and-selection).

For the selected common-descent candidate, the classifier-weighted
`modeled_usable` branch gives main80 m=-1.140485 +/-0.261272% and independent40
m=-0.542806 +/-0.185983%, versus actual-U -0.135051 +/-0.265483% and
+0.477637 +/-0.183980%. Errors are case standard errors in percentage points;
main and independent use h=0.02 and 0.05 respectively. The classifier predicts
the combined probability of detection and a usable measurement, so these
differences do not by themselves isolate raw detection error from measurement
failure or conditional-population mismatch. A near-target flow+emulator result
using observed U is not a complete predictive detection/selection calibration.
Both branches must remain visible in subsequent reporting.

### Classifier transfer to corrected measurement labels

The read-only `scripts/audit_classifier_measurement_transfer.py` audits the
frozen three-member eight-input classifier on all 200 corrected-domain cases,
using classifier train cases 40–99, tune cases 100–119 and unseen cases
0–39/120–199 separately. These are classifier splits, not flow splits.
There is no measured selection; the parent is the generated secondary role
with true r<26. The original classifier trained on the broader all-magnitude
parent and the earlier measurement labels.

Stable log arithmetic-mean probabilities preserve the classifier prediction
without clipping Bernoulli tails. Reports include BCE, Brier score, signed
probability gap, fixed probability-bin calibration, and case-bootstrap errors.
Identity-matched old/new comparisons retain label transition tables and input
feature differences; all new parents contribute to corrected calibration even
if absent from the older cache. Unmatched identities are explicitly counted.

The forward h=0.05 moment diagnostic regresses changes in predicted U, observed
U and matched detection on the applied random shear, with and without intrinsic
truth-shape weighting. It is not measured-shape response, coherent antithetic
calibration, or m. D denotes a crossmatch joined to a shape row;
`cross_without_shape` is reported separately. U also requires a usable fit.
No optimizer, response correction or checkpoint selection is involved.

CPU smoke 16579707 passes 3 tests in 12.93s and completes cases 0,40,100.
All corrected parents in the two old-cache cases match identities; features
and predictions are exactly unchanged on the overlap. Only 7/12 U labels
change in case40 g0/g05, and 10/11 in case100. This is preliminary evidence
against a large measurement-label shift, not a full classifier validation.
Full CPU audit 16579715 is frozen by `classifier_transfer_full_protocol.json`
and requires the successful smoke receipt; it uses no GPU.

Full audit16579715 completes in3m52s, including three passing tests and empty
stderr. Across 8,876,218 identity-matched parents in80 old-cache cases, g0/g05
have689/737 U-label changes (0.00776%/0.00830%), zero detection-label changes,
zero unmatched new parents, and exactly equal old/new features and predictions.
No crossmatch lacks a shape row in any corrected-domain case/leg.

On120 classifier-unseen cases, mean predicted-U minus observed-U is
+0.00034730 ±0.00008982 at g0 and +0.00032091 ±0.00008943 at g05
(probability units; case-bootstrap standard errors). Unusable fits among
matched detections comprise only0.00004611/0.00004723 of parents.
The truth-shape-weighted predicted-minus-observed U-response diagonal is
[+0.00005299, -0.00004218] with standard errors[0.00018124,0.00017938].
These particular random-direction moments agree within uncertainty. They do
not validate the coherently sheared selected joint measurement distribution.

Thus broad label drift is disfavored, and simply retraining the same classifier
on corrected labels is not justified as a fix for the roughly1pp modeled-U
joint-bias shift. Next separate coherent scene-shear usability response from
conditional measurement/selection coupling. The classifier has primary truth
shape and scalar crowding inputs but no explicit neighbour shape/shear tensor;
missing coherent-neighbour dependence is a hypothesis, not yet a diagnosis.
Do not infer classifier sufficiency from its mean probability calibration.

### Coherent-scene classifier response diagnostic

After the corrected random-shear audit, the next discriminating test uses the
same coherent +/-h catalogues and truth parents as the completed joint tests.
`scripts/audit_coherent_classifier_response.py` freezes four endpoints:
main80 (cases40–119, g1,h=0.02), independent40 (cases200–239,g1,h=0.05), and
fresh4 (cases20010–20013,h=0.02), separately for g1 and g2. Classifier
checkpoints and parent identities match the joint-result provenance. Per-leg
measurement reports must exactly replay their joint-evaluation counterparts.
Raw-crossmatch D is distinguished from usable-measurement U; missing or failed
measurements retain their own per-leg events rather than a matched intersection.

For fixed intrinsic shape e0, the reported diagnostic is
`sum[(pU_plus-pU_minus)*{1,e01,e02}]/(2*h*N)`, alongside the identical
observed-U and observed-D moments. N is the full parent count in the relevant
truth bin. This is an event response moment, not measured shear response or m.
The fixed bins are all parents, r<24, 24<=r<25, 25<=r<25.8 and25.8<=r<26;
there is no measured cut. Cases, amplitudes and axes are never pooled.
Case bootstraps preserve numerator/parent-count pairing and report joint
predicted-minus-observed differences. A one-case smoke has no case uncertainty.

Three analytic tests cover entering/leaving detections, measurement failures,
pooled parent normalization, undefined one-case uncertainty, and invalid event/
parent rejection. CPU smoke16579795 passes them in63.40s and completes all four fixed
single-case endpoints in2m50s, with exact measurement-report replay and empty
stderr. Full CPU16579820 uses all declared cases; its protocol additionally
pins the successful smoke receipt. No flow samples, optimizer, response correction,
classifier training or checkpoint selection are involved. The test can expose
coherent usability-response mismatch, but cannot alone identify its coupling
to selected joint measurement density.

Full coherent CPU16579820 completes all128 endpoint/case evaluations with
empty stderr. The all-parent predicted-minus-observed U moment weighted by
the intrinsic shape along the sheared axis is -0.00055125 ±0.00019267
(main80,g1) and -0.00042568 ±0.00011345 (independent40,g1). The declared
r<24 bin has positive errors +0.00111134 ±0.00037383 and
+0.00055361 ±0.00025850, whereas 25<=r<25.8 has negative errors
-0.00144792 ±0.00030860 and -0.00092359 ±0.00021167. These are diagnostic
response units, not percent m; bin results are exploratory conditional checks.
Fresh g1/g2 all-parent errors are +0.00033956 ±0.00020955 and
+0.00006708 ±0.00085399, with only four scenes. All bins and off-diagonal
moments remain in the receipt; no favorable subgroup is used for acceptance.

The coherent main/independent moments expose a mismatch not resolved by the
random-direction aggregate. That does not by itself explain the much larger
joint-response difference: the cached decomposition below locates that
primarily in blending-response weighting. Mean U probabilities alone cannot
validate either conditional response or detection/blending coupling.

### Usability discrepancy is concentrated in the blending contribution

`scripts/analyze_usability_blend_coupling.py` replays all cached joint responses
for actual U and modeled U, separately for flow, blending and total numerators.
All three existing cuts and four validation endpoints are retained. It uses
paired case bootstraps and checks the response replay and component sums to
1e-12. CPU16579853 completes in46s, including two analytic tests in0.40s.

At the primary measured radius>0.6 arcsec/MAG_AUTO<25.8 cut, replacing actual
per-leg U with classifier probabilities changes the model response as follows
(response units, diagonal for the tested shear axis; paired standard errors):

| Endpoint | Flow change | Blending change | Total change |
|---|---:|---:|---:|
| Main80,g1,h=.02 | +0.00000734 ±0.00010642 | +0.00696902 ±0.00005598 | +0.00697636 ±0.00011944 |
| Independent40,g1,h=.05 | -0.00012007 ±0.00008918 | +0.00715813 ±0.00008008 | +0.00703806 ±0.00010132 |
| Fresh4,g1,h=.02 | +0.00051503 ±0.00027553 | +0.00642603 ±0.00022905 | +0.00694106 ±0.00015500 |
| Fresh4,g2,h=.02 | -0.00003011 ±0.00062012 | +0.00650100 ±0.00021889 | +0.00647089 ±0.00043003 |

An exact numerator/denominator attribution averages both algebraic substitution
orders, avoiding a preferred reference normalization. For the blending component,
the main numerator/denominator contributions are +0.00697425/-0.00000523;
independent contributions are +0.00721359/-0.00005547. The relative mean-leg mass
changes are only +0.00003224 (main) and +0.00034305 (independent). This is not
primarily a baseline mass-normalization discrepancy.

The evidence locates the modeled-U shift in emulator-response weighting; it
is not a general flow-response derivative failure. Missing conditional
association between usability and blending severity is a supported hypothesis,
not yet a causal identification of which scene information is missing. The
current classifier sees primary truth shape/photometry and three scalar crowd
fluxes. A principled next experiment can test additional truth-scene geometry
features with held-out probability scoring and fixed downstream validation.
Do not subtract the measured shift, rescale emulator predictions, or use the
cross-normalized diagnostic quantities as production predictions. The full
coherent classifier audit and matched signed-image comparison remain relevant.

#### Next controlled classifier experiment

Compare the current eight-input feature contract against a truth-scene geometry
extension, using corrected full-parent true-r<26 U labels. Preserve train
cases40–99/tune100–119 and hold out independent200–239 and fresh20010–20013.
Use one fixed seed and matched architecture/training budget in both arms;
choose checkpoints by tune probability loss without reading joint bias.
The existing three-seed classifier remains the fixed reference, not a model to
replace before the full downstream comparison. No training has yet been run.

A useful bounded first feature set is log1p neighbour/primary flux-ratio sums
in fixed separation annuli [0,.5),[.5,1),[1,2),[2,4),[4,infinity), in the
existing pair-table distance coordinate. Crucial unit check:
`blendemu/data_utils.py::rescale` defines `distance_scaled` as separation over
`sqrt(Re_primary**2 + PSF_half_light_radius**2)`, not PSF FWHM. Scaled magnitude
differences preserve flux ratios because the aperture-noise magnitude offset
is common. The existing pairs carry all needed truth fields for80 main cases;
ignore their learned `response` column when constructing geometry features.
Verify exact parent identities, pair hashes, compatible baseline contexts,
feature units, no self/duplicate pairs, and explicit missing-parent reporting.
No-neighbour parents retain zero flux-ratio sums. Prepare the geometry with
the same function for training and validation; do not introduce measured
radius/flux or U-dependent features. Test preparation before training.

#### Geometry preparation and matched classifier training

Implemented `scripts/scene_classifier_features.py`,
`scripts/prepare_scene_classifier.py` and `scripts/train_scene_classifier.py`
with feature/preparation/training tests and scheduler launchers. The geometry
uses five log1p flux-ratio sums in the previously declared scaled-distance
annuli. Pair predictions and measured outputs are never read as features.
Source units and observation conditions are checked, pair/parent hashes are
verified, and corrected labels must match the geometry parent exactly.
The inherited pair support has at most19 neighbours per primary; the feature
is explicitly a summary of that support, not total flux from the entire sky.

Preparation smoke16579927 passes7 tests in0.44s and completes two cases in41s.
Full preparation16579955 completes80 cases in1m52s. Training smoke16579954
passes8 tests in17.18s and completes both CPU arms plus exported-checkpoint
replay in1m31s. No parent drops or unexpected pair primaries occur in the
smoke; full manifests retain counts for every case. Independent/fresh geometry
preparation16580007 uses exactly the same feature function without loading
validation labels or fitting any model.

Full training16580012 runs both arms sequentially on one GPU. Both use seed
20260913, train40–99/tune100–119, the same paired-row order, 64-wide two-layer
SiLU networks, up to100 epochs, tune-BCE learning-rate scheduling and15-epoch
patience. Added geometry weights start at zero and the remaining initial core
is identical, as tested. Corrected strict true-r<26 labels are shared by both
arms; this differs from the historical classifier's all-magnitude training
population and is controlled by the matched eight-input arm. Selection is
minimum mean-case tuning BCE; joint bias and emulator predictions are not read.
Models remain experimental and cannot be promoted on probability loss alone.

`training_full_protocol.json` pins prepared inputs, sources and both successful
smokes. Matched-model array16577323 now requires `afterok:16574297` and
`afterany:16580012`: it cannot overlap its two GPUs with classifier training,
and a classifier failure cannot strand the independent matched diagnostic.
The temporary scheduler holds were released and the final dependencies verified.
The exact commands are preserved in the external training-execution receipt.

#### Completed geometry training and shared-draw validation

Full training16580012 completes in8m54s. The control selects epoch95 with
mean-case tune BCE0.19845420 and Brier0.05783104; geometry selects epoch94 with
BCE0.13817770 and Brier0.03845942. Parameter counts are4801/5121. Selected hashes
are `b07200327383caef23e79303c5d8def2c45ef896ea44271563b18a6bbb5ed33a`
and `913b70c0c1a3c6a8fbb1b5d26ca82fb7e93d7c355ef74b93339ccc0f395c1ae7`.
This large probability-score improvement supports further testing, not a
joint-calibration claim. Both arms and the historical ensemble remain visible.

Full preparation covers8,876,218 train/tune parents with zero identity drops;
validation preparation16580007 completes in1m48s, covering4,437,748 independent
parents and444,404 fresh parents. Full test job16580019 passes542 tests with
4 skips and one existing warning in81.17s.

Added `scripts/scene_classifier_joint.py`, `scripts/evaluate_scene_classifier_joint.py`
and `scripts/validate_scene_classifier.py`. The evaluator deliberately extends
the frozen explicit-axis evaluator SHA902e016d... with three classifier arms
on the same CRN flow/emulator draws. It preserves actual-U and historical
classifier branches for exact replay, checks geometry by parent identity, and
records classifier-weighted four-output density diagnostics as well as joint
response. All cuts and all main/independent/fresh endpoints are retained.
Paired arm comparisons use the same measured response and case bootstrap;
no bias result chooses a classifier checkpoint.

CPU smoke16580095 passes the expanded545-test suite (4 skips, existing warning,
45.72s), then stops before model evaluation because its small-fixture case was
incorrectly specified as20010. The fixture manifest identifies20000. The empty
partial preparation is preserved as `validation_smoke_failed_16580095`; revised
launcher/protocol16580123 uses20000, retains the failure record and adds tests
for reference replay and paired-arm arithmetic. No threshold/model/source
catalogue is altered to accommodate the mistake.

Revised smoke16580123 completes in1m39s, including five targeted tests in7.93s.
All original branches' sufficient statistics replay exactly (maximum absolute
difference0.0), and response components agree to the fixed1e-12 tolerance.
The corrected small fixture is explicitly operational; its bias is not used
for selection or acceptance.

Frozen full array16580144 (tasks0–3, throttle2) evaluates main80, independent40,
fresh g1 and fresh g2 with64 common draws, seed7301 and batch1024. Every task
uses the selected common-descent flow, fixed disk emulator and all three
classifier arms. It also retains the original actual-U/matched-U branches and
replays the original reference results before reporting paired classifier
changes. Classifier-weighted predictive moments and four-dimensional histograms
are included. The array follows `afterany:16577323` solely for the two-GPU
limit; it does not require that independent physics diagnostic to succeed.
No joint result selects an epoch, seed, cut or classifier arm.

Full matched-model tasks16577323_0/1 are progressing normally. Snapshot16580149
at23:05 CEST replaces cancelled16579893 and includes the new classifier jobs
and artifacts. It is a Slurm log capture, not an agent wakeup. While the GPU
comparisons run, a useful independent next check is probability/response
calibration of the new arms on the fixed independent/fresh labels. Any later
extension of the flow's scene inputs must follow density evidence, rather than
assuming that better U prediction alone completes the joint factorization.

#### Held-out geometry calibration improves while coherent response worsens

`scripts/audit_scene_classifier_calibration.py` evaluates the three frozen
classifier arms on every previously declared coherent endpoint. It reuses the
exact truth parents, per-leg detection D and usable-measurement U labels, and
identity-checked geometry. Stable log probabilities preserve the historical
arithmetic classifier ensemble without probability clipping. The read-only
audit records BCE, Brier score, reliability bins, intrinsic-shape-weighted
U response, and five geometry-weighted probability residuals. It performs no
flow/emulator evaluation or model selection. All paired score differences use
10,000 case bootstraps, seed20260914, without pooling endpoints.

Smoke16580204 completes1m02s (3 tests,25.11s); full16580213 completes4m54s
(3 tests,16.94s), both with empty stderr. Counts replay exactly against the
old coherent audit; maximum per-row probability-score discrepancy is
2.11e-10, below the frozen2e-7 CPU tolerance. Result/protocol/receipts are in
`scene_classifier_geometry_20260918_v1/calibration_full.json`,
`calibration_full_protocol.json` and `calibration_execution.json` beneath the
cache root. The tests cover hidden geometry coupling despite zero mean
probability error, pooled-case arithmetic, and malformed feature rejection.

The probability-score improvement generalizes. On independent40, plus-leg BCE
is0.19828851/0.19816398/0.13806142 for historical/control/geometry; Brier is
0.05772911/0.05771097/0.03844612. Averaging both legs, geometry minus control
BCE is -0.06012486 +/-0.00015810 and Brier is -0.01927303 +/-0.00005513.
The corresponding BCE improvements are -0.06052024 +/-0.00012654 on main80,
-0.06007201 +/-0.00049642 on fresh g1, and -0.06022181 +/-0.00045087 on fresh
g2. Main80 overlaps train/tune; the other cases are unseen by all classifiers.
Errors here and below are case-bootstrap standard errors, not acceptance bands.

The five residual moments are full-parent means of each fixed geometry input
times the even-leg probability error, G_j*((p+ - U+)+(p- - U-))/2.
On independent40 the control moments are
[0.00211046,0.00644998,0.01923374,-0.00522215,0.01163566], versus geometry
[0.00006869,-0.00002380,-0.00018050,0.00086757,0.00114063], with geometry SE
[0.00001469,0.00001752,0.00004391,0.00012760,0.00037113]. Thus neighbour
correlation decreases substantially but is not eliminated. Reliability-bin
errors remain, and the fresh-scene mean probability gaps remain about-0.001.

**The coherent U-response diagnostic worsens substantially.** All-parent
intrinsic-shape-weighted errors along the applied shear axis are:

| Endpoint | Historical | Matched control | Geometry |
|---|---:|---:|---:|
| Main80, g1, h=0.02 | -0.00055125 +/-0.00019267 | -0.00028151 +/-0.00019238 | +0.00868193 +/-0.00019210 |
| Independent40, g1, h=0.05 | -0.00042568 +/-0.00011345 | -0.00015848 +/-0.00011317 | +0.00879871 +/-0.00011913 |
| Fresh4, g1, h=0.02 | +0.00033956 +/-0.00020955 | +0.00061798 +/-0.00020373 | +0.00959303 +/-0.00020966 |
| Fresh4, g2, h=0.02 | +0.00006708 +/-0.00085399 | +0.00026461 +/-0.00085736 | +0.00931890 +/-0.00083787 |

These are event-response moments without a measured-output cut, not m or
percentage points. All four fixed magnitude bins also have positive geometry
errors on both axes; the complete bin tables remain in the result. This is
evidence against treating improved probability loss as sufficient detection
calibration. The geometry features stay fixed when primary ellipticity is
sheared and omit neighbour shape/orientation; a conditional model's shape
dependence need not reproduce the intervention in which every scene source is
sheared. Missing scene dependence and conditional-response misspecification
are hypotheses, not identified causes or justification for a fitted offset.

Retain the frozen joint array16580144 and all its arms/cuts/endpoints: it will
measure how the improved baseline blending weights and worsened U response
combine with the unchanged flow/emulator, including density diagnostics. Do
not promote this classifier, choose a favorable branch, or launch a flow-input
extension based only on the probability improvement. A subsequent experiment
must address the response failure explicitly and preserve independent tests.

#### Signed-image usability response localizes the geometry failure

Added `scripts/audit_signed_classifier_response.py` and three analytic tests,
using the completed image probe on the already fixed full scene20011. It
reuses the matched-model preparation's catalogue/identity/zero-replay checks,
but evaluates only classifiers and U. For each axis, the fixed intrinsic
basis [1,e01,e02] multiplies (U+ - U-)/(2h), normalized by the full parent count
in each of the five established truth-magnitude bins. Multiplying signed-leg
differences by each target's own Rademacher sign estimates the self term;
coherent minus self estimates the neighbour term. All generated sources were
independently signed, including nonparents. No measured-output cut is applied
to these event moments, so they are not the selected joint response or m.

CPU16580345 completes55s, including3 tests passing18.91s with empty stderr.
The tests prove self/neighbour extraction on a linear response matrix averaged
over all independent sign assignments, preserve binary entering/leaving
events, and reject invalid probabilities/signs. Ruff and bash syntax pass.
All three arms exactly replay the earlier case20011 coherent moments; signed
probabilities also exactly equal the appropriate coherent plus/minus values
swapped by target sign. Thus static geometry adds no neighbour shear response
in these classifiers, as expected from their input contract. Output and frozen
protocol are `scene_classifier_geometry_20260918_v1/signed_response.json` and
`signed_response_protocol.json` under the cache root.

All-parent intrinsic-shape-weighted diagonal moments on this scene are:

| Quantity | g1 | g2 |
|---|---:|---:|
| Image coherent U | -0.00737908 | -0.00562083 |
| Image self U | -0.00834745 | -0.00743210 |
| Image neighbour U | +0.00096837 | +0.00181127 |
| Conditional two-probe SE | 0.00079361 | 0.00023799 |
| Historical predicted self/coherent | -0.00645946 | -0.00643364 |
| Control predicted self/coherent | -0.00620146 | -0.00623629 |
| Geometry predicted self/coherent | +0.00282475 | +0.00309244 |
| Geometry minus image self | +0.01117220 | +0.01052455 |

The geometry model has the wrong sign for the self event response on both
axes. Missing neighbour response alone therefore does not explain the observed
failure: the measured neighbour term is smaller and positive on this scene.
This revises the causal hypothesis suggested by the coherent audit; it does
not establish a universal self/neighbour decomposition. There are only two
probes on one scene, the SE is conditional on that scene, and finite-h mixed
nonlinear terms can contaminate diagonal extraction. No component is used as
an additive correction or as a calibration acceptance criterion.

The next read-only check compares the same three classifiers on the actual
random-shear training60/tuning20 pairs. A response error already present there
would implicate the learned conditional dependence or the probability-only
objective, rather than transfer to coherent scenes alone. Joint array16580144
remains unchanged while these CPU diagnostics run.

#### Geometry response failure is already present on the training pairs

`scripts/audit_classifier_training_response.py` reuses the exact corrected
prepared data, all three frozen classifiers, and random-direction forward
h=0.05 moment estimator from the earlier transfer audit. It verifies every
identity, usable/detected flag, context and shear against the preparation
receipts. Training cases40–99 and tuning cases100–119 remain separate. This is
a diagnostic of fit behavior, not independent validation. No response result
selects an epoch or changes a checkpoint.

CPU16580358 completes1m53s with the three existing analytic transfer/moment
tests passing22.74s, empty stderr, and exact historical sufficient-statistic
replay (normal, number and truth-shape maximum differences all0.0). Ruff and
bash syntax checks pass. Result and protocol are
`scene_classifier_geometry_20260918_v1/training_response.json` and
`training_response_protocol.json` below the cache root.

| Population | Image U diagonal (g1,g2) | Control predicted diagonal | Geometry predicted diagonal |
|---|---|---|---|
| Training60 | (-0.00611471,-0.00601602) | (-0.00609291,-0.00610274) | (+0.00285296,+0.00290104) |
| Tuning20 | (-0.00641225,-0.00624633) | (-0.00610447,-0.00609025) | (+0.00287987,+0.00286968) |

Geometry-minus-image errors are (+0.00896767,+0.00891706), with case SE
(0.00019447,0.00022373), on training; tuning errors are
(+0.00929212,+0.00911601), with SE(0.00037216,0.00049654). Historical ensemble
training errors are(-0.00023983,-0.00029050) and tuning errors are
(+0.00004780,-0.00004734). Complete off-diagonal/number/detection components
are retained in the result. These intrinsic-shape-weighted event moments are
not measured-shape response or m and use a different extraction from the
coherent audit; their numbers must not be pooled.

The geometry classifier's failure therefore precedes transfer to coherent
scenes and is not explained by that transfer alone. Probability-only training
can improve average loss while missing response information already present
in paired labels. A future classifier experiment should explicitly test
conditional response fit while retaining held-out probability performance;
neither an empirical offset nor a favorable joint-bias checkpoint is justified.

One feature-contract issue also merits checking before a new model: cached
pair distances use the original major-axis Re in the PSF-convolved scale
(`BlendEMU nz_utils.icat2reg` passes catalogue Re directly to `data_utils.rescale`),
whereas the eight-input classifier uses circularized Re. At fixed circularized
size, the former depends on intrinsic ellipticity. Holding those annuli fixed
while replacing intrinsic shape by sheared shape is thus not the same as
recomputing annuli from the latter shape. This is an identified coordinate
distinction, not proof that changing it will cure the failure. Any alternate
geometry must specify the same convention in preparation, training and
evaluation and be compared as a new, declared experiment. Frozen live/queued
evaluator sources and the current classifier artifacts remain unchanged.

#### Major-axis coordinate transport explains most of the geometry response failure

Added `scripts/scene_geometry_coordinates.py` and
`scripts/audit_geometry_coordinate_response.py`. At fixed circularized size Rc
and shape modulus e, major-axis size is Rc/sqrt((1-e)/(1+e)). The helper
reconstructs the cached primary scale sqrt(Rmajor^2+Rpsf^2), verifies it against
`Re_input_p_scaled`, and transports distances by the old/new scale ratio.
Physical separations and flux ratios are unchanged. Zero-shear features must
replay exactly. PSF half-light radius is calculated from the inherited FWHM
0.73 and Moffat beta2.224; this PSF number is distinct from the owner's
**measured galaxy FLUX_RADIUS>0.6 arcsec** primary cut.

The diagnostic retains the frozen geometry checkpoint and compares its
original static features with annuli recomputed from each sheared primary
shape. All independent40 cases and both fresh4 axes are retained, along with
the established magnitude bins, probability scores and event responses.
No measured-output cut, flow or emulator is evaluated. It is a counterfactual
coordinate check, not a validated replacement classifier: the checkpoint was
trained using static geometry in both legs.

Smoke16580393 fails on an unnecessarily tight raw aggregate replay threshold:
the discrepancy is about2e-11 per parent. Revision16580397 still executes the
superseded assertion despite revised source (the traceback shows the old
operation at changed line numbers). Both failures are preserved. The launcher
now compiles the exact source bytes whose hash it checks against the frozen
protocol, recording that executed hash. Prediction moments use the existing
2e-7 per-parent audit tolerance; counts and observed-U/D sufficient statistics
must replay exactly, and actual numerical differences are recorded. No
scientific result, checkpoint, population or threshold on bias is changed.

Revised smoke16580451 completes33s with3 tests passing0.13s. Full16580457
completes2m55s, also3 tests in0.13s, with empty stderr and exact static moment
replay on every full case. Tests verify major/circularized conversion,
preserved physical separation, zero-shear replay, rotation invariance and
rejection of an inconsistent cached radius or invalid shape. Ruff/bash checks
pass. Files are `scene_classifier_geometry_20260918_v1/coordinate_full.json`
and `coordinate_full_protocol.json`; protocol revisions and failed logs are
retained in the same artifact history.

| Endpoint | Static U-response error | Transported-coordinate error |
|---|---:|---:|
| Independent40, g1, h=0.05 | +0.00879871 +/-0.00011913 | -0.00059140 +/-0.00013096 |
| Fresh4, g1, h=0.02 | +0.00959303 +/-0.00020966 | -0.00043555 +/-0.00034267 |
| Fresh4, g2, h=0.02 | +0.00931890 +/-0.00083787 | -0.00075385 +/-0.00126950 |

These are all-parent intrinsic-shape event-response errors, not m. The
independent residual remains statistically nonzero; its faintest fixed bin is
-0.00124514 +/-0.00039121. Independent plus-leg BCE changes0.13806142 to
0.13797733 and Brier0.03844612 to0.03842003. Fresh plus-leg BCE changes
0.13581829 to0.13580541 on g1, but0.13570383 to0.13575448 on g2. All legs and
bins remain available. Geometry changes for356969/4437748 independent parents
in the plus leg, versus16159/444404 and16156/444404 in the two fresh axes.

The large response failure is therefore largely attributable to this
coordinate distinction in the frozen model's derivative, rather than a need
to fit an empirical response offset. The residual and the mismatch between
training/evaluation conventions prevent an acceptance claim. A new experiment
must use the state-major-axis convention in both training and evaluation.

#### Consistent-coordinate classifier preparation and training

Added `scripts/prepare_transported_geometry.py`, one additional preservation
test and its CPU launcher. It creates a separate
`scene_classifier_state_major_geometry_v2` dataset with explicit
`coordinate_convention=state_major_axis`. Every label, identity and first eight
inputs is preserved; zero-leg geometry must match exactly, while g05 geometry
is computed from the folded g05 shape at the same physical separations.
Original data and live evaluator sources are unchanged. The old trainer's
format check deliberately rejects this new data contract.

Preparation smoke16580469 completes18s with4 tests passing6.10s and empty
stderr. Zero-leg changed rows are0; g05 changes8874 rows in case40 and8945 in
case100. The test verifies bin transport, unchanged labels/base inputs and no
mutation of source arrays. Full80-case preparation16580483 is submitted with
the successful smoke and completed coordinate audit pinned.

`scripts/train_transported_classifier.py` deliberately preserves the original
classifier recipe and initialization, changing only the accepted data format
and adding explicit state-major-axis metadata to protocol/checkpoint. It
retains seed20260913, train40–99/tune100–119, the same architecture/optimizer,
and selection by minimum mean-case tune BCE. No response or joint-bias result
selects a checkpoint. CPU execution is recorded as a difference from the
earlier GPU training; numerical trajectories need not be identical. This
avoids extra GPU use or changes to queued joint16580144. Two-epoch training
smoke16580484 starts after the prepared-data smoke; its checkpoint is
operational only. Full training requires both full preparation and a successful
smoke. Candidate evaluation will require a new coordinate-aware adapter;
neither current likelihood configuration nor frozen classifier-arm evaluation
is changed.

Full consistent-coordinate preparation16580483 completes2m36s with empty
stderr, retaining8,876,218 parents. Zero-leg features change for0 parents;
g05 geometry changes for713595. The four targeted tests pass. Training
smoke16580484 completes38s and verifies checkpoint/tune-score replay after two
epochs; its loss is operational and is not compared to full scientific models.
Full CPU training16580498 is submitted using both successful receipts and the
full prepared manifest. The original 100-epoch/patience15 recipe remains
fixed; no independent/fresh responses enter its checkpoint selection.
The full job verifies inputs and completes its first two epochs with tune BCE
0.15431464/0.14887911, no clipped gradients and empty stderr; epoch2 takes12.79s.

#### Coordinate-aware joint evaluator and predeclared validation stages

Added `scripts/transported_classifier_joint.py`,
`scripts/evaluate_transported_classifier_joint.py`,
`scripts/validate_transported_classifier.py`, and an operational smoke runner.
The new evaluator deliberately extends the frozen three-classifier evaluator;
the original sources and queued16580144 are unchanged. It evaluates five arms
with identical flow/emulator draws:

| Arm | Checkpoint and geometry |
|---|---|
| `modeled_usable` | Historical three-member classifier ensemble, eight inputs |
| `modeled_control` | Fixed matched single classifier, eight inputs |
| `modeled_geometry` | Original single geometry classifier, static annuli |
| `modeled_coordinate` | Same original geometry checkpoint, state-dependent annuli; counterfactual diagnostic |
| `modeled_transported` | Newly trained single geometry classifier, state-dependent annuli in training and evaluation |

Each arm uses the same single flow and fixed disk emulator; these are model
comparisons, not a flow ensemble. Actual-U and matched-U branches also remain.
All three output cuts, classifier-weighted density moments/histograms and ten
paired classifier comparisons are retained. The primary cut stays strict
measured FLUX_RADIUS>0.6 arcsec and MAG_AUTO<25.8. The coordinate helper verifies
the observation conditions, baseline geometry and cached major-axis scale,
and preserves the flow's eight-input frames. Full evaluation rejects a smoke
checkpoint or a checkpoint without the new coordinate-contract metadata.

CPU smoke16580542 completes2m01s with3 tests passing26.08s and empty stderr.
The tests check identity/coordinate transport without changing flow inputs,
model-contract rejection, and all ten paired changes on an analytic fixture.
Both old and new evaluators run on the same CPU host with their exact
hash-verified source bytes. Every old branch replays exactly (maximum
sufficient-statistic difference0.0). The new arms are exercised on operational
case20000 using the two-epoch candidate; those bias numbers have no calibration
interpretation. Receipt: `transported_joint_smoke_completed.json` under the
scene-classifier artifact root. Ruff and launcher syntax checks pass.

`scripts/audit_transported_classifier_calibration.py` extends the fixed
probability/event-response audit to all five arms, using the identical
state-dependent geometry helper and stable log probabilities. It retains
main80, independent40, fresh g1 and fresh g2, all magnitude bins and geometry
residuals, without checkpoint reselection. CPU16580567 follows successful
training16580498. These endpoints are development tests: independent/fresh
labels are disjoint from classifier fitting but have informed this
investigation, so they are not untouched final acceptance samples.

`scripts/freeze_transported_classifier.py` binds predeclared validation
templates to the completed training output. It verifies the fixed split,
seed, stopping rule and minimum-tune-BCE checkpoint against the full history,
then pins the selected checkpoint and receipts. It does not choose an epoch
from response results. For joint validation it also verifies each completed
original reference against its recorded result hash and already pinned
upstream protocol, allowing all original branches to be replayed on the full
case sets. Two direct receipt/stopping tests and three deferred-reference
integrity cases are submitted as CPU16580623.

Joint binding16580568 follows both training16580498 and original joint
array16580144. New array16580569, tasks0–3 with throttle2, follows successful
binding and evaluates the same main80/independent40/fresh-axis endpoints.
This serializes both GPU arrays and preserves the two-GPU total without
changing the original comparison. Source/model/data contracts and successful
smoke receipts are pinned in `transported_calibration_template.json` and
`transported_joint_full_template.json`; completed-stage hashes will be added
to their frozen execution protocols before validation. No model or configured
likelihood is promoted from these preparations.

Receipt/stopping and deferred-reference job16580623 completes32s with5 tests
passing20.74s. Full-suite16580631 completes44s, with565 passing tests,4 skips
and the existing corner-extrapolation warning (36.40s). These support the
implementation and provenance checks; they do not establish scientific
calibration of the new checkpoint.

#### Completed consistent-coordinate classifier

Training16580498 completes22m24s, selecting epoch94 with mean-case tune
BCE0.13808794354 and Brier0.03843268363. The checkpoint is
`scene_classifier_geometry_20260918_v1/transported_classifier_full/selected.pt`,
SHA `e3077bf30426925e6ea3bdaff0a438b6770b4a2df8eb293e842b222752d24f76`.
Compared with the original static geometry classifier's tune
BCE0.13817769501/Brier0.03845941677, this preserves essentially the same
probability-score performance. It does not establish response or joint bias.
No downstream response entered the epoch choice, and production acceptance
remains false.

Audit16580567 starts automatically, passes5 receipt/statistic tests in7.36s,
verifies the completed training history and binds the exact selected hash in
`transported_calibration_protocol.json` before entering the fixed main80 cases.
The original and new joint arrays retain their declared dependency chain.

### Fresh continuation comparisons on both shear axes

The fixed four full-area scenes 20010–20013 are complete for initial, control
and augmented flows (16567110, comparisons 16567262). With coherent antithetic
h=0.02, measured MAG_AUTO<25.8 and radius>0.6 arcsec, actual per-leg U, and the
same fixed disk emulator and 64 CRN draws/seed7301:

| Flow | g1 m (%) | g2 m (%) |
|---|---:|---:|
| Initial corrected | +0.378600 ±0.818046 | -0.389917 ±0.723574 |
| Control continuation | +1.002768 ±0.818495 | +2.227450 ±0.744882 |
| Shape-augmented continuation | +1.664921 ±0.823413 | +3.233682 ±0.764681 |

Errors are case-bootstrap standard errors. Paired changes from initial are
+0.624168 ±0.003388pp (control g1), +2.617366 ±0.021522pp (control g2),
+1.286320 ±0.008903pp (augmented g1), and +3.623599 ±0.041717pp
(augmented g2). These are strong paired failures despite the weak absolute
precision of only four scenes. They corroborate, rather than replace, the
main and independent comparisons. The common-descent candidate has its own
separate frozen validation; it is not the control or augmented flow.
Selected common-descent validation16575483 subsequently completes both axes.
Its actual-U primary m is +0.402785 ±0.805590% (g1) and -0.469881 ±0.733269%
(g2). Paired shifts from initial are +0.024185 ±0.013944pp and
-0.079965 ±0.011723pp. Modeled-U values are -0.603985 ±0.808292% and
-1.395712 ±0.770364%. The roughly one-percentage-point modeled-U discrepancy
therefore persists in fresh scenes. The primary g1 model change is compatible
with the main/independent paired shifts; the new g2 change has the opposite
sign. Neither axis has sufficient absolute precision for acceptance.

All branches, including modeled U and the one-case h=0.04 diagnostic, remain
in the external results. No axis or branch is selected for acceptance.

## Acceptance and resource constraints

Track held-out density NLL, joint flux/radius/shape distributions, full 2x2
responses, and response errors over magnitude, radius, and crowding. A small
pooled number cannot certify cancellation between flow and emulator errors.
Keep simulation-family, population, usability, extraction, and random-stream
definitions explicit. Fresh multi-axis simulation acceptance and numerical
integration checks remain necessary before promotion; see
[acceptance boundary](INFERENCE.md#v35-like-status-and-acceptance-boundary).

All substantial work runs through Slurm with at most two SBSI GPUs in total.
Current objective audit outputs are under
`$DATA_DIR/sbsi_caches/guard_population_audit_20260917_v1/`.

### Consistent-coordinate classifier event-response results

CPU audit16580567 completes the predeclared main80, independent40 and both
fresh four-scene axes. Its selected epoch94 is unchanged. Results and every
magnitude-bin/off-diagonal diagnostic are in
`scene_classifier_geometry_20260918_v1/transported_calibration_full.json`;
`transported_calibration_summary.json` retains all five arms and the source hash.

| Endpoint | Static geometry shape-weighted U-response error | Consistent-coordinate error | New error case SE |
|---|---:|---:|---:|
| main80 g1 | +0.00868193 | -0.00059614 | 0.00019347 |
| independent40 g1 | +0.00879871 | -0.00044032 | 0.00013123 |
| fresh4 g1 | +0.00959303 | -0.00027800 | 0.00033659 |
| fresh4 g2 | +0.00931890 | -0.00064544 | 0.00126993 |

These are full-parent intrinsic-shape-weighted event-response errors, **not m**.
Independent plus-leg BCE is0.13796711, versus0.13806142 for static geometry
and0.19816398 for the matched eight-input control. The coordinate correction
removes most of the static model's response failure while retaining its
probability-score gain. A residual remains on independent40; fresh g2 is
imprecise. The joint selected-output response/density results remain necessary.
Detection and measurement usability are included in U; raw crossmatch D and
U-minus-D diagnostics remain separately reported. These development endpoints
have informed this investigation and are not untouched final acceptance data.

### Matched-image latent repeat completed

Matched-model array16577323 finishes both seeds7301/7302 (1h21m21s/1h15m39s).
Both use the identical full scene20011,111741 parents, two source-sign probes,
actual per-leg U,64 draws,h=0.02 and strict measured radius>0.6 arcsec with
MAG_AUTO<25.8. Input identity/provenance fields agree across repeats.

| Flow | Seed | g1 exact endpoint m (%) | g2 exact endpoint m (%) |
|---|---:|---:|---:|
| Initial | 7301 | -1.704418 | +0.006902 |
| Initial | 7302 | -1.717076 | -0.059693 |
| Common-descent | 7301 | -1.637731 | -0.080926 |
| Common-descent | 7302 | -1.692979 | -0.113045 |

The largest seed change is0.066595 percentage points. For common-descent g1,
the own-baseline-normalized physical self-response error is+0.01968488 and
+0.02014509; neighbour errors are-0.00822322 and-0.00828764. Their total errors
are+0.01146166 and+0.01185745. This cancellation persists across the two latent
seeds, but one scene and two sign probes cannot establish population bias or
provide a population confidence interval. Conditional probe SE is not a
Gaussian significance measure. Both models, axes, seeds and normalization
conventions remain in `flow_shape_augmentation_20260918_v1/` files
`matched_model_probe_full_seed7301.json` and `...seed7302.json`. These actual-U
results do not validate the new predictive classifier. No averaging into a
flow ensemble, empirical correction or checkpoint reselection is performed.

### Joint-validator runtime correction

Original joint array16580144 main/independent tasks showed roughly8–9 seconds
per1024-parent batch, implying that the declared four-hour limit was too short.
A read-only stack sample found CPU XGBoost tree prediction active while a GPU
utilization snapshot was zero; this supports a CPU emulator bottleneck, not the
initial untested suspicion about density histograms. The scientific sources,
protocols, checkpoints, draws and all endpoints stay unchanged.

Slurm denied increasing the already-running tasks' limits. Tasks0/1 were
requeued under their existing array identities with36-hour limits; their
startup logs were preserved with `.before_walltime_requeue` suffixes. Pending
tasks2/3 and dependent array16580569 also receive36-hour limits. The scheduler
started fresh-axis tasks2/3 first. Existing success dependencies and throttle2
remain intact, including binder16580568. This is a scheduling adjustment with
no scientific selection or change in numerical execution. Full results and
reference replay remain pending; no model is promoted.

### Signed-probe validation of the coordinate correction

CPU16580775 completes1m11s with five prerequisite tests passing18.06s and
empty stderr. `scripts/audit_signed_transported_classifier.py` extends the
original frozen signed audit to all five arms using the existing transported
geometry helper. It changes no live joint-evaluator sources. All original
signed-model fields replay exactly (maximum difference0.0). All signed
probability swaps are exact; coherent moments match the separate completed
five-arm audit to at most6.25e-9, below its existing2e-7 probability tolerance.
Artifacts are `scene_classifier_geometry_20260918_v1/signed_transported.json`
and `signed_transported_protocol.json`, with source/model/reference hashes.

The fixed full scene20011, two source-sign probes, h=0.02 and intrinsic-shape
basis are unchanged. There is no measured-output cut in this U-response
diagnostic. These response errors are not multiplicative shear bias m.

| Classifier arm | g1 self error | g2 self error | g1 coherent error | g2 coherent error |
|---|---:|---:|---:|---:|
| Historical ensemble | +0.00188800 | +0.00099846 | +0.00091962 | -0.00081281 |
| Matched single control | +0.00214600 | +0.00119582 | +0.00117763 | -0.00061546 |
| Static geometry | +0.01117220 | +0.01052455 | +0.01020383 | +0.00871327 |
| Original checkpoint, transported coordinates | +0.00063099 | -0.00025640 | -0.00033739 | -0.00206767 |
| Consistently trained coordinates | +0.00080171 | -0.00013598 | -0.00016666 | -0.00194726 |

The image self response is-0.00834745/-0.00743210 for g1/g2. The new model
predicts-0.00754574/-0.00756809. Thus the previously large, wrong-sign static
geometry response was predominantly a self-coordinate failure. Every present
classifier has zero neighbour-shape response by construction: its geometry
uses fixed neighbour fluxes/distances and primary-dependent scaling. The image
neighbour contribution is+0.00096837/+0.00181127; every model therefore has
neighbour error-0.00096837/-0.00181127. For the new model g2 this dominates the
remaining coherent error on this scene. Conditional two-probe SE is
0.00079361/0.00023799; it is neither population uncertainty nor a reliable
Gaussian significance measure.

This diagnoses a possible missing interaction if joint calibration fails;
it does not impose a new requirement that each physical component separately
meet0.3%. All arms and both axes remain reported. The practical decision still
rests on reproducible total selected-output bias and density with the fixed
measured0.6-arcsec cut, evaluated by the already frozen joint jobs. No response
result is used to reselect or refit the classifier in this audit.

### Reference prediction device restored

The earlier36-hour scheduling adjustment is superseded. Inspection shows
that all original joint references used `experimental_response.prediction_device`
CUDA, while the three-/five-classifier launch protocols specified CPU. The
CPU bottleneck was therefore an accidental device change. A bounded benchmark
of the fixed disk model on three development parts finds CPU/CUDA raw tree
prediction differences up to1.1920928955e-7, despite representing the same
trained model. Warm CUDA prediction including input construction takes about
0.017s per42k rows versus0.30–0.31s CPU on the sampled node. This is a finite
numerical/performance check, not full evaluator equivalence or calibration.

No replay tolerance is relaxed. New versioned protocols restore CUDA to match
the references, keeping all scientific inputs and diagnostics. The cancelled
CPU array16580144 and downstream16580568/16580569 retain their logs/protocols;
none had completed a scientific endpoint. Replacement16580808 runs both fresh
axes first and then main/independent; binder16580811 and five-arm16580812 follow
successfully completed reference receipts. Both arrays retain throttle2 and
12-hour task limits, with termination/success dependencies preserving the
two-GPU total. New artifacts have `_cuda` suffixes; original names are not
reused or overwritten. Sources and model checkpoints are unchanged. Full
strict replay against reference sufficient statistics remains the decisive
execution check before interpreting new classifier results.

### Fresh GPU classifier reference comparison

Array16580808 tasks0/1 complete6m22s/5m31s. Both reference comparisons have
maximum sufficient-statistic replay difference0.0, with verified result and
protocol hashes. Sources/models/tolerances are unchanged from their frozen
contracts; CUDA response prediction matches the earlier reference execution.
Main80 and independent40 tasks2/3 start automatically afterward.

For the same four development scenes20010–20013, antithetic h=0.02,64 CRN
draws seed7301, fixed common-descent single flow and disk emulator, strict
measured FLUX_RADIUS>0.6 arcsec and MAG_AUTO<25.8:

| Usability arm | g1 m (%) | g2 m (%) |
|---|---:|---:|
| Actual observed U (oracle) | +0.402785 ±0.805590 | -0.469881 ±0.733269 |
| Historical predicted-U ensemble | -0.603985 ±0.808292 | -1.395712 ±0.770364 |
| Matched single control | -0.584590 ±0.807950 | -1.382080 ±0.770555 |
| Static geometry single classifier | -0.400438 ±0.819196 | -1.301380 ±0.772857 |

Errors are case-bootstrap SE, not finite-draw/training uncertainty. Four
scenes do not provide precise absolute calibration. Static geometry improves
g2 relative to historical by+0.094331±0.010729 percentage points in paired
case differences, while its intrinsic-shape U-response diagnostic is worse;
that diagnostic must therefore not substitute for the measured joint response.
The consistently trained and coordinate-only arms are still pending and are
not represented by the static-geometry row.

Fixed four-dimensional joint-histogram TV for the modeled-U arms ranges from
0.03814 to0.03916 across plus/minus legs and axes. Changes among these arms are
small despite large differences in classifier BCE. These coarse histograms
include sampling noise and do not establish density equivalence or an
acceptance threshold. All cut/moment/response fields remain in
`joint_fresh_g1_cuda.json` and `joint_fresh_g2_cuda.json` under the scene-classifier
root; `fresh_cuda_reference_summary.json` includes every arm and both axes
with source hashes. No axis, classifier or cutoff is selected from these
results, and no configuration/model is promoted.

### Normalized selected-density comparison of the two single flows

CPU16580912 completes32s, with three analytic tests passing14.37s and empty
stderr. This adds a missing normalization to the existing transported physical
NLL comparison; it does not fit or alter either flow, emulator, classifier,
cut or response estimate. Inputs are the frozen main80 initial/common-descent
response sufficient statistics and `transported_density_full.json`. The code
checks flow/emulator/population identities, measured numerator/count replay,
per-leg measurement receipts and observed density counts. Output is
`joint_probability_refinement_full/normalized_selected_density.json` in the
flow-augmentation root, with source/script hashes and every case/leg/cut term.

Define a uniformly chosen parent identity i from the supplied finite parent
population. With observed usability U_i held fixed, and measured cut S, the
joint law of parent identity and physical output y conditional on selection is

`q(i,y | S,U_actual) = U_i q_i(y) 1_S(y) / sum_j U_j P_q(S|j)`.

Thus the candidate-minus-reference mean selected NLL is
`delta_raw_NLL + log(M_candidate/M_reference)`, where M is total generated
selected mass. This is a pooled parent-and-output score. It is **not** the
likelihood conditional separately on each parent (which would require each
parent's own pass probability), nor a marginalized output-only score. The
actual-U conditioning makes this an oracle density diagnostic, not validation
of predicted detection. The normalizer is part of the stated probability law;
it is not an empirical response offset or rescaling.

For the unchanged primary measured MAG_AUTO<25.8 and FLUX_RADIUS>0.6 arcsec:

| Leg | Raw NLL change | Log selection-mass ratio | Normalized NLL change ± case SE |
|---|---:|---:|---:|
| +0.02 | +0.00314135 | -0.00364851 | -0.00050716 ±0.00002575 |
| -0.02 | +0.00312024 | -0.00365095 | -0.00053071 ±0.00002367 |

Units are nat/selected object; negative favors common-descent over initial.
The normalized primary score therefore improves slightly even though its raw
restricted score worsens. The uncut changes remain+0.00076405±0.00003963 and
+0.00075929±0.00003707 nat/object because the uncut normalizer ratio is one.
The retained secondary0.75-arcsec diagnostic gives-0.00191409±0.00002228 and
-0.00200771±0.00002090; this is not substituted for the primary0.6-arcsec cut.

The calculation preserves the original density audit's2000 paired case
bootstrap replicas/seed20260918. Tests prove invariance when density within
the cut and pass probability scale together, distinguish pooled versus
per-case conditioning, and reject invalid masses/weights. Selection masses
still use the existing64-draw integration; these errors omit finite-draw and
training uncertainty. The score is relative to the initial corrected single
flow and does not establish absolute density adequacy, predicted-U accuracy,
or0.3% calibration. It does resolve the misleading interpretation of the raw
selected NLL increase while retaining the small uncut density cost.

### Corrected classifier on both fresh axes and case influence

Independent reference16580808_3 first completes56m49s with exact replay.
Its historical/control/static-geometry primary m values are
-0.542806±0.185983%,-0.524321±0.186046%, and-0.340327±0.187548%; these are
not the corrected-coordinate candidate. Static geometry raises its flow
contribution by0.00200092 versus control but reduces blending by0.00328012,
so total improvement alone conceals opposing component changes.

With the main reference still using one GPU, early binder16581486 verifies
the completed training and two fresh references, then array16581487 runs
both predeclared fresh axes serially on the other GPU. Full binder16580811
also waits for this early array to terminate, preserving two GPUs total.
These are identical scientific arguments to the queued full comparisons,
with separate `_early` output names; no cases/axes/models are selected by
result. The early tasks finish5m04s/5m56s with maximum reference sufficient
statistic difference0.0. Full main/independent results are still pending.

For all four fixed scenes20010–20013,64 shared draws seed7301,h=0.02, one
unchanged common-descent flow and disk emulator, measured FLUX_RADIUS>0.6
arcsec and MAG_AUTO<25.8:

| Classifier | g1 m (%) | g2 m (%) |
|---|---:|---:|
| Historical classifier ensemble | -0.603985 ±0.808292 | -1.395712 ±0.770364 |
| Matched single control | -0.584590 ±0.807950 | -1.382080 ±0.770555 |
| Static geometry single | -0.400438 ±0.819196 | -1.301380 ±0.772857 |
| Original geometry weights, transported coordinates | -0.039801 ±0.819371 | -0.857183 ±0.784445 |
| Consistently trained coordinates, single classifier | -0.045290 ±0.819529 | -0.862256 ±0.784785 |

Errors are case-bootstrap SE. Static-to-consistent paired changes are
+0.355147±0.011685pp and+0.439124±0.021469pp. Most improvement follows from
the coordinate contract, not reselection of a favorable trained checkpoint.
No averaging of these model arms into a flow ensemble is performed. The g1
point estimate is close to zero, but g2 remains appreciably negative and both
have large four-scene uncertainty. These results do not establish0.3% bias.

CPU influence audit16581503 completes32s with two tests passing14.71s and
empty stderr. It checks every arm/cut and recomputes the pooled response
ratio after each possible scene omission, without actually excluding any
scene from the reported estimate. For the consistent model, primary LOO
ranges are[-0.836247,+0.648088]% g1 and[-1.277733,+0.037561]% g2; largest
changes are0.790957/0.899817pp, both associated with case20012. The corresponding
paired gains over historical remain0.542911–0.570011pp and0.501398–0.547313pp
across all omissions. Thus the *improvement* persists under these influence
checks, while the *absolute bias* is not stable enough to certify the target.
LOO ranges are not confidence intervals. All four cases are retained.

For the consistent classifier, predicted-U minus actual-U model response is
+0.00307194(g1)/+0.00272766(g2). Its flow differences are+0.00008881/-0.00033784,
while blending differences remain+0.00298312/+0.00306550. These algebraic
components are not the physical self/neighbour decomposition. Fixed joint
histogram TV remains0.03819–0.03915(g1) and0.03838–0.03902(g2), essentially
unchanged from static geometry; it is a coarse descriptive metric without
an acceptance threshold.

The residual suggests checking conditional probability/output/blending
coupling if the larger samples also fail. The flow already uses
`nbr_flux_near`, `nbr_flux_far`, and `nbr_flux_max`; it does not use the added
five annular geometry features. Any proposed geometry extension must test
its incremental information beyond these existing inputs. This is a
hypothesis, not evidence that adding flow inputs or applying a response
correction is currently justified. No empirical offset is applied.

All full early outputs and comparisons are in the scene-classifier root as
`transported_joint_fresh_g[12]_cuda_early.json` and corresponding receipts;
`transported_early_influence.json` retains every omission/model/cut with
source hashes. Later full confirmation and its all-endpoint influence audit
remain queued. These scenes are development data and are not untouched
final acceptance samples.

### Full corrected classifier and predicted-U selected density

Full four-endpoint validation16580812 completes with exact reference replay
(maximum sufficient difference0.0) for every endpoint. Fresh results reproduce
the early execution. Primary measured FLUX_RADIUS>0.6 arcsec, MAG_AUTO<25.8,
single fixed common-descent flow and disk emulator:

| Endpoint | Historical predicted U m (%) | Corrected predicted U m (%) | Actual U oracle m (%) |
|---|---:|---:|---:|
| Main80, g1 h=0.02 | -1.140485 ±0.261272 | -0.608908 ±0.261944 | -0.135051 ±0.265483 |
| Independent40, g1 h=0.05 | -0.542806 ±0.185983 | -0.013201 ±0.187994 | +0.477637 ±0.183980 |
| Four additional scenes, g1 h=0.02 | -0.603985 ±0.808292 | -0.045290 ±0.819529 | +0.402785 ±0.805590 |
| Same four scenes, g2 h=0.02 | -1.395712 ±0.770364 | -0.862256 ±0.784785 | -0.469881 ±0.733269 |

Errors are case-bootstrap standard errors. All arms/cuts remain in the full
artifacts. Main leave-one-out estimates range[-0.670045,-0.515687]%; independent
range[-0.088174,+0.038848]%. The main residual survives every single-scene
omission. The improvement is about0.53pp on both large samples, but the
independent near-zero point cannot substitute for the main result. The
samples differ in both scene realization and shear amplitude. Predicted-U
minus actual-U model response remains+0.00327034(main)/+0.00336741(independent),
principally the blend contribution+0.00345776/+0.00368203. These quantities
diagnose weighting and are not offsets to subtract.

`scripts/audit_classifier_selected_density.py` evaluates the finite-parent
selected law proportional to `p_a(U|i) q(y|i) S(y)`, with normalization
`M_a=sum_i p_a(U|i) P_q(S|i)`. The common fixed flow/emulator density cancels
in log ratios between classifier arms. Candidate-minus-historical selected
mean NLL is mean_selected(log(p_historical/p_candidate)) plus
log(M_candidate/M_historical). This is a joint parent/output score, not a
conditional score separately normalized for each parent, nor an output-only
marginal score. Source/model/measurement receipts are checked; all arms/cuts,
legs and endpoints are retained. The mass uses the evaluator's64 shared draws.

The corrected primary normalized NLL changes are-0.02378085/-0.02379564nat
per object on main plus/minus and-0.02353859/-0.02353554 on independent.
Case SEs are0.00006650/0.00006814 and0.00009164/0.00008021. All four fresh
axis/leg scores improve by0.02334–0.02353nat. These relative gains include
predicted detection/usability, but do not establish absolute density adequacy
or the shear-calibration target. Bootstrap errors omit finite-draw and training
uncertainty. Artifacts: `classifier_selected_density_observed.json`,
`classifier_selected_density_full.json`, and `transported_full_influence.json`
in the scene-classifier cache root.

### Controlled smooth-crowding classifier experiment

The fixed continuous-crowding audit finds predicted-minus-observed U residual
partial correlations0.010924,0.091685,0.138767,0.078285 on independent40 after
linear projection on the13 static classifier inputs and intercept. The four
Gaussian kernel scales0.5,1,2,4 are all retained; fresh axes give similar
patterns. This is an association, not proof that these summaries add
information beyond a nonlinear use of existing inputs or cause a shear bias.

The next controlled experiment appends all four smooth flux-ratio summaries
to the13 consistent state-major inputs. It uses the same existing corrected
labels, case splits, seed20260913,64-wide two-layer network, optimizer and
minimum mean-case tune-BCE selection rule. Four additional columns add256
weights. Geometry is recomputed in each shear state using the same audited
major-axis coordinate mapping in preparation and evaluation. No measured
output or response target is an input or checkpoint-selection criterion.
All five previous classifier arms remain alongside the new arm; the joint
validator must replay their sufficient statistics before accepting the run.
The same four endpoints/three cuts and both fresh axes are predeclared.
No new images or flow ensemble are introduced.

### Completed smooth classifier and reproducibility repeats

Full training16586130 completes19m29s, selecting epoch99 solely by minimum
mean-case tune BCE0.11856610059211203 (previous corrected classifier0.13808794).
The single17-input classifier has5377 parameters; checkpoint SHA256
`ad975edf7020aacce00ee3efcc30353a87c5bc1eec030b2017f67369c3c78f3c`.
Joint16586160 completes all four endpoints with exactly unchanged previous
five-arm sufficient statistics (maximum replay difference0.0). The flow and
disk emulator remain unchanged. Primary measured>0.6-arcsec/MAG_AUTO<25.8:

| Endpoint | Smooth predicted-U m (%) | Change from prior corrected classifier (pp) | Leave-one-scene-out m range (%) |
|---|---:|---:|---:|
| Main80, g1 h=0.02 | -0.199979 ±0.263649 | +0.408929 ±0.004146 | [-0.260896,-0.106446] |
| Independent40, g1 h=0.05 | +0.402474 ±0.188664 | +0.415675 ±0.004709 | [+0.327589,+0.454937] |
| Four additional scenes, g1 h=0.02 | +0.347168 ±0.817220 | +0.392458 ±0.016440 | [-0.435775,+1.050883] |
| Same four scenes, g2 h=0.02 | -0.443430 ±0.794924 | +0.418826 ±0.018523 | [-0.848716,+0.469608] |

Errors are case-bootstrap SE. Omission ranges measure influence, not confidence
or scene-exclusion rules. The gain versus historical remains positive under
every omission, about0.94pp on the two large samples. All original scenes,
cuts and axes remain in their point estimates. This is a candidate at roughly
the requested response scale, not proof from a favorably selected endpoint.

The modeled-minus-actual-U response discrepancy is now+0.00044627(main),
+0.00051353(independent),+0.00037981(fresh g1), and-0.00018311(fresh g2),
corresponding to predicted-versus-oracle bias differences of roughly0.03–0.08pp.
Thus the classifier's former large discrepancy is largely removed without
fitting to joint response. This does not separately identify raw detection
and measurement-failure calibration or eliminate flow/emulator uncertainty.

Density collector16586194 completes18m09s and full summary16586197 completes43s;
three tests pass16.03s at the final stage. Smooth-minus-historical primary
selected NLL is-0.0335900/-0.0336275nat/object main plus/minus and
-0.0333313/-0.0333498 independent; fresh axis/legs range-0.03306 to-0.03331.
This adds a0.00972–0.00983nat improvement over the previous corrected classifier.
Coarse uncut histogram TV remains about0.0354(main),0.0365–0.0370(independent),
0.0382–0.0392(fresh). Primary mean-shape offsets and percent-level variance
discrepancies remain visible. There is no new absolute-density threshold.
Artifacts are `smooth_joint_*.json`, `smooth_density_full.json`, and
`smooth_full_influence.json` under the scene-classifier cache root.

The epoch99 model is now frozen for reproducibility checks. A second64-draw
latent stream7302 repeats every endpoint with identical model/data/evaluator
provenance; the existing latent-repeat comparator requires exact observed
statistics and reports all branch changes. Separately, classifier seed20260914
repeats the same training recipe, splits and minimum-tune-BCE selection rule.
Its joint and selected-density validation retain all four endpoints/three cuts.
Neither repeat is averaged into deployment or chosen by a favorable bias.
The latter assesses classifier training variation only; it does not represent
independent flow or emulator retraining. GPU evaluations are serialized by
array dependencies and throttle2; CPU classifier training overlaps safely.

### Smooth classifier response-vector validation

CPU16587517 completes35s and one unequal-mass, paired-vector test passes13.37s.
`scripts/summarize_smooth_response_vectors.py` reads verified existing sufficient
statistics, retaining every cohort, cut and arm. It uses the original10000
scene-bootstrap replicates/seed20260914 and reproduces the saved diagonal
points and SEs. No simulations, model changes or response corrections are added.
The result is `smooth_response_vectors.json` in the scene-classifier cache.

For the fixed primary measured>0.6-arcsec cut, define cross leakage as
`100*(R_measured_ij-R_model_ij)/R_model_jj`, with i different from j.
This is a cross-component residual normalized by the diagonal, not the
multiplicative bias m. Smooth predicted-U results, with paired case SE:

| Endpoint | Cross component | Leakage (%) | Raw response residual |
|---|---|---:|---:|
| Main80 | R21 | +0.049824 ±0.294751 | +0.000341992 |
| Independent40 | R21 | -0.199118 ±0.184632 | -0.001366900 |
| Four additional scenes, g1 | R21 | -1.387767 ±0.346398 | -0.009515282 |
| Same four scenes, g2 | R12 | -0.788142 ±0.262163 | -0.005430219 |

On the same four scenes, rows(e1,e2)/columns(g1,g2) give measured
`[[0.688034335,-0.006803738],[-0.010215189,0.685934749]]`, versus smooth
model `[[0.685653966,-0.001373519],[-0.000699907,0.688989937]]`.
Observed-U oracle cross leakages remain-1.488643 ±0.304932% and
-0.839496 ±0.298054%, respectively. Historical predicted-U values are also
similar. Thus improving predicted usability has not removed this discrepancy.

Four scenes provide weak uncertainty estimates, and these data have already
been used for development. Nevertheless, neither limitation justifies
discarding the off-diagonal discrepancy or calling the full matrix calibrated
from the diagonal m values alone. The large samples provide only the g1
column and cannot substitute for the missing large-sample g2 check. Next,
inspect per-scene influence and the existing second latent stream, then
localize whether the cross response comes from image measurement/selection,
finite scene anisotropy, or the flow/emulator response. Density adequacy and
property-resolved response remain separate requirements.

Follow-up CPU16587535 completes40s with empty stderr, saving
`smooth_cross_response_audit.json`. It retains every case/cut/arm, verifies
the existing seed7302 comparison receipts and exact measured statistics,
and uses the already-tested response-vector helper. For the primary smooth
cross residual, every leave-one-scene-out estimate remains negative:
R21 ranges[-1.636382,-1.054891]% and R12[-1.041605,-0.569279]%.
Seed7302 gives-1.387614% and-0.778226%, changes of
+0.000153 +/-0.011507pp and+0.009916 +/-0.019688pp. Thus neither a single
scene nor the difference between these two latent streams explains the signal.
These omission ranges are influence diagnostics, not confidence intervals.

The uncut smooth cross residuals remain-1.257697%(R21) and-0.355919%(R12).
The uncut matched-usable branch, which uses the same measured usable objects
in both legs, gives-1.322374% and-0.560332%. This rules out per-leg usability
turnover or the measured photometric thresholds as a complete explanation,
but does not validate the shape response conditional on that matched sample.
The four image manifests declare distinct scene seeds20133–20136 and distinct
noise seeds2036291831,2037291834,2038291837,2039291840. Shared noise metadata
hashes describe the noise settings, not pixel identity. A direct image-noise
audit has not been added. Case20010 already contains both axes at h=0.04;
the other three contain h=0.02 only, providing a limited existing-data route
to test measured amplitude dependence before modifying any response model.

### Existing-image cross-response amplitude audit

CPU16587781 completes22s with empty stderr. The new
`scripts/audit_existing_cross_amplitude.py` verifies parent, crossmatch and
measurement hashes and replays the saved h=0.02 observed sufficient statistics
for both actual-U and matched-U branches, every cut. It reads case20010's
existing eight nonzero legs, without fitting a model or generating images.
All four declared masking diagnostics and all three original cuts are saved
in `existing_cross_amplitude.json` in the scene-classifier cache.

The primary sample selected and usable in **all eight legs** has78,404 objects.
Its fixed membership is solely a mechanism diagnostic, not the acceptance
population. Measured response matrices (rows e1/e2, columns g1/g2) are:

* h=0.02: `[[0.713881743,0.006149838],[-0.007809566,0.732475632]]`.
* h=0.04: `[[0.713928704,0.003553976],[0.000215271,0.724674008]]`.

Thus the g1 cross response changes by+0.008024837 while R11 changes only
+0.000046961 on identical objects. The uncut common-usable sample likewise
changes R21 from-0.005093295 to+0.001729898. For the actual per-leg primary
selection, R21 changes from-0.005785428 to+0.000549584 and R12 from
-0.001110768 to+0.004858442. Neither amplitude is selected as the answer;
no h=0.04 model response has been evaluated here.

Bare intrinsic mean cross Jacobians over the four generated parents are
+0.000072967,+0.000596979,-0.000129833,-0.000323447; the all-eight selected
case20010 value is+0.000593528. These describe input orientation anisotropy,
not measured responsivity or a correction. They do not by themselves explain
the h=0.02 measured cross term, but cannot rule out blend-conditioned effects.

Only one scene has the two amplitudes, so the result does not distinguish
finite-step effects, noisy/nonsmooth measurement response, or missing model
physics. It does establish that a constant correction inferred from this
scene's h=0.02 cross response would be poorly supported. Next inspect spatial
and per-object contributions in the same measurements and measurement/noise
provenance before changing the response model.

Separately, independent40 latent repeat16587387_3 completes48m23s. Smooth
primary m=+0.391591 +/-0.187383%, a change of-0.010883 +/-0.005890pp
from seed7301 (paired95% interval[-0.022495,+0.000471]pp). Actual-U gives
+0.469058 +/-0.182518%, a change of-0.008579 +/-0.007011pp. Both streams
remain reported. Replica density collector16587401 completes18m13s with
empty stderr and all four endpoints; the normalized replica density result
still depends on the pending replica joint evaluations.

### Spatial and object contributions to measured cross response

CPU16587813 completes1m09s; the unequal-block-population unit test passes20.43s.
`scripts/audit_measured_cross_contributions.py` verifies input hashes and
replays canonical per-leg measured sufficient statistics before reporting
fixed8x8 equal-width sky cells and object contributions. All three cuts,
both axes and all four scenes remain. The masks requiring selection in all
four h=0.02 legs, or all eight case20010 legs, are conditional mechanism
diagnostics and do not replace the original per-leg acceptance population.
Artifact: `measured_cross_contributions.json` in the scene-classifier cache.

For case20010's same78,404 primary objects selected in all eight legs,
R21 at h=0.02 is-0.007809566 with spatial-block jackknife SE0.00840159;
at h=0.04 it is+0.000215271 with SE0.00553364. The paired amplitude change
is+0.008024837 +/-0.00779344. R12's paired change is
-0.002595862 +/-0.00707766. These are within-scene spatial diagnostics,
not scene-bootstrap errors or a calibrated replacement uncertainty for m.
They weaken an interpretation of the observed amplitude difference as a
demonstrated systematic finite-step effect: the difference is noisy on
the measured common-object sample.

Individual h=0.02 cross derivatives span approximately-47 to+49, despite
each endpoint satisfying the original shape usability rule. On the common
four-leg primary samples, the largest1% by absolute cross derivative
contribute-0.0130534 and-0.0209442 to the full R21 means of cases20012/20013
(-0.0177140/-0.0185556), and-0.00606767/-0.0170467 to their R12 means
(-0.00853115/-0.01346233). This documents heavy tails; it is not an
exclusion rule or permission to clip, winsorize, or select a favorable mean.
No single spatial block accounts for the full negative R21 in these scenes.

The source worker constructs the same per-object optimizer RandomState anew
for every leg. Large changes can still arise from nonlinear fits, moving
detection centroids, blend association, or image fluctuations; the audit
does not identify which. Next inspect the retained largest-contribution
objects' paired shapes, centroids and fit flags to distinguish these mechanisms.
The original four-scene cross discrepancy and its small-sample limitations
remain reported; it is neither corrected nor declared resolved.

### Tail-object inspection and owner priority

CPU16587840 completes60s with empty stderr. New
`scripts/audit_cross_tail_objects.py` verifies the measurement receipts,
fit-status shape replay and the previous contribution audit's means and
largest-object identities. All three cuts and both axes are retained.
On each four-leg common-selected primary sample,81–86% of the largest1%
of absolute cross derivatives have opposing fitted shape vectors, versus
about1.9–2.0% overall. Their median centroid shifts are0.023–0.029 pixels;
only0.5–2.0% shift by more than one pixel. All selected fits have zero
recorded flags. The tails therefore are not predominantly large detection
centroid jumps, but this does not establish optimizer failure or justify
removing any object. Artifact: `cross_tail_objects.json`.

A bounded same-image probe was submitted as CPU16587861 before the owner's
priority clarification. BlendEMU's new `probe_cross_tail_optimizer.py`
reuses the frozen measurement worker for the union of40 response-ranked
tail objects (or fewer if the two top20 lists overlap) and40 uniformly
sampled generated-parent controls. It includes all nine existing case20010
legs and three fixed optimizer streams; the original stream must replay.
It writes separate diagnostics and cannot replace canonical measurements.
`cross_optimizer_protocol.json` freezes code/inputs and the control seed.

The owner then clarified that cross-components have **less priority**.
Further cross-component development is paused. The main focus returns to
diagonal calibration, density and training/sampling reproducibility.
Replica GPU array16587398 is changed to throttle1 and released after
verifying its completed binder16587397 and absence of any dependency on
unfinished latent-repeat outputs. This overlaps one replica task with
main80 latent repeat16587387_2, preserving two GPUs total. Model, data,
evaluator and comparison protocols are unchanged. Throttle2 may be restored
after the remaining latent repeat finishes.

### Consolidated reproducibility evidence report

Added `scripts/report_smooth_reproducibility.py` and dependent CPU16587896.
After the latent array, replica paired comparisons/influence and normalized
replica density summary complete, it will verify result/receipt identities
and collect every endpoint, cut and classifier arm into
`reproducibility_report/evidence.json` and a primary-cut `report.md` under
the scene-classifier cache. It retains the original candidate and displays
original latent7301, repeated latent7302, and classifier seed20260914/latent7301
separately. No amplitudes are pooled, no seed is chosen by bias, and the
report does not automatically declare calibration or promote configuration.
Ruff and launcher syntax pass; full-data runtime validation is pending.

The first replica endpoint16587398_0 completes5m22s with unchanged original
five-arm sufficient statistics (maximum difference0.0). Smooth primary
fresh-g1 diagonal m=+0.410181 +/-0.810691%, versus original
+0.347168 +/-0.817220%. The0.063013pp point shift is retained; its paired
training-change uncertainty will be reported by the already-queued common
comparison job. Four scenes alone do not establish training robustness.
Replica g2 starts next while the main80 latent repeat continues.

### Completed sampling and classifier-training reproducibility

All four latent repeats and all four classifier-training-repeat endpoints
complete successfully. Replica comparisons/influence16587400 completes72s,
normalized replica density16587403 completes51s, and consolidated evidence
16587896 completes27s; their error logs are empty. The automatic throttle
restoration16587872 also succeeds, preserving two GPUs total throughout.

Primary diagonal m (%) with case-bootstrap SE (percentage points):

| Endpoint | Original classifier, latent7301 | Same model, latent7302 | Classifier seed20260914, latent7301 |
|---|---:|---:|---:|
| Main80, g1 h=0.02 | -0.199979 ±0.263649 | -0.209397 ±0.263432 | -0.134885 ±0.263934 |
| Independent40, g1 h=0.05 | +0.402474 ±0.188664 | +0.391591 ±0.187383 | +0.467089 ±0.188454 |
| Four additional scenes, g1 h=0.02 | +0.347168 ±0.817220 | +0.289840 ±0.824666 | +0.410181 ±0.810691 |
| Same four scenes, g2 h=0.02 | -0.443430 ±0.794924 | -0.425409 ±0.777288 | -0.384972 ±0.800023 |

On main/independent samples the latent changes are-0.009417 ±0.005147pp
and-0.010883 ±0.005890pp. Classifier-training changes are
+0.065095 ±0.001101pp and+0.064615 ±0.001095pp. These paired errors describe
scene variation in the difference, not uncertainty over all possible training
seeds. Replica selected-density gains versus the historical classifier remain
-0.033446/-0.033483nat per object on main plus/minus and
-0.033170/-0.033200 on independent plus/minus. All other endpoints also retain
the improvement. Replica leave-one-scene-out primary m ranges are
[-0.195772,-0.041150]% main and[+0.392346,+0.519838]% independent; no scene
is removed. All retained old-five-arm statistics replay exactly.

This supports practical diagonal response stability to the tested latent and
classifier streams, without a required ensemble or selecting a favorable
seed. It does not establish independence from the trained flow itself or
absolute density equivalence. The original epoch99 classifier and step2 flow
remain the candidate. Cross-components stay a lower-priority caveat under
the owner's instruction. The complete all-arm/cut evidence is in
`reproducibility_report/evidence.json`; the readable table is `report.md`.

### Fixed-recipe flow-refinement repeat

To address the remaining flow-side randomness question, freeze one additional
common-descent refinement. `refine_joint_guard_probability_flow_repeat.py`
is byte-for-byte the original source after adding101 to its eight declared
random-seed constants: NLL subsets, gradient, line-search, validation response
and validation probability streams. `check_flow_refinement_repeat_protocol.py`
checks that this is the only source difference and verifies29 input/code pins
from `flow_refinement_repeat_protocol.json` in the augmentation cache.

The initial flow, reference integrals, target banks, six-step recipe,
learning-rate grid,0.001nat NLL budget and response/probability selection gates
are unchanged. No joint validation result enters checkpoint selection. The
original candidate is retained regardless of the repeat's bias, and the two
flows will not be averaged. This tests stochastic refinement conditional on
the same pretrained initial checkpoint; it is **not independent pretraining**
or emulator retraining. CPU smoke16591198 precedes GPU full16591200.
Ruff, shell syntax and source-difference/pin checks pass. A first full-job
submission used the CPU partition and was rejected without starting a job;
the accepted submission uses inter/cip with one A40. Downstream response and
density comparisons must retain all declared endpoints and the fixed original
smooth classifier; they remain pending until the repeat is trained.

### Predicted-U density check for the flow repeat

Added `scripts/audit_flow_repeat_density.py` and GPU array16591245, dependent
on the complete flow-repeat response array16591210 with throttle2. It compares
the original and repeated single flows using the same disk emulator, original
smooth classifier, observations, shear amplitudes and both components' endpoint
catalogues. All four endpoints and three cuts are retained. It evaluates the
physical inverse-transport log density, including the existing Jacobian, and
then adds the log ratio of selected masses from the frozen joint evaluations.

Because the classifier and its truth/geometry inputs are identical, its
observed log-probability factor cancels in the raw flow-density difference.
It does **not** cancel from the two integrals over generated selected outputs;
each integral uses the classifier-weighted mass of the corresponding flow.
The audit verifies that only the flow changes before using that cancellation.
It reports all six predicted-U arms plus the actual-U oracle; the matched-U
branch is a different conditioning diagnostic and is not substituted here.
The primary result is the fixed original smooth classifier's normalized score.

Density protocol `flow_refinement_repeat_density_protocol.json` pins code,
tests and the joint template. The completed joint receipts must match their
results/comparisons; measured counts and shape sums replay before scoring.
CPU16591247 completes12s, four contract/normalization tests pass7.30s and
stderr is empty. The tests include rejecting classifier/emulator changes
that would invalidate factor cancellation and preserving a selected law
when both raw density and pass mass rescale. Ruff and shell syntax pass.
Full-data density results remain pending under
`flow_refinement_repeat_results/density_*.json`. No density result selects
between flows or changes the original candidate.

### Flow-refinement repeat trained and frozen for validation

GPU16591200 completes13m38s and selects step3, checkpoint SHA256
`33fc71c0636b5f8e030fc152319022736d904e9ec4e20e617d50b583e9fbd5db`.
Full validation NLL is-4.767842449654797 versus initial-4.768710489910869,
cost0.000868040256nat/object within the fixed0.001 budget. Soft/hard primary
probability residuals are[0.0010542616,0.0012273332], improved relative to
the repeat's initial checkpoint under the unchanged gates. Validation guard
loss falls0.4706957998 to0.3243362633. The original flow selected step2;
guard losses across the two different integration streams are not a common
deterministic objective comparison. All fourth-step line-search trials fail,
so no extra steps or tuning are introduced.

The full SBSI suite16591268 also completes67s:589 passed,4 skipped,1 existing
small-sample NumPy warning in57.57s, with no regression/snapshot failures.
Binder16591209 then starts its identity checks before releasing the fixed
four-endpoint response array. The original candidate remains frozen, and the
new checkpoint has no joint-calibration acceptance claim from training alone.

### First flow-repeat diagonal checks and absolute density review

Flow-repeat response tasks16591210_0/_1 complete on all four existing
two-axis scenes. At the fixed primary measured radius>0.6 arcsec and
MAG_AUTO<25.8 cut, predicted smooth-U m is+0.352174757 +/-0.815751151%
for g1 and-0.442323108 +/-0.793093017% for g2 (case-bootstrap SE).
Paired changes from the retained original are+0.005007011 +/-0.002370006pp
and+0.001106905 +/-0.001916953pp. These small repeat shifts are conditional
on the same initial flow checkpoint and emulator; four scenes do not resolve
the underlying calibration to0.3%. All arms/cuts remain in the comparison
artifacts. Main80 and independent40 tasks start with successful first-scene
progress and empty stderr; normalized density remains dependent on completion.

`scripts/report_smooth_density_diagnostics.py` and its CPU launcher review
the original candidate's saved predictive diagnostics at all four endpoints
and both legs, with source hashes and exact retained-arm replay checks.
The primary-cut figure reports mean residuals divided by observed standard
deviations and variance residuals divided by observed variances. These are
effect sizes, not significance tests. All cut moment/covariance summaries,
their existing paired case errors, and physical-support diagnostics are
retained in `absolute_density_report/evidence.json`. Histogram TV remains
explicitly uncut, includes invalid photometry and sampling noise, and has no
new acceptance threshold. Pass fractions are conditional on usable objects,
not absolute detection rates. No new simulation, fit, scene exclusion or
checkpoint selection enters this descriptive review. Cross-components remain
lower priority under the owner's guidance.

CPU16591324 completes25s with empty stderr; all result/replay identities and
histogram normalizations pass, and the exported PNG is visually checked.
Primary mean residuals are below0.6% of observed coordinate SD across all
eight endpoint/leg combinations. Shape variances remain underpredicted:
e1 by1.32–1.73%, e2 by2.41–2.56%. Primary pass-fraction residuals range
+0.0846 to+0.2029pp conditional on usable objects. Uncut 4D coarse TV is
3.54–3.92%, including finite sampling. All uncut generated outside-disk
fractions are about5.7–6.8e-6 and invalid-photometry fractions3.9–5.4e-7;
corresponding observed fractions are zero. Primary undefined-log mass is
zero to roundoff. These retained, nonzero density discrepancies prevent a
density-equivalence claim, despite the properly normalized likelihood gain.

### Combined flow-refinement evidence report

`scripts/report_flow_refinement_repeat.py` collects the completed original,
latent-repeat and classifier-repeat report with all four flow-refinement
response and normalized-density endpoints. CPU16591355 waits for the entire
density array16591245. It verifies source/protocol/receipt identities, fixed
scenes and shear amplitudes, every response branch, and all42 density
arm/cut/leg combinations per endpoint. Tables present diagonal m, paired
changes and both legs' properly normalized predicted-U density differences.
The original absolute-density diagnostics are included. No averaging,
favorable-seed choice or automatic acceptance decision is performed.

Outputs will be `flow_refinement_repeat_report/report.md` and `evidence.json`
in the scene-classifier cache. Ruff and shell syntax pass; runtime validation
remains pending on completion of the scientific jobs. The report explicitly
distinguishes conditional flow-refinement randomness from independent flow
pretraining or emulator retraining, and preserves the development-sample
limitation and lower priority of cross-component investigation.

### Completed flow-refinement response and density repeat

All four response tasks16591210 and density tasks16591245 complete with
zero exit status and empty error logs. Main80 response takes1h35m20s;
independent40 takes49m34s. Density takes44s on each small endpoint,
4m08s on main80 and2m28s on independent40; all workers pass their four
contract/normalization tests. Combined report16591355 completes3m38s and
passes all source, protocol, receipt and branch-completeness checks.

| Endpoint | Original primary m (%) | Flow repeat m (%) | Paired shift (pp) | Paired case SE (pp) |
|---|---:|---:|---:|---:|
| main80, g1, h=0.02 | -0.199979 | -0.198491 | +0.001489 | 0.000818 |
| independent40, g1, h=0.05 | +0.402474 | +0.404368 | +0.001894 | 0.000431 |
| four scenes, g1, h=0.02 | +0.347168 | +0.352175 | +0.005007 | 0.002370 |
| same four scenes, g2, h=0.02 | -0.443430 | -0.442323 | +0.001107 | 0.001917 |

The repeat's case-bootstrap SEs on m are0.263670,0.188783,0.815751 and
0.793093pp, respectively. Properly normalized primary predicted-U selected
NLL changes (repeat minus original, plus/minus leg) are main
+0.00000604/+0.00000787nat/object and independent
+0.00000453/+0.00000960nat/object. Small-endpoint changes are all positive
and below0.000008nat/object. Thus the repeat is slightly worse by this score,
but its change is tiny; the original is retained under the predeclared rule,
not selected on this comparison. No direct comparisons of losses from
different refinement integration streams are substituted for these checks.

The combined report preserves all four variants (original, latent, classifier
and flow-refinement repeat), every declared endpoint/arm/cut, and absolute
density caveats. These results support stability to the tested refinement
randomness. They do not test an independently pretrained flow, emulator
retraining, or untouched acceptance scenes. The original development model
and configured production likelihood remain unchanged.

### Fixed-recipe emulator training repeat

The completed latent, classifier-training and flow-refinement repeats do not
test randomness in the emulator. Its218-tree physical-moment XGBoost recipe
uses row subsampling0.7634540769 and feature subsampling0.9651944092, with
seed20260816. A bounded repeat changes only that seed to20260817. The
original emulator, flow and smooth classifier remain the retained candidate.

BlendEMU owns the new training copy, checker and32-CPU launcher; SBSI owns
the joint/density validation. A nine-pin training protocol and exact
one-replacement source check preserve all other settings and all existing
161,012,399 prepared pairs. Fit cases40–199, development cases0–39, the
physical-moment objective, scaling, intercept and218 trees are unchanged.
Neither development nor joint bias selects trees or a seed. CPU16592759
passes frozen-source checks and begins data preparation; no images are made.

Binder16592811 checks completed model/metadata identities and equality of
every recipe field except the seed and the corresponding training outputs.
Its127-pin template retains the original flow and all classifiers. GPU
array16592825, throttle2, evaluates fresh_g1/fresh_g2/main/independent using
the same scenes, draws, latent seed7301, shear steps and fixed measured
radius>0.6 arcsec primary cut. Before paired comparison, input identities
must agree and all model selected denominators must replay exactly because
this shape transport preserves radius and flux.

Density array16592853 follows the full response array, throttle2. It uses
each emulator's own velocity and inverse-map Jacobian at the same observed
outputs, with the same flow density. Fixed classifier terms and exactly
equal selection masses cancel; the normalized score is still computed with
the existing tested formula. All four endpoints, three cuts, six predicted-U
arms and actual-U oracle are retained. No nonfinite output is dropped.

CPU16592812 passes the recipe contract test in13.25s; CPU16592854 runs the
new fixed-flow/selected-mass contract and existing normalization tests:
4 pass6.89s with empty stderr.
Full-data results remain pending under `emulator_repeat_results/` in the
scene-classifier cache. This repeat tests emulator fitting randomness
conditional on the same response labels; it does not create fresh acceptance
scenes or repeat full flow pretraining.

### Complete emulator-repeat evidence report

CPU16592966 runs `scripts/report_emulator_training_repeat.py` after successful
completion of every emulator-density task16592853. It preserves the original,
latent, classifier-training and flow-refinement comparisons and appends the
emulator-training repeat. Primary tables report all five variants and paired
shifts without pooling or choosing a preferred seed. Every response branch
and every density arm/cut/leg remains in JSON, along with the original absolute
density diagnostics and the previous flow-repeat scores. Model/protocol,
response/density receipt, scene/shear and full branch identities are verified.

The expected output directory is `emulator_training_repeat_report/` in the
scene-classifier cache. Ruff and shell syntax pass; runtime remains pending.
This is a one-factor-at-a-time robustness assessment. It does not test joint
interactions among changed training seeds, independent full flow pretraining,
or newly untouched scenes, and does not automatically promote a model.

### Emulator repeat trained and frozen

CPU16592759 completes14m44s with218 trees and model SHA256
`1a20393ef749e90d483ee0f7743b14e7e2611551c5a68a79fc45e3f9bd531baf`.
The fit uses all128,809,729 training rows and the unchanged intercept
0.00811731707072. Final training physical-moment RMSE is0.997070 versus
the original0.997075. Neither that difference nor downstream calibration
selects an emulator: the original remains fixed.

Binder16592811 completes59s, including1 passing recipe test in19.84s;
all fixed metadata/parameter fields and prescribed seed/hash identities pass.
Both training and binding error logs are empty. The resulting129-pin
`emulator_repeat_joint_protocol.json` releases the two small-axis tasks
16592825_0/_1 first, then the large endpoints under the same two-GPU cap.
No calibration or density claim follows from successful training alone.

### Completed emulator-training repeat

The fixed218-tree emulator repeat (training seed20260817 versus20260816)
completes all joint-response and normalized-density endpoints. CPU report
16592966 verifies the response and density receipts and writes
`scene_classifier_geometry_20260918_v1/emulator_training_repeat_report`.
The original model remains retained independently of the repeat's outcomes.

| Endpoint | Original m (%) | Repeated-emulator m (%) | Paired change (pp) |
|---|---:|---:|---:|
| Main80, g1, h=0.02 | -0.199979 +/-0.263649 | -0.166180 +/-0.263913 | +0.033799 +/-0.001339 |
| Existing40, g1, h=0.05 | +0.402474 +/-0.188664 | +0.437508 +/-0.188822 | +0.035034 +/-0.001790 |
| Four scenes, g1, h=0.02 | +0.347168 +/-0.817220 | +0.378163 +/-0.816583 | +0.030995 +/-0.003387 |
| Same four, g2, h=0.02 | -0.443430 +/-0.794924 | -0.412592 +/-0.792283 | +0.030838 +/-0.005118 |

Errors are case-bootstrap SE, not uncertainty over the distribution of
training seeds. The measured primary cut remains radius>0.6 arcsec and
MAG_AUTO<25.8, generated secondary true-r<26, with predicted detection/usability.
The normalized selected-density NLL changes (repeat minus original, nat/object)
are main+0.00000790/+0.00000276, existing40+0.00005594/+0.00000907,
four-scene g1+0.00001438/-0.00001254 and g2-0.00000621/+0.00001030
for plus/minus legs. All arms, cuts, paired errors and provenance are retained
in the report's `evidence.json`. These results add tested seed stability;
they do not establish a training-seed distribution or independent calibration.

### Prospective independent two-axis validation

The new dataset at
`blendemu_runs/joint_independent_validation_20260919_v1` freezes the original
candidate hashes and40 scene/noise seeds before generation. Cases22000–22039
have g0 and both axes at +/-0.05. Cases22000–22003 additionally have both axes
at +/-0.02. All40 scenes and both primary axes must be reported; auxiliary
amplitudes cannot replace the primary result. No model, cut, checkpoint or
sample size is selected from the outcomes. Cross-components remain secondary.

This fills two concrete gaps: the compatible existing samples have already
informed development, and the current g2 evidence has only four scenes.
Existing simulations continue to support the training and latent repeats.
The inventory is bounded to the documented compatible archives, not a claim
that every simulation family has been exhausted. The approximate0.3% target
does not introduce a requirement that every confidence interval fit within
+/-0.3%.

CPU generation array16593294 uses8 workers per scene and throttle8. It retains
the proven sims1 reference-prior replay and py31 measurement convention.
The preparation script checks the exact predeclared legs/seeds and completed
receipts, retains the generated secondary true-r<26 common rendered parent,
and reports unmatched identities. It prepares truth-only neighbour pairs and
geometry without using observed-g0 selection. The binder then freezes prepared
products for the unchanged evaluator. The two primary endpoints use all40
scenes; two auxiliary endpoints use the fixed first four. Every endpoint uses
64 antithetic draws, seed7301, predicted detection/usability and the measured
radius>0.6 arcsec/MAG_AUTO<25.8 primary selection independently per leg.

Artifacts live at `sbsi_caches/independent_joint_validation_20260919_v1`:
`preparation_protocol.json`, `joint_template.json`, eventual `joint_protocol.json`,
`results/`, and `report/`. Five targeted tests pass in job16594969; Ruff and
shell checks pass. Full scene preparation and numerical validation remain
pending. No calibration claim or production configuration change follows
from generation, binding, tests or a completed report alone.

The downstream chain is now queued: preparation16594999, binding16595000,
GPU response16595003 (array0–3, throttle2), then report16595004. Preflight
16594998 verifies all79 pins and compatible historical g1/g2 result schemas;
this smoke contributes no independent calibration data. The previous emulator
GPU chain and report are complete. A two-hour one-shot snapshot16595007 records
all stages; it does not wake the Codex task. Job handles and dependency details
are saved in the new cache's `jobs.json`. Each downstream stage still requires
startup and outcome review once generation completes.
