# V3.6-like domain-transfer investigation

For the newer unbounded/close-crowding/guard models, use the
[completed candidate comparison](V36_MODEL_COMPARISON.md#completed-comparison-of-saved-predictions).

**Scope correction:** this audit uses the September20 pinned V3.6-like model
set. September22–24 unbounded and close-crowding retrains were subsequently
found in a separate worktree. They were not evaluated here; the claim that all
three components use true-r<26 training applies only to the pinned set. See
[newer model identities](V36_RECOVERY_PLAN.md#model-identity-correction--newer-work-found)
before applying the recovery recommendations to current experiments.

## Decision and scope — 2026-09-24

The retained approximately 0.3% development result does **not** establish
stable calibration under changed selection. A new reanalysis of the saved
sufficient statistics exposes opposing response errors in two measured-radius
regions. Recovering a small aggregate bias alone would therefore be an
inadequate success criterion. No recovered model or uncut inference result has
yet been established.

The owner explicitly fixed the goal: approximately 0.3% on constant-shear
simulations, after measured radius >0.60 arcsec and MAG_AUTO <25.8, with no
truth-property cuts. All rendered parents enter the forward calculation;
usability and measured selection are independent per leg. Measured-cut transfer
below is a diagnostic of cancellation, not a proposal to change the goal cuts.
The cancelled independent-validation chain remains stopped. The follow-up
[recovery plan](V36_RECOVERY_PLAN.md#priority-and-evidence) records the corrected
training-coverage audit and an ordered experiment design.

## Controlled measured-radius transfer

Reanalysed the four frozen `smooth_joint_*.json` endpoints, verifying each
against its retained comparison receipt. No new model draws or checkpoint
selection enter these numbers. Parent: rendered secondary-role true-r<26.
Each shear leg uses its own usability, `MAG_AUTO<25.8`, and measured radius
cut. Shapes are ngmix reduced-shear ellipticities. Response is the coherent
antithetic difference of separately normalized selected means. Bias is
`m=100*(R_measured/R_model-1)` percent; see
[response extraction](CONVENTIONS.md#6-response-estimators) and
[bias definition](CONVENTIONS.md#7-multiplicative-bias).

| Development endpoint | Radius >0.60 arcsec m (%) | Radius >=0.75 arcsec m (%) | 0.60<radius<0.75 arcsec m (%) |
| --- | ---: | ---: | ---: |
| Main80, cases40–119, g1, h=.02 | −0.200 | +1.319 | −6.466 ±1.307 |
| Existing40, g1, h=.05 | +0.402 | +1.468 | −3.895 ±0.974 |
| Four-scene g1, h=.02 | +0.347 | +2.241 | −7.152 ±3.665 |
| Same four scenes, g2, h=.02 | −0.443 | +0.438 | −3.390 ±2.205 |

Errors shown for the newly derived shell are case-bootstrap standard errors
in percentage points, 10,000 resamples, seed20260924. Original flow integration
uses64 draws/latent seed7301. These errors exclude training and finite-draw
uncertainty. Four-scene bootstrap precision is limited. All endpoints are
retained development evidence, including the historically named “independent”
40-scene endpoint.

The independently frozen latent-seed7302 reanalysis retains the cancellation:
main80 shell−6.559798%, high-radius+1.333763%, union−0.209397%; existing40
shell−3.938095%, high-radius+1.466081%, union+0.391591%. Thus the observed
pattern persists across both existing latent streams. Two streams do not
establish complete integration convergence. CPU16689063 completed the repeat
after the final suite passed315 tests,2 skipped in29.70s; results are in
`cut_transfer_seed7302.json` beside the primary report.

The **paired** radius-threshold shifts are +1.518969 ±0.247439pp on main80
(95% interval[1.037837,2.011021]) and +1.065663 ±0.187459pp on existing40
([0.700725,1.433635]). Pairing resamples the same cases in both cuts; adding
the two headline error bars in quadrature would discard that information.

Replacing predicted usability by actual per-leg usability leaves shell biases
−6.361% and−3.806%, respectively. The corresponding paired threshold shifts
are+1.529249pp and+1.085240pp. Thus this cut dependence cannot reasonably be
assigned principally to the usable-event classifier. This control does not
identify the separate conditional-flow and disk-emulator errors.

The historical matched-image probe provides a complementary warning: on
scene20011, the common-descent flow had an own-baseline-normalized physical
self-response error about+0.0197 and a neighbour-response error about−0.0082
along g1, persisting across seeds7301/7302. This is one scene/two sign probes,
not a population estimate and not the new radius-shell decomposition. It
reinforces the need to validate both components before changing the flow alone;
see [matched-image latent repeat](../archive/research-2026-09-20/doc/JOINT_CALIBRATION.md#matched-image-latent-repeat-completed).

### Exact finite-step cancellation

For shell fraction f on each leg and shell/high-region selected means μs/μh,

```text
R_total = mean(f) R_shell + (1-mean(f)) R_high
          + [(f_plus-f_minus)/(2h)] [mean(mu_shell)-mean(mu_high)].
```

The last term retains shear-dependent selection migration. This identity is
exact for the finite antithetic difference, not an approximation that assumes
fixed population fractions. It is checked numerically in the reducer.

Main80 g1 contributions are:

| Contribution to R11 at radius>0.60 | Measured | Model | Measured−model |
| --- | ---: | ---: | ---: |
| Weighted 0.60–0.75 shell | 0.115170979 | 0.122612679 | −0.007441699 |
| Weighted >=0.75 population | 0.569854876 | 0.563783092 | +0.006071784 |
| Selection migration | −0.000000452 | +0.000002287 | −0.000002739 |
| Total | 0.685025402 | 0.686398057 | −0.001372655 |

The shell is36.19% of measured selected objects versus36.04% of predicted
selected mass. The small total error hides appreciably larger opposing terms.
The near equality of selected fractions does not imply accurate conditional
response. This is direct evidence of cancellation under this **measured-radius
partition**, not a claim to have separated truth-population and measurement
residual components.

Machine-readable result:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/cut_transfer.json`.
Implementation: `scripts/audit_v36_cut_transfer.py`. CPU job16688906 completed;
the identity test passed. An earlier test failure was a3.5e-16 zero-comparison
roundoff tolerance, repaired without changing scientific calculations.

## Expansion of the truth parent

New CPU audit16688812 scanned both coherent ±.02 g1 legs in all80 cases40–119.
It verified unique identities, equal rendered truth membership and intrinsic
properties between legs, and reproduced every retained measured numerator and
denominator against the frozen endpoint. Unmatched detections/measurements are
reported per case; undetected truth objects remain in the parent with U=0.
The measured selection stays strict radius>0.60arcsec and MAG_AUTO<25.8.

| Diagnostic parent | Parent rows | Selected plus-leg rows | Measured R11 |
| --- | ---: | ---: | ---: |
| Retained secondary-role true-r<26 | 8,876,218 | 6,473,204 | 0.685025402 |
| All rendered true-r<26 | 17,755,228 | 12,946,864 | 0.687183145 |
| All rendered, no truth cut | 55,965,430 | 14,530,239 | 0.672734807 |
| Added bright objects alone | 8,879,010 | 6,473,660 | 0.689341325 |
| True-r>=26 alone | 38,210,202 | 1,583,375 | 0.554582307 |

These are measured responses, **not expanded-population biases**. A matched
model prediction for each parent is needed. Correction from the retained
training protocols: all three retained components restrict target training to
true r<26, including the smooth 17-input classifier. The earlier statement
that this classifier had broader magnitude coverage was incorrect. Its
protocol/completion hashes identify the retained checkpoint; the flow
refinement pins the narrow domain, and the pinned emulator metadata declares
the same magnitude restriction. See the
[training coverage evidence](V36_RECOVERY_PLAN.md#training-coverage-and-labels).

All80 measured replays passed. Full suite at submission:313 passed,2 skipped
in43.08s. Summary CPU job16688980 completed. Reports, original source hashes,
and join accounting are under `v36_domain_recovery_20260924_v1/observed_16688812/`
and `observed_summary.json` in the cache root above.

### Paired forward pilot and numerical support

The new `scripts/audit_v36_domain_response.py` compares the retained, all-bright,
and all-rendered parents with the pinned single flow/classifier/emulator and
actual-U diagnostic arm. Cases40/41 are an operational development pilot,
not enough cases to demonstrate0.3% calibration. It reconstructs context and
response pairs from the same full rendered scene, checks against frozen retained
contexts, uses FP64 neural arithmetic and64 antithetic latent CRN across ±.02,
and applies the same per-leg measured cuts to data and model. Finite-draw
convergence is not established by this pilot.

Initial job16688813 and diagnostic retry16688904 stopped in the first block.
The saved retry identifies14 auxiliary-underflow draws among65536 draws on
three faint objects (input IDs522,848,923; true r28.516,27.454,26.414).
There are no nonfinite values and none passes the measured selection.
This is numerical/extrapolation evidence, not itself a contribution to selected
response bias. Failed-run receipts and the14 actual outputs are retained.

The revised evaluator integrates only the selected contribution. An auxiliary
failure that is **provably excluded by a finite failing measured bound**
contributes zero, with the original total draw count in the denominator.
It is neither repaired nor removed and renormalized away. Shape transport
changes no radius/flux, so evaluating it only for selected draws is exact for
these moments. Any invalid draw not provably excluded aborts evaluation.
Boundary-rounded shape draws are counted and retained; excursions beyond
1e-12 abort. A regression test checks preservation of the original draw mass.
This treatment does not validate the model's full conditional density.

Array16688944_0 was stopped after its first progress block to include the
explicit preflight context-replay check and preserve one final source version
for both workers; its partial results are not used. Final pilot jobs are
16688994_0 (case40) and16688944_1 (case41), at most two GPUs total. Outputs:
`forward_16688994_0/` and `forward_16688944_1/` under the same cache root.
Scientific settings and artifacts are unchanged. See the latest worklog for
startup/completion status.

### Completed uncut pilot

Both workers and reducer16689064 completed successfully (case40:14m40s;
case41:15m07s; reducer12s). They cover1,399,135 rendered parents and use the
same frozen source hash`5b780adf8a38…`, saved with each output. All measured
replays pass, as do the preflight and subsequent retained-context checks;
maximum context difference is below2.37e-7. No GPU job remains active.

| Two-case diagnostic population | m (%) with predicted U | m (%) with actual U |
| --- | ---: | ---: |
| Retained true-r<26 parent | +0.931886 | +1.217116 |
| All rendered true-r<26 | −0.116593 | +0.059484 |
| All rendered, no truth cut | +1.263967 | +0.990386 |
| Added bright objects alone | −1.177167 | −1.108140 |
| True-r>=26 alone | +7.489339 | +7.496259 |

These are pooled selected-mean **point estimates on two development scenes**,
not replacements for the80-scene retained result. Case40/41 uncut predicted-U
m are+0.638900/+1.894072%; their faint-only values are+9.050714/+5.852679%.
The stored two-case bootstrap is descriptive and cannot establish reliable
population confidence intervals or0.3% calibration. In particular, the
retained parent in these same two cases is itself+0.93%, so comparing the
pilot's+1.26% directly against the80-scene−0.20% would confound population
change with scene sampling. The matched retained→uncut shift is+0.332081pp;
the all-bright→uncut shift is+1.380561pp.

The **selected faint count** is overpredicted by28.61% on the plus leg
(28.58% minus), while all-bright selected counts agree to about0.1%.
Using actual U reduces the faint selected-count excess to12.10%/12.14%, but
leaves the faint response bias about+7.50%. Thus the faint flow's selected
probability and conditional flow-plus-emulator response require attention;
classifier improvement alone cannot account for these pilot residuals.
The total uncut selected-count excess is3.05% with predicted U and1.44–1.46%
with actual U. This is a direct same-scene forward comparison, independent of
the24m prior and likelihood sampler.

Across179,089,280 draws,474,893 auxiliary failures are provably unselected
and remain in the denominator. Some occur at true-r<26 as well. There is no
non-excluded auxiliary failure or shape excursion beyond1e-12. Full boundary-
rounding counts are preserved in the case reports; none is clipped or used to
renormalize a generated population. This establishes completion of the
selected-moment diagnostic, not full-density or latent-convergence acceptance.

Machine-readable pooled and per-case provenance:
`v36_domain_recovery_20260924_v1/forward_summary.json` under the cache root.
The pilot supplies a concrete expanded-domain failure to address. It does
not establish that the proposed training change will recover0.3%.

## Relationship to the failed likelihood run

The cut-transfer result above uses direct forward moments. It is independent
of the24m-atom integration, proposal, and numerical derivative failures in
[the inference review](V36_INFERENCE_REVIEW.md#likelihood-focused-diagnostics--2026-09-21).
Fixing a sampler alone cannot remove this measured-radius dependence. Conversely,
a repaired forward response would not establish accurate full-density shear
inference. Bright-tail atom-bank sensitivity, selection-normalizer convergence,
finite-step derivatives and score-solver convergence retain separate tests.

## Recovery experiment justified by the evidence

### An unfinished objective correction in the historical logs

The earlier common-descent refinement already used multiple response guards;
merely adding more thresholds is not a new cure. Its guard response used
measured-reference normalization. The completed normalization audit found that
most apparent primary-response improvement came from the normalization gap:
the selected flow's own-normalized soft self-response remained
[0.52899557,0.53052232], against measured[0.51920047,0.52039433]. The second
diagonal slightly worsened. See
[the historical audit](../archive/research-2026-09-20/doc/JOINT_CALIBRATION.md#completed-common-descent-joint-and-normalization-evidence).

The concrete objective correction is to differentiate the model's **own**
selected mean, including its selected mass D and shape numerator A. For
selection slope B and shape-numerator slope H, the response is
`R = [H - (A/D) outer B] / D`. Differentiation must include A and D, and a
separate loss must constrain the selected probability. Multiplying every
selection weight by a common factor must leave this response and its gradient
unchanged; this is a useful implementation check, not a fitted scaling.

A read-only historical full audit found a stable common response/probability
descent direction (independent-half cosine0.9994166) but made **zero updates**.
Its flow hash is `fade8aebfb48…`, the corrected predecessor, not retained
V3.6-like `5434cfafb94e…`. Response-only descent worsened selected probability
(gradient cosine about−0.5685). Consequently this is evidence that the
corrected objective is feasible to investigate, not a reusable validated
update direction for today's retained model or the expanded population.
Repeat integration and gradient checks at the current candidate and intended
population before training. Source records:
`flow_shape_augmentation_20260918_v1/own_normalized_guard_full/{protocol,result,reference}.json`
under the SBSI cache root.

### Proposed candidate and acceptance

The two-scene pilot is complete. Its exact bright/faint error allocation and
the retained training-support audit prioritize full-domain data coverage for
all components, followed by a controlled test of the corrected own-normalized
response objective. The concrete stages, numerical checks and independent
acceptance design are in the
[September 24 recovery plan](V36_RECOVERY_PLAN.md#ordered-experiments).
Conditional diagnostic bins need not each achieve 0.3%; their weighted
contributions must explain the aggregate and expose reliance on cancellation.
The final measured cuts remain fixed. No empirical correction, model promotion,
new training, or recovered 0.3% result is claimed.
