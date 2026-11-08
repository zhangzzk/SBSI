# Selection-response replication on ConstGold cases90–139

## Fixed comparison and estimators — 2026-09-12

The owner requested the same response-bias tests on another50 cases.
Cases90–139 are disjoint from the preceding cases40–89. Case40 is rerun
as a reproduction control and excluded from the new population results.
The experiment is external:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1/selection_replication_c90_139_v1`.

The recipe inherits
[the selection audit](CONSTGOLD_SELECTION_LOCALIZATION.md#question-and-fixed-diagnostic-recipe--2026-09-12)
and [explicit mocks](CONSTGOLD_GENERATIVE_MOCK_CLOSURE.md#completed-comparison--2026-09-12):

- Circularized grid-free shape+radius lambda10 flow, selected epoch154;
  ordinary-MSE trial9 blend emulator; circularized-BCE detector.
- Fixed truth support18<r<25.8 and0.37<semi-major Re<1.5arcsec, with complete
  neighbours. Actual unmatched and unusable measurements are counted.
- Central g1+/-0.02, denominator0.04. Pool per-leg moments before ratios.
  Report m=R_actual,11/R_model,11-1, the response-closure definition in
  [Conventions §8](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention). These are not SBSI inference estimates.
- B: fixed both-usable detections; U: observed independent usable detections;
  p: the frozen classifier. Keep these controls distinct.
- All8 combinations of |e|<0.6, MAG_AUTO<25.8, radius>=3.75pixels;
  all15 previously fixed threshold points. No threshold fitting.
- Exact measured=true+(measured-true) selected-response decomposition,
  cut interactions, drop-one-cut effects, and the existing subgroup controls.
- Sixteen independent64-draw QMC groups, seeds7301–7316, common draws across
  legs and cuts. Repeat the64 explicit mock catalogues using Bernoulli
  detections and the same original seed20260912 convention.

Case uncertainties retain pairing between cuts and controls within a block.
The new-minus-original comparison uses independent case bootstrap draws for
the two disjoint blocks, seed2026091201, and shared latent-group deletion
for integration contrasts. Case40 and case90 are not paired observations.
Case and integration uncertainty are reported separately. ConstGold results
do not select a checkpoint, fitted correction, loss weight, or threshold.

## Preparation repair and startup checks — 2026-09-12

All306 required raw catalogue files are available, as is the crowding cache.
Setup16462494 completed and wrote hashed per-case crowding shards. Initial
preparation16462495/16462496 failed before any completed case receipt because
the frozen flow environment does not install the optional BlendEMU package.
Pending dependents16462497–16462499 were cancelled. The failed workers,
manifests, submission, partial directories and accounting are preserved in
`failed_attempt_1/`; their original logs remain in `logs/`.

The repair separates blend-lookup generation into the original
`anchorblend_current_re037_v1` environment and its frozen BlendEMU snapshot.
It uses the same explicit trial9 model/metadata and load arguments. Its
source is hashed. Flow preparation and evaluation retain the frozen flow
environment. No shared environment or model was modified.

Replacement jobs: control preparation16462598; new-case preparation array
16462599; GPU control16462600; GPU case array16462601; review16462602.
Preparation of the new cases waits for control preparation. GPU evaluation
waits for its corresponding preparation and the successful control.
Case40 must reproduce its exact blend lookup, observed masks/shapes and
flow conditions, detector probabilities, and all23 cuts/three parents/four
response components in all16 groups. The latter comparisons allow only
the declared floating-point tolerance, not an empirical adjustment.

The control completed successfully. Its exact lookup/context checks and
detector probability checks pass; all16 groups reproduce. Independent
comparison of4416 statistic arrays finds maximum absolute difference
2.73e-12 (`independent_control_check.json`). The new-case GPU array has
started. Population-level replication results remain pending.

Pre-submission validation: AST parsing of all Python workers, `bash -n`
on six launchers, `CUT_FACTORIAL_SELFTEST_PASSED`, `MOCK_SELFTEST_PASSED`,
and import verification of the original BlendEMU environment. Production
source was not edited. Successful scientific completion still requires
case receipts, independent review and comparison outputs.

`monitoring.json` records a fresh one-shot check due at10:29UTC, approximately
two hours after startup, under the AGENTS.md monitoring rule. It supersedes
the submission-time `scheduled_wakeups=false` field without editing the
frozen manifest. The one-shot runner removes only its own tagged cron entry
after queuing the check. Earlier completed watches remain disabled.

## Outputs and interpretation

`submission.json` records current jobs and the failed attempt. `manifest.json`
and `workers.json` freeze inputs, recipes and workers. Per-case preparation,
16-group moments, detector probabilities and explicit mock outputs retain
hashes and IDs. Final reduction is intended to write `results.json`,
`final_review.json`, `review_sufficient.npz`, `mock/results.json`,
`replication_comparison.json`, and `completed.json`.

The principal comparison is m for no measured cuts, magnitude+radius only,
and all cuts, under B/U/p. For reference, original full-detector QMC m was
-1.12894%, -0.93410%, and -2.06772%, respectively. The replication tests
whether those discrepancies and their true/residual components generalize
to different cases. Agreement would support reproducibility, not identify
a causal mechanism or prove calibrated SBSI inference.

## Completed replication — 2026-09-12

All replacement stages completed with exit0 and empty stderr. The50 GPU
case tasks completed all16 groups and64 explicit mocks each; review16462602
passed source/output hashes, retained mock-row reconstruction, matching,
cut algebra, and exact response identities. A separate light
`completion_review.py/json` verifies all800 group-file hashes,150 case
receipts, frozen source/model/result hashes, and independently reconstructs
responses and integration errors from compact sums. No new draws or fitting.

New population: **4,855,279 truth objects; 4,338,226 matched usable pairs**.
The previous block had4,854,422 and4,338,426, respectively. Case40 is excluded
from these counts. All numbers below are percent m=R_actual,11/R_model,11-1,
with one standard error from case bootstrap; the same frozen model set is
used in both blocks.

| Detection treatment | Measured cuts | Cases40–89 | Cases90–139 |
|---|---|---:|---:|
| B: matched pairs | None | +0.234 ±0.222 | +0.402 ±0.201 |
| B: matched pairs | Magnitude + radius | +0.248 ±0.175 | +0.750 ±0.159 |
| B: matched pairs | All three | -0.724 ±0.293 | -0.428 ±0.294 |
| U: actual independent detections | None | -0.538 ±0.227 | -0.305 ±0.210 |
| U: actual independent detections | Magnitude + radius | -0.419 ±0.176 | +0.160 ±0.167 |
| U: actual independent detections | All three | -1.361 ±0.305 | -1.013 ±0.296 |
| p: model detections | None | -1.129 ±0.224 | -0.959 ±0.208 |
| p: model detections | Magnitude + radius | -0.934 ±0.176 | -0.429 ±0.170 |
| p: model detections | All three | -2.068 ±0.304 | -1.811 ±0.296 |

B fixes the both-usable parent, but measured cuts are still applied
independently to the data and model measurements. Only its uncut entry is
the matched shape-only control; its cut entries also include selection.
For p, QMC integration SE on m is0.00041/0.00200/0.00587 percentage points
for none/no-e/all, much smaller than case uncertainty. Do not add these
as independent components to case uncertainty.

Full-detector new R_actual/R_model is0.95213837/0.96136223 without cuts,
1.01031291/1.01466393 with magnitude+radius, and0.60738925/0.61859170
with all cuts. New-minus-old m changes are+0.16949 ±0.30207,
+0.50528 ±0.24620, and+0.25676 ±0.42837 percentage points, respectively,
using independent case-block resampling. Shared-group integration SE of
these changes is0.00052/0.00270/0.00904pp. The no-e change is about2.1 SE;
the other two are below1 SE. Without e cut, actual R changes by+0.005714,
versus only+0.000593 for the prediction. This is a fixed-model replication,
not evidence that any model was improved between blocks.

### Decomposition and cut dependence

For full model detections, new model-minus-data R11 errors are:

| Cuts | Selected true ellipticity | Selected measured minus true | Total |
|---|---:|---:|---:|
| None | +0.000285 ±0.000301 | +0.008939 ±0.001979 | +0.009224 ±0.001997 |
| Magnitude + radius | +0.000415 ±0.000706 | +0.003936 ±0.001670 | +0.004351 ±0.001727 |
| All three | -0.015914 ±0.001305 | +0.027117 ±0.001783 | +0.011202 ±0.001832 |

The exact identity R_selected_measured=R_selected_true+
R_selected_(measured-true) holds with the same denominators, including every
resample. The all-cut cancellation of large true/residual errors repeats.
Without e cut, the true-e error remains consistent with zero and the
measured-residual discrepancy is smaller than in the original block.
It remains appreciable with no measured cuts, so e selection alone does
not explain the full detector-composed bias.

The full-detector eight-cut m values are: none-0.959±0.208%,
shape-1.808±0.352%, magnitude-0.890±0.209%, shape+magnitude-1.442±0.355%,
radius-0.413±0.167%, shape+radius-1.852±0.289%, magnitude+radius
-0.429±0.170%, and all-1.811±0.296%.
With other cuts present, adding the shape/magnitude/radius cut changes the
R11 error by+0.006851±0.002030, -0.000253±0.000241,
and+0.002896±0.001414. Shape-radius interaction with magnitude fixed is
+0.007156±0.001528. This supports a joint-cut effect, not unique causal
attribution to a single observable. Shapley error allocations for
shape/magnitude/radius are+0.003835/-0.000610/-0.001246, respectively;
these are accounting conventions.

With other cuts fixed, shape thresholds0.50/0.55/0.60/0.65/0.70 give
m=-2.062/-0.988/-1.811/-1.319/-1.004%, with case SE
0.422/0.329/0.296/0.318/0.231pp. Magnitude thresholds25.3–26.3 give
-1.566 to-1.859%; radius thresholds0.6–0.9arcsec give-1.591 to-1.020%,
with nonmonotone intermediate values. Full paired curves and all15
preset thresholds are saved; no threshold is chosen from these results.
Subgroups remain fixed-B, all-cut diagnostics with unequal counts and
within-subgroup denominators, not an additive explanation of the p result.

### Explicit mocks, finite tails and limitations

The64 explicit mocks give m=-0.95357±0.20792%, -0.42558±0.16974%,
and-1.81532±0.29634%, with separate mock integration SE
0.00415/0.00789/0.02847pp. Mock/reference differences in m are
-0.00594/-0.00325/+0.00444pp, consistent with their integration uncertainty.
Bernoulli-minus-probability-weighted same-draw R11 differences are
-3.55e-5/-1.18e-5/-1.85e-5, each within about1.2 MC SE.
Explicit detection sampling therefore reproduces the numerical prediction;
it does not remove the image/model discrepancy.

Without e cut, actual-minus-mock midpoint offsets are
(+0.000851±0.000152, -0.001457±0.000180), similar in sign and scale to the
original block. The c1 proxy is+0.000839±0.000150; no g=0 image leg exists
to establish an independent zero-shear additive bias.

All9,943,611,392 QMC draws were finite, retaining5433 zero radii and26 zero
fluxes under the existing semantics. The621,475,712 explicit mock draws
contained359 zero radii and1 zero flux. Large positive tails remain
(QMC maximum radius9119pixels and flux2.30e7); finite outputs do not prove
tail calibration. Matching review reports0 cross-match rows without shapes,
320,889 shape rows without cross-match, and22,708,354 matched measurement
rows outside fixed truth support, summed across both legs; these are
reported exclusions, not added to the science population.

The replication supports a persistent all-cut bias and a shape-cut
contribution, while the no-e residual varies between case blocks. It does
not establish a unique flux, radius, detection, or conditioning mechanism,
and no result here is actual SBSI posterior-inference m. All models and
thresholds were frozen. On final review, the one-shot check was disabled
and only its own tagged cron entry removed; earlier watches are unchanged.
