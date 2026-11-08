# Calibration after paired-seed shape measurement

## Completed result

All 40 scenes, 640 additional chunks (4,411,805 attempted leg fits), and the
report completed successfully. Reuse the original 5,412,739 common IDs and
add 1,248,893 IDs eligible in at least one exposure. Each exposure applies its
own radius/flux cut and new shape validity; no common-four conditioning or
truth-property cuts. This is the selected calibration estimator, unlike the
earlier fixed-common-galaxy diagnostic.

At radius >3 pixels and magnitude <25:

| Quantity | Original measurement | Paired-seed measurement |
|---|---:|---:|
| m1 (%) | +0.8081 ±0.3102 | +0.5558 ±0.1314 |
| m2 (%) | −0.3335 ±0.2478 | +0.2101 ±0.1690 |
| Measured R11−R22 | +0.009653 ±0.003571 | +0.002487 ±0.001615 |
| New m1−m2 (percentage points) | — | +0.3457 ±0.1796 |

The paired change in measured diagonal gap is −0.007166 ±0.003504.
The new measured off-diagonal gap is +0.000563 ±0.001728. Thus the original
large directional mismatch has weakened, but the model residual is not removed.
The new m1−m2 difference is about 1.93 standard errors from zero; its paired
bootstrap 95% interval is [−0.00183, +0.69736] percentage points. Do not infer
exact equality or force symmetry.

## Response matrices and definitions

Rows are shape components and columns are shear components. Pool scene sums,
normalize each ±0.02 exposure independently, and difference over 0.04.
At the central cut, the unchanged model with predicted usability and generated
photometry gives

```
model:       [[ 0.899866, -0.002308],
              [ 0.000755,  0.900488]]
remeasured:  [[ 0.904867,  0.000324],
              [-0.000239,  0.902380]]
measurement SE:
             [[ 0.001185,  0.001344],
              [ 0.001079,  0.001584]]
```

m_i =100*(Rmeasured_ii/Rmodel_ii−1). Errors are whole-scene delete-one jackknife
standard errors over the same 40 scenes. The 10,000 paired bootstrap resamples
retain correlations between models, measurements, axes and cuts.
[Response convention](../../CONVENTIONS.md#6-response-estimators),
[bias convention](../../CONVENTIONS.md#7-multiplicative-bias),
[population/seed protocol](rule.md#objective-and-fixed-estimator).

## All nine target cuts

All entries are percent; uncertainty is in percentage points. These nine cuts
are highly correlated and are not independent replications.

| Radius lower bound (pixels) | Magnitude upper bound | New m1 (%) | New m2 (%) |
|---:|---:|---:|---:|
| 2.8 | 24.9 | +0.554 ±0.155 | +0.303 ±0.179 |
| 2.8 | 25.0 | +0.536 ±0.154 | +0.328 ±0.184 |
| 2.8 | 25.1 | +0.695 ±0.155 | +0.382 ±0.183 |
| 3.0 | 24.9 | +0.559 ±0.145 | +0.228 ±0.166 |
| 3.0 | 25.0 | +0.556 ±0.131 | +0.210 ±0.169 |
| 3.0 | 25.1 | +0.725 ±0.138 | +0.288 ±0.173 |
| 3.2 | 24.9 | +0.929 ±0.191 | +0.664 ±0.188 |
| 3.2 | 25.0 | +0.926 ±0.180 | +0.728 ±0.183 |
| 3.2 | 25.1 | +1.082 ±0.191 | +0.798 ±0.183 |

The joint RMS over both components and all nine cuts changes from 0.6052% to
0.6346%; paired change +0.0293 ±0.2159 percentage points. Better measurement
precision has not demonstrated better overall model calibration. Both components
show larger positive residuals at radius >3.2 pixels; paired cut contrasts are
reported in `frozen_comparison.json` and summarized below.

## Frozen training comparison and radius dependence

Re-evaluate both frozen model endpoints against the new measurements. The
probability-loss endpoint now has smaller absolute bias at all nine cuts for
both components, compared with its direct control (same shape weights).

| Quantity | Control | Probability-loss endpoint |
|---|---:|---:|
| Central m1 (%) | +0.67065 ±0.13188 | +0.55576 ±0.13143 |
| Central m2 (%) | +0.32372 ±0.16912 | +0.21011 ±0.16898 |
| Joint nine-cut RMS (%) | 0.75735 ±0.11238 | 0.63457 ±0.11008 |

Paired central shifts are −0.11488 ±0.00332 and −0.11361 ±0.00317 percentage
points. RMS change is −0.12278 ±0.00515pp (paired bootstrap95% interval
[−0.12826,−0.10435]). This retains the photometry-training improvement as a
useful partial gain. It does not prove the photometry distribution is correct.
The central average of m1/m2 remains +0.38294 ±0.12187%.

At magnitude <25, tightening radius >3.0 to >3.2 pixels changes:

| Quantity | Component1 | Component2 |
|---|---:|---:|
| Measured response increase | +0.053356 ±0.001217 | +0.054598 ±0.000862 |
| Model response increase | +0.049568 ±0.000144 | +0.049572 ±0.000139 |
| Change in model bias (pp) | +0.36999 ±0.12962 | +0.51807 ±0.09306 |

Both model and measurement predict increased response for the stricter cut,
but the model increase is smaller. The same direction repeats at magnitude
24.9 and25.1; these are correlated checks. The radius2.8→3.0 transition is much
smaller. This identifies sensitivity near a target selection boundary, not a
unique cause: generated photometry, conditional shapes, predicted usability,
and emulator transport can all contribute. Subpopulation errors do not
supersede the owner's joint nine-cut criterion.

Comparison job17824924 completes0:0 in4s, verifies80 control cache receipts,
replays historical control m and errors, and reproduces new weighted estimates.
An independent scalar saved-sum audit verifies60 additional cut-contrast/mean/
half-difference point and jackknife quantities (`contrast_audit.json`). No
new model prediction or training was needed for this comparison.

## Next localization step

This input substitution is now [complete](../paired_measured_photometry_20261108/findings.md#completed-result-and-interpretation): RMS improves0.635→0.255%, while the radius dependence survives. The planned experiment below is retained as its rationale.

Prioritize a matched measured-photometry/usability substitution using the new
fit outcomes, with all nine cuts and the same conditional shapes and emulator.
Compare how much of the radius3.0→3.2 discrepancy survives. That would distinguish
an input/selection problem from an error remaining in the conditional response
composition; it would not uniquely separate shape and emulator errors. Do not
reuse old oracle sums as if their membership and measurement inputs were current.
A second paired seed bank remains a useful measurement robustness check before
any final calibration claim. The input substitution subsequently completed; the second seed bank remains unlaunched.

## Validation and limitations

Workers17824712_0–11 complete 0:0 in 2m59s–8m34s each; report17824713 completes
0:0 in 56s. All checkpoint/source/data receipts, exact-once coverage and same-seed
repeat checks pass. Original measured sums and bias point/error values replay.
The independent scalar audit reproduces 126 matrix/bias point and jackknife
checks from saved sums (`completed_audit.json`). Within the loose candidate
domain, 3,189 old-invalid leg fits become valid and 3,029 old-valid fits become
invalid; both effects are retained. Unmatched shape rows remain excluded and
reported; no crossmatch row lacks a shape row. The model is unchanged.

Paired seeds reduce stochastic differencing noise; they do not establish the
accuracy of this nonlinear fitter. Hardware-dependent numerical fitting
variation remains possible. These reused development scenes are not independent
acceptance evidence. Do not claim a robust 0.3% calibration. Earlier oracle
photometry/usability predictions used old shape membership and cannot simply
be divided into the new response; they require a matched re-evaluation.

The frozen comparison and threshold analysis are complete. Broad17788286 remains
held; no automated follow-up timer or production model changes.
