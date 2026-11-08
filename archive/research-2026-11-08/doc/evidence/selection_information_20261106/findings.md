# Detailed neighbour information predicts the missing selection association

## Four held-out scenes

All preparation, four training folds and both reducers completed exit0. Full
report17807248 verifies disjoint scene splits, exact model reload, untouched test
targets and numerical replay of the preceding selection audit. Every test scene
was excluded from training and validation for its fold. The diagnostic models
were selected by validation log loss, never response calibration.

| Selection predictor | Measured minus predicted selected-mean reference coefficient |
|---|---:|
| Existing weak-nine flow | +0.003599 +/-0.000078 |
| Logistic boosting, existing eight flow inputs | +0.002619 +/-0.000128 |
| Same inputs, larger depth9 capacity | +0.002584 +/-0.000125 |
| Existing inputs plus detailed-scene coefficient | -0.000010 +/-0.000133 |

The coefficient is evaluated at radius3.5/magnitude24.5 for all modelled
neighbours. It is a selection/scene-association diagnostic, not measured R11 or
the full constant-shear calibration error. Uncertainties are whole-case bootstrap
SEs with fitted models fixed; four development scenes do not provide independent
acceptance or training uncertainty. See [complete metrics](results.md).

Per-scene residuals for the larger-capacity versus added-scene-score arms are:

| Test scene | Existing-input capacity control | Added scene information |
|---|---:|---:|
| 1000 | +0.002260 | -0.000353 |
| 1040 | +0.002502 | +0.000145 |
| 1080 | +0.002957 | +0.000337 |
| 1120 | +0.002618 | -0.000167 |

Added information also improves held-out soft-label log loss in every scene,
by0.000204--0.000325 relative to the capacity control. The pooled paired gain is
-0.000247 +/-0.000024 (jackknife0.000028). Thus the result is not solely a score
calibration statistic chosen after training. The score uses truth scene features
and a frozen emulator at fixed photometry, with no measured target quantities.

## Mechanism supported by the interventions

For selection indicator S and neighbour coefficient B, the selected response
contains E[S B]/E[S]. Matching the overall pass fraction E[S] does not ensure the
correct covariance between selection and neighbour sensitivity. The flow's
three scalar flux-crowding summaries compress information that the pair emulator
uses in more detail, including neighbour sizes, profiles and separations.

The [fixed-photometry intervention](../photometry_fixed_cell_20261106/findings.md#four-scene-fixed-photometry-result)
preserves the original+0.006547 coefficient contrast even when emulator radius
and magnitude are identical within bins. The selection audit finds that
generated photometry underselects high-response configurations despite nearly
matching the overall pass fraction. The new held-out regressions show that
additional scene information predicts most of this selection association beyond
the tested existing-input predictors. Merely increasing their capacity does
little for this specific residual.

This supports richer neighbour conditioning as the next flow experiment. It
does not prove information-theoretic insufficiency of the existing eight inputs,
identify a unique omitted physical feature, or establish that one scalar feature
recovers the full photometry density. Some practical underfitting and conditioning
differences may also matter. All tests condition on actual usability in all four
legs, which differs from per-leg usability conditioning; population-level
calibration must be rechecked separately.

No offsets, production model changes or new flow training were made. In
particular the -0.000010 diagnostic result is not a new constant-shear bias.
The separately identified neighbour-induced selection response remains a distinct
part of the budget and is not removed by this result.

## Complementary image experiment and execution state

The [primary/neighbour grid design](../primary_neighbour_grid_20261106/design.md#four-shear-corners-and-exact-closure)
specifies four shear corners, paired noise/optimizer seeds, source sampling
weights, coherent grid-to-original closure and a2048-primary operational pilot.
It would directly separate primary and all-neighbour image responses. No image
jobs were launched while its timing remains tentative.

All current focused jobs are complete. Cancel redundant status-only job17806329;
broad evaluation17788286 remains held. Final syntax/serialization checks passed
for9 Python files,14 JSON files and7 shell launchers across these new diagnostics
and the replay design. Numerical gates passed in the actual scheduled jobs.
