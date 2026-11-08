# Both learning limitations and useful additional neighbour information

## Main conclusion

The completed test finds evidence for both mechanisms. Most of the conspicuous close-neighbour mean residual from the new plots can be learned from the flow's existing eight inputs. Separately, adding a truth-only detailed neighbour score improves held-out predictions beyond both tested existing-input predictors, while a shuffled version does not give a comparable improvement. This is a practical predictive distinction, not a proof that no possible function of the original inputs could perform as well.

## What was tested

Fit 15 separate diagnostic predictors: five input/capacity arms, each predicting radius, magnitude or the probability of passing the joint cut. Use 128 original training scenes for fits (262,144 sampled galaxies), 32 different original training scenes for early stopping (65,536), and the exact 163,840 objects in the 40 validation scenes from our preceding plots for test. Model rounds are chosen only by early-stopping validation loss. All sources and model files are hash recorded; all joins have zero unmatched IDs, and saved models reproduce test predictions exactly on reload.

The original-input arms have tree depths 4 and 8. Each is compared with an otherwise matched predictor receiving a ninth input: the frozen emulator's neighbour coefficient sum evaluated at fixed radius 3.5 px and magnitude 24.5. The score uses truth scene information, not current measured photometry. It is a learned scene summary, not a direct physical observable. A fifth arm uses the depth-8 model with scores shuffled within each scene. The original and nine-cut flows remain unchanged and serve as fixed comparators.

## The visible close-neighbour error is mostly learnable from existing inputs

For the previously identified crowding interval 0.00477–0.01860:

| Predictor | Radius error [pixels] | Magnitude error [mag] | Joint passing-fraction error [percentage points] |
|---|---:|---:|---:|
| Nine-cut flow | −0.0639 ± 0.0067 | +0.0222 ± 0.0019 | −2.064 ± 0.202 |
| Existing inputs, depth 8 | −0.0106 ± 0.0070 | +0.00284 ± 0.00195 | −0.369 ± 0.211 |
| Added score, depth 8 | −0.0079 ± 0.0069 | +0.00276 ± 0.00197 | −0.309 ± 0.209 |

Errors are predicted minus measured; positive magnitude means too faint. The original-input diagnostic reduces the point-estimate size/magnitude/selection residual by about 83%/87%/82%, respectively. These ratios are descriptive and are not precise causal fractions. The added score produces little further change in this particular bin. Thus its conspicuous residual should not be explained solely as a lack of neighbour inputs. The flow's density/response objectives, finite optimization and architecture differ from these direct mean/probability predictors; this test does not determine which learning change would reproduce that improvement in a full flow.

## Extra neighbour information nevertheless improves prediction

The following are relative reductions in held-out loss; errors are paired 40-scene jackknife SEs scaled by the pooled reference loss:

| Change | Radius squared-error reduction | Magnitude squared-error reduction | Joint-cut log-loss reduction |
|---|---:|---:|---:|
| Depth 4 → 8, existing inputs | 0.54% ± 0.14% | 0.82% ± 0.11% | 2.66% ± 0.12% |
| Add real score at depth 8 | 2.48% ± 0.51% | 1.44% ± 0.18% | 0.662% ± 0.068% |
| Add shuffled score at depth 8 | −0.031% ± 0.059% | −0.019% ± 0.055% | 0.047% ± 0.031% |

The real-score gains at depth 8 occur in 36/40 scenes for radius, 36/40 for magnitude and 38/40 for cut log loss. Gains also occur at depth 4. The benefit therefore persists after increasing capacity and is not reproduced by an extra uninformative input. The size and magnitude columns use squared error; the cut column uses Bernoulli log loss, so their percentages should not be read as equivalent changes in shear calibration.

## Connection to the earlier selection-information test

The earlier four-scene, common-all-four-usable diagnostic found selected-neighbour score error +0.002619 with the original inputs, +0.002584 with more capacity, and −0.000010 with an added score. It supported additional information but tested selection under a different usability condition.

On the present per-leg-usable test population, selected-mean score errors are +0.002606 ± 0.000506 for the nine-cut flow, +0.001466 ± 0.000501 for the depth-8 existing-input predictor, and −0.000924 ± 0.000493 with the score. The added-input predictor shifts past zero; it does not reproduce the earlier near-zero result. Preserve this limitation alongside the improvements in proper predictive losses. It is not a calibrated correction or a measured response.

## Practical interpretation and limits

Improved conditional learning and richer neighbour information are both justified directions for a controlled flow experiment. The local photometry anomaly is largely recoverable with current inputs; additional scene information offers further predictive value across the population. The useful extra score does not reveal which individual physical descriptor is missing, nor how much of the coherent-shear bias each mechanism explains.

The diagnostic predictors are not globally superior replacements for the flow. In particular, the larger existing-input predictor has worse whole-population radius squared error and joint-cut Brier score than the nine-cut flow after accounting for finite-draw noise, despite improving the highlighted conditional mean residual. Better local calibration and better overall prediction loss are distinct tests.

Naive whole-population magnitude squared-error comparisons against the flow are dominated by the previously identified extreme generated faint tail in scene 180, with correspondingly large scene uncertainty. Do not interpret the apparent large numerical MSE reduction as a robust global improvement. The capacity-versus-score comparisons among the diagnostic regressors do not depend on those flow draws.

Uncertainty fixes the trained predictors and excludes training/split uncertainty. The 40 test scenes are held out from these diagnostic fits and early stopping but have been reused in earlier model development and plots. Several fits reach the fixed 500-round limit; this is a bounded capacity comparison, not an optimization-to-convergence claim. No new images, full flow training, classifier changes, emulator changes or production model replacements were performed.

## Validation and artifacts

CPU training job 17817608 completed all 15 fits in 3m17s; report/plot job 17817627 completed in 11s. Input hashes, case disjointness, zero unmatched IDs, exact model reload and finite predictions passed. Object-level train/validation/test data, model artifacts and fitting histories are retained under `/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_photometry_information_20261107/`. `report.json` contains all losses, paired contrasts, scene-level values, subpopulation errors and Monte Carlo-corrected flow comparisons; `protocol.md` records the fixed design.
