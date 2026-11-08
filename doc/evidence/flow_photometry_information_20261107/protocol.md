# Existing-input learning versus additional neighbour information

## Frozen question and scope

Extend the earlier four-scene selection-information diagnostic to the zero-shear, per-leg-usable population used in the new truth-property plots. Test whether a flexible predictor using the original eight inputs can match the measured size, magnitude and central cut probability, and whether an added detailed neighbour score supplies further predictive value. Diagnostic regressors are not replacements for the flow or classifier, and no shear calibration claim follows.

## Data and splits fixed before fitting

Use the original 160 flow-training scenes: deterministically shuffle with seed 202611071901; first 32 are early-stopping validation, remaining 128 are training. Sample 2,048 usable g0 objects per scene without replacement with seed 202611071902 + case. Use the exact 163,840 objects (4,096 each in the 40 original flow-validation scenes) from the preceding plots for test. Verify all case sets are disjoint, sample IDs unique and source hashes correct. All source rows already meet the domain validity and true magnitude <30 predicates. No output radius or magnitude cuts are applied before fitting. Weight each row by its eligible scene population divided by its sample size, rescaled to mean one within each split.

The test scenes have already informed diagnostic plots and historical model development. This is held-out from these fits and early stopping, not a new independent scientific acceptance sample.

## Targets and fixed predictor arms

Three targets: measured FLUX_RADIUS in pixels; measured MAG_AUTO (30 − 2.5 log10(flux)); and binary pass of radius >3 px AND magnitude <25. Fit separate XGBoost squared-error regressors for radius/magnitude and a logistic predictor for the joint cut. Use five arms:

1. Existing eight inputs, depth 4.
2. Existing eight inputs, depth 8.
3. Existing inputs + neighbour score, depth 4.
4. Existing inputs + neighbour score, depth 8.
5. Existing inputs + independently within-scene shuffled neighbour score, depth 8 (negative control).

The extra input is the existing frozen-emulator sum of neighbour coefficients evaluated at fixed radius 3.5 pixels and magnitude 24.5. It is computed from truth scene properties with no current measured radius/flux labels. It is a learned, model-based scene summary, not an observed R11 or unique physical feature. Its source hashes and ID joins must match cached receipts. It compresses detailed neighbour sizes, profiles, brightness and geometry beyond the original scalar crowding inputs. The shuffled arm is a feature-use/overfitting control, not a formal conditional-independence test.

All arms use the same rows, target, split and fitting budget: histogram trees, max_bin 256, eta 0.05, min_child_weight 50, lambda 5, no row/column subsampling, seed 202611071903, up to 500 rounds, early stop after 40 without improvement on validation RMSE or log loss. Capacity and added-input contrasts are reported separately. Only validation loss selects rounds; test outcomes do not select hyperparameters.

## Reporting and interpretation

Report test squared error for radius/magnitude, Bernoulli log loss and Brier score for the cut, mean residuals, and selected-mean reference-score error. Include existing original/nine-cut flow estimates as fixed comparators where applicable. Do not compare flow log loss from 64-draw pass probabilities, which can be exactly 0/1; their Brier score includes Monte Carlo error.

Evaluate paired loss contrasts using delete-one-test-scene jackknife SEs over 40 scenes, with fitted models fixed. Report per-scene gains and the previously identified close-crowding residual; no fitted offsets or post-hoc corrections. Test loss may disagree with selected-score calibration; preserve both. This test does not recover the full photometry distribution, isolate a unique omitted feature, prove information-theoretic insufficiency, or quantify coherent-shear bias. The existing flow's full conditional-density objective differs from these mean/probability objectives.

One CPU allocation on cluster, 8 CPUs and 12 GB, with all preparation and fits in persistent workers sequentially. Record runtime/resources, best rounds, checksums, serialization replay, sample joins and progress. No new images, flow retraining, model replacement, broad evaluation resumption, or GPUs.

## Analytical Monte Carlo sensitivity, specified before reading test results

For comparisons against the existing flow's mean and cut predictions, also report quadratic losses using the product `(prediction_A − measurement) * (prediction_B − measurement)` from the two independent complete antithetic 32-draw banks. Bank A is the saved half-bank; B is twice the full-bank estimate minus A. This has the correct expected squared error without the extra variance of a finite-draw prediction, provided the two banks are independent conditional on the inputs (their underlying normal draws are disjoint). It changes neither predictions nor training, and supplements the ordinary squared error/Brier comparison. These flow comparisons remain secondary to the identical-objective, capacity-by-information contrasts among diagnostic predictors.
