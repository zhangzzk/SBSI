# Neighbour-only cut crossings and selected response

## Measured contrast on the existing four images

Use the unchanged 2,048 probability-sampled IDs, the original two ngmix optimizer repeats, all four scenes, all nine cuts, the original inclusion weights, and the full-field detector/matcher receipts. Only g1 was injected: report R11 and R21, not R22. At each fixed primary sign compare opposite neighbour shears (−0.02,+0.02), then average the two contrasts. Primary analysis is fitted ellipticity with actual usability and independent hard cuts. Repeat with fixed catalogue intrinsic ellipticity to diagnose membership without noisy shape fits. Never impute shapes to failed legs.

At each primary sign, let C pass on both legs, A pass only on +, and X pass only on −. On each leg normalize the selected sum by its own selected weight W. Recalculate the common-only mean c on that leg. Relative to the response of c, entrants contribute sum_A w(e_plus−c_plus)/W_plus and exits contribute −sum_X w(e_minus−c_minus)/W_minus, divided by 0.04 and averaged over primary signs. This exact descriptive removal of migrating IDs includes denominator changes. Divide each migration term into cases where the opposite corner was usable but missed a photometric cut and cases where the opposite corner was unusable. For multiple shape seeds average the weighted counts and sums first, as in the published four-corner estimator. Save results by seed, scene, primary sign, and cut. No classifier/flow label, or assumption that measured migration equals an absolute model error.

Report paired deleted-scene jackknife and finite-population stratified sampling SE, and seed sensitivity. Keep both legs, primary signs and fits together in these calculations. Check all components sum to the original physical result and reproduce the archived counts, response and both reference errors. The 17 distinct changers are not 17 independent scene replicates. No new images, training, or automated follow-ups.

## Frozen-shape readout

Use all eight completed full-field outputs and the existing dependent report. Compare fixed weighted and matched frozen-control endpoints on all scenes, both axes, actual and predicted usability with generated cuts, and actual usability with measured cuts. Show component changes, signed and absolute residual changes. Report paired uncertainties separately from absolute calibration. No endpoint or loss choice on these scenes. Conventions: [catalogues and response](../../CONVENTIONS.md#6-response-estimators).
