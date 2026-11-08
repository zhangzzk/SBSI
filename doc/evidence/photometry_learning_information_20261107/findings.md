# Completed comparison: better probabilities, no shear-calibration improvement

All four fixed step-64 endpoints and all eight coherent field/axis evaluations completed successfully. The extra conditional-cut loss improves held-out cut-probability prediction. Adding the static neighbour summary improves that score slightly further. Neither change produces a useful shear-calibration improvement in this matched experiment: the loss increases the central bias, and the extra input alone has little central effect.

## What was held fixed

All arms start from the original nine-cut regularized checkpoint (888fb044), with exactly identical initial density and physical draws. Only photometry parameters train. Every conditional-shape weight stays fixed, including zero weights on the new input. Input-off arms remain invariant to the extra score. Input-on arms change five additional photometry tensors. Classifier, emulator, row/latent streams, training budget and evaluation data are fixed. This tests a specific 64-update training intervention and a particular neighbour summary, not convergence or all possible neighbour representations.

## Held-out prediction

The score is the mean conditional cut-probability squared prediction error, estimated with independent draw banks to remove integration variance. Lower is better. It still contains irreducible Bernoulli outcome variation; it is not the squared error to a known true probability. Results average the nine target cuts and both original data legs across 40 validation scenes.

| Arm | Mean probability score | Change versus control | Relative reduction |
|---|---:|---:|---:|
| Matched control | 0.07182869 | — | 0.000% |
| Extra loss | 0.07171782 | -0.00011087 ± 0.00000655 | 0.154% |
| Extra neighbour input | 0.07181782 | -0.00001088 ± 0.00000187 | 0.015% |
| Both | 0.07169145 | -0.00013724 ± 0.00000785 | 0.191% |

All three interventions reduce this score at each of the nine cuts. The combined probability-score interaction is -0.00001549 ± 0.00000335: the combined improvement exceeds the sum of the two separate improvements on this score. That interaction does not translate into a detectable central shear-bias benefit.

The added input alone slightly worsens photometry negative log likelihood: +0.00083752 ± 0.00007083. The extra loss alone changes it by -0.00023371 ± 0.00011932. Thus improved cut probabilities are not evidence that every part of the photometry distribution improves.

## Coherent shear calibration

Central cut: radius >3 pixels and magnitude <25. Model selection and usability are generated/predicted; simulation selection and usability are measured. Bias is m=100(R_sim/R_model−1). Four reused development scenes are paired across arms; quoted SEs resample whole scenes with fitted models fixed. Percentages are biases; differences are percentage points (pp).

| Arm | m1 (%) | m2 (%) | Nine-cut RMS, axis 1 (%) | Nine-cut RMS, axis 2 (%) |
|---|---:|---:|---:|---:|
| Matched control | +0.7019 ± 0.7205 | +1.3840 ± 0.4858 | 1.0179 | 1.2505 |
| Extra loss | +0.7716 ± 0.7194 | +1.4694 ± 0.4858 | 1.0638 | 1.3085 |
| Extra neighbour input | +0.6968 ± 0.7156 | +1.3817 ± 0.4830 | 1.0232 | 1.2512 |
| Both | +0.7662 ± 0.7137 | +1.4696 ± 0.4849 | 1.0719 | 1.3129 |

| Change versus matched control | Δm1 (pp) | Δm2 (pp) | Δ nine-cut RMS, axis 1 (pp) | Δ nine-cut RMS, axis 2 (pp) |
|---|---:|---:|---:|---:|
| Extra loss | +0.0697 ± 0.0049 | +0.0854 ± 0.0048 | +0.0459 ± 0.0092 | +0.0580 ± 0.0053 |
| Extra neighbour input | -0.0050 ± 0.0059 | -0.0023 ± 0.0059 | +0.0053 ± 0.0009 | +0.0007 ± 0.0032 |
| Both | +0.0644 ± 0.0075 | +0.0856 ± 0.0091 | +0.0540 ± 0.0074 | +0.0624 ± 0.0082 |

Extra loss alone worsens pooled absolute bias at 8/9 cuts on axis 1 and 9/9 on axis 2. Both changes together worsen it at 7/9 and 8/9. Added input alone has mixed small cut-dependent effects and slightly increases pooled nine-cut RMS. There is no robust improvement across the target cuts.

Paired changes are much more precise than absolute biases because the same scenes and draws are compared. These small paired SEs do not establish independent acceptance or include retraining-seed uncertainty.

## Why the prediction improvement does not fix calibration

The measured total response is compared with the sum of flow and neighbour-emulator contributions. For extra loss versus control at the central cut:

| Axis | Change in flow contribution | Change in emulator contribution | Change in total model response |
|---|---:|---:|---:|
| 1 | -0.0014594 | +0.0008371 | -0.0006224 |
| 2 | -0.0015797 | +0.0008251 | -0.0007545 |

The emulator increase is outweighed by the flow decrease. The model total falls, so the already-positive bias rises. These are model prediction contributions; they do not independently identify physical component errors.

Fixed shape weights do not fix the selected mean shape: changing photometry changes which generated draws pass and how shape is averaged. A selection diagnostic supports this distinction. With measured usability but generated cuts, the extra loss still worsens central bias; with measured usability and measured membership, it reduces central bias by about 0.14 pp on both axes. The latter mode still uses generated photometry/model draws and has much larger absolute biases (~4–5%); it is a diagnostic, not an improved inference pipeline or a replacement by measured inputs.

On the zero-shear validation leg, the measured-minus-generated selected-neighbour coefficient discrepancy also falls from 0.003317 to 0.002328 with both changes, while the passing-fraction discrepancy falls from 0.414 to 0.199 pp. Better static selection and neighbour association therefore coexist with worse coherent shear response.

## Relation to the earlier comparisons

The earlier distribution panels compare different quantities: size-density differences count how many galaxies occupy each size bin, whereas the neighbour-coefficient panel compares conditional neighbour influence within size/flux cells. Peaks in different bins are not contradictory, and matching marginal histograms does not ensure the right conditional response.

The earlier tree diagnostic showed that existing inputs recover much of the local crowding anomaly, while extra neighbour information adds predictive value. The present flow experiment supports usefulness for cut-probability prediction, but does not establish that either intervention cures the shear error.

The earlier two-arm conditional-loss trial reduced central biases by 0.103/0.115 pp. It started from an already neighbour-trained checkpoint (6fc0c373), unlike this four-way experiment starting from the original nine-cut checkpoint (888fb044). Its control predictions differ. Opposite outcomes imply that the calibration benefit of the same kind of loss depends on the starting model and the balance of response contributions; the results must not be pooled as one matched comparison. The separate wider replication of that older frozen pair remains a distinct experiment.

## Interpretation and next step

We have evidence for useful extra neighbour information and for improving a particular conditional prediction with a changed training objective. We do not yet have a successful shear-calibration intervention, a proof that existing inputs are fundamentally insufficient, or a convergence test. The added score summarizes static truth geometry at reference photometry; it does not introduce a neighbour-shear-varying channel.

The next targeted question is why the selected-shape response changes incorrectly when cut probabilities improve. Use the saved fixed models to localize response changes near the size/flux boundaries and by neighbour environment, then test an objective that controls the relevant selected-shape response alongside probabilities. Longer training or extra inputs alone should not be treated as a demonstrated fix. No new training was launched while interpreting this completed experiment.

## Files and validation

`factorial_summary.png` separates conditional prediction, central bias, nine-cut RMS, and measured-selection diagnostics. `all_cut_changes.png` shows every target cut. Machine-readable tables remain in `report.json`, `probability_report.json`, and `own_report.json`. `completion_audit.json` records the independent saved-sum reconstruction.

Checks/training/pilot/evaluation/report jobs 17817862, 17817874, 17817875_0–2 and 17817876 completed with exit 0:0. Resource job 17817944 restored the separate replication to six workers. All replay, frozen-shape, checkpoint reload, input-invariance, measured-count and pooled-emulator checks passed. The broad paused array remains held.
