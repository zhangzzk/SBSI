# What changes the selected response

## Main finding

The extra probability loss increases the effective selected sample size by about 0.35–0.38% relative to the control, depending on shear leg and axis. The selected-shape numerator increases, but not enough to keep the mean response unchanged. The resulting model response decrease reproduces the previously measured worsening of shear bias. This is a change in the selected distribution; the required normalization is not an estimator bug and should not be removed.

The largest negative total size contributions are from true radii 0.25–0.7 arcsec. The lowest-neighbour-influence quartile contributes negatively, while the highest contributes positively. A second decomposition shows changes within the groups as well as population reweighting, so this is not solely an artifact of assigning a common denominator change to every group.

All statements describe extra-loss minus matched control at radius >3 pixels and magnitude <25, with predicted usability, unless stated otherwise. They localize the *change caused by the training intervention*, not the entire original model error.

## Three terms that sum to the observed change

All response values in the following tables are multiplied by 1000. Errors are paired delete-one-scene standard errors from four reused development scenes, with trained models fixed.

| Term | Axis 1, ΔR ×1000 | Axis 2, ΔR ×1000 |
|---|---:|---:|
| Cut-membership change | +1.7264 ± 0.0361 | +1.5765 ± 0.0238 |
| Output-value change | +0.9196 ± 0.0017 | +0.9254 ± 0.0032 |
| Selected-mass normalization | -3.2683 ± 0.0209 | -3.2565 ± 0.0190 |
| Total | -0.6224 ± 0.0449 | -0.7545 ± 0.0422 |

Cut membership and output values contribute positively, but the negative normalization term is larger. The total decreases by0.0006224/0.0007545 on axes1/2, corresponding to the earlier bias increases of0.0697/0.0854 percentage points. This is consistent with the earlier flow/emulator decomposition: they are different ways to partition the same total change.

The terms use a symmetric exact identity under the same latent draws. Output value includes both generated shape and the emulator-induced shape shift. Frozen shape weights preserve the shape density at fixed photometry, but photometry changes can still change the generated shape and its selection.

## Where the changes occur

| True radius [arcsec] | Total contribution, axis 1 ×1000 | Total contribution, axis 2 ×1000 |
|---|---:|---:|
| <0.15 | +0.0393 ± 0.0352 | +0.0808 ± 0.0237 |
| 0.15–0.25 | +0.4905 ± 0.0699 | +0.4154 ± 0.0137 |
| 0.25–0.4 | -0.5666 ± 0.0522 | -0.6734 ± 0.0430 |
| 0.4–0.7 | -0.4895 ± 0.0138 | -0.4840 ± 0.0151 |
| ≥0.7 | -0.0962 ± 0.0009 | -0.0933 ± 0.0034 |

These bins describe intrinsic true radius in arcseconds. They are different from the measured/generated FLUX_RADIUS bins in pixels in the earlier plots; they must not be equated directly. Each bin contributes to the global response change using the global selected mass, rather than its own denominator.

| Neighbour-influence quartile | Total contribution, axis 1 ×1000 | Total contribution, axis 2 ×1000 |
|---|---:|---:|
| Q1: lowest | -0.8462 ± 0.0388 | -1.0124 ± 0.0284 |
| Q2 | -0.3651 ± 0.0274 | -0.2454 ± 0.0124 |
| Q3 | -0.1303 ± 0.0607 | -0.2228 ± 0.0212 |
| Q4: highest | +0.7192 ± 0.0314 | +0.7262 ± 0.0141 |

Quartiles use the truth-only frozen reference neighbour coefficient within each scene. Approximate quartile boundaries are0.012,0.071–0.073 and0.336–0.343; exact values are retained in report.json. This is a neighbour-influence score, not neighbour separation or galaxy count. A positive group contribution offsets the global response decrease, but does not prove that the group itself is correctly calibrated.

## Population composition versus changes within groups

For each shear leg, write the global selected mean as the sum of selected group fractions times their within-group mean shapes. A symmetric decomposition separates changes of these fractions from changes of the within-group means. Both contributions are recomputed for each jackknife sample and add to the original response change.

| Partition | Population-composition term, axis 1 ×1000 | Within-group term, axis 1 ×1000 | Population-composition term, axis 2 ×1000 | Within-group term, axis 2 ×1000 |
|---|---:|---:|---:|---:|
| Five true-size bins | -0.3361 ± 0.0025 | -0.2863 ± 0.0444 | -0.3341 ± 0.0030 | -0.4205 ± 0.0439 |
| Four influence quartiles | -0.0479 ± 0.0027 | -0.5744 ± 0.0462 | -0.0419 ± 0.0036 | -0.7126 ± 0.0402 |
| Twenty size × influence cells | -0.2369 ± 0.0046 | -0.3855 ± 0.0460 | -0.2322 ± 0.0047 | -0.5223 ± 0.0426 |

With the joint size × neighbour partition, roughly38%/31% of the net response decrease is assigned to composition and62%/69% to changes within the cells (axes1/2). These are descriptive central-value fractions, depend on the chosen partition, and are not unique physical causes. Within-cell changes can still include finer selection changes.

The independently normalized within-quartile response also falls in Q1–Q3 and rises in Q4 on both axes. For example Q1 changes by−0.000960±0.000146 /−0.001535±0.000094, while Q4 changes by+0.001179±0.000199 /+0.001242±0.000093. This verifies that the sign pattern is not merely the common-denominator allocation.

## Which cut crossings matter

In the cut-membership term, radius-only switches contribute +0.870/+0.763 ×10⁻³, flux-only switches +0.851/+0.798 ×10⁻³, and switches in both component cuts +0.005/+0.015 ×10⁻³ (axes1/2). Radius and flux are comparably important. These are contributions of paired generated draws changing selection state, not an estimate of a physical causal response for an individual galaxy.

## Added neighbour input and the combined model

The neighbour-input-only arm has a positive Q4 contribution of+0.326/+0.282 ×10⁻³, largely canceled by negative contributions in the lower quartiles. Its total remains only+0.0448/+0.0201 ×10⁻³, consistent with no resolved central shear improvement. The combined arm strengthens the same opposing pattern and still decreases the total response.

Across all nine cuts, the extra-loss cut-membership term is positive and its normalization term negative on both axes. Its total response falls at8/9 cuts on axis1 and9/9 on axis2. Thus the central result is not an isolated cut choice.

## Implication for the next test

The earlier photometry comparisons supported both imperfect learning of existing inputs and useful extra neighbour information. This localization adds a more specific failure: improving passing probabilities changes the population mix and selected-shape response in an unfavorable balance. More accurate passing probabilities alone do not constrain that balance.

A targeted follow-up should test conditional selected-shape response jointly with passing mass, using fixed true-size and neighbour bins and evaluating the final normalized response. Keep the existing density and probability checks. The current trial already has a global response penalty; this finding motivates testing its conditional structure rather than assuming that adding static neighbour information or simply increasing training time solves the error. The subgroup decomposition is exploratory and should be checked on additional scenes before being treated as a general physical conclusion. No new training was launched for this interpretation.

## Validation and files

All evaluation and report jobs completed exit0:0:17818806,17818807_0–2,17818808. The saved total response changes replay to within1.87×10⁻¹²; original aggregate sums pass the declared replay tolerance. Independent interpretation job17820925 completed0:0, reproducing1,512 reported values/SEs and checking both partition and alternative-decomposition closure. Both figures were visually inspected. The broad paused evaluation remains held.

Figures: response_localization.png/pdf shows the three original terms, size/influence contributions and boundary crossings; population_composition.png/pdf separates population fractions from within-group changes. Detailed tables: report.json and interpretation.json. These are four-scene development results with fixed models, not an independent calibration acceptance.
