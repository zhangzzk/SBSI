# Constant-shear error investigation: matched four-state test

## Current conclusion

The matched test below localizes part of the joint error to response to other
galaxies' shear. Its negative-T mechanism remains unproven. A shared underlying population
should preserve cancellation of an unchanged conditional self-response error.
The earlier cell comparison does not demonstrate why that cancellation fails:
the joint-minus-standalone point difference is0.014219, with0.013270 outside
the highlighted low-crowding size cell. The experiments also contain different
neighbour perturbations and selection responses. The matched contrast is now available.

## Completed controls

The four-state model predictions are cached for all four constant scenes.
All four runs reproduce every original all-plus/all-minus selected count and
shape numerator for four cuts and both predicted/actual usability. This checks
the new evaluator against the original reported error before interpreting new
mixed states. The sum of separate A/B neighbour emulator predictions also
reproduces the unsplit prediction on each case's first block.

Analytical fixtures validate the exact own/opposite-group decomposition,
population and flow/emulator additivity, normalization, and the measured
shape/selection product-and-ratio identity. These are implementation checks,
not scientific results. See `contrast_verification.json`,
`reducer_verification.json`, and `fit_reducer_verification.json`.

## Optimizer instability confirmed

Two independent seeded re-fits of the same original pixels give component RMS
change0.22005 across62767 selected usable case/leg rows. The median vector
change is0.00369, but12.10% exceed0.1 and the99th percentile is1.5414: a tail
of large changes dominates the RMS. Radius3–3.2px objects comprise14.16% of
these rows but contribute35.65% of squared g1 changes. This is a descriptive
sample decomposition, not a population response-error attribution.

An independent uniform512-ID case1000 replay (1021 detected case/leg rows)
reproduces the seeded results to1e-10. All2042 independent fits report flags=0,
yet their component RMS change is0.20836. Thus this is not explained by ignoring
nonzero fitter-status flags or by catalogue misalignment. Some successful fits
return nearly opposite, near-unit ellipticities with very different likelihoods.
The installed ngmix Runner retries only nonzero flags, accepting the first
nominal success; it does not compare multiple successful starts.

Recording fitted parameters localizes much of this scatter. In1001 selected,
both-success case/leg rows from the uniform replay sample,7.94% of fits have
negative deconvolved second-moment size T. Between the two starts,5.29% switch
T sign; those switches account for75.52% of squared g1 changes. All16 deliberately
selected largest-change examples switch T sign and reverse ellipticity direction;
their PSF parameters change by at most5.6e-17. This points to the galaxy fit's
signed-size solutions rather than a stochastic PSF change. The original code's
`FlatPrior(-10,1.e6)` explicitly permits negative T. Its reported success and
the later finite/disk-shape usability test retain these fits. T here is ngmix's
deconvolved moment size; the catalogue selection still uses FLUX_RADIUS.
See [parameter report](fit_parameters_report.json). No positivity cut, extra
starts, or changed measurement definition has been applied. A mean response
bias cannot be inferred from this scatter attribution.

The original unseeded result and two seeded repetitions give the following
changes on the same sampled targets (8000 IDs/case,4 cases):

| Difference in measured response | Change | Conditional subsampling SE | Four-case bootstrap SE |
|---|---:|---:|---:|
| Repeat0 minus original | -0.024919 | 0.042633 | 0.033344 |
| Repeat1 minus original | +0.038328 | 0.041563 | 0.046609 |
| Repeat1 minus repeat0 | +0.063247 | 0.043024 | 0.028862 |

These sample changes do not establish a systematic shift. Independent-object
ratio-influence scaling from the two seeded repeats estimates a single-fit
response SE0.007437 for the full four-scene selected catalogue. This estimate
is conditional on images and optimizer independence and uses only two repeats.
If those scenes represent the140-case sample, scaling by sqrt(4/140) gives
0.001257, close to the previously quoted0.001274 case-bootstrap SE. Optimizer
noise can already be present in case-to-case scatter; do not add this term in
quadrature or claim the older SE is underestimated from this check alone.

Source: [repeat response report](fit_report.json),
[seeded replay and distribution report](fit_details_report.json).
The mean joint calibration error remains unresolved. Matching the model's
training measurement algorithm does not guarantee its response is learned,
but optimizer randomness alone is not evidence of a mean model bias.

## Matched measurement test

The four original scenes have fixed random A/B groups. Two new mixed states
shear the groups oppositely at +/-0.02. Together with the original coherent
states, these identify the response to shearing the other group while holding
each target's own shear fixed. Own-group includes same-group neighbours.
The contrasts sum exactly to the original coherent response.

Report all four cuts, predicted/per-state actual/common usability, and both
global and within-group normalizations. On common usability, measured changes
split exactly into shape changes and crossing selection cuts. In an opposite-
group contrast the model's flow and generated selection stay fixed, directly
testing the neighbour response represented by emulator transport.

An unchanged-input control must reproduce original image pixels and detection
tables. The completed seeded fitter audit above is not a bound on all
uncertainty or a substitute for scene sampling.

## Completed matched-shear response decomposition

All eight mixed-image/shape jobs completed. The unchanged-input control
reproduces science/PSF pixels and input/detection/crossmatch tables exactly.
Reducer17753575 completed after allowing absolute5e-16 roundoff only in the
renderer's derived intrinsic e1/e2 columns; q, angle and all other non-shear
inputs still match exactly. All original coherent response sums replay.

With actual per-state usability, radius>3px and MAG_AUTO<25, signed absolute
response errors (simulation minus model) are:

| Exact contribution | Error | Paired four-case bootstrap SE |
|---|---:|---:|
| Own-group shear | +0.013958 | 0.004683 |
| Opposite-group shear, target own shear fixed | +0.007892 | 0.001789 |
| Coherent total | +0.021850 | 0.005689 |

These add exactly under the global selected denominator; their SEs are correlated.
Own-group includes same-group neighbours and is not a pure flow error. The
opposite-group contrast directly measures missing response to changing other
galaxies' shear, including selection and normalization. Do not double it into
a claimed full-neighbour correction. Predicted usability gives own+0.012594
and opposite+0.007127, total+0.019721 on the same four scenes.

On the common-all-four usable cohort, within-group normalization removes the
opposite group's selection-denominator influence. Group A's opposite-shear
simulation/model responses are0.091280/0.084224 (error0.007056 +/-0.004473);
group B gives0.094921/0.084830 (error0.010091 +/-0.004485). Exact measured
selection terms are0.002715 +/-0.000855 and0.003136 +/-0.000226; the remaining
measured shape terms are0.088566 and0.091785. The model flow/selection remain
fixed in these contrasts, so model response is entirely emulator transport.
Both missing neighbour-induced selection and residual shape response are
present at the point-estimate level. These common-U terms are for a different
cohort and are not pasted into the original total budget.

This result does not establish whether negative-T switching causes the
neighbour response shortfall. It does establish that there is a neighbour-
related discrepancy on the constant scenes themselves, which the standalone
flow test cannot assess. Source: [full numerical report](report.json),
[compact table](results.md).

## Execution and remaining limits

Image simulations and model evaluation completed, with at most two GPUs during
model evaluation. The first reduction stopped at a derived-column roundoff
check; the repaired reduction completed with the provenance and replay checks
above. Job identities are in `jobs.json`. Numerical results are in `report.json`
and `results.md`; optimizer results are in `fit_report.json`.

Four cases are a diagnostic with conditional paired case-bootstrap errors.
They are insufficient for a precision acceptance claim. No model has been
retrained or corrected. See the [predeclared rules](rule.md#four-state-image-experiment)
and [response conventions](../../CONVENTIONS.md#6-response-estimators).
