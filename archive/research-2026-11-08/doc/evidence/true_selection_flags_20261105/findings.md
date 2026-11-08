# Measured usability and selection flags, 2026-11-05

## Definitions and scope

Older 16-cut GalSBI-dev flow and its original emulator/classifier, with unchanged
checkpoints. Report signed absolute response error, `Delta R = R_sim - R_model`;
positive means model underprediction. Follow [response conventions](../../CONVENTIONS.md#6-response-estimators)
and [flag definitions](../../CONVENTIONS.md#5-usability-detection-and-selection).
Here true detection means actual usable measurement U, which is stricter than
raw detection. True selection means actual per-leg measured FLUX_RADIUS > 3 pixels
and MAG_AUTO < 25; it does not mean cutting on intrinsic galaxy properties.

The requested branch replaces generated selection with measured membership,
retaining generated shapes and generated radius/flux as emulator inputs. It
does not require generated measurements to pass the cut as well. Classifier
probabilities are replaced by U. See [predeclared rule](rule.md).

## Joint constant-shear result

All three rows use exactly cases 1000, 1040, 1080, 1120, g1 = +/-0.02,
64 original antithetic flow draws and matched full-block random seeds. The
simulation response is 0.905810912. Errors are paired whole-case bootstrap
standard errors (10,000 resamples, seed 20260925), conditional on fixed models
and draws. Four cases are a diagnostic; they do not support a robust estimate
of all statistical and systematic uncertainty.

| Flags and selection | Flow contribution | Emulator contribution | Joint response | Delta R +/- SE |
|---|---:|---:|---:|---:|
| Predicted U, generated cut | 0.713230134 | 0.172859332 | 0.886089466 | +0.019721446 +/-0.005763188 |
| True U, generated cut | 0.712720562 | 0.171240031 | 0.883960592 | +0.021850320 +/-0.005688872 |
| True U, measured cut | 0.692227524 | 0.171371811 | 0.863599335 | +0.042211577 +/-0.005503867 |

Replacing only selection in the true-U branch increases the error by
0.020361257 +/-0.001133698. The flow contribution drops by 0.020493037 and
the emulator contribution rises by only 0.000131780. Contributions are not
separately observable component errors on these simulations.

The earlier 140-case errors +0.021855 +/-0.001274 and +0.023719 +/-0.001280
remain historical results for predicted/true U with generated selection. They
are not the matched control for this four-case replay. Full numbers and the
historical rows are preserved in [joint report](report.json).

## Correction to the earlier substitution interpretation

Replacing both auxiliary radius/flux and selection with measurements produces
flow/emulator responses 0.692227524/0.185213298, so the joint response is
0.877440822 and its error +0.028370090 +/-0.005417515. The increase in the
emulator contribution does not explain a reduction in the total error: the
flow contribution also changes, and the total error increases.

The flags-only result also clarifies the order dependence. With generated
emulator inputs, switching selection changes the emulator contribution by
only +0.000132. With measured inputs, the same switch changes it by +0.012546.
Neither number is a unique causal share of the original joint error. The
[previous findings](../constant_error_localization_20261104/findings.md#finding)
have been corrected accordingly.

## Standalone checks on their own data families

The flow check uses the same 40 held-out training-family cases as the old
standalone row (5,817,708 matched usable pairs), not the 160 fitting cases.
Use 16 original antithetic draws and common random numbers, with the same
forward influence-regression half-trace estimator on simulation and model.
The bootstrap now recomputes the measured baseline selected mass, mean shape,
and shear normal matrix in each resample. The original soft-cut responses
reproduce to within 6e-15; its SE changes negligibly after this adjustment.

| Cut and selection source | Simulation | Model | Delta R +/- SE |
|---|---:|---:|---:|
| Soft cut, generated selection (original) | 0.722234284 | 0.716535448 | +0.005698836 +/-0.002486060 |
| Soft cut, measured selection weight | 0.722234284 | 0.694761632 | +0.027472652 +/-0.002510189 |
| Hard cut, generated selection | 0.727716164 | 0.719728789 | +0.007987374 +/-0.002743722 |
| Hard cut, measured selection | 0.727716164 | 0.698053007 | +0.029663156 +/-0.002779413 |

Thus the hard-cut standalone flow error rises by 0.021675782, close to the
joint shift 0.020361257. This shows that the flags-substitution effect also
occurs without the emulator; it does not identify the cause of the original
generated-selection joint error. See [flow report](flow_report.json).

The standalone emulator already uses actual usable labels and measured baseline
selection, so its response error remains -0.002556339 +/-0.002293462 per
represented primary after weighting by mean retained neighbour multiplicity
10.798280. This is the same 40-case training-family development check, with
simulation/model responses 0.079403706/0.081960045. It is a summed projected
pair diagnostic, not coherent shear with per-leg selection migration. See
[neighbour-weighted report](../neighbour_weighted_response_20261104/report.json).

## Interpretation limits

An actual object mask does not condition an independent generated shape on the
measurement realization that passed the cut. It replaces the generated
within-object correlation between shape, size and flux with a different
coupling. Consequently a nonzero or increased error in this test does not by
itself prove that the usability or selection predictor caused the original
error, even for a well-specified conditional distribution. It is the requested
flags-substitution diagnostic, not a deployable calibration correction.

For example, at fixed context x and usability, the ordinary selected numerator
is E[S e | x]. An independent generated shape multiplied by observed membership
instead has expectation E[S | x] E[e | x]. Their difference is Cov(S, e | x).
Its shear derivative can affect the response even when the conditional model
is correct. The same issue applies to the output-dependent emulator shift.

Standalone flow and emulator errors use different training-family populations
and response estimators. They cannot be added to explain the coherent-shear
joint error. No models or production code changed, and the stopped single-cut
experiment was not restarted.

## Replay validation

Array 17750012 completed all four cases successfully with at most two GPUs.
Each case reproduced all four earlier substitution branches and original
true-U sums within declared tolerances. Measured counts and shape sums matched
the simulation reference. The requested mask's selected counts exactly matched
the all-measured branch. Reducer 17750016 completed successfully and verified
the model and simulation denominator match in every bootstrap resample.
Standalone job 17750022 ran after the joint array, completed 0:0 in 3m45s,
and reproduced the original soft-cut point values within the 2e-10 threshold.
Its per-case sufficient statistics are saved in flow_sums.npz for re-reduction.
