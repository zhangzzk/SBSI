# Conditional shapes at measured photometry: zero-shear check

## Completed result

On249,433 zero-shear usable rows from the existing fixed validation sample,
119,008 pass radius >3 pixels and magnitude <25. The current conditional model
slightly underpredicts both mean squared ellipticity components. The difference
between those squared components is consistent with the measured value at
this precision. This establishes a static conditional-shape discrepancy but
does not identify the coherent shear-response error.

| Central-cut quantity | Measured | Current prediction | Prediction − measurement ± case jackknife SE |
|---|---:|---:|---:|
| Mean e1 | −0.00012435 | +0.00002233 | +0.00014668 ±0.00083992 |
| Mean e2 | +0.00129652 | −0.00031764 | −0.00161416 ±0.00085590 |
| Mean e1² | 0.12592562 | 0.12496735 | −0.00095827 ±0.00039622 |
| Mean e2² | 0.12636282 | 0.12509748 | −0.00126535 ±0.00037313 |
| Mean e1²−e2² | −0.00043720 | −0.00013013 | +0.00030708 ±0.00061863 |

These are dimensionless shape moments, not multiplicative-bias percentages.
The second moments are about0.76%/1.00% low relative to their measured values;
those fractions must not be equated with shear bias. Raw second moments are
reported, not exactly mean-subtracted variances.

At radius >2.8 pixels the deficits are larger and more clearly resolved; at
radius >3.2 pixels they are smaller and compatible with zero at current scene
precision. The squared-component difference is compatible with zero residual
at all nine cuts. The cuts overlap heavily. See [all cuts and four model
comparisons](report.json).

## Which model comparison is meaningful

Original NLL, original nine-weak, neighbour-conditioned NLL and current shapes
all use measured radius/flux, actual membership, identical base preprocessing
and common shape latent draws. The current probability-loss and control
endpoints share the last conditional shape distribution exactly.

The current shape model improves several conditional prediction scores relative
to original nine-weak; that does not imply a better coherent shear response.
For example current-minus-nine-weak conditional squared scores for e1²/e2²
are−0.00003258±0.00000544 /−0.00002369±0.00000311. Relative to its actual
neighbour-NLL ancestor, those score changes are instead
+0.00000143±0.00000132 /+0.00000163±0.00000091. The latter continuation includes
NLL and response optimization; it is not a response-penalty-only intervention.
All six moment scores and paired errors are retained, including unfavorable
or unresolved changes. These scores contain irreducible conditional outcome
variation; their absolute values are not pure model error.

## Implication for the next training choice

The [current measured-input coherent test](../current_measured_photometry_20261107/findings.md#completed-result-and-decision)
leaves m2 near−0.96%, whereas this static zero-shear check finds no resolved
error in the squared-component difference. Matching that single static moment
is therefore not currently justified as the calibration remedy. Its influence
on disk transport can also depend on neighbour weighting and shear; a correct
unweighted zero-shear mean does not certify the full response.

Likewise, making the shape distribution wider to fix the second moments could
improve a static diagnostic while changing coherent response unfavorably. The
previous photometry/moment trials already demonstrated that risk. Do not add
an empirical shape rescaling or another loss solely to make these means match.

The next scientific target is the conditional-shape change under primary shear
and its composition with neighbour response, using compatible population and
normalization. Existing primary matched-pair targets can test that restricted
question; they cannot settle neighbour-only usability/selection channels or
provide independent acceptance. Full physical attribution remains unresolved.

## Sampling, numerical checks and limits

Use only g0 rows inside the original500k validation-NLL sample across40 scenes.
Retain the original full block/latent ordering before skipping g05 rows; no new
sample, truth cut or common-leg intersection is introduced.64 antithetic draws
are split into two independent32-draw banks. For a given moment, the product
of bankA-minus-observed and bankB-minus-observed residuals estimates conditional
squared prediction loss without the extra finite-draw variance term. Both banks
use the same object. Whole-scene bootstrap10000 and jackknife preserve model/
cut correlations. [Fixed protocol](rule.md#quantities-and-interpretation).

GPU17821432 completed0:0 in37s, with all four models packed in one allocation.
Input/target state equality, score alignment, measured-photometry exactness,
conditional reconstruction and finite-output gates pass. Independent audit
reproduces2,160 point/jackknife scalars exactly and replays prior selected counts
and observed shape sums on the same rows; quadrupole identities and the
independent-bank scoring fixture pass. [Audit receipt](audit.json).

Half-bank differences at the central cut are0.00007424 for mean e1²,
0.00006848 for mean e2² and0.00000576 for their difference; respective case SEs
are0.00010071,0.00013341 and0.00014984. This is an integration sensitivity
check, not an exhaustive uncertainty study.

These reused validation scenes, fixed checkpoints and existing measurement
realizations provide development evidence. No training, emulator change,
production promotion, new images or broad-array restart was performed.
