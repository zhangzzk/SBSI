# Measured-input localization after paired-seed shape refits

## Completed result and interpretation

Replacing generated size, flux and predicted usability/selection with measured
photometry and NEW independently selected usable membership reduces the joint
nine-cut RMS from0.63457% to0.25525%. The paired change is−0.37932±0.17053
percentage points; paired bootstrap95% interval[−0.62268,−0.03180]pp. This is
substantial improvement on these40 development scenes, not a deployed calibration
result or an independent acceptance test. The oracle RMS95% interval extends
to0.45306%, so its point estimate below0.3% does not establish a robust0.3% claim.

At radius>3pixels and magnitude<25:

| Prediction inputs | m1 (%) | m2 (%) |
|---|---:|---:|
| Generated photometry, predicted usability | +0.55576 ±0.13143 | +0.21011 ±0.16898 |
| Measured photometry, new actual membership | −0.13281 ±0.12329 | −0.41639 ±0.16471 |
| Paired shift (percentage points) | −0.68858 ±0.03605 | −0.62651 ±0.02716 |

The substitution moves both biases downward and changes their central mean from
+0.38294% to−0.27460±0.11311%. It improves the nine-cut objective, but overshoots
zero at the central cut. The remaining component difference0.28358±0.18298pp
is not strong evidence of unequal component biases. Inputs and membership are
substituted jointly; this is not a unique allocation to photometry or classifier.

## Radius dependence survives the substitution

At magnitude<25, tightening radius>3.0 to>3.2pixels changes the biases by:

| Prediction inputs | Delta m1 (pp) | Delta m2 (pp) |
|---|---:|---:|
| Generated/predicted | +0.36999 ±0.12962 | +0.51807 ±0.09306 |
| Measured/actual | +0.26997 ±0.09676 | +0.45970 ±0.09411 |
| Change in the above contrast | −0.10002 ±0.04814 | −0.05838 ±0.05201 |

The remaining oracle differences are about2.8 and4.9 jackknife standard errors
from zero. They have the same sign at magnitude24.9 and25.1, which are correlated
checks, not independent replications. At radius>3.2pixels/magnitude<25, oracle
biases are+0.13716±0.15023% and+0.04330±0.16563%, close to zero. Thus the joint
radius dependence remains even though the original large bias at the stricter
cut improves. Generated photometry and classifier membership are insufficient
as a complete explanation within this frozen composition.

A remaining error in conditional shapes, neighbour transport or the measurement
reference can produce this pattern. Model contribution sizes alone cannot
separate those causes. The next direct localization should compare paired-seed
primary-only and neighbour-only measured responses on matched populations near
the radius cuts. Preserve the useful photometry-training gain; do not force
symmetry or add a fitted calibration offset.

## Full target grid

Errors are whole-scene jackknife SE in percentage points. All cuts remain in
the objective, and correlated cuts are not counted as independent observations.

| Radius lower bound (pixels) | Magnitude upper bound | Measured-input m1 (%) | Measured-input m2 (%) |
|---:|---:|---:|---:|
| 2.8 | 24.9 | -0.209 ±0.149 | -0.453 ±0.169 |
| 2.8 | 25.0 | -0.208 ±0.145 | -0.395 ±0.176 |
| 2.8 | 25.1 | -0.007 ±0.146 | -0.297 ±0.175 |
| 3.0 | 24.9 | -0.143 ±0.140 | -0.427 ±0.160 |
| 3.0 | 25.0 | -0.133 ±0.123 | -0.416 ±0.165 |
| 3.0 | 25.1 | +0.056 ±0.129 | -0.317 ±0.167 |
| 3.2 | 24.9 | +0.159 ±0.169 | -0.017 ±0.172 |
| 3.2 | 25.0 | +0.137 ±0.150 | +0.043 ±0.166 |
| 3.2 | 25.1 | +0.286 ±0.160 | +0.104 ±0.171 |

The additional36-scene cohort has RMS0.67231→0.22658%, while the original four
have0.44285→0.67682%. The latter cohort is small and noisy; both are retained in
`report.json`. The declared primary comparison uses all40 scenes and no subset
was chosen from its outcome.

## Central response matrices

Rows are shape components; columns are shear components. Independently normalize
each leg, then difference±0.02 over0.04. Define m_i=100*(Rmeasured_ii/Rmodel_ii−1).

```
Measured:         [[ 0.904867,  0.000324],
                   [-0.000239,  0.902380]]
Generated inputs: [[ 0.899866, -0.002308],
                   [ 0.000755,  0.900488]]
Measured inputs:  [[ 0.906070, -0.002015],
                   [ 0.000958,  0.906153]]
```

The oracle diagonal contributions are flow0.721039/0.721102 plus disk transport
0.185032/0.185051. These are model bookkeeping terms, not measured error budgets.
[Response conventions](../../CONVENTIONS.md#6-response-estimators),
[bias conventions](../../CONVENTIONS.md#7-multiplicative-bias),
[protocol](rule.md#question-and-estimator).

## Validation, resources and limitations

Preparation17824947 completes0:0 in39s. All6,218 changed leg memberships are
included; photometry and unchanged-row mathematical predictions are preserved.
The signed update uses newly evaluated model contributions, never an empirical
correction. Original measured/old oracle sums and new measured sums replay.
Unmatched rows are retained in the audit: 228,683 shape rows
without crossmatch are excluded; 0 crossmatch rows lack a shape row.
These totals cover40 scenes/four exposures once, not repeated model arms.

Pilot17824967 completes0:0 in64s. Full selected evaluations for case1000/axis1
and1129/axis2 replay old oracle responses within2.19e-9 and signed-update new
responses within2.17e-9 (gate1e-6), with exact counts. Conditional equality of
both frozen photometry endpoints, reconstruction, latent invariance, direct
neighbour feature grouping and original disk-boundary handling pass.

Four persistent GPU workers17824973_0–3 complete0:0 in45–71s. CPU report17824974
completes0:0 in3s;168 independent scalar point/jackknife checks pass. All80 output
receipts, source/checkpoint pins, half-bank identities and selected counts pass.
Draw-half differences at the central cut are+0.00788±0.01136 and+0.00193±0.01458pp;
this is an integration-sensitivity check, not a bound on fitting systematics.
The model response change caused solely by updated membership is small:
Delta R11=−0.00000780, Delta R22=+0.00001014, now explicitly accounted for.

The old reference fitter can still have numerical or optimization bias; paired
seeds do not prove its accuracy. These reused scenes do not establish independent
acceptance. No model training/preset changes, broad-array resume or automatic
timers. All jobs in this follow-up are complete; the split-response follow-up
has not been launched.
