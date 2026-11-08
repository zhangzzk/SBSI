# Full joint measured-input diagnostic

## Full forty-scene result

The complete40-scene, two-axis evaluation has finished successfully. This
supersedes the four-scene pilot below. At measured radius>3px/magnitude<25,
with actual per-leg usability and both models evaluated at measured photometry:

| Flow | Delta R11 | m1 (%) | Delta R22 | m2 (%) |
|---|---:|---:|---:|---:|
| Cartesian NLL | -0.011041 | -1.2025 ± 0.3093 | -0.012808 | -1.4070 ± 0.2400 |
| Weak nine-cut | -0.000024 | -0.0026 ± 0.3128 | -0.006628 | -0.7331 ± 0.2417 |

Errors in this table are whole-case jackknife SEs in percentage points. The
[full report](full_results.md) also provides bootstrap errors, which agree
closely. The measured-input substitution does not remove the full residual:
NLL overpredicts both diagonal responses, while weak-nine agrees in R11 but
still overpredicts R22. All80 case/axis outputs pass the numerical checks.

On matched available scenes, the substitution lowers m by1.04--1.15 percentage
points. The NLL comparison uses40 scenes and the original weak-nine comparison
uses22; do not compare the22-scene generated value directly with the40-scene
measured-input value. The model emulator contribution rises0.01296--0.01339,
partly offset by flow-contribution decreases0.00247--0.00401. This isolates a
prediction shift under the intervention, not a standalone emulator error.

The largest full-sample difference between the two32-draw antithetic banks is
0.0082pp in m; case SEs are0.24--0.31pp. The four-scene pilot's positive m2 for
weak-nine did not persist in the full sample. The residual must be interpreted
from the40-scene result, not selected pilot scenes. Fits and existing image
measurement realizations remain fixed; this is development evidence.

A [paired-axis audit](../neighbour_flow_selection_20261106/completed_audit.json)
gives m2-m1=-0.2045 ±0.3944pp for NLL and-0.7304 ±0.3985pp for weak-nine
(case jackknife). The latter difference is only about1.8 SE, so it does not
establish a distinct axis-specific failure. The trace response bias is
-1.3043% ±0.1945pp for NLL and-0.3672% ±0.1961pp for weak-nine.

## Four-scene result

Cases1000/1040/1080/1120, both coherent shear axes, h=0.02. Use each leg's
actual usable measurements, measured FLUX_RADIUS>3px (=0.6arcsec) and
MAG_AUTO<25. Draw flow shapes conditional on that leg's measured photometry,
then apply the original emulator's full neighbour transport at that photometry.
The original Cartesian NLL and original weak-nine checkpoints are frozen in
[the manifest](manifest.json); this is not the new neighbour-conditioned trial.
Definitions and estimator provenance: [protocol](rule.md#authorized-diagnostic).

The automatically generated [results](four_results.md) and
[numerical report](four_report.json) preserve paired bootstrap and jackknife
errors. For these four scenes, the following table uses the slightly larger
whole-case jackknife SE. Errors on m are percentage points.

| Flow | Delta R11 | m1 (%) | Delta R22 | m2 (%) |
|---|---:|---:|---:|---:|
| Cartesian NLL | -0.013124 | -1.4282 ± 0.6554 | -0.000846 | -0.0930 ± 0.3810 |
| Weak nine-cut | -0.002080 | -0.2291 ± 0.6608 | +0.005386 | +0.5960 ± 0.3842 |

DeltaR=simulation minus model. The simulation diagonal responses are0.905811
and0.909096. NLL predicts0.918935 and0.909942; weak-nine predicts0.907890 and
0.903711. These are development results on four reused scenes with fitted
models and measurement realizations fixed, not a0.3% calibration demonstration.

## Paired change and interpretation

Using generated photometry and selection, with the same actual usability on
the same four scenes, NLL has m1=-0.1925% and m2=+1.0022%; weak-nine has
m1=+0.9544% and m2=+1.6348%. Substitution lowers m by1.04--1.24 percentage
points across the four comparisons. Paired bootstrap SEs are0.10--0.14pp;
jackknife SEs are0.12--0.16pp.

The model emulator contribution rises by0.01300--0.01360. The conditional flow
contribution decreases by0.00164--0.00437, so the total model response rises by
0.00924--0.01138. These are changes in model contributions, not independently
measured component errors. The emulator artifact is unchanged.

Measured photometry itself changes under coherent shear and also conditions the
flow's shape distribution. This intervention therefore cannot be interpreted as
an additive removal of a single causal error, or as proof that the emulator
alone is unbiased. Selection changes are included in the full response; the
previous opposite-group selection term must not be added again.

## Numerical checks and continuation

All eight case/axis tasks pass conditional reconstruction, fixed-photometry
preservation, exact selected counts, original simulation-sum replay and the
full/half-bank moment identity. Two separate32-draw antithetic banks differ in
aggregate m by0.031--0.068pp, smaller than the scene uncertainty. No clipping,
row removal or empirical corrections were applied. The rare representable disk
boundary values are counted in the report.

All nine nearby cuts are retained. Their point estimates still vary: NLL m1
ranges from-1.97% to-0.52%, m2 from-0.83% to+0.20%; weak-nine m1 ranges from
-0.67% to+0.61%, m2 from-0.07% to+0.88%. Full paired cut results are in JSON;
these ranges alone are not significance tests.

The same frozen test subsequently completed on all40 existing scenes above. Comparisons with
the old generated-input results use only matched available scenes:40 for NLL,
22 for weak-nine. The older broad evaluation remains held.
