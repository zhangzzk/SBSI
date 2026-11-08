# Full joint response with measured photometry and selection

40 scenes; actual usability and cuts independently per shear leg.

Target: radius>3 pixels and magnitude<25. Absolute response error is simulation minus model. m is in percent; its SE is in percentage points.

| Flow | Delta R11 | m1 (%) | Delta R22 | m2 (%) |
|---|---:|---:|---:|---:|
| baseline | -0.01104 ± 0.00281 | -1.20247 ± 0.30618 | -0.01281 ± 0.00215 | -1.40697 ± 0.23648 |
| nine_weak | -0.00002 ± 0.00281 | -0.00265 ± 0.30970 | -0.00663 ± 0.00215 | -0.73306 ± 0.23813 |

Matched comparison against generated photometry and generated selection (actual usability in both):

| Flow | Axis | Cases | Generated m (%) | Measured-input m (%) | Paired change (pp) |
|---|---|---:|---:|---:|---:|
| baseline | 1 | 40 | -0.05021 ± 0.30497 | -1.20247 ± 0.30618 | -1.15226 ± 0.03589 |
| nine_weak | 1 | 22 | +1.06895 ± 0.41895 | -0.06376 ± 0.41619 | -1.13271 ± 0.05691 |
| baseline | 2 | 40 | -0.32750 ± 0.24347 | -1.40697 ± 0.23648 | -1.07946 ± 0.02733 |
| nine_weak | 2 | 22 | +0.46789 ± 0.33558 | -0.57525 ± 0.31816 | -1.04314 ± 0.03773 |

Full matrices, all nine cuts, per-case values, jackknife errors and 32-draw subbank differences are in the JSON report.
Measured photometry changes under coherent shear. This intervention is not an additive causal error budget. Selection changes are already included; do not add the opposite-group selection term.
Paired whole-case bootstrap and jackknife; fits and measurement realizations fixed. Development evidence.
