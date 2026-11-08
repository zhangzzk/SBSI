# Full joint response with measured photometry and selection

4 scenes; actual usability and cuts independently per shear leg.

Target: radius>3 pixels and magnitude<25. Absolute response error is simulation minus model. m is in percent; its SE is in percentage points.

| Flow | Delta R11 | m1 (%) | Delta R22 | m2 (%) |
|---|---:|---:|---:|---:|
| baseline | -0.01312 ± 0.00522 | -1.42819 ± 0.56884 | -0.00085 ± 0.00302 | -0.09296 ± 0.33199 |
| nine_weak | -0.00208 ± 0.00521 | -0.22906 ± 0.57345 | +0.00539 ± 0.00303 | +0.59598 ± 0.33484 |

Matched comparison against generated photometry and generated selection (actual usability in both):

| Flow | Axis | Cases | Generated m (%) | Measured-input m (%) | Paired change (pp) |
|---|---|---:|---:|---:|---:|
| baseline | 1 | 4 | -0.19246 ± 0.60686 | -1.42819 ± 0.56884 | -1.23573 ± 0.13980 |
| nine_weak | 1 | 4 | +0.95441 ± 0.61411 | -0.22906 ± 0.57345 | -1.18347 ± 0.14059 |
| baseline | 2 | 4 | +1.00219 ± 0.42064 | -0.09296 ± 0.33199 | -1.09516 ± 0.10436 |
| nine_weak | 2 | 4 | +1.63483 ± 0.42041 | +0.59598 ± 0.33484 | -1.03885 ± 0.10420 |

Full matrices, all nine cuts, per-case values, jackknife errors and 32-draw subbank differences are in the JSON report.
Measured photometry changes under coherent shear. This intervention is not an additive causal error budget. Selection changes are already included; do not add the opposite-group selection term.
Paired whole-case bootstrap and jackknife; fits and measurement realizations fixed. Development evidence.
