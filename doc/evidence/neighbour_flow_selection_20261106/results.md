# Selection association on original flow data

Measured minus generated selected mean reference coefficient; not a shear response. Whole-case paired bootstrap and jackknife with fixed fits. All rows are usable in their original leg. Score-at-mean predictions are input interventions.

Target: radius>3 pixels and magnitude<25. Each split uses500,000 fixed rows. Errors below are case-bootstrap SEs.

| Flow | Training g0 score error | Validation g0 score error | Validation pooled score error |
|---|---:|---:|---:|
| baseline | +0.003320 ± 0.000416 | +0.003117 ± 0.000513 | +0.002793 ± 0.000399 |
| nine_weak | +0.003803 ± 0.000417 | +0.003608 ± 0.000512 | +0.003293 ± 0.000399 |
| control_nll | +0.002994 ± 0.000416 | +0.002780 ± 0.000512 | +0.002463 ± 0.000398 |
| neighbour_nll | +0.002678 ± 0.000414 | +0.002520 ± 0.000513 | +0.002190 ± 0.000399 |
| control_nine | +0.003591 ± 0.000417 | +0.003382 ± 0.000512 | +0.003058 ± 0.000399 |
| neighbour_nine | +0.003322 ± 0.000414 | +0.003148 ± 0.000511 | +0.002823 ± 0.000400 |
| neighbour_nll_mean_score | +0.003094 ± 0.000415 | +0.002832 ± 0.000517 | +0.002515 ± 0.000402 |
| neighbour_nine_mean_score | +0.003600 ± 0.000416 | +0.003355 ± 0.000516 | +0.003029 ± 0.000405 |

Validation likelihood change from adding neighbour information (matched training, lower is better):

| Stage | Joint NLL change | Photometry NLL change | Conditional-shape NLL change |
|---|---:|---:|---:|
| nll | -0.003668 ± 0.000337 | -0.001885 ± 0.000086 | -0.001783 ± 0.000307 |
| nine | +0.000295 ± 0.000370 | +0.001722 ± 0.000252 | -0.001427 ± 0.000273 |

The two mean-score arms replace only the extra input with its training mean; they diagnose learned dependence and are not calibrated models.
All nine cuts, original g05 rows, paired changes, quintile profiles and half-bank checks are in report.json.
