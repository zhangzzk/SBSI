# Photometry and selection factorial

Weak nine-cut flow, common actual usability; equal A/B within-group opposite response. Errors=simulation-model; fixed-model case SE.
Cases: [1000, 1040, 1080, 1120]. Target radius>3pixels, magnitude<25.

| Inputs / selection | Model shape response | Shape error | Error including selection |
|---|---:|---:|---:|
| G_inputs_G_selection | +0.084958 ± 0.000316 | +0.005217 ± 0.001492 | +0.008142 ± 0.001773 |
| G_inputs_M_selection | +0.084784 ± 0.000337 | +0.005391 ± 0.001546 | +0.008316 ± 0.001844 |
| M_inputs_G_selection | +0.085613 ± 0.000361 | +0.004562 ± 0.001500 | +0.007487 ± 0.001780 |
| M_inputs_M_selection | +0.091514 ± 0.000417 | -0.001339 ± 0.001549 | +0.001586 ± 0.001842 |

G=generated; M=measured. Crossed branches are diagnostic substitutions.

| Change in model response | Value |
|---|---:|
| photometry_first | +0.000655 ± 0.000055 |
| photometry_after_membership | +0.006730 ± 0.000083 |
| membership_first | -0.000174 ± 0.000126 |
| membership_after_photometry | +0.005901 ± 0.000128 |
| symmetric_photometry | +0.003693 ± 0.000068 |
| symmetric_membership | +0.002863 ± 0.000126 |
| interaction | +0.006075 ± 0.000032 |
| diagonal_change | +0.006556 ± 0.000171 |

Measured selection response: +0.002925 ± 0.000418.
M/M change from the earlier draw bank: +0.000002 ± 0.000011.

## Coefficient and scene controls

Mean scalar emulator coefficient change: +0.006547 ± 0.000172.
Remaining finite shape-transport geometry change: +0.000009 ± 0.000011.

| Scene | G/G response | M/M response | Diagonal change | Interaction |
|---|---:|---:|---:|---:|
| 1000 | +0.084072 | +0.090161 | +0.006089 | +0.005982 |
| 1040 | +0.085601 | +0.091965 | +0.006365 | +0.006150 |
| 1080 | +0.084659 | +0.091557 | +0.006898 | +0.006053 |
| 1120 | +0.085503 | +0.092372 | +0.006869 | +0.006115 |

Delete-one-scene jackknife SEs: diagonal 0.000199, interaction 0.000037. These supplement the bootstrap SEs; neither includes model-training uncertainty.

Antithetic half-bank difference in diagonal change: -0.000005.

- Photometry swaps also condition shape on the substituted values.
- Measured masks on independent flow draws are diagnostics, not corrected predictions.
- Symmetric effects are algebraic model contrasts; interaction and both orders are retained.
- Measured neighbour selection response is separate; these contributions cannot be doubled into coherent full-population errors.
