# Photometry and selection factorial

Weak nine-cut flow, common actual usability; equal A/B within-group opposite response. Errors=simulation-model; fixed-model case SE.
Cases: [1000]. Target radius>3pixels, magnitude<25.

| Inputs / selection | Model shape response | Shape error | Error including selection |
|---|---:|---:|---:|
| G_inputs_G_selection | +0.084072 (one scene) | +0.006129 (one scene) | +0.009023 (one scene) |
| G_inputs_M_selection | +0.083713 (one scene) | +0.006489 (one scene) | +0.009382 (one scene) |
| M_inputs_G_selection | +0.084539 (one scene) | +0.005663 (one scene) | +0.008556 (one scene) |
| M_inputs_M_selection | +0.090161 (one scene) | +0.000040 (one scene) | +0.002933 (one scene) |

G=generated; M=measured. Crossed branches are diagnostic substitutions.

| Change in model response | Value |
|---|---:|
| photometry_first | +0.000466 (one scene) |
| photometry_after_membership | +0.006448 (one scene) |
| membership_first | -0.000359 (one scene) |
| membership_after_photometry | +0.005623 (one scene) |
| symmetric_photometry | +0.003457 (one scene) |
| symmetric_membership | +0.002632 (one scene) |
| interaction | +0.005982 (one scene) |
| diagonal_change | +0.006089 (one scene) |

Measured selection response: +0.002893 (one scene).
M/M change from the earlier draw bank: +0.000033 (one scene).

## Coefficient and scene controls

Mean scalar emulator coefficient change: +0.006056 (one scene).
Remaining finite shape-transport geometry change: +0.000033 (one scene).

| Scene | G/G response | M/M response | Diagonal change | Interaction |
|---|---:|---:|---:|---:|
| 1000 | +0.084072 | +0.090161 | +0.006089 | +0.005982 |

Antithetic half-bank difference in diagonal change: +0.000055.

- Photometry swaps also condition shape on the substituted values.
- Measured masks on independent flow draws are diagnostics, not corrected predictions.
- Symmetric effects are algebraic model contrasts; interaction and both orders are retained.
- Measured neighbour selection response is separate; these contributions cannot be doubled into coherent full-population errors.
