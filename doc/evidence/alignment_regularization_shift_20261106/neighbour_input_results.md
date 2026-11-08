# Matched neighbour-response input diagnostics

Common-all-four usability; equal average of separately normalized A/B opposite-group responses. Errors=simulation-model; case bootstrap SE; fixed checkpoints.

Target: measured/generated radius>3pixels (0.6arcsec), magnitude<25 as indicated.
Measured shape response: +0.090175 ± 0.001643; measured selection response: +0.002925 ± 0.000418.

| Inputs | Predicted shape response | Shape error | Error including selection |
|---|---:|---:|---:|
| generated_baseline | +0.085487 ± 0.000317 | +0.004688 ± 0.001488 | +0.007613 ± 0.001769 |
| measured_photometry_baseline | +0.091775 ± 0.000428 | -0.001600 ± 0.001546 | +0.001325 ± 0.001838 |
| generated_nine_weak | +0.084958 ± 0.000316 | +0.005217 ± 0.001492 | +0.008142 ± 0.001773 |
| measured_photometry_nine_weak | +0.091512 ± 0.000427 | -0.001337 ± 0.001546 | +0.001588 ± 0.001839 |
| measured_shape_and_photometry | +0.091449 ± 0.000392 | -0.001274 ± 0.001458 | +0.001651 ± 0.001762 |

## Paired changes in model response

| Flow | Generated to measured photometry | Conditional to measured shape |
|---|---:|---:|
| baseline | +0.006288 ± 0.000176 | -0.000326 ± 0.000115 |
| nine_weak | +0.006554 ± 0.000175 | -0.000063 ± 0.000115 |

## Limits

- Four development scenes, not independent acceptance.
- Generated-to-measured-photometry substitution changes photometry and selection as well as conditioning; not a unique causal separation of those effects.
- Conditional-to-measured-shape comparison retains the same measured photometry and selection weights.
- These within-group opposite responses are not the globally normalized coherent contribution and must not be doubled.

Generated/measured selected-mass ratios on the common cohort: baseline +1.000196 ± 0.001625, nine_weak +0.998324 ± 0.001624. Matching counts does not establish matching response-weighted input distributions.
