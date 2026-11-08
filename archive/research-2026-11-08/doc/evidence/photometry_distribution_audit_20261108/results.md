# Conditional photometry distribution audit

## Validation: central cut

Passing-rate errors are model minus measured, percentage points. B error is measured minus model in reference-neighbour-coefficient units. These are not shear biases.

| Model | Radius contribution (pp) | Flux contribution (pp) | Association contribution (pp) | Joint passing error (pp) | Selected B error |
|---|---:|---:|---:|---:|---:|
| baseline | +0.3097 ±0.0320 | -0.1079 ±0.0227 | -0.3354 ±0.0264 | -0.1337 ±0.0437 | +0.0028 ±0.0004 |
| nine_weak | +0.2285 ±0.0319 | -0.1514 ±0.0227 | -0.3002 ±0.0264 | -0.2231 ±0.0438 | +0.0033 ±0.0004 |
| neighbour_nine | +0.2283 ±0.0319 | -0.1304 ±0.0227 | -0.2791 ±0.0263 | -0.1812 ±0.0434 | +0.0028 ±0.0004 |
| latest | +0.2295 ±0.0319 | -0.1440 ±0.0228 | -0.3114 ±0.0261 | -0.2260 ±0.0434 | +0.0016 ±0.0004 |

## Where the central selected-neighbour error comes from

Largest absolute contributions are descriptive rankings on these validation scenes. Each model’s 25 cells add exactly to its full selected-B error.

### baseline

- true_radius_x_score/q3_q5: +0.000787 ±0.000148
- true_radius_x_score/q2_q5: +0.000564 ±0.000171
- true_radius_x_score/q4_q5: +0.000489 ±0.000134
- true_radius_x_score/q1_q1: +0.000210 ±0.000027
- true_radius_x_score/q2_q1: +0.000207 ±0.000041

### nine_weak

- true_radius_x_score/q3_q5: +0.000924 ±0.000148
- true_radius_x_score/q2_q5: +0.000715 ±0.000172
- true_radius_x_score/q4_q5: +0.000595 ±0.000134
- true_radius_x_score/q1_q5: +0.000203 ±0.000124
- true_radius_x_score/q1_q1: +0.000201 ±0.000027

### neighbour_nine

- true_radius_x_score/q3_q5: +0.000810 ±0.000146
- true_radius_x_score/q2_q5: +0.000601 ±0.000171
- true_radius_x_score/q4_q5: +0.000527 ±0.000134
- true_radius_x_score/q1_q1: +0.000179 ±0.000026
- true_radius_x_score/q2_q1: +0.000175 ±0.000041

### latest

- true_radius_x_score/q3_q5: +0.000555 ±0.000147
- true_radius_x_score/q2_q5: +0.000399 ±0.000171
- true_radius_x_score/q4_q5: +0.000279 ±0.000133
- true_radius_x_score/q1_q1: +0.000137 ±0.000026
- true_radius_x_score/q2_q1: +0.000136 ±0.000040

## Limits

- Association is within each specified population, including remaining context heterogeneity; not an identified per-object conditional copula.
- One measured realization per row; individual prediction residuals are not individual systematic errors.
- Original usable training-domain populations; no classifier-error or coherent-shear attribution.
- Fixed models and samples; scene jackknife omits retraining variability. Original data are not newly paired-seed shape fits.
