# Independent primary and neighbour shear on four development fields

2,048 fixed probability-sampled primaries, two paired optimizer seeds. R11 residual = simulation minus model. Each ± is a whole-source-case jackknife SE; sampling SEs condition on the four original images, and must not be added to case SEs. Central cut: measured/generated radius > 3 pixels and magnitude < 25 as indicated.

| Flow | Usability / selection | Primary residual ± case SE | Neighbour residual ± case SE | Coherent residual ± case SE | Sampling SEs: primary / neighbour / coherent |
|---|---|---:|---:|---:|---|
| initial | predicted_generated | +0.048007 ± 0.074482 | +0.089605 ± 0.092495 | +0.137612 ± 0.072287 | 0.08125 / 0.08859 / 0.12849 |
| initial | actual_generated | +0.051430 ± 0.079891 | +0.091416 ± 0.088185 | +0.142846 ± 0.069911 | 0.08097 / 0.08889 / 0.12843 |
| initial | actual_measured_selection | +0.071640 ± 0.093748 | +0.080822 ± 0.073416 | +0.152462 ± 0.093755 | 0.07691 / 0.08710 / 0.12176 |
| control | predicted_generated | +0.059145 ± 0.075395 | +0.090416 ± 0.092658 | +0.149561 ± 0.072561 | 0.08127 / 0.08860 / 0.12851 |
| control | actual_generated | +0.063641 ± 0.080365 | +0.092283 ± 0.088246 | +0.155924 ± 0.070122 | 0.08097 / 0.08890 / 0.12847 |
| control | actual_measured_selection | +0.082910 ± 0.093080 | +0.081634 ± 0.073572 | +0.164543 ± 0.092735 | 0.07688 / 0.08712 / 0.12178 |
| weighted | predicted_generated | +0.062955 ± 0.075642 | +0.087876 ± 0.092383 | +0.150830 ± 0.070489 | 0.08120 / 0.08858 / 0.12845 |
| weighted | actual_generated | +0.067777 ± 0.079545 | +0.090284 ± 0.088469 | +0.158061 ± 0.066871 | 0.08091 / 0.08889 / 0.12841 |
| weighted | actual_measured_selection | +0.084051 ± 0.092903 | +0.081287 ± 0.073595 | +0.165338 ± 0.092517 | 0.07688 / 0.08711 / 0.12177 |

See report.json for all cuts, R21, paired covariance, optimizer and spatial sampling diagnostics. This g1-only variance pilot cannot establish R22 or independent calibration. No model or larger sample is selected.
