# Fixed-factor constant-shear comparison

Fixed-factor intervention effects; not standalone component errors. DeltaR=simulation-model; m=100(Rsim_jj/Rmodel_jj-1). Four reused development scenes; paired whole-case SEs conditional on fitted models. All modes use generated emulator photometry.

Target: radius>3px, magnitude<25; actual usability and generated selection. m in percent, errors in percentage points.

| Product | m1 | m2 |
|---|---:|---:|
| initial | -0.2840 ± 0.6015 | +0.3901 ± 0.4080 |
| control | +1.0564 ± 0.6024 | +1.3794 ± 0.4120 |
| weighted | +1.1300 ± 0.6094 | +1.7015 ± 0.4156 |
| photo_control_shape_initial | -0.2250 ± 0.5970 | +0.4497 ± 0.4076 |
| photo_initial_shape_control | +0.9959 ± 0.6068 | +1.3191 ± 0.4125 |
| photo_weighted_shape_control | +1.0322 ± 0.6076 | +1.3523 ± 0.4140 |
| photo_control_shape_weighted | +1.1551 ± 0.6042 | +1.7298 ± 0.4133 |

Changes in the flow contribution, in absolute response units:

| Contrast | Axis | Photometry at old shapes | Shapes at old photometry | Interaction | Total |
|---|---|---:|---:|---:|---:|
| response_regularization | 1 | +0.000225 ± 0.000053 | -0.011389 ± 0.000019 | +0.000000 ± 0.000002 | -0.011165 ± 0.000061 |
| response_regularization | 2 | +0.000235 ± 0.000011 | -0.008427 ± 0.000020 | +0.000004 ± 0.000004 | -0.008188 ± 0.000013 |
| neighbour_moment | 1 | -0.002268 ± 0.000072 | -0.000983 ± 0.000013 | +0.000005 ± 0.000001 | -0.003246 ± 0.000078 |
| neighbour_moment | 2 | -0.002250 ± 0.000075 | -0.002980 ± 0.000012 | +0.000011 ± 0.000004 | -0.005219 ± 0.000068 |

All cuts, flag modes, emulator/total response effects and jackknife SEs are retained in report.json. No model promotion or new training.
