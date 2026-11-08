# Conditional cut-probability training with frozen shapes

DeltaR=simulation-model; m=100(Rsim_jj/Rmodel_jj-1). Whole-case paired bootstrap/jackknife, fixed fits; four development scenes. All three modes retain generated emulator photometry.

Target: radius>3px and magnitude<25. m in percent, SE in percentage points.

| Endpoint | Predicted usability m1 | Actual usability m1 | Predicted usability m2 | Actual usability m2 |
|---|---:|---:|---:|---:|
| initial | -0.5112 ± 0.6172 | -0.2840 ± 0.6032 | +0.2167 ± 0.4194 | +0.3901 ± 0.4143 |
| previous_nine | +0.8196 ± 0.6201 | +1.0564 ± 0.6041 | +1.2025 ± 0.4215 | +1.3794 ± 0.4184 |
| control | +0.7410 ± 0.6284 | +0.9728 ± 0.6119 | +1.1172 ± 0.4259 | +1.2865 ± 0.4226 |
| weighted | +0.6377 ± 0.6226 | +0.8747 ± 0.6077 | +1.0025 ± 0.4240 | +1.1727 ± 0.4204 |

Paired effect of the added conditional probability score in absolute response units (weighted minus control). Component changes are model contributions, not standalone component errors. Residual change is the negative sum of the two component changes.

| Axis | Usability / selection | Flow contribution change | Emulator contribution change | Residual change |
|---|---|---:|---:|---:|
| 1 | predicted / generated | -0.000792 ± 0.000070 | +0.001714 ± 0.000010 | -0.000923 ± 0.000061 |
| 1 | actual / generated | -0.000854 ± 0.000069 | +0.001726 ± 0.000014 | -0.000872 ± 0.000059 |
| 1 | actual / measured | +0.000314 ± 0.000009 | +0.000309 ± 0.000004 | -0.000624 ± 0.000011 |
| 2 | predicted / generated | -0.000697 ± 0.000108 | +0.001718 ± 0.000004 | -0.001020 ± 0.000106 |
| 2 | actual / generated | -0.000720 ± 0.000121 | +0.001729 ± 0.000009 | -0.001010 ± 0.000116 |
| 2 | actual / measured | +0.000321 ± 0.000007 | +0.000316 ± 0.000003 | -0.000637 ± 0.000006 |

All endpoints were fixed before response evaluation. Full matrices, all nine cuts, standalone probes and paired changes are retained in JSON.

The initial row is the historical NLL model; previous_nine is the exact training start. Control and weighted are the matched photometry-only continuations.
