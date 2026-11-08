# Frozen conditional shape: photometry-only training

DeltaR=simulation-model; m=100(Rsim_jj/Rmodel_jj-1). Whole-case paired bootstrap/jackknife, fixed fits; four development scenes. All three modes retain generated emulator photometry.

Target: radius>3px and magnitude<25. m in percent, SE in percentage points.

| Endpoint | Predicted usability m1 | Actual usability m1 | Predicted usability m2 | Actual usability m2 |
|---|---:|---:|---:|---:|
| initial | -0.5112 ± 0.6172 | -0.2840 ± 0.6032 | +0.2167 ± 0.4194 | +0.3901 ± 0.4143 |
| previous_nine | +0.8196 ± 0.6201 | +1.0564 ± 0.6041 | +1.2025 ± 0.4215 | +1.3794 ± 0.4184 |
| control | +0.7410 ± 0.6284 | +0.9728 ± 0.6119 | +1.1172 ± 0.4259 | +1.2865 ± 0.4226 |
| weighted | +0.7701 ± 0.6260 | +1.0149 ± 0.6113 | +1.1451 ± 0.4227 | +1.3307 ± 0.4169 |

Paired effect of the added moment in absolute response units (weighted minus control). Component changes are model contributions, not standalone component errors. Residual change is the negative sum of the two component changes.

| Axis | Usability / selection | Flow contribution change | Emulator contribution change | Residual change |
|---|---|---:|---:|---:|
| 1 | predicted / generated | -0.003256 ± 0.000047 | +0.002997 ± 0.000009 | +0.000259 ± 0.000043 |
| 1 | actual / generated | -0.003271 ± 0.000037 | +0.002897 ± 0.000017 | +0.000374 ± 0.000034 |
| 1 | actual / measured | -0.000824 ± 0.000006 | +0.000249 ± 0.000004 | +0.000575 ± 0.000009 |
| 2 | predicted / generated | -0.003248 ± 0.000038 | +0.003000 ± 0.000008 | +0.000248 ± 0.000036 |
| 2 | actual / generated | -0.003299 ± 0.000068 | +0.002908 ± 0.000014 | +0.000391 ± 0.000057 |
| 2 | actual / measured | -0.000811 ± 0.000006 | +0.000264 ± 0.000004 | +0.000547 ± 0.000007 |

All endpoints were fixed before response evaluation. Full matrices, all nine cuts, standalone probes and paired changes are retained in JSON.

The initial row is the historical NLL model; previous_nine is the exact training start. Control and weighted are the matched photometry-only continuations.
