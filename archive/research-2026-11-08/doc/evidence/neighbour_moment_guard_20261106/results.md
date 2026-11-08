# Nine-cut response plus selected-neighbour moment

DeltaR=simulation-model; m=100(Rsim_jj/Rmodel_jj-1). Whole-case paired bootstrap/jackknife, fixed fits; four development scenes. All three modes retain generated emulator photometry.

Target: radius>3px and magnitude<25. m in percent, SE in percentage points.

| Endpoint | Predicted usability m1 | Actual usability m1 | Predicted usability m2 | Actual usability m2 |
|---|---:|---:|---:|---:|
| initial | -0.5112 ± 0.6172 | -0.2840 ± 0.6032 | +0.2167 ± 0.4194 | +0.3901 ± 0.4143 |
| previous_nine | +0.8196 ± 0.6201 | +1.0564 ± 0.6041 | +1.2025 ± 0.4215 | +1.3794 ± 0.4184 |
| control | +0.8196 ± 0.6201 | +1.0564 ± 0.6041 | +1.2025 ± 0.4215 | +1.3794 ± 0.4184 |
| weighted | +0.8812 ± 0.6274 | +1.1300 ± 0.6114 | +1.5126 ± 0.4275 | +1.7015 ± 0.4219 |

Paired effect of the added moment in absolute response units (weighted minus control). Component changes are model contributions, not standalone component errors. Residual change is the negative sum of the two component changes.

| Axis | Usability / selection | Flow contribution change | Emulator contribution change | Residual change |
|---|---|---:|---:|---:|
| 1 | predicted / generated | -0.003230 ± 0.000075 | +0.002682 ± 0.000006 | +0.000549 ± 0.000079 |
| 1 | actual / generated | -0.003246 ± 0.000078 | +0.002594 ± 0.000013 | +0.000652 ± 0.000085 |
| 1 | actual / measured | -0.001131 ± 0.000014 | +0.000397 ± 0.000003 | +0.000734 ± 0.000013 |
| 2 | predicted / generated | -0.005201 ± 0.000067 | +0.002456 ± 0.000003 | +0.002745 ± 0.000065 |
| 2 | actual / generated | -0.005219 ± 0.000067 | +0.002379 ± 0.000010 | +0.002840 ± 0.000072 |
| 2 | actual / measured | -0.003142 ± 0.000013 | +0.000173 ± 0.000002 | +0.002969 ± 0.000014 |

All endpoints were fixed before response evaluation. Full matrices, all nine cuts, standalone probes and paired changes are retained in JSON.
