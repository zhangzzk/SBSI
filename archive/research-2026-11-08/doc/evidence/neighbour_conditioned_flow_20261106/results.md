# Neighbour-conditioned flow: four-scene test

DeltaR=simulation-model; m=100(Rsim_jj/Rmodel_jj-1). Paired whole-case SE with fixed fits; four development scenes, not acceptance.

Target: measured radius>3px and magnitude<25. Values below are m in percent; errors are percentage points.

| Flow | Predicted usability m1 | True usability m1 | Predicted usability m2 | True usability m2 |
|---|---:|---:|---:|---:|
| baseline | -0.4351 ± 0.6153 | -0.1925 ± 0.5991 | +0.8178 ± 0.4201 | +1.0022 ± 0.4170 |
| control_nll | -0.4964 ± 0.6113 | -0.2614 ± 0.5990 | +0.3017 ± 0.4137 | +0.4870 ± 0.4099 |
| neighbour_nll | -0.5112 ± 0.6109 | -0.2840 ± 0.5972 | +0.2167 ± 0.4141 | +0.3901 ± 0.4092 |
| control_nine | +0.8058 ± 0.6216 | +1.0507 ± 0.6082 | +1.2624 ± 0.4204 | +1.4466 ± 0.4201 |
| neighbour_nine | +0.8196 ± 0.6138 | +1.0564 ± 0.5981 | +1.2025 ± 0.4162 | +1.3794 ± 0.4132 |

Reference-coefficient selection errors below are diagnostics, not calibrated R11:

| Flow | Measured minus generated selected-mean score |
|---|---:|
| control_nll | +0.002785 ± 0.000076 |
| neighbour_nll | +0.002484 ± 0.000082 |
| control_nine | +0.003375 ± 0.000076 |
| neighbour_nine | +0.003112 ± 0.000081 |

All nine cuts, full response components, true-selection rows, paired changes and standalone response probes are retained in report.json.
No checkpoint was chosen by these results. The static new feature does not implement neighbour-induced selection response.
