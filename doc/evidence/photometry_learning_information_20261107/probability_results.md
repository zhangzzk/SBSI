# Conditional probability and shape diagnostics

Per-row conditional probability squared error; independent 32-draw banks remove integration variance. Shape errors here are single-leg conditional usable-population means, not coherent response or m.

| Split | Hard-cut score: control | Hard-cut score: weighted | Paired change |
|---|---:|---:|---:|
| train | +0.0714983 ± 0.0002008 | +0.0713930 ± 0.0002010 | -0.0001054 ± 0.0000080 |
| validation | +0.0718287 ± 0.0001964 | +0.0717178 ± 0.0001945 | -0.0001109 ± 0.0000065 |

Errors are paired delete-one-case standard errors. All nine cuts, both original legs, hard/soft targets, smoothing sensitivities and selected shape moments are retained in probability_report.json. Shape weights are fixed; selected-shape behavior need not be fixed.
