# Conditional probability and shape diagnostics

Per-row conditional probability squared error; independent 32-draw banks remove integration variance. Shape errors here are single-leg conditional usable-population means, not coherent response or m.

| Split | Hard-cut score: control | Hard-cut score: weighted | Paired change |
|---|---:|---:|---:|
| train | +0.0713774 ± 0.0002013 | +0.0712391 ± 0.0002012 | -0.0001383 ± 0.0000096 |
| validation | +0.0716847 ± 0.0001948 | +0.0715439 ± 0.0001953 | -0.0001408 ± 0.0000089 |

Errors are paired delete-one-case standard errors. All nine cuts, both original legs, hard/soft targets, smoothing sensitivities and selected shape moments are retained in probability_report.json. Shape weights are fixed; selected-shape behavior need not be fixed.
