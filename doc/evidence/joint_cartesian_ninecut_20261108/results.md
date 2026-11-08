# Frozen joint Cartesian: full comparison

## Both components and all nine cuts

Same40 scenes, new paired-seed measured reference, independently selected
per-leg populations, generated photometry and predicted usability. Values are
m=100*(Rmeasured/Rmodel−1), in percent; errors are whole-scene jackknife standard
errors in percentage points. Both models use64 portable antithetic draws and
the same case/block seeds. No training or empirical calibration correction.

| Radius lower bound (pixels) | Magnitude upper bound | Joint m1 (%) | Joint m2 (%) | Latest m1 (%) | Latest m2 (%) |
|---:|---:|---:|---:|---:|---:|
| 2.8 | 24.9 | -0.3734 ±0.1538 | +0.8396 ±0.1816 | +0.5537 ±0.1553 | +0.3025 ±0.1793 |
| 2.8 | 25.0 | -0.3328 ±0.1528 | +0.9023 ±0.1871 | +0.5364 ±0.1544 | +0.3279 ±0.1838 |
| 2.8 | 25.1 | -0.1166 ±0.1530 | +1.0045 ±0.1856 | +0.6952 ±0.1554 | +0.3824 ±0.1831 |
| 3.0 | 24.9 | +0.6092 ±0.1455 | +2.7818 ±0.1703 | +0.5591 ±0.1447 | +0.2275 ±0.1662 |
| 3.0 | 25.0 | +0.6553 ±0.1329 | +2.7386 ±0.1729 | +0.5558 ±0.1314 | +0.2101 ±0.1690 |
| 3.0 | 25.1 | +0.8687 ±0.1383 | +2.7952 ±0.1768 | +0.7249 ±0.1381 | +0.2883 ±0.1729 |
| 3.2 | 24.9 | +1.0304 ±0.1913 | +3.0146 ±0.1898 | +0.9293 ±0.1908 | +0.6637 ±0.1881 |
| 3.2 | 25.0 | +1.0620 ±0.1813 | +3.0195 ±0.1851 | +0.9258 ±0.1799 | +0.7282 ±0.1834 |
| 3.2 | 25.1 | +1.2476 ±0.1902 | +3.0320 ±0.1836 | +1.0820 ±0.1911 | +0.7984 ±0.1832 |

## Joint nine-cut objective

- joint: +1.8034 ±0.1088%.
- latest: +0.6346 ±0.1101%.
- joint_minus_latest_pp: +1.1689 ±0.0696 percentage points.

## Central matrices and random-stream check

measured_matrix:

```
[[0.9048671238613875, 0.00032416559586618236], [-0.0002390198035154162, 0.9023800298477103]]
```

joint_matrix:

```
[[0.8989762978433261, 0.00245478809763969], [-0.0004927161725066392, 0.8783257918689055]]
```

latest_matrix:

```
[[0.8998659863046026, -0.002307930310482303], [0.0007549050540656707, 0.9004879858079566]]
```

The prior small recheck used historical CUDA draws. This run uses the same
portable stream as the latest flow. The old central estimate is retained:

- historical_cuda_m1_percent: +0.6514 ±0.1328.
- portable_m1_percent: +0.6553 ±0.1329.
- portable_minus_historical_pp: +0.0038 ±0.0072.

## Verification and scope

All80 case/axis output receipts, measured sums, fixed model identities and
half-bank identities passed; 168 independent scalar point/jackknife checks passed.
The A40-8Q pilot separately replays the historical stream and gates the full
run. Per-cut half-bank sensitivity, paired contrasts and confidence intervals
are preserved in `report.json`. The reported uncertainties condition on the
fitted models and existing scenes; they are not training-seed uncertainties or
fresh acceptance evidence. The held broad array remains paused.
[Response conventions](../../CONVENTIONS.md#6-response-estimators),
[protocol](rule.md#authorized-comparison).
