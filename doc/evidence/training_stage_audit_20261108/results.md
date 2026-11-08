# Training-stage audit with paired-seed measurements

## Full 40-scene comparison

| Recipe | Central m1 (%) | Central m2 (%) | Nine-cut RMS (%) |
|---|---:|---:|---:|
| initial_nll | -0.50484 ±0.13105 | +0.02729 ±0.16848 | 0.39999 ±0.06862 |
| latest | +0.55576 ±0.13143 | +0.21011 ±0.16898 | 0.63457 ±0.11008 |

## Matched four-scene stage comparison

The following rows use only the same four scenes; do not compare their absolute values to the full40 table.

| Recipe | Central m1 (%) | Central m2 (%) | Nine-cut RMS (%) |
|---|---:|---:|---:|
| initial_nll | -0.85017 ±0.39263 | -0.39632 ±0.67060 | 0.64857 ±0.18839 |
| control_nll | -0.91118 ±0.38930 | -0.90619 ±0.66287 | 0.86002 ±0.25893 |
| neighbour_nll | -0.92600 ±0.38823 | -0.99016 ±0.66063 | 0.91840 ±0.26374 |
| control_nine | +0.38554 ±0.39755 | +0.04296 ±0.66616 | 0.53431 ±0.25735 |
| neighbour_nine | +0.39930 ±0.38944 | -0.01628 ±0.66761 | 0.55598 ±0.26050 |
| latest | +0.21821 ±0.39116 | -0.21382 ±0.66524 | 0.44285 ±0.25694 |

Errors are whole-scene jackknife standard errors. Normal generated selection/predicted usability, independent per-leg normalization, m=100*(Rmeasured/Rmodel−1).
Independent scalar diagonal checks: 2016. Latest-model/new-measurement replay passes. No model evaluation or retraining.
[Conventions](../../CONVENTIONS.md#6-response-estimators).
Reused development scenes; historical caches without original data hashes are hashed now, with original source/model receipts and measured-sum replay. Stage sequence necessity is not established by unrelated checkpoints.

## All nine cuts on the full 40 scenes

| Cut | Initial NLL m1 (%) | Initial NLL m2 (%) | Latest m1 (%) | Latest m2 (%) |
|---|---:|---:|---:|---:|
| mag24.9_radius_gt0.56 | -0.62731 | +0.06526 | +0.55374 | +0.30254 |
| mag25_radius_gt0.56 | -0.68733 | +0.08276 | +0.53641 | +0.32790 |
| mag25.1_radius_gt0.56 | -0.56722 | +0.13279 | +0.69517 | +0.38236 |
| mag24.9_radius_gt0.60 | -0.46928 | +0.04361 | +0.55907 | +0.22750 |
| mag25_radius_gt0.60 | -0.50484 | +0.02729 | +0.55576 | +0.21011 |
| mag25.1_radius_gt0.60 | -0.36380 | +0.10864 | +0.72486 | +0.28833 |
| mag24.9_radius_gt0.64 | -0.01071 | +0.50953 | +0.92928 | +0.66372 |
| mag25_radius_gt0.64 | -0.03987 | +0.57941 | +0.92575 | +0.72819 |
| mag25.1_radius_gt0.64 | +0.09871 | +0.65934 | +1.08200 | +0.79839 |
