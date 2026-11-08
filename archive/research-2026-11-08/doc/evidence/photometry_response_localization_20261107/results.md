# Selected-response localization completed

New minus control model response; common latent draws, symmetric decomposition. Contributions use global per-leg normalization, not subgroup response. Errors are paired scene jackknife with models fixed.

Central cut; extra-loss model minus control. Entries below are response changes, not bias percentages.

| Contribution | Axis 1 | Axis 2 |
|---|---:|---:|
| cut_membership | +0.00172636 ± 0.00003607 | +0.00157655 ± 0.00002385 |
| output_value | +0.00091963 ± 0.00000173 | +0.00092544 ± 0.00000323 |
| normalization | -0.00326834 ± 0.00002090 | -0.00325652 ± 0.00001896 |

Full arm/cut/mode, true-size and neighbour-quartile tables are in report.json. Decomposition is exact for the common-draw pairing and uses symmetric averages; it is not a unique causal attribution. No new model training or simulated images.
