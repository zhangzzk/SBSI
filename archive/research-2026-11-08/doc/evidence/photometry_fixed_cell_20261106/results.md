# Fixed-photometry response audit

Measured minus generated selection-weighted scalar opposite-neighbour coefficient; original or fixed fine-cell photometry.
Cases: [1000, 1040, 1080, 1120].

| Fine-bin term | Original photometry | Exactly fixed photometry | Original minus fixed |
|---|---:|---:|---:|
| distribution | -0.000209 ± 0.000280 | -0.000207 ± 0.000280 | -0.000002 ± 0.000001 |
| within_bin | +0.005938 ± 0.000242 | +0.005929 ± 0.000244 | +0.000009 ± 0.000018 |
| exclusive | +0.000817 ± 0.000011 | +0.000824 ± 0.000011 | -0.000007 ± 0.000001 |
| diagonal_change | +0.006547 ± 0.000172 | +0.006547 ± 0.000162 | -0.000000 ± 0.000017 |

| Scene | Original total contrast | Fixed total contrast | Original within-bin | Fixed within-bin |
|---|---:|---:|---:|---:|
| 1000 | +0.006056 | +0.006102 | +0.005567 | +0.005609 |
| 1040 | +0.006386 | +0.006372 | +0.006468 | +0.006446 |
| 1080 | +0.006872 | +0.006886 | +0.006399 | +0.006397 |
| 1120 | +0.006871 | +0.006827 | +0.005385 | +0.005330 |

- Fixed-cell inputs are diagnostic substitutions, not corrected predictions.
- Fine bins hold photometry exactly fixed; coarse aggregates contain multiple fixed fine-cell values.
- Truth-context differences include target properties and neighbour properties.
- Fixed-model development-scene uncertainties; no training uncertainty or independent acceptance.
