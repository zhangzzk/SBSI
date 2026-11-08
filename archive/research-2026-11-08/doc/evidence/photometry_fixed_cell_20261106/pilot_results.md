# Fixed-photometry response audit

Measured minus generated selection-weighted scalar opposite-neighbour coefficient; original or fixed fine-cell photometry.
Cases: [1000].

| Fine-bin term | Original photometry | Exactly fixed photometry | Original minus fixed |
|---|---:|---:|---:|
| distribution | -0.000292 (one scene) | -0.000294 (one scene) | +0.000002 (one scene) |
| within_bin | +0.005567 (one scene) | +0.005609 (one scene) | -0.000042 (one scene) |
| exclusive | +0.000782 (one scene) | +0.000787 (one scene) | -0.000005 (one scene) |
| diagonal_change | +0.006056 (one scene) | +0.006102 (one scene) | -0.000046 (one scene) |

| Scene | Original total contrast | Fixed total contrast | Original within-bin | Fixed within-bin |
|---|---:|---:|---:|---:|
| 1000 | +0.006056 | +0.006102 | +0.005567 | +0.005609 |

- Fixed-cell inputs are diagnostic substitutions, not corrected predictions.
- Fine bins hold photometry exactly fixed; coarse aggregates contain multiple fixed fine-cell values.
- Truth-context differences include target properties and neighbour properties.
- Fixed-model development-scene uncertainties; no training uncertainty or independent acceptance.
