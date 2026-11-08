# Selection versus scene sensitivity

Reference scene coefficient at radius3.5/magnitude24.5, all modelled neighbours. Not a measured or calibrated R11.
Cases: [1000].

| Reference coefficient | Generated pass rate | Measured pass rate | Measured minus generated |
|---|---:|---:|---:|
| -inf, 0.0 | +0.584363 | +0.578053 | -0.006310 |
| 0.0, 0.025 | +0.444175 | +0.442998 | -0.001177 |
| 0.025, 0.05 | +0.447938 | +0.444732 | -0.003206 |
| 0.05, 0.1 | +0.436773 | +0.435732 | -0.001040 |
| 0.1, 0.2 | +0.455410 | +0.459742 | +0.004331 |
| 0.2, 0.4 | +0.449454 | +0.457831 | +0.008378 |
| 0.4, 0.8 | +0.479085 | +0.494776 | +0.015691 |
| 0.8, inf | +0.545977 | +0.553362 | +0.007385 |

Selected-mean reference coefficient change: +0.003446.

| Control partition | Distribution | Within-cell | Exclusive support |
|---|---:|---:|---:|
| reference_coefficient_bins | +0.002953 | +0.000494 | +0.000000 |
| intrinsic_cells | +0.000218 | +0.003228 | -0.000000 |
| intrinsic_and_close050_cells | +0.000237 | +0.003221 | -0.000012 |

- Common-all-four actual usability; this does not test the usability classifier.
- Coarse intrinsic/crowding cells do not hold every flow input exactly fixed.
- No conditional-density or omitted-feature causal conclusion follows from association alone.
