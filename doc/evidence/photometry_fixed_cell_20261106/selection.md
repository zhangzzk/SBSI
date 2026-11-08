# Selection versus scene sensitivity

Reference scene coefficient at radius3.5/magnitude24.5, all modelled neighbours. Not a measured or calibrated R11.
Cases: [1000, 1040, 1080, 1120].

| Reference coefficient | Generated pass rate | Measured pass rate | Measured minus generated |
|---|---:|---:|---:|
| -inf, 0.0 | +0.586404 ± 0.001329 | +0.579783 ± 0.001696 | -0.006621 ± 0.000710 |
| 0.0, 0.025 | +0.444315 ± 0.000271 | +0.441828 ± 0.001182 | -0.002487 ± 0.001027 |
| 0.025, 0.05 | +0.447892 ± 0.000794 | +0.445353 ± 0.000505 | -0.002540 ± 0.000427 |
| 0.05, 0.1 | +0.435409 ± 0.000938 | +0.434263 ± 0.001294 | -0.001146 ± 0.000702 |
| 0.1, 0.2 | +0.456068 ± 0.000495 | +0.460852 ± 0.000746 | +0.004784 ± 0.001097 |
| 0.2, 0.4 | +0.449123 ± 0.000598 | +0.461445 ± 0.001518 | +0.012322 ± 0.001895 |
| 0.4, 0.8 | +0.482632 ± 0.001080 | +0.496031 ± 0.001516 | +0.013400 ± 0.001317 |
| 0.8, inf | +0.547746 ± 0.001080 | +0.555571 ± 0.001210 | +0.007825 ± 0.000818 |

Selected-mean reference coefficient change: +0.003599 ± 0.000078.

| Control partition | Distribution | Within-cell | Exclusive support |
|---|---:|---:|---:|
| reference_coefficient_bins | +0.003060 ± 0.000107 | +0.000539 ± 0.000046 | +0.000000 ± 0.000000 |
| intrinsic_cells | +0.000195 ± 0.000305 | +0.003404 ± 0.000325 | +0.000000 ± 0.000000 |
| intrinsic_and_close050_cells | +0.000199 ± 0.000302 | +0.003401 ± 0.000322 | -0.000001 ± 0.000003 |

- Common-all-four actual usability; this does not test the usability classifier.
- Coarse intrinsic/crowding cells do not hold every flow input exactly fixed.
- No conditional-density or omitted-feature causal conclusion follows from association alone.
