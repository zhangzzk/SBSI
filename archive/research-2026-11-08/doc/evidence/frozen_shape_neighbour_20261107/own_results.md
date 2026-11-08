# Frozen-shape photometry training on its own data

Measured minus generated mean neighbour score among galaxies passing the hard target cut (unsheared usable rows). Paired whole-case jackknife SEs; fixed fitted models. This is not shear response.

| Split | Untouched nine-cut start | Frozen control | Frozen + neighbour loss | Added loss minus frozen control | Added loss minus start |
|---|---:|---:|---:|---:|---:|
| train | +0.003322 ± 0.000423 | +0.003540 ± 0.000423 | +0.000877 ± 0.000423 | -0.002663 ± 0.000030 | -0.002446 ± 0.000027 |
| validation | +0.003148 ± 0.000523 | +0.003367 ± 0.000523 | +0.000762 ± 0.000533 | -0.002605 ± 0.000027 | -0.002386 ± 0.000026 |
