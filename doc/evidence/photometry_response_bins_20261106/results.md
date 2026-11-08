# Photometry-bin response decomposition

Measured minus generated conditional-model response. Same common-U four-edge normalization as parent factorial.
Cases: [1000, 1040, 1080, 1120].

| Bins | Distribution term | Within-bin response term | Exclusive support | Total change |
|---|---:|---:|---:|---:|
| fine | -0.000204 ± 0.000281 | +0.005938 ± 0.000241 | +0.000822 ± 0.000011 | +0.006556 ± 0.000171 |
| fine_with_intrinsic_size | -0.000246 ± 0.000323 | +0.006001 ± 0.000277 | +0.000801 ± 0.000014 | +0.006556 ± 0.000171 |
| medium | -0.000171 ± 0.000264 | +0.006094 ± 0.000224 | +0.000633 ± 0.000010 | +0.006556 ± 0.000171 |
| medium_with_intrinsic_size | -0.000232 ± 0.000318 | +0.006160 ± 0.000270 | +0.000628 ± 0.000010 | +0.006556 ± 0.000171 |
| coarse | +0.000174 ± 0.000262 | +0.006337 ± 0.000225 | +0.000045 ± 0.000003 | +0.006556 ± 0.000171 |
| coarse_with_intrinsic_size | +0.000045 ± 0.000341 | +0.006467 ± 0.000303 | +0.000043 ± 0.000003 | +0.006556 ± 0.000171 |

All terms use finite conditional-flow shape transport. Scalar-coefficient decomposition is also preserved in JSON.

- Descriptive bin decomposition, not a unique causal attribution.
- Within-bin contrasts can include differing intrinsic properties and neighbour environments.
- Development scenes at fixed checkpoints; conditional SE excludes model-training uncertainty.
- Rare endpoint cut-crossers retain their original weights and all bin support.
