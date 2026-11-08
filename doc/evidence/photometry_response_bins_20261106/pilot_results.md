# Photometry-bin response decomposition

Measured minus generated conditional-model response. Same common-U four-edge normalization as parent factorial.
Cases: [1000].

| Bins | Distribution term | Within-bin response term | Exclusive support | Total change |
|---|---:|---:|---:|---:|
| fine | -0.000286 (one scene) | +0.005588 (one scene) | +0.000787 (one scene) | +0.006089 (one scene) |
| fine_with_intrinsic_size | -0.000283 (one scene) | +0.005658 (one scene) | +0.000715 (one scene) | +0.006089 (one scene) |
| medium | -0.000257 (one scene) | +0.005743 (one scene) | +0.000603 (one scene) | +0.006089 (one scene) |
| medium_with_intrinsic_size | -0.000317 (one scene) | +0.005821 (one scene) | +0.000585 (one scene) | +0.006089 (one scene) |
| coarse | +0.000141 (one scene) | +0.005904 (one scene) | +0.000044 (one scene) | +0.006089 (one scene) |
| coarse_with_intrinsic_size | +0.000062 (one scene) | +0.005987 (one scene) | +0.000040 (one scene) | +0.006089 (one scene) |

All terms use finite conditional-flow shape transport. Scalar-coefficient decomposition is also preserved in JSON.

- Descriptive bin decomposition, not a unique causal attribution.
- Within-bin contrasts can include differing intrinsic properties and neighbour environments.
- Development scenes at fixed checkpoints; conditional SE excludes model-training uncertainty.
- Rare endpoint cut-crossers retain their original weights and all bin support.
