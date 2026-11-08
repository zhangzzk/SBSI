# Model artifacts

## Current GalSBI models

The repository includes both exact flow checkpoints and their shared GalSBI
C8 classifier and disk emulator (including metadata):

- `like-1.0-galsbi`: current fiducial, control NLL plus weak nine-cut response.
- `galsbi-multistage`: retained weighted multistage comparison, SHA43673e5d.

Their manifests are `configs/models_like_1_0_galsbi.json` and
`configs/models_galsbi_multistage.json`. With no external-root overrides,
`python -m sbsi validate-model like-1.0-galsbi galsbi-multistage` checks the
included bytes. Both use the standard `load_measurement_model` entry point.
The multistage flow requires an explicit reference neighbour coefficient;
missing values are rejected, never substituted by observed-photometry response.
See [current model/inference boundary](../doc/CURRENT_INFERENCE.md#models-and-compatibility).

## Historical compatibility

Small bundled JSON defaults and `SHA256SUMS` support the configured finite-
catalogue likelihood. Historical V3.5/V3.6/V3.9 model presets and artifact
hashes remain unchanged; their larger checkpoints use explicit external roots.
The 17-input V3.6 preparation cannot be relabelled as V3.9. See
[history](../doc/HISTORY.md#earlier-model-and-finite-catalogue-work) and
[the preserved model guide](../archive/research-2026-11-08/snapshots/models/README.md).
