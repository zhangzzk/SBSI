# Likelihood artifacts

`models/` contains the three small JSON artifacts used by the current
catalogue likelihood. The measurement-flow checkpoint is supplied separately;
no `.pt` checkpoint is bundled in this repository.

```text
models/
  blendemu/
    emulator_metadata_lsst_r_extnbr_v22.json
    classification_model_lsst_r_extnbr_ho.json
  derisk/v22_reweighted_vector_optuna30_all40_v1/
    best_weighted_model.json
  SHA256SUMS
```

## The `v3.2-like` likelihood release

`v3.2-like` names a likelihood composition, not a Python API version or an
SBSI package version. Its components are:

- the externally supplied seed-501 original-E measurement flow,
  `measurement_flow_mixed_g0_g005_E_s501_swaavg.pt`, with SHA-256
  `38a76bb9bbece61f403ce2781883f38779b419101d0e690439edffb82b93c51e`;
- the bundled BlendEMU metadata and its legacy classification booster. This
  detector has seven spin-0 inputs, uses the nearest neighbour inside its
  3-arcsec training aperture, and can therefore be cached across the
  shape-only shear views used by the likelihood; and
- the bundled response-regression artifact used to build an optional fixed,
  atom-aligned `R_blend` cache.

The seed-501 flow is intentionally explicit: likelihood-generated closure
mocks must be generated and scored with the same density. An ensemble is a
separate model-robustness study, not part of this likelihood identity.

The numerical pipeline using this likelihood is independently named
`v1.1-infer`. `configs/likelihood.json` records the likelihood release and
artifact identities; `configs/inference.json` records sampling and solver
choices. No likelihood or inference behavior should branch on either release
string alone: paths, hashes, feature metadata, and geometry are the executable
contract.

### Not the `V3.2` model preset

`v3.2-like` is deliberately distinct from `get_model("V3.2")`.
The latter is a path-only convenience preset for the four-seed original-E flow
ensemble, the response emulator, and an external transition-aware SBSI
detection classifier using shape-dependent, impact-ranked neighbours. The
`v3.2-like` catalogue likelihood instead uses one seed-501 flow and the legacy
spin-0 BlendEMU classifier described above. Do not substitute one name for the
other or infer likelihood behavior from a model-preset name.

## The `V3.3-like` validated model set

`V3.3-like` names the model composition validated on the cases-40--59
ConstGold response ladder on 2026-09-02:

- the single seed-501 500/500 Flow-E checkpoint with both full-matrix shape and
  magnitude/log-size derivative supervision,
  `measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt`, SHA-256
  `4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73`;
- the frozen transition-aware SBSI detector, SHA-256
  `9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455`,
  evaluated separately on each sheared view with the 3-arcsec, impact-ranked,
  exponent-one neighbour rule; and
- the same response emulator and atom-aligned `R_blend` convention as V3.2.

`get_model("V3.3-like")` is a path-only convenience for these artifacts.  The
set has one flow seed, so it has no flow-seed ensemble uncertainty.  It is not
the likelihood selected by `configs/likelihood.json`: the production inference
runner still uses the deployed `v3.2-like` spin-0 detector contract and cannot
silently reuse a transition probability across shear views.  Promoting
`V3.3-like` to an inference likelihood therefore requires an explicit config
and shear-dependent detector integration, not a release-string substitution.

## Supplying the flow

Pass the seed-501 checkpoint as an explicit user path in the likelihood
configuration or command invocation. A typical external layout is:

```text
/path/to/models/mixed_shear_cde/
  measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
```

The checkpoint is 5,576,968 bytes in the frozen release. Verify both its size
and digest before building model, proposal, or mock caches.

BlendEMU is needed only to load the JSON classifier/response artifacts. Install
the BlendEMU checkout into the same environment as SBSI; the measurement flow
itself needs only SBSI and PyTorch.

## Integrity

Verify the bundled files from this directory:

```bash
sha256sum -c SHA256SUMS
```

Verify the separately supplied flow against the full digest printed above:

```bash
sha256sum /path/to/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
```

`SHA256SUMS` also records the external filename, size, and digest as comments;
the three ordinary checksum lines are intentionally limited to files present in
the checkout so the standard verification command remains portable.

Training catalogues, scene priors, derived response caches, proposal caches,
and scheduler outputs are user data and do not belong in `models/`.
