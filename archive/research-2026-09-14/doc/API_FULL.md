# SBSI API — scope, releases, and public contract

`AGENTS.md` carries the short project rules. This document defines the code and
data boundary behind them.

## Scope boundary

- SBSI owns conditional measurement flows, detection classifiers, response
  prediction, finite-scene likelihoods, sampling, validation, and shear
  inference.
- BlendEMU owns image simulation, measurement, simulation-catalogue production,
  and response-emulator training. SBSI consumes its supported catalogue and
  model interfaces; it does not copy those implementations.
- SBSI prepares response-emulator inputs from an explicit truth catalogue. This
  includes validation, neighbour finding, pair selection, training-matched
  cuts and scaling, and keyed alignment of predictions.
- A new simulator, measurement, or emulator-training feature belongs in
  BlendEMU unless the project owner explicitly changes this boundary.

## Release identity

SBSI has one inference workflow with two independent scientific release labels:

| role | release | authoritative definition |
| --- | --- | --- |
| numerical pipeline | `v1.1-infer` | `configs/inference.json` |
| catalogue likelihood | `v3.2-like` | `configs/likelihood.json` |

The labels are provenance, not dispatch keys. Code validates the effective
configuration, artifact hashes, target order, feature metadata, geometry, and
cache identities; it must not select behavior from a release string alone.
They are also separate from the package version in `pyproject.toml`.

`v3.2-like` is not the path-only `get_model("V3.2")` preset. The likelihood
uses the single seed-501 original-E flow and the bundled seven-input spin-0
BlendEMU classifier. The `V3.2` preset points to a four-flow ensemble and an
external transition-aware detector. `models/README.md` records the complete
distinction and the frozen artifact hashes.

Model presets remain optional compatibility conveniences. `get_model("V3.1")`,
`get_model("V3.2")`, and `get_model("V3.3-like")` return paths only; no
catalogue, selection, response, or inference behavior may branch on those
names. `V3.3-like` is the validated single-seed 500/500 Flow-E plus
transition-aware detector set, not the likelihood currently selected by
`configs/likelihood.json`. Broken presets for retired checkpoint trees are
intentionally not kept.

## General contract

- Every training, validation, prior, and inference catalogue is supplied by the
  user as a DataFrame or explicit path. Project data paths are never API
  defaults.
- Flow checkpoints, classifier artifacts, response models, derived caches, and
  outputs are explicit and provenance-checked.
- Model support comes from checkpoint metadata or an explicit configuration,
  never from a filename.
- Unmatched keyed rows are rejected or intentionally dropped and reported.
  Alignment failure is never represented as a zero blending response.
- Common random numbers are retained across all finite-difference views. A
  sampled atom set and its proposal probabilities stay fixed throughout an
  inference stencil.
- Loaded mocks and caches must match their declared inputs, models, selection,
  shear transform, target order, and implementation identity.

## Public Python surface

The package root is intentionally small and lazy. It exports
`EmulatorPairingConfig`, `ModelPaths`, `ResponsePredictor`, `example_path`,
`get_model`, `load_catalogue`, `load_emulator`, `prepare_forward_catalogue`,
`predict_blend_response`, and `sample_measurement`.

Specialized workflows import their implementation modules directly:

- `sbsi.flow` — conditional-flow training and explicit tuning;
- `sbsi.scene_prior` — finite catalogue atoms and the guarded neighbour graph;
- `sbsi.catalogue_likelihood` — flow, detection, measured-selection, and
  population-normalization terms;
- `sbsi.catalogue_sampling` — defensive importance sampling;
- `sbsi.catalogue_blend` — fixed atom-aligned external response caches;
- `sbsi.catalogue_null` — score/information diagnostics and exact local
  validation;
- `sbsi.catalogue_closure` and `sbsi.image_closure` — likelihood- and
  image-generated closure adapters; and
- `sbsi.shear_map`, `sbsi.measurement_model`, and `sbsi.selection_model` — the
  shared scientific transformations and learned-model loaders.

Reusable scientific behavior belongs in these modules, not in scheduler
wrappers or one-off scripts.

## Operational entry points

The production path is deliberately small:

- `configs/inference.json` defines `v1.1-infer` sampling, finite differences,
  optimization, precision, and chunking;
- `configs/likelihood.json` defines `v3.2-like` model artifacts, target order,
  shear transform, geometry, and observing conditions;
- `scripts/run_inference.py` is the single inference runner; and
- `jobs/job_inference.sh` is the cluster-local reference wrapper.

Reusable inputs are prepared separately:

- `scripts/build_scene_prior.py` builds the guarded finite scene store;
- `scripts/build_catalogue_blend_response.py` builds the optional fixed,
  atom-aligned `R_blend` cache; and
- `scripts/prepare_image_closure_mock.py` converts one declared BlendEMU image
  leg into the flow's measured-target convention.

`scripts/run_catalogue_closure.py` and the retained exact, powered, and nonzero
closure programs are validation tools. They are not alternative production
pipelines or release choices.

## Likelihood and response invariants

For prior atoms `z_j` with masses `pi_j`, retained measurements use

```text
A_i(g) = sum_j pi_j P_det(S_g z_j) p_flow(xhat_i - b_j(g) | S_g z_j)
B_W(g) = sum_j pi_j P_det(S_g z_j) P_pass,j(g)
p(xhat_i | detected, W, g) = A_i(g) / B_W(g)

b_j(g) = R_blend,j [e(S_g z_j) - e(z_j)]
```

The current project shear transform changes intrinsic ellipticity only. Flux,
size, positions, pair separations, and neighbour membership remain fixed;
magnification is outside this likelihood. A measured cut cancels from the
numerator for retained objects, but its population probability does not cancel
from `B_W`.

Response reporting always uses both components:

```text
R_model = R_flow + R_blend
m = R_sim / R_model - 1
```

An atom with no supported neighbour has a physical zero external response and
is reported separately from a failed catalogue join.

## Current scientific limitation

The `v3.2-like` detector has only seven spin-0 inputs, so the software may cache
its probability across shape-only shear views. That is a model contract, not
evidence that image detection is shear invariant. Precision image closure
still requires a detector/usable-measurement model that represents the missing
detection-selection response. A likelihood-generated closure validates the
likelihood implementation; it does not by itself validate population transfer
or image selection.
