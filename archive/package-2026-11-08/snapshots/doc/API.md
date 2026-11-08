# SBSI API and scope

## Scope boundary

SBSI owns conditional flows, usable-event classifiers, response prediction,
validation, population priors, selected likelihoods and Bayesian inference.
BlendEMU owns image simulation, measurement, simulation-catalogue production
and response-emulator training. Catalogue keys must be aligned explicitly;
unmatched rows are rejected or dropped and reported.

## Current model loading

```python
from sbsi.models import get_model, load_detection_classifier, load_disk_response
from sbsi.measurement_model import load_measurement_model

paths = get_model("like-1.0-galsbi")
paths.validate()
flow = load_measurement_model(paths.flow_checkpoints[0])
classifier = load_detection_classifier(paths)
disk_response = load_disk_response(paths)
```

The bundled artifacts resolve under `models/` unless `SBSI_CACHE_DIR` or
`BLENDEMU_RUNS_DIR` overrides their roots. The physical flow models radius,
flux and Cartesian shapes jointly. Its eight informative raw inputs match the
current inference summary; the appended control column is zero.
`galsbi-multistage` additionally requires its explicit reference-neighbour
coefficient and is rejected by the current inference entry until that adapter
is validated. See [model contracts](CURRENT_INFERENCE.md#models-and-compatibility).

## Likelihood contract

The classifier predicts usability; the flow predicts joint measured outputs.
The disk emulator uses output-conditioned photometry and Möbius transport.
Measured-output cuts and population normalization remain explicit. A disk
velocity coefficient is not an additive response matrix. The continuous
selected likelihood is defined in [the inference mathematics](INFERENCE.md#selected-likelihood).

Older V3.5/V3.6/V3.9 presets, additive response helpers and finite-catalogue
configurations remain available for compatibility. They do not select the
current inference model or make old feature caches compatible. Their exact
contracts are retained in the [historical API](../archive/research-2026-11-08/snapshots/doc/API.md#release-and-model-identities)
and [history](HISTORY.md#earlier-model-and-finite-catalogue-work).

## Public Python surface

The package root exports only lightweight prediction helpers:

- `EmulatorPairingConfig`
- `ModelPaths`
- `ResponsePredictor`
- `example_path`
- `get_model`
- `load_catalogue`
- `load_emulator`
- `prepare_forward_catalogue`
- `predict_blend_response`
- `sample_measurement`

Specialized workflows import their modules directly:

- `sbsi.galsbi_scene_prior` and `sbsi.galsbi_prior` — continuous population prior;
- `sbsi.galsbi_summary` — compressed disk likelihood;
- `sbsi.galsbi_target_coordinates` and `sbsi.mcmc_proposals` — posterior sampling coordinates and proposals;
- `sbsi.scene_prior` — finite scene atoms and model views;
- `sbsi.catalogue_likelihood` — likelihood and model caches;
- `sbsi.catalogue_sampling` — proposals and importance sampling;
- `sbsi.catalogue_null` — score/information inference estimators;
- `sbsi.catalogue_blend` — atom-aligned response caches;
- `sbsi.selection_normalization` — exact and fitted selection normalization;
- `sbsi.measurement_model` and `sbsi.selection_model` — model loaders;
- `sbsi.shear_map` — the shared reduced-shear transformation.

`python -m sbsi show-model like-1.0-galsbi` prints the current model identity.
The continuous inference entry is `scripts/run_galsbi_inference.py`;
`scripts/run_inference.py` retains the finite-catalogue compatibility interface.

## Input and cache rules

- Catalogues and artifact paths are user inputs; project data paths are not API
  defaults.
- Caches record their input, model, configuration, implementation, and random
  stream identities.
- Common random numbers are retained across every finite-difference view.
- A sampled atom set and proposal probabilities remain fixed across a stencil.
- Loaded caches must match their declared scene, flow, classifier, selection,
  shear transform, target order, and implementation.
- Existing output paths are not overwritten.

The cache formats and scalable workflow are specified in
[`CATALOGUE_PRIOR.md`](CATALOGUE_PRIOR.md). The numerical estimator is specified
in [`INFERENCE.md`](INFERENCE.md) and [`MATH.md`](MATH.md).
