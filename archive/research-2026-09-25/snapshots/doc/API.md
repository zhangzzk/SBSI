# SBSI API and scope

## Scope boundary

SBSI owns:

- finite scene priors and their guarded neighbour graph;
- conditional measurement-flow loading and evaluation;
- usable-measurement classifier loading and evaluation;
- measured-output selection and population normalization;
- fixed atom-aligned `R_blend` composition;
- likelihood caches, importance sampling, validation, and inference.

BlendEMU owns image simulation, measurement, simulation-catalogue production,
and response-emulator training. SBSI accepts the resulting catalogues and model
artifacts through explicit paths. Unmatched keyed rows are rejected or dropped
and reported; they are never represented by a zero response.

The archived training and response-development code under `archive/` is
provenance, not part of the supported API. See the
[archive map](REPOSITORY.md#research-archive).

## Release and model identities

The configured default is numerical pipeline `v1.1-infer` with likelihood
`v3.2-like`. `v1.2-infer` is the available tilted-stratified numerical
estimator.

`V3.6-like` is the retained development model set, pinned in
`configs/models_v3_6_like.json`: one common-descent flow, one physical-moment
disk-response emulator, and one 17-input smooth-crowding classifier. Its
primary cuts are strict measured radius>0.6 arcsec and MAG_AUTO<25.8.
The owner stopped further validation on 2026-09-20. See
[retained evidence](JOINT_CALIBRATION.md#retained-evidence).

`V3.5-like` remains the historical path-only model set:

1. lambda10 epoch154 joint four-output measurement flow;
2. trial9 response emulator used to build fixed atom-aligned `R_blend`;
3. equal-probability ensemble of three nine-input coherent-`U` classifiers.

It is defined in `configs/likelihood_v3_5_like.json` and `sbsi.models`. The name
must never select behavior by itself: effective configuration, feature and
target metadata, paths, and hashes remain authoritative. The model set has not
demonstrated 0.3% calibration and is not the default likelihood.

## V3.6-like model loading

Set `SBSI_CACHE_DIR` and `BLENDEMU_RUNS_DIR` before importing `sbsi.models`.
The first root contains the flow and classifier caches, the second the
BlendEMU disk-response run. Both default to the checkout's `models/` directory.

```python
from sbsi.models import get_model, load_detection_classifier, load_disk_response
from sbsi.measurement_model import load_measurement_model
from sbsi.crowding import FEATURES, classifier_features

paths = get_model("V3.6-like")
paths.validate()  # all four pinned artifact files
flow = load_measurement_model(paths.flow_checkpoints[0])
classifier = load_detection_classifier(paths)
disk_response = load_disk_response(paths)  # optional XGBoost dependency
```

`classifier_features(ids, context, target_shape, pairs, psf_radius)` builds
the 17 inputs in `FEATURES` order. Supply the original eight-column context,
aligned target shapes and the truth neighbour pairs in original convolved
major-axis coordinates. The helper preserves physical separations while
recomputing annuli and smooth crowding for the target shape. Parent identities
must be unique and every pair primary must belong to the parent set.

Use the response predictor on radius/flux from each joint flow draw and
compose disk velocity using `sbsi.disk_response_transport` with `mobius`.
`load_emulator` rejects this backend because its fixed additive interface
cannot represent it. The original six-arm evaluator and training recipes are
in the research archive. This runtime API does not promote V3.6-like into
`run_inference.py`; the configured likelihood contract below is unchanged.

## Likelihood contract

For scene atom `j` and shear `g`, the model composes

```text
p(U_j(g) | x_j(g))
p(e1, e2, FLUX_RADIUS, flux-from-MAG_AUTO | U_j(g), x_j(g))
R_blend,j [e_j(g) - e_j(0)]
```

`U` is the actual per-leg usable-measurement event. The classifier does not
replace measured selection. The magnitude and radius cuts are integrated over
the joint flow outputs (up to invertible coordinate transforms), so
`MAG_AUTO` and `FLUX_RADIUS` must remain in the measurement target. For the
historical V3.5-like sample:

```text
MAG_AUTO < 25.8
FLUX_RADIUS >= 0.75 arcsec
no measured-|e| cut
```

The blend term is evaluated once per atom at zero shear and applied as

```text
e_model,j(g) = e_flow,j(g) + R_blend,j [e_truth,j(g) - e_truth,j(0)].
```

Under selection, `R_blend` is averaged over the same passing population as the
flow outputs.

### Replacement fixed-cohort training contract

The 2026-09-14 replacement training run is deliberately distinct from the
configured V3.5-like likelihood. Its flow and BlendEMU response emulator use
objects selected once on the measured g=0 leg by strict
`MAG_AUTO < 25.8` and `FLUX_RADIUS > 0.60 arcsec`, with exact object keys
carried to the sheared leg and no analysis truth cut. The usable-event
classifier instead uses the complete simulation target-role parent and no
measured selection or analysis truth cut.

The blending-response cohort is additionally restricted to the simulator's
structural primary target role, identified by stable
`input_index < floor(N_generated / 2)`. Shape placeholders do not define that
role. Response-pair rows require finite, strictly interior ngmix shapes in
both label legs; this only makes the response target well defined and does not
alter the fixed-g0 cohort predicate.

Because the replacement flow is conditional on fixed cohort membership, it
must not be combined with the current per-leg measured-selection normalizer.
New model artifacts do not become a public/configured model set until a
separate fixed-cohort likelihood contract and rebuilt caches pass validation.

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

- `sbsi.scene_prior` — finite scene atoms and model views;
- `sbsi.catalogue_likelihood` — likelihood and model caches;
- `sbsi.catalogue_sampling` — proposals and importance sampling;
- `sbsi.catalogue_null` — score/information inference estimators;
- `sbsi.catalogue_blend` — atom-aligned response caches;
- `sbsi.selection_normalization` — exact and fitted selection normalization;
- `sbsi.measurement_model` and `sbsi.selection_model` — model loaders;
- `sbsi.shear_map` — the shared reduced-shear transformation.

`python -m sbsi show-model V3.5-like` prints the path preset. The production
inference entry point remains `scripts/run_inference.py`.

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
