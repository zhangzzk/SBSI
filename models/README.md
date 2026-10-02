# Model artifacts

## V3.9-like

The retained model is **F8 flow + primary C8 classifier + original disk emulator**.
[The manifest](../configs/models_v3_9_like.json) pins all artifact hashes,
features, cuts, classifier seed and development evidence.

Set `SBSI_CACHE_DIR` to the external cache root containing
`v36_f8_20260924_v1/` and `v36_classifier_f8_inputs_20260925_v1/`.
Set `BLENDEMU_RUNS_DIR` to the root containing
`disk_response_moment_20260918_v1/`. Artifact directory names retain their
original provenance; checkpoint bytes are unchanged.

```bash
python -m sbsi show-model V3.9-like
python -m sbsi validate-model V3.9-like
```

Both flow and classifier use the eight physical-crowding features in
[the feature contract](../doc/V39_MODEL.md#feature-and-prediction-contract).
The old17-input `classifier_features` helper is for V3.6 only. Loading and
prediction are supported; V3.9 is not yet integrated into the historical
V3.6 inference preparation. See [scope](../doc/V39_MODEL.md#loading-and-inference-boundary).

## Historical V3.6-like

The historical development model is one common-descent flow, one physical-moment
disk-response emulator, and one 17-input smooth-crowding classifier. Full
paths, hashes, cuts, and evidence are pinned in
[the model manifest](../configs/models_v3_6_like.json).

Set `SBSI_CACHE_DIR` to the external SBSI cache root containing
`flow_shape_augmentation_20260918_v1/` and
`scene_classifier_geometry_20260918_v1/`. Set `BLENDEMU_RUNS_DIR` to the root
containing `disk_response_moment_20260918_v1/`. If unset, each root defaults
to this `models/` directory; large checkpoints are not bundled.

```bash
python -m sbsi show-model V3.6-like
python -m sbsi validate-model V3.6-like
```

Use `sbsi.models.load_disk_response` for its disk-velocity predictor and
`sbsi.crowding.classifier_features` for its state-dependent inputs. See
[the API](../doc/API.md#v36-like-model-loading) and
[scientific status](../doc/JOINT_CALIBRATION.md#retained-evidence).

## Historical V3.5-like

`get_model("V3.5-like")` still identifies lambda10 epoch154, trial9 additive
response, and the equal-probability three-seed nine-input coherent-U ensemble.
Its configuration remains `configs/likelihood_v3_5_like.json`, including the
0.75-arcsec selection. Historical details are preserved in the
[previous model guide](../archive/research-2026-09-20/models/README.md).

## Bundled defaults

The small JSON artifacts here support the configured `v3.2-like` default.
Their hashes are listed in `SHA256SUMS`. Model-set naming does not change
the configured likelihood or numerical estimator.
