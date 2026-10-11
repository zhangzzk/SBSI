# V3.9-like model artifacts

This directory holds the frozen **V3.9-like** release artifacts. Together they are what
`get_model("V3.9-like")` names.

```
models/
  v3_9_like/
    flow.pt                 measurement flow (shape, size and flux given truth and crowding)
    classifier.pt           probability that a galaxy is detected with a usable measurement
    blending/model.json     blending-response emulator (xgboost)
    blending/metadata.json  its feature and target description
  SHA256SUMS
```

Total about 30 MB. The flow and classifier take eight inputs per galaxy: its true
ellipticity, Sersic index, magnitude, circularized radius and three counts of how crowded
its surroundings are (`sbsi.crowding.V39_FEATURES`). The blending emulator adds, for every
neighbour within 10 arcsec, how much that neighbour shifts the measured shape. It is
applied to every flow draw, so the flow and blending parts of the response come from the
same draws.

## Using these files

`sbsi/models.py` anchors preset paths on the imported SBSI checkout, so
`get_model("V3.9-like")` finds this directory from any working directory and needs no
configuration. Loading needs SBSI, PyTorch and xgboost only; BlendEMU is not required.

## Integrity

Every file is listed in `SHA256SUMS`:

```bash
cd models && sha256sum -c SHA256SUMS
```

The same hashes are pinned in [`configs/models_v3_9_like.json`](../configs/models_v3_9_like.json),
and `ModelPaths.validate()` checks them. Those pins, not the filenames, fix the model's
identity; the config also records where each file was copied from.

## What is *not* here

- **Earlier models.** The V3 16-flow ensemble and its BlendEMU emulator, and the V3.5-/V3.6-like
  presets, were removed when V3.9-like replaced them. They remain in the git history.
- **Training data and intermediate caches.** They live under `$DATA_DIR` and are not
  redistributable at this size.
