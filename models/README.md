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
    classifier_replicate.pt same classifier retrained with another seed (check only, not used)
    evidence/               training and validation records behind the choice of these files
  SHA256SUMS
```

Total about 30 MB. The flow and classifier take eight inputs per galaxy: its true
ellipticity, Sersic index, magnitude, circularized radius and three flux-weighted measures of how crowded
its surroundings are (`sbsi.crowding.V39_FEATURES`). The blending emulator adds, for every
neighbour within 10 arcsec, how much that neighbour shifts the measured shape. It is
applied to every flow draw, so the flow and blending parts of the response come from the
same draws.

## Training domain

All three parts were trained on the same 200 simulated scenes (cases 0–199), rendered
with 0.2 arcsec pixels, a Moffat PSF of FWHM 0.73 arcsec (beta 2.224), zero point 30 and
pixel noise rms 0.312. Images taken under other conditions are outside the training domain.
No constant-shear scene was used for training or for choosing a checkpoint.

| Part | Scenes | Galaxies |
|---|---|---|
| Flow | 0–199, split at random into 160 training and 40 validation | every validly measured galaxy with true r < 30; no measured cuts |
| Classifier | trained on 40–99, checkpoint chosen on 100–119 | every generated target galaxy; no truth cuts. It predicts passing flux radius > 0.6 arcsec (3 pixels) and `MAG_AUTO` < 25.8 |
| Blending emulator | fitted on 40–199, checked on 0–39 | target galaxies with true r < 26, a usable shape at zero shear and a positive radius, paired with every neighbour within 10 arcsec; labels from a shear step of 0.2 |

Typical flow inputs (mean ± spread): true r 25.4 ± 1.3, circularized radius
0.29 ± 0.27 arcsec, Sersic index 1.5 ± 1.0. Two consequences:

- The emulator was trained only to true r = 26, so for fainter galaxies the blending part
  is used beyond its training range.
- The recorded bias, m = +0.02% ± 0.27% (bootstrap over scenes, shear on g1 only), was
  measured on scenes 40–119, which overlap the training scenes. It is a development check,
  not an independent test, and its error leaves out training and sampling noise.

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
