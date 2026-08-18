# V3 model artifacts

This directory holds the frozen **V3** release artifacts: the 16-seed measurement
flow ensemble and the companion BlendEMU blending emulator. Together they are what
`get_model("V3")` names.

```
models/
  ablation/   measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s{seed}_swaavg.pt   x16
  derisk/     v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json
  blendemu/   emulator_metadata_lsst_r_extnbr_v22.json
              classification_model_lsst_r_extnbr_ho.json
  SHA256SUMS
```

The 16 seeds are `SHAPE_SEEDS` in `sbsi/models.py`: 501, 502, 503 and 505–517.
Total about 93 MB.  The classification booster is the per-galaxy detection model named
by the metadata's `classification` task; `BlendingPredictor` loads it unconditionally,
so it must sit next to the metadata file.  Through it the emulator answers
`predict_detection(catalogue)`: the detection probability of each galaxy from its own
properties and its single nearest neighbour, considering neighbours only within the
classifier's 3-arcsec training aperture (galaxies with no such neighbour take the
isolated-galaxy branch).  The pair-level form is
`predict_on_pairs(pairs, task="detection")` — note that pair tables built for the
response emulator's 10-arcsec aperture are outside the classifier's training domain.

## Using these files

`sbsi/models.py` resolves the presets against this directory by default: with the
environment unset, `get_model("V3")` resolves entirely inside a clone of this
repository and needs no access to the original cluster paths. Identity is fixed
by the checksums below, not by the location.

Two environment variables override the roots, for cluster caches, custom
layouts, or non-editable installs:

```bash
export SBSI_CACHE_DIR=/path/to/caches    # ablation/ and derisk/ roots
export BLENDEMU_MODELS=/path/to/models   # the metadata + classifier directory
```

`BLENDEMU_ROOT` is not read for model resolution; it names the BlendEMU code
checkout (with `~/.config/sbsi/blendemu_root` as a file-based fallback when the
variable is unset). When `blendemu` is not otherwise importable, `load_emulator`
inserts that location into `sys.path` and imports it from there, so the emulator step
needs no `PYTHONPATH` entry of its own; the flow checkpoints need only SBSI and
torch. When an artifact named by a preset
is missing, `load_emulator` raises a `FileNotFoundError` naming it, rather than
deferring to BlendEMU's filename fallbacks.

## Integrity

Every file is listed in `SHA256SUMS`:

```bash
cd models && sha256sum -c SHA256SUMS
```

The emulator's hash is additionally pinned in `sbsi/models.py` as
`emulator_sha256`, and `ModelPaths.validate()` checks it. That pin, not the filename,
is what fixes the emulator's identity.

## What is *not* here

- **V3b.** `get_model("V3b")` names a different flow ensemble
  (`..._dom6x6_s{seed}_swaavg.pt`) and a different emulator. Those files are not in
  this directory, so on a fresh clone `load_emulator` reports them as missing; point
  `SBSI_CACHE_DIR` and `BLENDEMU_MODELS` at a tree that carries them (on this
  project's cluster, the original caches).
- **Training data and intermediate caches.** They live under `$DATA_DIR` and are not
  redistributable at this size.
- **Superseded checkpoints.** 77 earlier experimental flows that used to sit here were
  retired on 2026-08-17 to
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/retired_models_2026-08-17/`.
  No shipped code path ever read them.
