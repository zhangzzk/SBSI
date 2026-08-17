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
Total about 93 MB.  The classification booster is the per-pair detection model named
by the metadata's `classification` task; `BlendingPredictor` loads it unconditionally,
so it must sit next to the metadata file.  Through it the emulator also answers
`predict_on_pairs(pairs, task="detection")`, the detection probability of each
paired galaxy.

## Using these files

`sbsi/models.py` resolves preset paths through environment variables, so point
them at this directory:

```bash
export SBSI_CACHE_DIR="$PWD/models"
export BLENDEMU_MODELS="$PWD/models/blendemu"
```

With those set, `get_model("V3")` resolves entirely inside this repository and needs
no access to the original cluster paths. Unset, the presets fall back to their
frozen locations under `/project/ls-gruen/users/zekang.zhang/sbsi_caches` and
`~/blendemu/models`, which reproduces the milestone on the machine where it was made.

The emulator itself is loaded by BlendEMU, so `load_emulator` still requires BlendEMU
on the `PYTHONPATH`. The flow checkpoints need only SBSI and torch.

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
  this directory, so V3b resolves only against the original cluster paths.
- **Training data and intermediate caches.** They live under `$DATA_DIR` and are not
  redistributable at this size.
- **Superseded checkpoints.** 77 earlier experimental flows that used to sit here were
  retired on 2026-08-17 to
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/retired_models_2026-08-17/`.
  No shipped code path ever read them.
