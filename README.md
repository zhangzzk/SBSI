# Simulation-Based Shear Inference

SBSI evaluates a finite-scene likelihood for weak-lensing shear. The active
repository contains one path: scene-prior preparation, joint measurement-flow
likelihoods, usable-measurement probabilities, measured selection,
atom-aligned `R_blend`, importance sampling, and inference.

## Identities

The numerical and model identities are independent:

| Role | Identity | Status |
| --- | --- | --- |
| Default numerical estimator | `v1.1-infer` | configured in `configs/inference.json` |
| Tilted-stratified estimator | `v1.2-infer` | available, not default |
| Default likelihood | `v3.2-like` | configured in `configs/likelihood.json` |
| Development model set | `V3.5-like` | path-only, not 0.3%-validated |

`V3.5-like` is exactly:

- lambda10 epoch154 `physical_disk_affine` measurement flow;
- joint measured outputs `(e1, e2, FLUX_RADIUS, flux-from-MAG_AUTO)`, with
  magnitude and log-radius obtained by invertible transforms;
- trial9 atom-aligned `R_blend`; and
- the arithmetic mean of three coherent-`U` classifiers trained with seeds
  20260913, 20260914, and 20260915 from the same nine conditions, including
  `R_blend`.

The sample keeps `MAG_AUTO < 25.8`, convolved SExtractor
`FLUX_RADIUS >= 0.75 arcsec`, and no measured-`|e|` cut. Magnitude and radius
remain stochastic flow outputs because they define selection. The V3.5-like
config applies the equivalent stored-coordinate bounds: radius at least 3.75
pixels and flux-from-`MAG_AUTO` above 47.863 at zero point 30.

The model set is pinned in
[`configs/likelihood_v3_5_like.json`](configs/likelihood_v3_5_like.json) and is
available through `get_model("V3.5-like")`. Set `SBSI_CACHE_DIR` to the external
model store when using that path preset.

## Scientific status

SBSI has not yet demonstrated 0.3% image-level calibration. On the reused
cases120--139 mechanism screen, the V3.5-like classifier with the frozen joint
flow gave `m=-0.372 +/- 0.223%`; its truth-population and measurement-residual
contributions were `-0.549pp` and `+0.177pp`, so it failed the anti-cancellation
gate. With actual per-leg usable flags, the broader cases40--139 flow oracle
gave `m=-0.129 +/- 0.125%`. This identifies learned usable-probability weighting
as the main remaining development problem without proving final calibration.

## Run inference

```bash
python scripts/run_inference.py \
  --inference-config configs/inference.json \
  --likelihood-config configs/likelihood_v3_5_like.json \
  --scene-store /path/to/scene \
  --measurement-model /path/to/epoch154.pt \
  --blend-response-cache /path/to/rblend-cache \
  --model-cache /path/to/model-cache \
  --proposal-cache /path/to/proposal-cache \
  --output /new/output/path
```

All catalogues, large checkpoints, caches, and output paths are explicit.
Configurations pin their scientific identity and hashes; they do not hide
project data paths. `jobs/job_inference.sh` is the minimal scheduler wrapper
for the configured default likelihood.

Supporting scripts build the finite scene prior, `R_blend` cache, selection
normalization, model-QMC cache, partitions, and final summary. See
[`doc/INFERENCE.md`](doc/INFERENCE.md) for the command sequence.

## Repository layout

- `sbsi/` — supported likelihood and inference modules.
- `scripts/` — active preparation, inference, combination, and summary commands.
- `jobs/` — scheduler wrappers for cache building and inference.
- `tests/` — core regression tests only.
- `configs/` — numerical, likelihood, and prior identities.
- `doc/` — the seven canonical documents listed in `AGENTS.md`.
- `archive/research-2026-09-14/` — preserved training, response-tuning,
  ConstGold, plotting, and failed-branch provenance; not supported code.

## Install and test

```bash
python -m pip install --config-settings editable_mode=compat -e /path/to/blendemu
python -m pip install -e /path/to/SBSI
python -m pytest -q
```

BlendEMU owns image simulation, measurement, simulation-catalogue production,
and emulator training. SBSI consumes its catalogues and trained response
artifacts.

## Documentation

- [`doc/API.md`](doc/API.md) — scope and public contract.
- [`doc/INFERENCE.md`](doc/INFERENCE.md) — releases and inference workflow.
- [`doc/CATALOGUE_PRIOR.md`](doc/CATALOGUE_PRIOR.md) — finite prior and caches.
- [`doc/MATH.md`](doc/MATH.md) — score/information derivation.
- [`doc/CONVENTIONS.md`](doc/CONVENTIONS.md) — catalogue and response definitions.
- [`doc/ENVIRONMENT.md`](doc/ENVIRONMENT.md) — interpreters, tests, and scheduler.
- [`doc/WORKLOG.md`](doc/WORKLOG.md) — concise active change log.
