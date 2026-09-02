# Simulation-Based Shear Inference (SBSI)

SBSI is a scientific library for weak-lensing shear inference with a finite
scene prior, a conditional measurement flow, detection probabilities, measured
selection, and an optional atom-aligned blending response.

The current releases are:

| role | release | definition |
| --- | --- | --- |
| numerical inference pipeline | `v1.1-infer` | [`configs/inference.json`](configs/inference.json) |
| catalogue likelihood | `v3.2-like` | [`configs/likelihood.json`](configs/likelihood.json) |

These are independent scientific release labels. `v1.1-infer` identifies how
an explicit likelihood and prior are sampled and solved; `v3.2-like` identifies
the flow, detector, response artifact, and geometry used by that likelihood.
Neither label replaces the SBSI package version, and neither should control code
behavior without validating its configuration and artifact hashes.

`v3.2-like` is also not the same thing as the path-only
`get_model("V3.2")` preset. The distinction and frozen hashes are documented in
[`models/README.md`](models/README.md).

The validated seed-501 500/500 Flow-E plus transition-aware detector set is
named `V3.3-like` and is available as the path-only
`get_model("V3.3-like")` preset. It is not yet the likelihood selected by
`configs/likelihood.json`; see [`models/README.md`](models/README.md) for the
exact artifacts and promotion boundary.

## Inference workflow

The production workflow has one entry point:

```bash
python scripts/run_inference.py \
  --inference-config configs/inference.json \
  --likelihood-config configs/likelihood.json \
  --scene-store /path/to/scene \
  --measurement-model /path/to/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --model-cache /path/to/model-cache \
  --proposal-cache /path/to/proposal-cache \
  --output /new/output/path
```

All catalogues, checkpoints, caches, and outputs are explicit user inputs. The
repository configurations define the named scientific setup; they do not hide
project data paths. The cluster-local reference wrapper is
[`jobs/job_inference.sh`](jobs/job_inference.sh).

Supporting commands prepare reusable inputs:

- `scripts/build_scene_prior.py` builds a guarded finite scene prior.
- `scripts/build_catalogue_blend_response.py` builds the optional fixed
  atom-aligned `R_blend` cache.
- `scripts/prepare_image_closure_mock.py` maps one declared image leg into the
  measurement-flow target convention.

The inference result records the release labels, complete effective
configuration, model/cache/input hashes, random streams, object partition, and
per-object score/information moments needed for exact partition combination.
Common random numbers are retained across every finite-difference view and
nested draw rung.

## Library areas

Reusable behavior lives in `sbsi/`:

1. `sbsi.flow` trains and tunes conditional measurement flows.
2. `sbsi.scene_prior`, `sbsi.catalogue_likelihood`, and
   `sbsi.catalogue_sampling` implement the finite-prior likelihood and defensive
   importance sampler.
3. `sbsi.catalogue_blend` and `sbsi.response` handle the external blending
   response without copying BlendEMU's simulation or training implementation.
4. `sbsi.catalogue_closure` and `sbsi.image_closure` provide likelihood- and
   image-closure adapters and validation.

## Image simulation and measurement

BlendEMU owns image rendering, measurement, simulation-catalogue construction,
and emulator training. SBSI consumes its supported catalogues and trained
artifacts; it does not carry a second copy of that implementation. The example
Slurm wrapper [`examples/job_blendemu.sh`](examples/job_blendemu.sh) runs the
BlendEMU production steps from a user-owned configuration.

## Flow training and tuning

Copy [`examples/flow_training.yaml`](examples/flow_training.yaml), replace its
catalogue and artifact paths, and run inside an appropriate compute allocation:

```bash
python -m sbsi flow --config my_flow.yaml --mode train
python -m sbsi flow --config my_flow.yaml --mode tune
```

An editable install provides the equivalent `sbsi` command. Training writes the
declared checkpoint and averaged `*_swaavg.pt` checkpoint. Tuning evaluates an
explicit candidate list against a separate validation catalogue and writes a
ranked JSON manifest. Existing artifacts are never overwritten.

## Installation

Use one Python environment for SBSI and its BlendEMU dependency:

```bash
python -m pip install --config-settings editable_mode=compat -e /path/to/blendemu
python -m pip install -e /path/to/SBSI
```

BlendEMU is required only when its classifier or response emulator is loaded.
The measurement-flow checkpoint itself needs SBSI and PyTorch. Do not add
checkout paths to `PYTHONPATH` or depend on the process working directory.

The repository bundles only three small JSON likelihood artifacts. The
seed-501 measurement flow is supplied externally and verified by SHA-256; see
[`models/README.md`](models/README.md).

## Documentation

- [`doc/API.md`](doc/API.md) defines the scope and public contract.
- [`doc/INFERENCE.md`](doc/INFERENCE.md) defines the current releases and the
  estimator.
- [`doc/CATALOGUE_PRIOR.md`](doc/CATALOGUE_PRIOR.md) documents the finite-prior
  likelihood and caches.
- [`doc/CONVENTIONS.md`](doc/CONVENTIONS.md) fixes catalogue, shear, response,
  seed, and reported-`m` conventions.
- [`doc/ENVIRONMENT.md`](doc/ENVIRONMENT.md) gives local interpreter, test, and
  scheduler instructions.
- [`doc/WORKLOG.md`](doc/WORKLOG.md) is the preserved, newest-first scientific
  record.
