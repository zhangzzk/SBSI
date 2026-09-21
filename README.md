# Simulation-Based Shear Inference

SBSI provides model loading and prediction, finite-scene likelihoods,
selection normalization, importance sampling, and Bayesian shear inference.
BlendEMU owns image simulation, measurement, catalogue production, and
response-emulator training.

## Model and numerical identities

| Role | Identity | Status |
| --- | --- | --- |
| Retained development model | `V3.6-like` | single flow, disk-response emulator, single smooth-crowding classifier |
| Historical model preset | `V3.5-like` | epoch154, trial9, three-classifier ensemble |
| Default likelihood | `v3.2-like` | `configs/likelihood.json` |
| Default estimator | `v1.3-infer` | `configs/inference.json` |
| Superseded estimator | `v1.2-infer` | `configs/inference_v1_2.json` |
| Superseded estimator | `v1.1-infer` | `configs/inference_v1_1.json` |

V3.6-like pins the original common-descent step2 flow, the physical-moment
disk-response emulator with Möbius composition, and the epoch99 single
17-input classifier. Its primary population uses measured
`FLUX_RADIUS > 0.6 arcsec` and `MAG_AUTO < 25.8`, selected independently in
each shear leg, from the declared true-r<26 parent.

The retained development joint biases are -0.200 ±0.264% on main80,
+0.402 ±0.189% on the existing40, and +0.347 ±0.817% / -0.443 ±0.795% on
the four-scene g1/g2 checks (case-bootstrap standard errors). Training and
sampling repeats are recorded separately. The owner stopped the investigation
on 2026-09-20; the new 40-scene validation was cancelled before evaluation.
See [model evidence and scope](doc/JOINT_CALIBRATION.md#retained-evidence).

The identity is [configs/models_v3_6_like.json](configs/models_v3_6_like.json).
It is a model manifest, not an inference configuration: the configured
likelihood still uses fixed additive response and does not yet implement the
V3.6-like disk composition and state-dependent classifier.

## Inspect models

Set `SBSI_CACHE_DIR` to the external SBSI cache root and `BLENDEMU_RUNS_DIR`
to the external BlendEMU runs root, then:

```bash
python -m sbsi show-model V3.6-like
python -m sbsi validate-model V3.6-like
```

`get_model("V3.6-like")` returns pinned paths. Loading and feature construction
are described in [the model API](doc/API.md#v36-like-model-loading).

## Run inference

```bash
python scripts/run_inference.py \
  --inference-config configs/inference.json \
  --likelihood-config configs/likelihood.json \
  --scene-store /path/to/scene \
  --measurement-model /path/to/configured-flow.pt \
  --blend-response-cache /path/to/rblend-cache \
  --model-cache /path/to/model-cache \
  --proposal-cache /path/to/proposal-cache \
  --output /new/output/path
```

Inputs and model/cache identities must match the selected likelihood config.
See [the inference workflow](doc/INFERENCE.md#active-workflow).

## Repository

- `sbsi/`: supported model, likelihood, and inference modules.
- `scripts/`: scene preparation, cache construction, inference, and reporting.
- `jobs/`: four scheduler wrappers for inference and response-cache building.
- `tests/`: core regression tests, model loading, and retained feature contracts.
- `configs/`: numerical/likelihood configurations and model manifests.
- `doc/`: current documentation; start with [the repository map](doc/REPOSITORY.md#active-layout).
- `archive/`: dated, hash-indexed research snapshots; excluded from imports and tests.

## Installation and tests

```bash
python -m pip install -e /path/to/SBSI
python -m pytest tests -q
```

Install BlendEMU in the same environment for the historical additive emulator.
The V3.6-like disk-response loader also requires XGBoost. Heavy work and full
test runs use the scheduler; see [environment and resources](doc/ENVIRONMENT.md#tests).

## Documentation

- [API and scope](doc/API.md#scope-boundary)
- [Inference](doc/INFERENCE.md#current-identities)
- [Finite catalogue prior](doc/CATALOGUE_PRIOR.md)
- [Mathematical derivation](doc/MATH.md)
- [Scientific conventions](doc/CONVENTIONS.md)
- [V3.6-like evidence](doc/JOINT_CALIBRATION.md#retained-evidence)
- [Change log](doc/WORKLOG.md)
