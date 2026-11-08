# Simulation-Based Shear Inference

SBSI loads measurement models, predicts selected shear response, and infers
shear with a GalSBI population prior. BlendEMU owns image simulation,
measurement, catalogue production and response-emulator training.

## Current pipeline and models

| Role | Identity | Guide |
| --- | --- | --- |
| Core continuous-scene inference | `infer-1.0-galsbi` | [Algorithm and implementation](doc/CURRENT_INFERENCE.md#name-and-reference-mock) |
| Current fiducial model | `like-1.0-galsbi` | [Identity, training and evidence](doc/LIKE_1_0_GALSBI.md#decision-and-identity) |
| Retained multistage comparison | `galsbi-multistage` | [Compatibility and limits](doc/CURRENT_INFERENCE.md#models-and-compatibility) |

Both GalSBI flow checkpoints and their shared classifier and disk emulator
are included under `models/`, with original hashes pinned in `configs/`.
With no external artifact-root overrides:

```bash
python -m sbsi validate-model like-1.0-galsbi galsbi-multistage
python scripts/run_galsbi_inference.py --help
```

The fiducial's central paired-seed response biases are +0.386 ±0.398% and
+0.043 ±0.666% on four development scenes. The historical inference mock
used a different model and mixed sampler settings. Neither result establishes
independent acceptance of end-to-end inference with the current fiducial.
The multistage flow requires an extra reference-neighbour input and is not yet
supported by the current inference adapter.

## Documentation

- [Inference math](doc/INFERENCE.md#selected-likelihood) and [detailed derivation](doc/MATH.md#1-the-model)
- [Scientific conventions](doc/CONVENTIONS.md#1-model-composition)
- [API and scope](doc/API.md#scope-boundary)
- [Finite-catalogue compatibility](doc/CATALOGUE_PRIOR.md)
- [Environment and tests](doc/ENVIRONMENT.md#tests)
- [Research history and previous attempts](doc/HISTORY.md#how-to-read-this-history)
- [Repository and archive map](doc/REPOSITORY.md#active-layout)
- [Recent changes](doc/WORKLOG.md)

## Installation and tests

```bash
python -m pip install -e /path/to/SBSI
mkdir -p logs
sbatch jobs/job_tests.sh
```

Full tests and substantial CPU/GPU work use the scheduler. Optional scientific
runtime dependencies are described in the environment guide. Older model
presets and finite-catalogue configurations remain compatibility interfaces;
their names and defaults have not been reassigned to the current pipeline.
