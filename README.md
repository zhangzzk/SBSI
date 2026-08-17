# SBSI — simulation-based shear-bias inference

SBSI measures the multiplicative shear bias `m` from a catalogue of detected, measured
galaxies, using the parameter-free response decomposition

```
m = R_sim / (R_flow + R_blend) - 1
```

where `R_sim` is the response measured directly from the simulation, `R_flow` is the
isolated-galaxy shape response from a trained conditional flow, and `R_blend` is the
neighbour-blending response.

## Relationship to blendemu

**SBSI consumes finished catalogues. It does not simulate or measure anything.**

You run [blendemu](../blendemu) yourself to render the simulations and produce the
detection/measurement catalogues, then point SBSI at the output directory. Nothing in the
main pipeline imports blendemu.

There are exactly two places the boundary is crossed, and both are isolated:

| crossing | where | why |
|---|---|---|
| the blending-response emulator | `sbs_shear/emulator.py` | `R_blend` is evaluated by blendemu's trained `BlendingPredictor`. This is the only runtime dependency, and only for the R_blend step. |
| the catalogue bridge | `scripts/build_detection_measurement_catalogue.py` | Turns an existing blendemu *rendering* into an SBSI input catalogue. Optional — skip it if you already have catalogues. |

Emulator-retraining jobs, which run entirely inside blendemu, live in
`jobs/blendemu_side/` and are not part of the SBSI pipeline.

## Configuration

Every external path is resolved in `sbs_shear/paths.py` from an environment variable with
a fallback default — no path is hardcoded in a script. To run against your own data:

```bash
export SBSI_CATALOGUE_DIR=/path/to/your/catalogues   # finished blendemu catalogues
export SBSI_CACHE_DIR=/path/to/caches                # SBSI-derived lookups and harvests
export SBSI_SIM_DIR=/path/to/lsst_sims               # half-shear sim set
export SBSI_CONST_SIM_DIR=/path/to/lsst_sims_const   # constant-shear ("gold") sim set
export BLENDEMU_ROOT=/path/to/blendemu               # only needed for R_blend
```

Print what is currently resolved, and whether it exists:

```bash
python -m sbs_shear.paths
```

## Layout

```
sbs_shear/       importable library
  paths.py         all filesystem roots (environment-overridable)
  emulator.py      the single blendemu boundary + survey conditions
  measurement_model.py, spline_flow.py    the R_flow conditional flow
  selection_model.py, scene_model.py      selection / scene models
  response.py, shear_map.py, coordinates.py, preprocessing.py
  forward_model.py, posterior_shape.py
scripts/         runnable entrypoints (also an importable package; siblings reuse each other)
jobs/            Slurm submission scripts
  blendemu_side/   jobs that drive blendemu, not SBSI
plotting/        figure scripts
tests/           unit tests
archive/         superseded scripts, kept for provenance
```

## Environment

```bash
conda activate sims1                                  # Python 3.9
export PYTHONPATH="$PWD:$PYTHONPATH"                  # repo is not pip-installed
```

Scripts also self-bootstrap: each walks up from its own location to find the repo root,
so `python scripts/foo.py` works without `PYTHONPATH` set.

## Tests

`pytest` is not installed in `sims1`, and not every test module has a `__main__` guard, so
use the bundled runner (it needs no `PYTHONPATH`):

```bash
python tests/run_tests.py             # all modules
python tests/run_tests.py shear_map   # only matching modules
```

If you do have pytest available, `python -m pytest tests/` works too.

## Conventions

- Never run compute on the login node — submit with `sbatch jobs/job_*.sh`.
- Large data belongs under `$DATA_DIR`, not in the repo.
- Record substantive changes in `WORKLOG.md` (newest first).
- Current framing and estimator math: `INFERENCE.md`, `MATH.md`. Certified numbers:
  `Gold-V1.md`, `Gold-V2.md`, `Gold-V3.md`.
