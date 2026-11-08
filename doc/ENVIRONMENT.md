# SBSI environment and resources

This document records the local execution contract. Scientific behavior belongs
in code and release configuration, not in shell setup.

## Installation

Install SBSI into the interpreter that will run it:

```bash
python -m pip install -e /path/to/SBSI
```

The package metadata declares the core Python dependencies and installs the
`sbsi` command. Imports must work from any directory; do not add checkout paths
to `PYTHONPATH` or depend on the current working directory.

BlendEMU is required only when loading its response emulator or classifier
artifacts. Install its checkout into the same environment:

```bash
python -m pip install --config-settings editable_mode=compat -e /path/to/blendemu
```

SBSI does not search `BLENDEMU_ROOT` or a user configuration file for source
code. Model-store locations may be supplied explicitly or through the optional
`SBSI_CACHE_DIR` and `BLENDEMU_MODELS` path overrides.

## Local interpreters

Two project environments are currently available:

- `/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python` — Python 3.10,
  pytest, and PyTorch; use this for the test suite and light validation.
- `/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python` — Python 3.9
  runtime used by existing scheduled science jobs. It does not provide pytest
  and its login-node SciPy/sklearn stack has a `GLIBCXX` incompatibility.

Both environments need the checkout installed editable after a clean setup.
Environment names are deployment details, not part of either scientific
release identity.

### Historical scene-prior runtime compatibility

Fresh full-scene calibration development must generate catalogues and render
with `sims1`, setting `LD_LIBRARY_PATH` to its `lib` directory for those
commands. Keep corrected measurement and tests in `py31`. Exact replay of
original scene0 (seed123) in job16564191 reproduced all699,568 objects only
with Python3.9.20/NumPy1.26.4/pandas2.2.3. The identical loader and source
catalogue produce69,956,780 prior rows there, versus69,967,938 in `py31`,
with position-angle precision also differing. The precise numerical cause
is not yet isolated. Do not silently substitute a runtime or change cuts to
match the old count; require full original-scene replay. See
[runtime replay](../archive/research-2026-09-20/doc/JOINT_CALIBRATION.md#historical-runtime-replay-and-full-scene-restart).

## Tests

From the repository root:

```bash
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -m pytest tests -q
```

The default pytest configuration searches only `tests/`. The suite is CPU
only and must pass without BlendEMU installed; the optional cross-check against
BlendEMU skips when that package is unavailable.

The cleanups archive experiment-only tests alongside their scripts;
they are not discovered by default. The retained suite covers core inference,
model loading, and retained feature/disk contracts. V3.9 freeze receipts are
checked by `tests/test_model_presets.py`. Run the full suite via
Slurm using `sbatch jobs/job_tests.sh` from the checkout root. Set the external
artifact roots first to include the V3.9 artifact prediction smoke test; without
them it skips. The launcher uses py31 and one CPU thread.

Use targeted tests while editing, then run the whole suite for changes to
`sbsi/`, production scripts, configuration validation, or artifact loading.

## Login node and scheduler

The login node is for edits, syntax checks, metadata inspection, checksum
verification, and small CPU tests. Training, simulations, full-catalogue scans,
response-cache construction, and other substantial CPU/GPU work go through the
local scheduler.

### Job packing

The owner's 2026-11-06 preference is fewer, larger, longer-lived scheduler jobs
instead of many short jobs for small chunks. Pack multiple independent cases
or shear legs into each allocation, using persistent workers or a bounded
worker pool. Reuse initialized models and other shared setup when practical
to reduce scheduler and startup overhead. Keep enough workers to use the
authorized resources, and allocate only resources the remaining work can use.

Retain per-case outputs, completion receipts and restart checkpoints so a
worker can resume unfinished work without repeating completed cases. Preserve
case identities, random streams, model identities, pilot gates, report
dependencies and study-wide CPU/GPU ceilings. This preference concerns job
packing; numerical batch sizes and scientific settings remain separately
validated choices.

### Job monitoring

The joint-development chain was cancelled by the owner on 2026-09-20. Do not
restart its research/recovery/snapshot jobs without a new request. The policy
below applies to newly authorized work, including cleanup verification jobs.

Follow new jobs through startup and first meaningful progress, using brief
status/log checks about every 30--60 seconds to catch early failures. Short
preparation, validation, and test stages (normally under 30 minutes) may be
followed through completion, including the startup of a dependent stage.
After a long job demonstrates normal progress, or queue/startup waiting becomes
prolonged, yield and use a bounded scheduled check about every two hours.
Do not keep a turn occupied polling an entire long run. Report meaningful
progress or failures rather than repeated unchanged states. Immediate checks
requested by the owner remain allowed. This is the owner's 2026-09-10 revision
to the earlier blanket prohibition, including historical work-log notes.

### Scheduler launchers

`jobs/job_tests.sh` runs the retained CPU suite. Current science enters through
`scripts/run_galsbi_inference.py`; use explicit model-matched inputs, settings
and allocation limits. The population/bright wrappers preserve their existing
snapshot-based reference assumptions and are not generic validated launchers
for the new models. Finite-catalogue cache/inference wrappers are retained for
compatibility. Completed study launchers live in the archive.

## Data and models

- Training, validation, simulation, prior, cache, mock, and output catalogues
  are user data. They do not belong in the repository and are never hidden
  defaults.
- `models/` bundles both GalSBI flow checkpoints and their classifier/disk
  companions, plus small historical compatibility artifacts. Exact identities
  are in the model manifests; external roots may override bundled locations.
- `configs/inference.json` and `configs/likelihood.json` contain portable
  scientific settings and artifact identities. They intentionally omit
  machine-specific data and output paths.

## Documentation workflow

`doc/WORKLOG.md` is the short active log. Start with
[the current inference guide](CURRENT_INFERENCE.md#entry-points),
[model guide](LIKE_1_0_GALSBI.md#decision-and-identity), and
[history](HISTORY.md#how-to-read-this-history). Old full logs, completed
experiments and output artifacts are indexed in the
[repository map](REPOSITORY.md#research-archive).
