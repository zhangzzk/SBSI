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

The 2026-09-20 cleanup archives experiment-only tests alongside their scripts;
they are not discovered by default. The retained suite covers core inference,
model loading, and V3.6-like geometry/disk contracts. Run the full suite via
Slurm using the py31 interpreter and one CPU thread.

Use targeted tests while editing, then run the whole suite for changes to
`sbsi/`, production scripts, configuration validation, or artifact loading.

## Login node and scheduler

The login node is for edits, syntax checks, metadata inspection, checksum
verification, and small CPU tests. Training, simulations, full-catalogue scans,
response-cache construction, and other substantial CPU/GPU work go through the
local scheduler.

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

`jobs/job_inference.sh` is the cluster-local reference launcher for
`v1.1-infer` with the configured `v3.2-like` likelihood. A V3.5-like run must
explicitly select `configs/likelihood_v3_5_like.json`. Every launcher resolves
every
user-owned catalogue, checkpoint, cache, and output path explicitly. Other
scheduler wrappers are validation or deployment aids, not public Python APIs.

V3.6-like is a separate model manifest, not a replacement likelihood config.
Its artifact roots are `SBSI_CACHE_DIR` and `BLENDEMU_RUNS_DIR`; see
[model loading](API.md#v36-like-model-loading).

`jobs/job_inference_hybrid_chain.sh` is the multi-GPU iterated launcher.  It
derives its worker count from the Slurm allocation (or `N_GPUS` outside
Slurm), partitions both prior-atom normalization and observation work across
all visible GPUs, and strictly combines the resulting shards.  Requesting
more GPUs therefore changes the partitioning automatically; scientific
settings such as `K`, `M`, cuts, seeds, and stencil step remain unchanged.
When `SELECTION_NORMALIZATION_CACHE` names a validated surrogate, the launcher
skips the atom preparation and reuses that cache for every pass.  Without it,
exact normalization preparation is automatically sharded and repeated at each
iteration centre.  Set `BUILD_SELECTION_NORMALIZATION_QUADRATIC=0` when the
exact chain is intended only as inference/validation and no reusable surrogate
has yet met the induced-shear accuracy gate.
Set `SCORE_ROOT_BFGS=1` only for an iterated custom chain that should solve the
successive catalogue scores with a positive-definite BFGS secant matrix rather
than require every intermediate observed Hessian to be positive definite.
Bright-only workers may persist indefinite shard-level moments because those
shards are not estimators; the combined full-catalogue solve retains the
positive-definite requirement.

## Data and models

- Training, validation, simulation, prior, cache, mock, and output catalogues
  are user data. They do not belong in the repository and are never hidden
  defaults.
- `models/` contains only the three small JSON artifacts described in
  `models/README.md`. The seed-501 measurement flow is supplied externally and
  verified by its recorded SHA-256.
- `configs/inference.json` and `configs/likelihood.json` contain portable
  scientific settings and artifact identities. They intentionally omit
  machine-specific data and output paths.

## Documentation workflow

`doc/WORKLOG.md` is a concise newest-first active log. Detailed pre-cleanup
provenance is preserved under `archive/research-2026-09-14/`. The maintained
operational contract is in
`doc/API.md`, `doc/INFERENCE.md`, `doc/CATALOGUE_PRIOR.md`, and the two
numerical release configurations plus the selected likelihood configuration.
