# SBSI environment and resources

Split out of `AGENTS.md` and `CLAUDE.md` on 2026-08-18. No code behavior lives
here; this is how to run what the repository already contains.

## Interpreters and installation

- Conda env: `conda activate sims1` (Python 3.9). The repo is
  editable-installed in `sims1` and `py31`
  (`pip install -e . --no-build-isolation`, no declared deps), so `import sbsi`
  works without PYTHONPATH; the PYTHONPATH contract still works as before:

  ```bash
  conda activate sims1
  export PYTHONPATH="$PWD:$PYTHONPATH"        # add BlendEMU only for the emulator step
  ```

- BlendEMU is NOT needed for this — add it only when loading the emulator
  (`sbsi.models.load_emulator`), which is the sole place SBSI imports it. Model
  artifacts resolve through `SBSI_CACHE_DIR` / `BLENDEMU_MODELS`, defaulting to
  the repository's `models/` release tree; `BLENDEMU_ROOT` is not read for
  model paths — `load_emulator` reads it only to import `blendemu` when it is
  not already importable (falling back to a one-line
  `~/.config/sbsi/blendemu_root` file when the variable is unset, for
  JupyterHub-style launches that skip shell exports).
- `pip install -e .` also installs the `sbsi` console script.

## Running the tests

- **`sims1` has no pytest** — the obvious `conda activate sims1 && python -m
  pytest` fails with `No module named pytest`. Run the suite with the `py31`
  interpreter instead (pytest 9 + torch), from the repo root with PYTHONPATH
  set:

  ```bash
  PYTHONPATH="$PWD" /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -m pytest tests/ -q
  ```

- Whole suite is ~80 s, pure CPU, no catalogue access — fine on the login node,
  and worth running after any edit to `sbsi/`. The suite must pass on a
  checkout that has SBSI only; the single BlendEMU cross-check in
  `tests/test_api.py` skips when BlendEMU is absent.
- Pytest discovery is pinned to `tests/`; archived historical tests are
  provenance and are not collected by the default command.
- `sims1`'s scipy fails to import on the LOGIN node (`GLIBCXX_3.4.30 not
  found`, via `sklearn`); it is fine on compute nodes. Login-node smoke tests
  of the trainers must therefore use `py31`, or avoid importing sklearn.

## Login node and scheduler

- The login node has no GPU (32c/376G); perform only edits, syntax checks,
  metadata inspection, and tiny smoke tests on it.
- Never run nontrivial work directly on the login node — training,
  full-catalogue scans, response production, simulations, and other nontrivial
  CPU/GPU work go through the local scheduler.
- `jobs/job_infer_v1.sh` is the cluster-local reference launcher for the
  canonical Infer V1 defaults. Other scheduler wrappers are deployment and
  provenance aids rather than public SBSI APIs.

## Data

- Every training, validation, and inference catalogue is an explicit user
  input. Do not add a project catalogue path as an API default.

## Gotchas

- `doc/WORKLOG.md` is large and newest-first — read only the top.
- Current shape estimator is ngmix (`NGMIX_G1/G2`), superseding SExtractor
  moments.
- `archive/` holds superseded scripts; don't resurrect without checking
  WORKLOG.
