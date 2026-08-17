# CLAUDE.md

@AGENTS.md

The project overview, scope boundaries (SBSI consumes finished `blendemu` catalogues; don't move classifier/flow/inference code into blendemu), the `WORKLOG.md` requirement, the Slurm resource policy, and model-specific development notes all live in `AGENTS.md` above. Only Claude-Code-specific setup that AGENTS.md omits is below.

**Before building any new number, read `CONVENTIONS.md`** — it fixes which catalogue, the population-cut order, true vs measured cuts, unsheared/sheared/measured shapes, leg matching, the response estimators, and the single definition of `m`. Update it when a convention changes.

## Environment
- Conda env: `conda activate sims1` (Python 3.9). Repo is NOT pip-installed.
- Imports rely on PYTHONPATH: `export PYTHONPATH="$PWD:$PYTHONPATH"` from the repo root.
  BlendEMU is NOT needed for this — add it only when loading the emulator
  (`sbs_shear.models.load_emulator`), which is the sole place SBSI imports it. Its location
  comes from `BLENDEMU_ROOT` / `BLENDEMU_MODELS`, not from a hardcoded path.
- `pip install -e .` is an optional alternative (see `pyproject.toml`); it also installs the
  `sbsi` console script.
- Login node has no GPU (32c/376G); all training/GPU work must go through Slurm.

## Tests
- **`sims1` has no pytest** — the obvious `conda activate sims1 && python -m pytest` fails with
  `No module named pytest`. Run the suite with the `py31` interpreter instead (pytest 9 + torch),
  from the repo root with PYTHONPATH set:
  `/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -m pytest tests/ -q`
- Whole suite is ~8 s, pure CPU, no catalogue access — fine on the login node, and worth running
  after any edit to `sbs_shear/`.
- No pytest config; `tests/` covers `sbs_shear/` only. The one BlendEMU cross-check in
  `tests/test_api.py` skips when BlendEMU is absent, so the suite passes on a bare checkout.
- Note `sims1`'s scipy fails to import on the LOGIN node (`GLIBCXX_3.4.30 not found`, via
  `sklearn`); it is fine on compute nodes. So login-node smoke tests of the trainers must use
  `py31`, or avoid importing sklearn.

## Compute jobs
- Never run nontrivial work directly on the login node.
- The repository has no supported job scripts. Users call the Python API from
  their own Slurm or other scheduler environment.
- Scheduler logs and wrappers live outside the SBSI source workflow.

## Data
- Every training, validation, and inference catalogue is an explicit user input.
  Do not add a project catalogue path as an API default.

## Gotchas
- `WORKLOG.md` is large and newest-first — read only the top.
- Current shape estimator is ngmix (`NGMIX_G1/G2`), superseding SExtractor moments.
- `archive/` holds superseded scripts; don't resurrect without checking WORKLOG.
