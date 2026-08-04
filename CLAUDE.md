# CLAUDE.md

@AGENTS.md

The project overview, scope boundaries (SBSI consumes finished `blendemu` catalogues; don't move classifier/flow/inference code into blendemu), the `WORKLOG.md` requirement, the Slurm resource policy, and model-specific development notes all live in `AGENTS.md` above. Only Claude-Code-specific setup that AGENTS.md omits is below.

**Before building any new number, read `CONVENTIONS.md`** — it fixes which catalogue, the population-cut order, true vs measured cuts, unsheared/sheared/measured shapes, leg matching, the response estimators, and the single definition of `m`. Update it when a convention changes.

## Environment
- Conda env: `conda activate sims1` (Python 3.9). Repo is NOT pip-installed.
- Imports rely on PYTHONPATH — always set both repos:
  `export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"`
- Login node has no GPU (32c/376G); all training/GPU work must go through Slurm.

## Tests
- **`sims1` has no pytest** — the obvious `conda activate sims1 && python -m pytest` fails with
  `No module named pytest`. Run the suite with the `py31` interpreter instead (pytest 9 + torch),
  from the repo root with PYTHONPATH set:
  `/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -m pytest tests/ -q`
- Whole suite is ~8 s, pure CPU, no catalogue access — fine on the login node, and worth running
  after any edit to `sbs_shear/`.
- No pytest config; `tests/` covers `sbs_shear/` only, never `scripts/`.
- Note `sims1`'s scipy fails to import on the LOGIN node (`GLIBCXX_3.4.30 not found`, via
  `sklearn`); it is fine on compute nodes. So login-node smoke tests of the trainers must use
  `py31`, or avoid importing sklearn.

## Jobs
- Submit with `sbatch jobs/job_*.sh`. Never run nontrivial work directly on the login node.
- Job body pattern: activate `sims1`, set PYTHONPATH (both repos), `cd` to repo, `python -u scripts/...`.
- Slurm logs: `/home/z/Zekang.Zhang/logs/*_%j.{out,err}` (outside repo) and `jobs/logs/`.

## Data
- `SBSI/data/` is empty. Real catalogues live at
  `/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/`
  (e.g. `det_meas_ngmix_ap7_g0.0_train.feather`). Jobs read from there.

## Gotchas
- `WORKLOG.md` is large and newest-first — read only the top.
- Current shape estimator is ngmix (`NGMIX_G1/G2`), superseding SExtractor moments.
- `jobs/archive/` and `archive/` hold superseded scripts; don't resurrect without checking WORKLOG.
