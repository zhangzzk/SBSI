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
- `python -m pytest tests/` from repo root (env active, PYTHONPATH set). No pytest config.

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
