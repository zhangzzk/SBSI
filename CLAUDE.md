# CLAUDE.md

@AGENTS.md

The project overview, scope boundaries (SBSI consumes finished `blendemu` catalogues; don't move classifier/flow/inference code into blendemu), the `WORKLOG.md` requirement, the Slurm resource policy, and model-specific development notes all live in `AGENTS.md` above. Only Claude-Code-specific setup that AGENTS.md omits is below.

## Environment
- Conda env: `conda activate sims1` (Python 3.9). Repo is NOT pip-installed.
- `export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"` is enough — blendemu no
  longer needs to be on PYTHONPATH; `sbs_shear.emulator` resolves it from `BLENDEMU_ROOT`.
- Scripts also self-bootstrap (they walk up to the repo root), so `python scripts/foo.py`
  works with no PYTHONPATH at all.
- Login node has no GPU (32c/376G); all training/GPU work must go through Slurm.

## Tests
- `pytest` is NOT installed in `sims1`, and most test modules lack a `__main__` guard.
  Use `python tests/run_tests.py` (needs no PYTHONPATH). 17 tests currently.

## Paths
- Nothing is hardcoded; `sbs_shear/paths.py` resolves every root from env vars.
  `python -m sbs_shear.paths` prints what is resolved and flags anything missing.

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
