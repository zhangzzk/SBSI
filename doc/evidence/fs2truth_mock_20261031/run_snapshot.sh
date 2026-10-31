#!/bin/bash
# Run a script of this study inside the frozen symmetric-run source snapshot (same code as the measured runs).
# usage: run_snapshot.sh <script.py> [args...]
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
unset PYTHONPATH
cd /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_signsym_20261028/cacheoff_validation_v1/source_snapshot
sha256sum --check SHA256SUMS --quiet
hostname; nvidia-smi -L || true
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();runpy.run_path(sys.argv.pop(1),run_name="__main__")' "$@"
