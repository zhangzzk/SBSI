#!/bin/bash
#SBATCH --job-name=sbsi-tempered
#SBATCH --partition=inter
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Persistent-scene MH (scripts/check_galsbi_local_sampler.py) from a frozen source snapshot.
# Usage: SBSI_TEMPER_ROOT=... sbatch jobs/job_galsbi_tempered.sh <label> --reference <dir> --shear g1 g2 \
#        --seed <s> --initial-bank-seed <b> --observation-indices <ids...> [--betas ... for mag < 23]
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=${SBSI_CACHE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches}
export BLENDEMU_RUNS_DIR=${BLENDEMU_RUNS_DIR:-/project/ls-gruen/users/zekang.zhang/blendemu_runs}
unset PYTHONPATH
: "${SBSI_TEMPER_ROOT:?set tempering experiment root}"
SBSI_TEMPER_SCRIPT=${SBSI_TEMPER_SCRIPT:-scripts/check_galsbi_local_sampler.py}
cd "$SBSI_TEMPER_ROOT/source_snapshot"
sha256sum --check SHA256SUMS --quiet
label=$1
shift
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();runpy.run_path(sys.argv.pop(1),run_name="__main__")' \
    "$SBSI_TEMPER_SCRIPT" --output "$SBSI_TEMPER_ROOT/$label" --device cuda "$@"
