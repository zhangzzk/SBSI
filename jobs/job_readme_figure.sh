#!/bin/bash
#SBATCH --job-name=sbsi-readme
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/readme_figure_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-readme-${SLURM_JOB_ID:-$$}"
export SBSI_CACHE_DIR=${SBSI_CACHE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches}
unset PYTHONPATH
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.plot_readme_figure "$@"
