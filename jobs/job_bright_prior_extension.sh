#!/bin/bash
#SBATCH --job-name=sbsi-bright-prior
#SBATCH --partition=inter
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bright_prior_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-bright-pycache-${SLURM_JOB_ID}"
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
SBSI_BRIGHT_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
SBSI_BRIGHT_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_bright_extension_20260925_v1
if [[ "$1" == evaluate ]]; then
  "$SBSI_BRIGHT_PYTHON" "$SBSI_BRIGHT_ROOT/bright_prior_extension.py" evaluate --shard "$SLURM_ARRAY_TASK_ID"
else
  "$SBSI_BRIGHT_PYTHON" "$SBSI_BRIGHT_ROOT/bright_prior_extension.py" "$@"
fi
