#!/bin/bash
#SBATCH --job-name=sbsi-v39-study
#SBATCH --partition=inter
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v39_study_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-v39-${SLURM_JOB_ID}"
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
SBSI_STUDY_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export SBSI_STUDY_ROOT="${SBSI_STUDY_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_v39_mock20k_distance_20260925_v1}"
if [[ "$1" == features ]]; then
  "$SBSI_STUDY_PYTHON" "$SBSI_STUDY_ROOT/prepare_v39_study_features.py" --shard "$SLURM_ARRAY_TASK_ID"
else
  "$SBSI_STUDY_PYTHON" "$SBSI_STUDY_ROOT/run_v39_sampling_study.py" "$@"
fi
