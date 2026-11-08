#!/bin/bash
#SBATCH --job-name=sbsi-v39-recenter-setup
#SBATCH --partition=inter
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v39_recenter_setup_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-recenter-setup-${SLURM_JOB_ID}"
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
: "${SBSI_STUDY_ROOT:?Set reduced-prior source}"
: "${SBSI_RECENTER_ROOT:?Set inference root}"
SBSI_REDUCED_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
if [[ "$1" == features ]]; then
  for (( shard=SLURM_ARRAY_TASK_ID; shard<20; shard+=4 )); do
    SLURM_ARRAY_TASK_ID="$shard" "$SBSI_REDUCED_PYTHON" "$SBSI_STUDY_ROOT/prepare_v39_reduced_prior.py"
  done
elif [[ "$1" == prepare ]]; then
  for (( shard=SLURM_ARRAY_TASK_ID; shard<20; shard+=4 )); do
    "$SBSI_REDUCED_PYTHON" "$SBSI_STUDY_ROOT/run_v39_sampling_study.py" prepare --shard "$shard"
  done
elif [[ "$1" == assemble ]]; then
  "$SBSI_REDUCED_PYTHON" "$SBSI_STUDY_ROOT/run_v39_sampling_study.py" assemble
  "$SBSI_REDUCED_PYTHON" "$SBSI_RECENTER_ROOT/run_v39_recenter.py" initialize
else
  "$SBSI_REDUCED_PYTHON" "$SBSI_STUDY_ROOT/run_v39_sampling_study.py" "$@"
fi
