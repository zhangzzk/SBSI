#!/bin/bash
#SBATCH --job-name=sbsi-v39-bright20
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v39_bright20_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-bright20-${SLURM_JOB_ID}"
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
: "${SBSI_BRIGHT_ROOT:?Set frozen continuation root}"
: "${SBSI_RECENTER_ROOT:?Set previous full-pass root}"
: "${SBSI_STUDY_ROOT:?Set matched model-mock source}"
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python "$SBSI_BRIGHT_ROOT/run_v39_bright_bfgs.py" "$@"
