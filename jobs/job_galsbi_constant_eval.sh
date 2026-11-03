#!/bin/bash
# GalSBI-dev forward evaluation on the GalSBI constant-shear family (scripts/evaluate_galsbi_constant.py).
#   sbatch --array=0-13 -p cluster --constraint=x86-64-v3 -c 8 --mem=32G jobs/job_galsbi_constant_eval.sh prep
#   sbatch -p cluster --constraint=x86-64-v3 -c 4 --mem=24G jobs/job_galsbi_constant_eval.sh pairs-check
#   sbatch --array=0-13 -p cluster --constraint=x86-64-v3 -c 8 --mem=24G jobs/job_galsbi_constant_eval.sh c8
#   sbatch --array=0-13%3 -p cip --gres=gpu:a40-16gb:1 -c 4 --mem=24G --time=12:00:00 jobs/job_galsbi_constant_eval.sh draws
#   sbatch -p cluster -c 2 --mem=8G jobs/job_galsbi_constant_eval.sh summary
#SBATCH --job-name=galsbi_const_eval
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_dev_constant_20261103/logs/%x_%A_%a.log
set -euo pipefail
S=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/scripts/evaluate_galsbi_constant.py
D=/project/ls-gruen/users/zekang.zhang
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
echo "stage=$1 task=${SLURM_ARRAY_TASK_ID:-} host=$(hostname) commit=$(git -C "$(dirname "$S")" rev-parse HEAD)"
cd /tmp
case "$1" in
  prep|c8|draws) CHUNK=$SLURM_ARRAY_TASK_ID ;;
  pairs-check|summary) CHUNK= ;;
  *) echo "unknown stage $1" >&2; exit 2 ;;
esac
case "$1" in
  prep|pairs-check|summary) PY=$D/envs/py31/bin/python ;;
  c8) PY=$D/sbsi_caches/v36_classifier_f8_inputs_20260925_v1/venv/bin/python ;;
  draws) PY=$D/sbsi_caches/v36_f8_20260924_v1/venv/bin/python ;;
esac
$PY -I -u "$S" "$1" $CHUNK
