#!/bin/bash
#SBATCH --job-name=hs_v22_regctl
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-1
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_v22_regctl_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_v22_regctl_%A_%a.err
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
TOKENS=(unwtg0 unwtg0mc20)
TAG=lsst_r_extnbr_v22_rpow${TOKENS[$SLURM_ARRAY_TASK_ID]}
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/build_halfshear_vector_closure.py \
  --catalogue "$CAT" --n-cases 40 --development-max 19 --shear 0.2 \
  --model-tag "$TAG" \
  --table-output "results/halfshear_vector_closure_${TAG}_c0-39.feather" \
  --output "results/halfshear_vector_closure_${TAG}_c0-39.json"
