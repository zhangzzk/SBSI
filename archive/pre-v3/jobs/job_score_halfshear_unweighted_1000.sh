#!/bin/bash
#SBATCH --job-name=hs_v22_t1000
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_v22_t1000_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_v22_t1000_%j.err
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
TAG=lsst_r_extnbr_v22_rpowunwt1000
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
cd "$ROOT"
"$PY" -u scripts/build_halfshear_vector_closure.py \
  --catalogue "$CAT" --n-cases 40 --development-max 19 --shear 0.2 \
  --model-tag "$TAG" \
  --table-output "results/halfshear_vector_closure_${TAG}_c0-39.feather" \
  --output "results/halfshear_vector_closure_${TAG}_c0-39.json"
