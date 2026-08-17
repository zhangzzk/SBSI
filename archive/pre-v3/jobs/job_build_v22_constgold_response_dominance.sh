#!/bin/bash
#SBATCH --job-name=v22cg_dom
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22cg_dom_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22cg_dom_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
export XGB_DEVICE=cpu
cd "$ROOT"
"$PY" -u scripts/build_v22_constgold_response_dominance.py \
  --cases $(seq 40 139) \
  --tag lsst_r_extnbr_v22 \
  --reference-lookup results/blend_lookup_v22_c40-139.feather \
  --output results/v22_constgold_response_dominance_c40-139.feather
