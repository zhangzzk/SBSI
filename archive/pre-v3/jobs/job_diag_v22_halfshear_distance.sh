#!/bin/bash
#SBATCH --job-name=v22hs_dist
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22hs_dist_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22hs_dist_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/diag_v22_halfshear_distance.py \
  --base /project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/base_c40-199.feather \
  --selfresp /project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather \
  --constgold-feather results/v22_constgold_gap_features_c40-139.feather \
  --case-min 40 --case-max 139 --development-max 89 \
  --mag-max 25.8 --re-min 0.5 \
  --output results/v22_constgold_vs_halfshear_distance_c40-139_s16.json
