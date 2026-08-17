#!/bin/bash
#SBATCH --job-name=v30ns_ana
#SBATCH --time=01:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v30ns_ana_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v30_nosize_grid
V29=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
F=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend
OUT=results/v30_nosize_grid_halfshear_seed501_screen.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/analyze_v27_scene_arms.py \
  --arms v30_m10_c16 v30_m12_c16 --seeds 501 \
  --features-half "$F/halfshear_rblend_base_c40-199_v22domain.feather" \
  --halfshear-dir "$C/halfshear_scores" \
  --halfshear-baseline-dir "$V29/halfshear_scores" \
  --halfshear-baseline-template 'v22_control_s{seed}.feather' \
  --skip-constgold --output "$OUT"
echo V30_NOSIZE_ANALYZE_DONE; date
