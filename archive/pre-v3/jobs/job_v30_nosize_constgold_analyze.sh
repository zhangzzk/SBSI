#!/bin/bash
#SBATCH --job-name=v30nscg_ana
#SBATCH --time=01:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v30nscg_ana_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v30_nosize_grid
V29=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
F=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend
OUT=results/v30_nosize_grid_constgold_seed501_screen.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/analyze_v27_scene_arms.py \
  --arms v30_m10_c16 v30_m12_c16 --seeds 501 \
  --features-half "$F/halfshear_rblend_base_c40-199_v22domain.feather" \
  --features-const "$V29/const_primary_features_c40-139.feather" \
  --halfshear-dir "$C/halfshear_scores" \
  --halfshear-baseline-dir "$V29/halfshear_scores" \
  --halfshear-baseline-template 'v22_control_s{seed}.feather' \
  --constgold-dir "$C/constgold_scores" \
  --constgold-baseline-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_shards \
  --output "$OUT"
echo V30_NOSIZE_CONSTGOLD_ANALYZE_DONE; date
