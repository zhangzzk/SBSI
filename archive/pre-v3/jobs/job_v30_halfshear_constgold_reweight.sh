#!/bin/bash
#SBATCH --job-name=v30_reweight
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v30_reweight_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v30_nosize_grid
V29=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
F=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend
OUT=results/v30_halfshear_constgold_population_reweight.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

"$PY" -u scripts/diag_halfshear_constgold_reweight.py \
  --features-half "$F/halfshear_rblend_base_c40-199_v22domain.feather" \
  --features-const "$V29/const_primary_features_c40-139.feather" \
  --constgold-glob "$C/constgold_scores/v30_m10_c16_s501_c*.feather" \
  --baseline-template "$V29/halfshear_scores/v22_control_s{seed}.feather" \
  --variant-dir "$C/halfshear_scores" \
  --arms v30_m10_c16 v30_m12_c16 --seeds 501 502 503 505 \
  --target-baseline results/response_target_crowd_rblend_snc_c0-99_6x6x5_v22.npz \
  --target-size4 results/v22_rblend_bin_design_conditional8_size4_target.npz \
  --target-mag10 results/v30_rblend_mag10_nosize_c16_target.npz \
  --target-mag12 results/v30_rblend_mag12_nosize_c16_target.npz \
  --closure-json results/v30_nosize_grid_fourseed_screen.json \
  --output "$OUT"

echo V30_REWEIGHT_DONE; date
