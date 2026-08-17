#!/bin/bash
#SBATCH --job-name=hsv22rbslp
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv22rbslp_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

SELF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather
RBLEND=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend/halfshear_rblend_base_c40-199_v22domain.feather
TARGET=results/response_target_crowd_rblend_snc_c0-99_6x6x5_v22.npz
OUT=results/v22_rblend_within_target_bin_slopes.json
CSV=results/v22_rblend_within_target_bin_slopes.csv
for file in "$SELF" "$RBLEND" "$TARGET"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
[ ! -e "$OUT" ] && [ ! -e "$CSV" ] || { echo "REFUSING overwrite $OUT or $CSV"; exit 1; }

echo "### V2.2 R_SELF SLOPES INSIDE R_BLEND TARGET BINS ###"; date
"$PY" -u scripts/analyze_v22_rblend_within_bin_slopes.py \
  --selfresp "$SELF" --rblend-base "$RBLEND" --response-target "$TARGET" \
  --output "$OUT" --csv "$CSV"
echo V22_RBLEND_WITHIN_BIN_SLOPES_DONE; date
