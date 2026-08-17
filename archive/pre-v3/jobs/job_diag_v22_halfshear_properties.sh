#!/bin/bash
#SBATCH --job-name=hsv22props
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv22props_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

SELF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather
RBLEND=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend/halfshear_rblend_base_c40-199_v22domain.feather
EDGES=results/v27_sixshell_four_seed_blendness.json
TARGET=results/response_target_crowd_rblend_snc_c0-99_6x6x5_v22.npz
CONST=results/v22_constgold_localization.csv
OUT=results/v22_halfshear_property_profiles_targetgrid.json
CSV=results/v22_halfshear_property_profiles_targetgrid.csv
for file in "$SELF" "$RBLEND" "$EDGES" "$TARGET" "$CONST"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
[ ! -e "$OUT" ] && [ ! -e "$CSV" ] || { echo "REFUSING overwrite $OUT or $CSV"; exit 1; }

echo "### V2.2 HALF-SHEAR PROPERTY PROFILES (16 seeds) ###"; date
"$PY" -u scripts/analyze_v22_halfshear_properties.py \
  --selfresp "$SELF" --rblend-base "$RBLEND" \
  --blend-edges-json "$EDGES" --response-target "$TARGET" --constgold-profile "$CONST" \
  --output "$OUT" --csv "$CSV"
echo V22_HALFSHEAR_PROPERTY_PROFILES_DONE; date
