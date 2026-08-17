#!/bin/bash
#SBATCH --job-name=v22rbdesign
#SBATCH --time=01:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22rbdesign_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAIN_RESULTS=/home/z/Zekang.Zhang/SBSI/results
DECILE=results/v22_rblend_bin_design_decile_target.npz
BASE=results/response_target_crowd_rblend_snc_c0-99_6x6x5_v22.npz
OUT=results/v22_rblend_bin_design.json
CSV=results/v22_rblend_bin_design.csv
for file in "$D/det_meas_crowd_g0.05_val_full.feather" \
            "$D/det_meas_crowd_conc_g0.0_train_full.feather" \
            "$MAIN_RESULTS/g0_lookup_c0-99.feather" "$BASE"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
[ ! -e "$DECILE" ] && [ ! -e "$OUT" ] && [ ! -e "$CSV" ] || {
  echo "REFUSING overwrite design outputs"; exit 1;
}

echo "### V2.2 R_BLEND BIN DESIGN — COUNTS ONLY, NO RETRAINING ###"; date
"$PY" -u scripts/compute_response_target_blend.py \
  --catalogue "$D/det_meas_crowd_g0.05_val_full.feather" \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --n-size 6 --n-crowd 10 --min-count 500 --max-case 99 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --snc-lookup "$MAIN_RESULTS/g0_lookup_c0-99.feather" \
  --output "$DECILE"

"$PY" -u scripts/analyze_v22_rblend_bin_design.py \
  --decile-target "$DECILE" --baseline-target "$BASE" \
  --flow-catalogue "$D/det_meas_crowd_conc_g0.0_train_full.feather" \
  --output "$OUT" --csv "$CSV"
echo V22_RBLEND_BIN_DESIGN_DONE; date
