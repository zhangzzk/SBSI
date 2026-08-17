#!/bin/bash
#SBATCH --job-name=v22c4target
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22c4target_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAIN_RESULTS=/home/z/Zekang.Zhang/SBSI/results
OUT=results/v22_rblend_bin_design_conditional8_size4_target.npz
for file in "$D/det_meas_crowd_g0.05_val_full.feather" "$MAIN_RESULTS/g0_lookup_c0-99.feather"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

echo "### V2.2 6x4x8 CONDITIONAL R_BLEND TARGET — NO TRAINING ###"; date
"$PY" -u scripts/compute_response_target_blend.py \
  --catalogue "$D/det_meas_crowd_g0.05_val_full.feather" \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend --crowd-conditional \
  --n-flux 6 --n-size 4 --n-crowd 8 --min-count 500 --max-case 99 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --snc-lookup "$MAIN_RESULTS/g0_lookup_c0-99.feather" \
  --output "$OUT"
echo V22_CONDITIONAL4_TARGET_DONE; date
