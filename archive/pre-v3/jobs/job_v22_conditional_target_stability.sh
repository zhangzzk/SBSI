#!/bin/bash
#SBATCH --job-name=v22cstable
#SBATCH --time=01:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22cstable_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAIN_RESULTS=/home/z/Zekang.Zhang/SBSI/results
T6=results/v22_rblend_bin_design_conditional8_target.npz
T4=results/v22_rblend_bin_design_conditional8_size4_target.npz
OUT=results/v22_conditional_target_stability.json
CSV=results/v22_conditional_target_stability.csv
for file in "$D/det_meas_crowd_g0.05_val_full.feather" "$MAIN_RESULTS/g0_lookup_c0-99.feather" "$T6" "$T4"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
[ ! -e "$OUT" ] && [ ! -e "$CSV" ] || { echo "REFUSING overwrite stability output"; exit 1; }

echo "### V2.2 CONDITIONAL TARGET CASE-STABILITY GATE — NO TRAINING ###"; date
"$PY" -u scripts/audit_v22_conditional_target_stability.py \
  --catalogue "$D/det_meas_crowd_g0.05_val_full.feather" \
  --snc-lookup "$MAIN_RESULTS/g0_lookup_c0-99.feather" \
  --target cond6=results/v22_rblend_bin_design_conditional8_target.npz \
  --target cond4=results/v22_rblend_bin_design_conditional8_size4_target.npz \
  --output "$OUT" --csv "$CSV"
echo V22_CONDITIONAL_TARGET_STABILITY_DONE; date
