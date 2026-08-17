#!/bin/bash
#SBATCH --job-name=v30tailaud
#SBATCH --time=02:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v30tailaud_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAIN_RESULTS=/home/z/Zekang.Zhang/SBSI/results
T10=results/v30_rblend_tailq10_6x4_target.npz
T12=results/v30_rblend_tailq12_6x4_target.npz
COUNT_JSON=results/v30_rblend_tail_grid_counts.json
COUNT_CSV=results/v30_rblend_tail_grid_counts.csv
STABLE_JSON=results/v30_rblend_tail_grid_stability.json
STABLE_CSV=results/v30_rblend_tail_grid_stability.csv
for file in "$D/det_meas_crowd_g0.05_val_full.feather" \
  "$D/det_meas_crowd_conc_g0.0_train_full.feather" \
  "$MAIN_RESULTS/g0_lookup_c0-99.feather" "$T10" "$T12"; do
  [ -f "$file" ] || { echo "MISSING $file"; exit 1; }
done
for file in "$COUNT_JSON" "$COUNT_CSV" "$STABLE_JSON" "$STABLE_CSV"; do
  [ ! -e "$file" ] || { echo "REFUSING overwrite $file"; exit 1; }
done

echo "### V3.0 TAIL GRID COUNT/STABILITY AUDIT — NO TRAINING, NO CONSTGOLD ###"; date
"$PY" -u scripts/summarize_v30_tail_grids.py \
  --target tailq10="$T10" --target tailq12="$T12" \
  --flow-catalogue "$D/det_meas_crowd_conc_g0.0_train_full.feather" \
  --output "$COUNT_JSON" --csv "$COUNT_CSV"
"$PY" -u scripts/audit_v22_conditional_target_stability.py \
  --catalogue "$D/det_meas_crowd_g0.05_val_full.feather" \
  --snc-lookup "$MAIN_RESULTS/g0_lookup_c0-99.feather" \
  --target tailq10="$T10" --target tailq12="$T12" \
  --output "$STABLE_JSON" --csv "$STABLE_CSV"
echo V30_TAIL_GRID_AUDIT_DONE; date
