#!/bin/bash
#SBATCH --job-name=abias_panelc
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
CURVES=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final_curves.csv
OUT=$ROOT/results/anchorblend_g002_bias_emulator_panelC_c700-899
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

test -s "$CURVES" || { echo "MISSING $CURVES"; exit 1; }
for suffix in csv pdf png; do
  test ! -e "$OUT.$suffix" || {
    echo "REFUSING existing $OUT.$suffix"; exit 1; }
done

cd "$ROOT"
"$PY" -u scripts/plot_anchor_bias_emulator_panel_c.py \
  --curves "$CURVES" \
  --output-prefix "$OUT"
echo ANCHOR_BIAS_EMULATOR_PANEL_C_DONE
