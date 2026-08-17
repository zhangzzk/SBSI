#!/bin/bash
#SBATCH --job-name=abiasetail
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abiasetail_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abiasetail_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
BASE=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final
OUT=${BASE}_scene_tail
test ! -e "$OUT.json" || { echo "REFUSING existing $OUT.json"; exit 1; }
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
"$PY" -u scripts/plot_anchor_bias_scene_tail_curves.py \
  --features results/anchorblend_g002_bias_features_v22_c400-899.feather \
  --predictions "${BASE}_test_predictions.feather" \
  --parent-json "${BASE}.json" \
  --parent-curves "${BASE}_curves.csv" \
  --output-prefix "$OUT"
echo ANCHOR_BIAS_SCENE_TAIL_JOB_DONE
