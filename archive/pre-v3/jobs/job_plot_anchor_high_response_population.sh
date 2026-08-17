#!/bin/bash
#SBATCH --job-name=anchor_rsp01
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
PREDICTIONS=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final_test_predictions.feather
BASE1=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c700-799
BASE2=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c800-899
OUT=$ROOT/results/anchorblend_g002_high_response_gt0p1_population_c700-899_v2
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for input in "$FEATURES" "$PREDICTIONS" "$BASE1" "$BASE2"; do
  test -e "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in csv json md pdf png; do
  test ! -e "$OUT.$suffix" || {
    echo "REFUSING existing $OUT.$suffix"; exit 1; }
done

cd "$ROOT"
"$PY" -u scripts/plot_anchor_high_response_population.py \
  --features "$FEATURES" \
  --predictions "$PREDICTIONS" \
  --manifest-dir "$BASE1" "$BASE2" \
  --pair-prefix pairs_renderer_v22 \
  --galaxy-shear 0.02 \
  --case-min 700 \
  --case-max 899 \
  --threshold 0.1 \
  --output-prefix "$OUT"
echo ANCHOR_HIGH_RESPONSE_POPULATION_DONE
