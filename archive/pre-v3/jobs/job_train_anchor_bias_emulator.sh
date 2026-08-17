#!/bin/bash
#SBATCH --job-name=ab_biasemu
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/py31
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
EDA=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899_eda.json
PREFIX=${AB_BIAS_PREFIX:-$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for input in "$FEATURES" "$EDA"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in .json .md .joblib _test_predictions.feather _curves.csv _maps.csv; do
  test ! -e "${PREFIX}${suffix}" || {
    echo "REFUSING existing ${PREFIX}${suffix}"; exit 1; }
done
cd "$ROOT"

python -u scripts/train_anchor_bias_emulator.py \
  --features "$FEATURES" --eda-json "$EDA" \
  --train-max 599 --tune-max 699 --seed 7301 \
  --curve-bins 12 --map-bins 8 --output-prefix "$PREFIX"
echo ANCHOR_BIAS_EMULATOR_JOB_DONE
date
