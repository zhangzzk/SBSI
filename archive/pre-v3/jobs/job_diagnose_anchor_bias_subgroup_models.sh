#!/bin/bash
#SBATCH --job-name=ab_biassub
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
OUT_JSON=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_subgroups.json
OUT_MD=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_subgroups.md
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -s "$FEATURES" || { echo "MISSING $FEATURES"; exit 1; }
for output in "$OUT_JSON" "$OUT_MD"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
python -u scripts/diagnose_anchor_bias_subgroup_models.py \
  --features "$FEATURES" --train-max 599 --tune-max 699 --seed 9182 \
  --output-json "$OUT_JSON" --output-md "$OUT_MD"
echo ANCHOR_BIAS_SUBGROUP_MODELS_JOB_DONE
date
