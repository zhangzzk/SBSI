#!/bin/bash
#SBATCH --job-name=ab_seq2_eval
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3
SCORE=$RUN_ROOT/anchor_scores_c400_899
ANCHOR_FEATURE=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUT=$ROOT/results/v22_crossfit_sequential_optuna50_v3_anchor_c400-899
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

for case in $(seq 400 899); do
  test -s "$SCORE/case${case}.feather"
  test -s "$SCORE/case${case}.json"
done
python -u scripts/evaluate_anchor_crossfit_sequential.py \
  --score-dir "$SCORE" --anchor-features "$ANCHOR_FEATURE" \
  --output-prefix "$OUT"
echo ANCHOR_CROSSFIT_SEQUENTIAL_EVALUATION_JOB_DONE
date
