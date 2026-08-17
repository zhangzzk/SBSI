#!/bin/bash
#SBATCH --job-name=v22grs_pairplot
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22grs_pairplot_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22grs_pairplot_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1
PREFIX=$ROOT/results/v22_grouped_rscene_pair_residual_vs_prediction_validation_c20-39
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/plot_v22_grouped_rscene_pair_residual_validation.py \
  --source-cache /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache \
  --scene-cache "$RUN/cache" \
  --model-summary "$RUN/run/final.summary.json" \
  --reference results/v22_grouped_rscene_final_transfer_v1_c20-39_anchor_c400-899.json \
  --output-prefix "$PREFIX"
for suffix in json csv md pdf png; do
  test -s "$PREFIX.$suffix"
done
echo V22_GROUPED_RSCENE_PAIR_RESIDUAL_VALIDATION_PLOT_JOB_DONE
date
