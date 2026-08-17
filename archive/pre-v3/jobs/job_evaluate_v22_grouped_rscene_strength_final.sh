#!/bin/bash
#SBATCH --job-name=v22gaincheck
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
CURRENT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/run/final.summary.json
TUNED=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_tune_v1/run/final.summary.json
REFERENCE=$ROOT/results/v22_grouped_rscene_final_transfer_v1_c20-39_anchor_c400-899.json
STRENGTH=$ROOT/results/v22_grouped_rscene_strength_tuning_v1_sel160-199_dev0-19.json
OUT=$ROOT/results/v22_grouped_rscene_strength_final_half_shear_c20-39
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$STRENGTH"
python -u scripts/evaluate_v22_grouped_rscene_hparam_final.py \
  --source-cache "$SOURCE" --scene-cache "$SCENE" \
  --current-model-summary "$CURRENT" --tuned-model-summary "$TUNED" \
  --strength-selection "$STRENGTH" --reference "$REFERENCE" \
  --output-prefix "$OUT"
echo V22_RSCENE_STRENGTH_FINAL_EVALUATION_JOB_DONE
date
