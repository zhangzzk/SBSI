#!/bin/bash
#SBATCH --job-name=v22grpcand
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-4
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/run
STRENGTHS=(0 1 3 10 30)
STRENGTH=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}
STEM=$RUN/candidate_lambda_${STRENGTH}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

mkdir -p "$RUN"
test -s "$SOURCE/metadata.json"
test -s "$SCENE/metadata.json"
python -u scripts/train_v22_grouped_rscene_residual.py \
  --source-cache "$SOURCE" --scene-cache "$SCENE" \
  --group-strength "$STRENGTH" --train-case-min 40 --train-case-max 159 \
  --trees 180 --output-model "$STEM.json" --output-summary "$STEM.summary.json"
echo "V22_GROUPED_RSCENE_CANDIDATE_JOB_DONE strength=$STRENGTH"
date
