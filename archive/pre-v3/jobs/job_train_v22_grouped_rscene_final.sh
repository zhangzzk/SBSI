#!/bin/bash
#SBATCH --job-name=v22grpfinal
#SBATCH --time=03:30:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/run
STRENGTH=${GROUP_STRENGTH:?submit with GROUP_STRENGTH selected by candidate evaluation}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

python -u scripts/train_v22_grouped_rscene_residual.py \
  --source-cache "$SOURCE" --scene-cache "$SCENE" \
  --group-strength "$STRENGTH" --train-case-min 40 --train-case-max 199 \
  --trees 180 --output-model "$RUN/final.json" \
  --output-summary "$RUN/final.summary.json"
echo "V22_GROUPED_RSCENE_FINAL_JOB_DONE strength=$STRENGTH"
date
