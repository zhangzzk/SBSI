#!/bin/bash
#SBATCH --job-name=v22hpfinal
#SBATCH --time=03:00:00
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
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_tune_v1/run
SELECTION=$ROOT/results/v22_grouped_rscene_hparam_tuning_v1_c40-139_es140-159_sel160-199_dev0-19.json
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$SELECTION"
python -u scripts/train_v22_grouped_rscene_hparam_final.py \
  --source-cache "$SOURCE" --scene-cache "$SCENE" --selection "$SELECTION" \
  --output-model "$RUN/final.json" --output-summary "$RUN/final.summary.json"
echo V22_RSCENE_HPARAM_FINAL_JOB_DONE
date
