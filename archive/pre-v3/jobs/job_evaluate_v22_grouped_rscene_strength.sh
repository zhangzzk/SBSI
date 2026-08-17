#!/bin/bash
#SBATCH --job-name=v22gain
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
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_tune_v1/run
TREE_TUNING=$ROOT/results/v22_grouped_rscene_hparam_tuning_v1_c40-139_es140-159_sel160-199_dev0-19.json
OUT=$ROOT/results/v22_grouped_rscene_strength_tuning_v1_sel160-199_dev0-19
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$TREE_TUNING"
test -s "$RUN/candidate_01.summary.json"
python -u scripts/evaluate_v22_grouped_rscene_strength.py \
  --source-cache "$SOURCE" --scene-cache "$SCENE" \
  --candidate-summary "$RUN/candidate_01.summary.json" \
  --tree-tuning-result "$TREE_TUNING" --output-prefix "$OUT"
echo V22_RSCENE_STRENGTH_TUNING_JOB_DONE
date
