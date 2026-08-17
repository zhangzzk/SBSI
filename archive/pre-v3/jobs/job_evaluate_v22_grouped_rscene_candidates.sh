#!/bin/bash
#SBATCH --job-name=v22grpeval
#SBATCH --time=02:00:00
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
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/run
OUT=$ROOT/results/v22_grouped_rscene_candidates_v1_c40-159_dev0-19_val160-199
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

for strength in 0 1 3 10 30; do
  test -s "$RUN/candidate_lambda_${strength}.summary.json"
done
python -u scripts/evaluate_v22_grouped_rscene_candidates.py \
  --source-cache "$SOURCE" --scene-cache "$SCENE" \
  --candidate-summary "$RUN/candidate_lambda_0.summary.json" \
  --candidate-summary "$RUN/candidate_lambda_1.summary.json" \
  --candidate-summary "$RUN/candidate_lambda_3.summary.json" \
  --candidate-summary "$RUN/candidate_lambda_10.summary.json" \
  --candidate-summary "$RUN/candidate_lambda_30.summary.json" \
  --output-prefix "$OUT"
echo V22_GROUPED_RSCENE_CANDIDATE_EVAL_JOB_DONE
date
