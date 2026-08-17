#!/bin/bash
#SBATCH --job-name=v22hpeval
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
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
OUT=$ROOT/results/v22_grouped_rscene_hparam_tuning_v1_c40-139_es140-159_sel160-199_dev0-19
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

ARGS=()
for id in $(seq 0 17); do
  SUMMARY=$(printf '%s/candidate_%02d.summary.json' "$RUN" "$id")
  test -s "$SUMMARY"
  ARGS+=(--candidate-summary "$SUMMARY")
done
python -u scripts/evaluate_v22_grouped_rscene_hparam_candidates.py \
  --source-cache "$SOURCE" --scene-cache "$SCENE" \
  "${ARGS[@]}" --output-prefix "$OUT"
echo V22_RSCENE_HPARAM_EVALUATION_JOB_DONE
date
