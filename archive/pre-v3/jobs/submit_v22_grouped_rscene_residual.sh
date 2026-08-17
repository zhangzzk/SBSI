#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
cd "$ROOT"

CACHE_JOB=$($SBATCH --parsable jobs/job_prepare_v22_grouped_rscene_cache.sh)
TRAIN_JOB=$($SBATCH --parsable --dependency="afterok:${CACHE_JOB}" jobs/job_train_v22_grouped_rscene_candidates.sh)
EVAL_JOB=$($SBATCH --parsable --dependency="afterok:${TRAIN_JOB}" jobs/job_evaluate_v22_grouped_rscene_candidates.sh)
echo "cache=$CACHE_JOB candidates=$TRAIN_JOB evaluation=$EVAL_JOB"
