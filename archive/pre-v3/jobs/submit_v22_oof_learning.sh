#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
cd "$ROOT"

CACHE_JOB=$($SBATCH --parsable jobs/job_prepare_v22_oof_learning.sh)
BASE_JOB=$($SBATCH --parsable --dependency="afterok:${CACHE_JOB}" jobs/job_train_v22_oof_base.sh)
BASE_MERGE_JOB=$($SBATCH --parsable --dependency="afterok:${BASE_JOB}" jobs/job_merge_v22_oof_base.sh)
BIAS_JOB=$($SBATCH --parsable --dependency="afterok:${BASE_MERGE_JOB}" jobs/job_train_v22_oof_bias.sh)
BIAS_MERGE_JOB=$($SBATCH --parsable --dependency="afterok:${BIAS_JOB}" jobs/job_merge_v22_oof_bias.sh)
VARIANCE_JOB=$($SBATCH --parsable --dependency="afterok:${BIAS_MERGE_JOB}" jobs/job_train_v22_oof_variance.sh)
VARIANCE_WEIGHT_JOB=$($SBATCH --parsable --dependency="afterok:${VARIANCE_JOB}" jobs/job_train_v22_variance_weighted.sh)
WEIGHT_SCAN_JOB=$($SBATCH --parsable jobs/job_retrain_v22_positive_weight_weak.sh)
EVALUATION_JOB=$($SBATCH --parsable --dependency="afterok:${VARIANCE_WEIGHT_JOB}:${WEIGHT_SCAN_JOB}" jobs/job_evaluate_v22_oof_learning.sh)

printf 'cache=%s\nbase=%s\nbase_merge=%s\nbias=%s\nbias_merge=%s\nvariance=%s\nvariance_weighted=%s\nweight_scan=%s\nevaluation=%s\n' \
  "$CACHE_JOB" "$BASE_JOB" "$BASE_MERGE_JOB" "$BIAS_JOB" "$BIAS_MERGE_JOB" \
  "$VARIANCE_JOB" "$VARIANCE_WEIGHT_JOB" "$WEIGHT_SCAN_JOB" "$EVALUATION_JOB"
