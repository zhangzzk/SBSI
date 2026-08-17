#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
cd "$ROOT"

train_job=$(/opt/slurm/bin/sbatch --parsable jobs/job_train_v2_reweighted_vector_fixed.sh)
lookup_job=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$train_job" \
  jobs/job_build_lookup_v2_reweighted_vector_fixed.sh)
eval_job=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$lookup_job" \
  jobs/job_eval_m_swap_v2_reweighted_vector_fixed.sh)

printf 'train_job=%s\nlookup_job=%s\neval_job=%s\n' "$train_job" "$lookup_job" "$eval_job"
