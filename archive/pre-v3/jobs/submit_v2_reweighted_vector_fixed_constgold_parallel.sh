#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
cd "$ROOT"

lookup_job=$(/opt/slurm/bin/sbatch --parsable jobs/job_build_lookup_v2_reweighted_vector_fixed_array.sh)
merge_job=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$lookup_job" \
  jobs/job_merge_lookup_v2_reweighted_vector_fixed.sh)
eval_job=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$merge_job" \
  jobs/job_eval_m_swap_v2_reweighted_vector_fixed.sh)

printf 'lookup_array_job=%s\nmerge_job=%s\neval_job=%s\n' "$lookup_job" "$merge_job" "$eval_job"
