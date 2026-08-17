#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
cd "$ROOT"

lookup_job=$(/opt/slurm/bin/sbatch --parsable jobs/job_build_lookup_v22_positive010.sh)
eval_job=$(/opt/slurm/bin/sbatch --parsable --dependency=afterok:"$lookup_job" \
  jobs/job_eval_m_swap_v22_positive010.sh)

printf 'lookup_job=%s\neval_job=%s\n' "$lookup_job" "$eval_job"
