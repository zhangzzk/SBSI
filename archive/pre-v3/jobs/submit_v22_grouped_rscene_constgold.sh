#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
cd "$ROOT"

lookup_job=$(/opt/slurm/bin/sbatch --parsable \
  jobs/job_build_v22_grouped_rscene_constgold_lookup.sh)
eval_job=$(/opt/slurm/bin/sbatch --parsable --dependency="afterok:$lookup_job" \
  jobs/job_eval_v22_grouped_rscene_constgold.sh)
echo "lookup=$lookup_job eval=$eval_job"
