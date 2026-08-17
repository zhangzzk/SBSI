#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
cd "$ROOT"

score_job=$($SBATCH --parsable jobs/job_score_anchor_response_weak_weight_transfer.sh)
analysis_job=$($SBATCH --parsable --dependency="afterok:${score_job}" jobs/job_analyze_anchor_response_weak_weight_transfer.sh)
printf 'score=%s\nanalysis=%s\n' "$score_job" "$analysis_job"
