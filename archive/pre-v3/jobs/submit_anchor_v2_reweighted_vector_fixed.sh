#!/bin/bash
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
cd "$ROOT"
score_job=$(sbatch --parsable jobs/job_score_anchor_v2_reweighted_vector_fixed.sh)
analysis_job=$(sbatch --parsable --dependency=afterok:${score_job} jobs/job_analyze_anchor_v2_reweighted_vector_fixed.sh)
echo "score_job=$score_job"
echo "analysis_job=$analysis_job"
