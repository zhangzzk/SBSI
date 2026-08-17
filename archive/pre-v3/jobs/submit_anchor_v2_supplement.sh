#!/bin/bash
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
cd "$ROOT"
prep=$(sbatch --parsable jobs/job_prepare_anchor_v2_complement.sh)
sim=$(sbatch --parsable --dependency=afterok:${prep} jobs/job_sim_anchor_v2_complement.sh)
shape=$(sbatch --parsable --dependency=afterok:${sim} jobs/job_shape_anchor_v2_complement.sh)
response=$(sbatch --parsable --dependency=afterok:${shape} jobs/job_response_anchor_v2_complement.sh)
score=$(sbatch --parsable --dependency=afterok:${response} jobs/job_score_anchor_v2_supplement.sh)
analysis=$(sbatch --parsable --dependency=afterok:${score} jobs/job_analyze_anchor_v2_supplement.sh)
printf 'prep=%s\nsim=%s\nshape=%s\nresponse=%s\nscore=%s\nanalysis=%s\n' \
  "$prep" "$sim" "$shape" "$response" "$score" "$analysis"
