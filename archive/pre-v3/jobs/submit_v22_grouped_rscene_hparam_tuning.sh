#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
cd "$ROOT"

CANDIDATES=$($SBATCH --parsable jobs/job_train_v22_grouped_rscene_hparam_candidates.sh)
EVALUATION=$($SBATCH --parsable --dependency="afterok:${CANDIDATES}" jobs/job_evaluate_v22_grouped_rscene_hparam_candidates.sh)
FINAL=$($SBATCH --parsable --dependency="afterok:${EVALUATION}" jobs/job_train_v22_grouped_rscene_hparam_final.sh)
CHECK=$($SBATCH --parsable --dependency="afterok:${FINAL}" jobs/job_evaluate_v22_grouped_rscene_hparam_final.sh)
echo "candidates=$CANDIDATES evaluation=$EVALUATION final=$FINAL check=$CHECK"
