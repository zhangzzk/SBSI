#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
cd "$ROOT"

TRAIN=$($SBATCH --parsable jobs/job_train_v22_proxy_qd2_correction.sh)
ANCHOR=$($SBATCH --parsable --dependency="afterok:${TRAIN}" jobs/job_score_anchor_v22_proxy_qd2_correction.sh)
EVAL=$($SBATCH --parsable --dependency="afterok:${ANCHOR}" jobs/job_evaluate_v22_proxy_qd2_correction.sh)
echo "train=$TRAIN anchors=$ANCHOR evaluation=$EVAL"
