#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_weak_weight_transfer_g002_c400-899
TAG010=lsst_r_extnbr_v22_rpowposa010
TAG020=lsst_r_extnbr_v22_rpowposw0200
cd "$ROOT"

test ! -e "/home/z/Zekang.Zhang/blendemu/models/regression_model_${TAG020}.json"
test ! -e "/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_${TAG020}.json"
test ! -e "$PARTS/$TAG010"
test ! -e "$PARTS/$TAG020"
test ! -e results/anchorblend_response_alpha_scan_transfer_v22_g002_c400-899.json

train020=$($SBATCH --parsable jobs/job_train_v22_positive_weight_alpha020.sh)
score010=$($SBATCH --parsable --export="ALL,AB_MODEL_TAG=${TAG010}" jobs/job_score_anchor_response_high_alpha_transfer.sh)
score020=$($SBATCH --parsable --dependency="afterok:${train020}" --export="ALL,AB_MODEL_TAG=${TAG020}" jobs/job_score_anchor_response_high_alpha_transfer.sh)
analysis=$($SBATCH --parsable --dependency="afterok:${score010}:${score020}" jobs/job_analyze_anchor_response_high_alpha_transfer.sh)
printf 'train_alpha020=%s\nscore_alpha010=%s\nscore_alpha020=%s\nanalysis=%s\n' \
  "$train020" "$score010" "$score020" "$analysis"
