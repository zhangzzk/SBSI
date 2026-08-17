#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
SELECTION=$ROOT/results/v22_grouped_rscene_candidates_v1_c40-159_dev0-19_val160-199.json
cd "$ROOT"

STRENGTH=$(python - "$SELECTION" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
selected = payload["selection"]["selected_model"]
if selected is None:
    raise SystemExit("candidate protocol selected no final model")
row = next(
    item for item in payload["selection"]["candidate_rows"]
    if item["model"] == selected
)
print(row["strength"])
PY
)
FINAL_JOB=$($SBATCH --parsable --export="ALL,GROUP_STRENGTH=$STRENGTH" jobs/job_train_v22_grouped_rscene_final.sh)
ANCHOR_JOB=$($SBATCH --parsable --dependency="afterok:${FINAL_JOB}" jobs/job_score_anchor_v22_grouped_rscene.sh)
TRANSFER_JOB=$($SBATCH --parsable --dependency="afterok:${ANCHOR_JOB}" jobs/job_evaluate_v22_grouped_rscene_final_transfer.sh)
echo "strength=$STRENGTH final=$FINAL_JOB anchors=$ANCHOR_JOB transfer=$TRANSFER_JOB"
