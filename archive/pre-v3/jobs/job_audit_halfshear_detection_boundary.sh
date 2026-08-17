#!/bin/bash
#SBATCH --job-name=hs_det_audit
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_det_audit_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_det_audit_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/audit_halfshear_detection_boundary.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather \
  --table-output /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_detection_c0-39.feather \
  --output results/halfshear_detection_boundary_v22_c0-39.json
echo HALFSHEAR_DETECTION_BOUNDARY_JOB_DONE
date
