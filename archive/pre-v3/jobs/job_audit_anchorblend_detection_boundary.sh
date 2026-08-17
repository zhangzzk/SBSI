#!/bin/bash
#SBATCH --job-name=ab_detectaudit
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_detectaudit_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_detectaudit_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
TABLE=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_detection_c200-299.feather
OUT=$ROOT/results/anchorblend_detection_boundary_v22_c200-299.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$ROOT"
[ ! -e "$TABLE" ] || { echo "REFUSING existing $TABLE"; exit 1; }
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/audit_anchorblend_detection_boundary.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --table-output "$TABLE" --output "$OUT"
echo ANCHORBLEND_DETECTION_BOUNDARY_AUDIT_JOB_DONE; date
