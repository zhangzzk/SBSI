#!/bin/bash
#SBATCH --job-name=ab_rlaudit
#SBATCH --array=0-1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_rlaudit_%A_%a.out
set -euo pipefail
LAYER=$((SLURM_ARRAY_TASK_ID + 1))
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BASE=/project/ls-gruen/users/zekang.zhang
OUT=$ROOT/results/anchorblend_random_layer${LAYER}_catalogue_audit_c200-299.json
cd "$ROOT"
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/audit_anchorblend_random_catalogues.py \
  --root10 "$BASE/lsst_sims_fs2_25876_anchorblend_random_layer${LAYER}_local10_c200-299" \
  --root15 "$BASE/lsst_sims_fs2_25876_anchorblend_random_layer${LAYER}_local15_c200-299" \
  --cases $(seq 200 299) --g 0.05 --output "$OUT"
echo "ANCHORBLEND_RANDOM_LAYER${LAYER}_CATALOGUE_AUDIT_JOB_DONE"; date
