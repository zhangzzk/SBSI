#!/bin/bash
#SBATCH --job-name=ab_rsupport
#SBATCH --array=0-1
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_rsupport_%A_%a.out
set -euo pipefail

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
case "$TASK" in 0) RADIUS=10 ;; 1) RADIUS=15 ;; *) exit 1 ;; esac
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
SRC=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
OUT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_random_local${RADIUS}_c200-299
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

[ -d "$OUT" ] || { echo "Missing prepared output root $OUT"; exit 1; }
"$PY" -u scripts/prepare_anchorblend_guarded_catalogues.py \
  --source "$SRC" --output "$OUT" --cases $(seq 200 299) \
  --g 0.05 --support-only

echo "ANCHORBLEND_RANDOM_LOCAL${RADIUS}_SUPPORT_DONE"; date
