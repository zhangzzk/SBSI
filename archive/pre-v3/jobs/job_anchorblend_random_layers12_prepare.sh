#!/bin/bash
#SBATCH --job-name=ab_rlprep
#SBATCH --array=0-3
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_rlprep_%A_%a.out
set -euo pipefail
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
LAYER=$((TASK / 2 + 1))
if (( TASK % 2 == 0 )); then RADIUS=10; else RADIUS=15; fi
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
SRC=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
OUT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_random_layer${LAYER}_local${RADIUS}_c200-299
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
[ ! -e "$OUT" ] || { echo "REFUSING existing output root $OUT"; exit 1; }
"$PY" -u scripts/prepare_anchorblend_guarded_catalogues.py \
  --source "$SRC" --output "$OUT" --cases $(seq 200 299) \
  --g 0.05 --radius "$RADIUS" --min-separation 30.01 \
  --random-anchor-directions --direction-seed 25876 --anchor-layer "$LAYER"
echo "ANCHORBLEND_RANDOM_LAYER${LAYER}_LOCAL${RADIUS}_PREP_DONE"; date
