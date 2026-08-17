#!/bin/bash
#SBATCH --job-name=rbl2x2fit
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter,cip
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-1
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_fullneighbour_context_v1
PARAMS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_fullneighbour_fixed_v1/summary.json

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
  FIT_MIN=40
  FIT_MAX=199
  FINAL_MIN=0
  FINAL_MAX=39
  CONTEXT=full
  OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_fullneighbour_40_199_v1
else
  FIT_MIN=0
  FIT_MAX=159
  FINAL_MIN=160
  FINAL_MAX=199
  CONTEXT=half
  OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_halfneighbour_0_159_v1
fi

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
cd "$ROOT"

mkdir -p "$OUTPUT"
test -s "$SOURCE/metadata.json"
test -s "$FULL/metadata.json"
test -s "$PARAMS"
nvidia-smi -L
python -u scripts/train_newbase_oldway_fullneighbour_160_40.py \
  --source-cache "$SOURCE" \
  --full-neighbour-cache "$FULL" \
  --base-parameter-summary "$PARAMS" \
  --fit-case-min "$FIT_MIN" --fit-case-max "$FIT_MAX" \
  --final-case-min "$FINAL_MIN" --final-case-max "$FINAL_MAX" \
  --scene-context "$CONTEXT" \
  --output-dir "$OUTPUT"
echo "TRAIN_NEWBASE_OLDWAY_2X2_CELL_DONE context=$CONTEXT fit=$FIT_MIN-$FIT_MAX"
date
