#!/bin/bash
#SBATCH --job-name=ab_biaspairs
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-4
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

# Build model-only pair features from the exact renderer input catalogue used
# by build_anchorblend_response.py.  This avoids two rare but real differences
# from the generated latent catalogue: renderer edge clipping and float32 versus
# float64 feature rescaling at an XGBoost split boundary.

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BLOCKS=(400-499 500-599 600-699 700-799 800-899)
BLOCK=${BLOCKS[$SLURM_ARRAY_TASK_ID]}
START=${BLOCK%-*}
STOP=${BLOCK#*-}
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${BLOCK}
PAIR_PREFIX=pairs_renderer_v22
SUMMARY=$ROOT/results/anchorblend_g002_pairs_renderer_v22_c${BLOCK}.json
STEM=$ROOT/results/anchorblend_g002_dominance_renderer_v22_c${BLOCK}
DEV_MAX=$(( START + (STOP - START) / 2 ))
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

if [ ! -e "$SUMMARY" ]; then
  python -u scripts/build_anchorblend_pair_responses.py \
    --base "$BASE" --manifest-dir "$BASE" \
    --cases $(seq "$START" "$STOP") --g 0.02 \
    --catalogue-source rendered --pair-prefix "$PAIR_PREFIX" \
    --tag lsst_r_extnbr_v22 --summary-json "$SUMMARY"
else
  for case in $(seq "$START" "$STOP"); do
    test -s "$BASE/${PAIR_PREFIX}_case${case}.feather" || {
      echo "INCOMPLETE existing exact-pair block at case $case"; exit 1; }
  done
  echo "reusing completed exact-pair block $BLOCK"
fi

for output in "$STEM.feather" "$STEM.json" "$STEM.md"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
python -u scripts/localize_anchorblend_response_dominance.py \
  --manifest-dir "$BASE" --reference-base "$BASE" --reference-shear 0.02 \
  --pair-prefix "$PAIR_PREFIX" --tag lsst_r_extnbr_v22 \
  --case-min "$START" --case-max "$STOP" --development-max "$DEV_MAX" \
  --dominance-threshold 0.7752772106835227 \
  --output-feather "$STEM.feather" --output-json "$STEM.json" \
  --output-md "$STEM.md"
echo "ANCHOR_BIAS_PAIR_INPUTS_DONE block=$BLOCK"
date
