#!/bin/bash
#SBATCH --job-name=ab_v2_rwvfix
#SBATCH --time=00:40:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-99%20
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_v2_rwvfix_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_v2_rwvfix_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
MODEL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/weighted_model.json
MODEL_SHA=3cf70b6e74ad382f3ec59c6e8a2d0a5b9b0615d4c4677c7a71dd2344cbf35553
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/anchor_scores_c400-899
LABEL=v2_reweighted_vector_fixed_from_v22_trial15
START=$((400 + 5 * SLURM_ARRAY_TASK_ID))
STOP=$((START + 4))
HUNDRED=$((START / 100 * 100))
NEXT=$((HUNDRED + 99))
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${HUNDRED}-${NEXT}
REFERENCE=$ROOT/results/anchorblend_g002_response_v22_c${HUNDRED}-${NEXT}.feather

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$ROOT/scripts:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$MODEL"
test -s "$REFERENCE"
test -d "$BASE"
mkdir -p "$PARTS/$LABEL"
for case in $(seq "$START" "$STOP"); do
  env SLURM_PROCID=0 python -u scripts/score_anchor_response_model.py \
    --base "$BASE" --reference "$REFERENCE" \
    --model-tag lsst_r_extnbr_indom_tuned \
    --reg-file "$MODEL" --expected-reg-sha256 "$MODEL_SHA" \
    --candidate-label "$LABEL" --output-dir "$PARTS/$LABEL" \
    --case-offset "$case" --n-cases 1 --sign 0.02
done
echo "ANCHOR_V2_REWEIGHTED_VECTOR_FIXED_SCORE_DONE cases=$START-$STOP"
