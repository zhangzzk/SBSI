#!/bin/bash
# Score the frozen V2 refit without tuning or selection on anchor truth.
#SBATCH --job-name=lv2as_score
#SBATCH --time=00:40:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-99%20
#SBATCH --output=/home/z/Zekang.Zhang/logs/lv2as_score_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lv2as_score_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
LEGACY=$ROOT/archive/pre-v3
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1
MODEL=$RUN/weighted_model.json
MODEL_SHA=3cf70b6e74ad382f3ec59c6e8a2d0a5b9b0615d4c4677c7a71dd2344cbf35553
ORIGINAL_PARTS=$RUN/anchor_scores_fullv2_c400-899/original
COMPLEMENT_PARTS=$RUN/anchor_scores_fullv2_c400-899/complement
LABEL=v2_reweighted_vector_fixed_from_v22_trial15
START=$((400 + 5 * SLURM_ARRAY_TASK_ID))
STOP=$((START + 4))
HUNDRED=$((START / 100 * 100))
NEXT=$((HUNDRED + 99))
ORIGINAL_BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${HUNDRED}-${NEXT}
COMPLEMENT_BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_v2complement_c${HUNDRED}-${NEXT}
ORIGINAL_REFERENCE=$LEGACY/results/anchorblend_g002_response_v22_c${HUNDRED}-${NEXT}.feather
COMPLEMENT_REFERENCE=$LEGACY/results/anchorblend_g002_response_v2complement_c${HUNDRED}-${NEXT}.feather

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LEGACY:$LEGACY/scripts:$BE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$LEGACY"
mkdir -p "$ORIGINAL_PARTS" "$COMPLEMENT_PARTS"
for case in $(seq "$START" "$STOP"); do
  test ! -e "$ORIGINAL_PARTS/case${case}.feather"
  env SLURM_PROCID=0 python -u scripts/score_anchor_response_model.py --base "$ORIGINAL_BASE" \
    --reference "$ORIGINAL_REFERENCE" --model-tag lsst_r_extnbr_indom_tuned --reg-file "$MODEL" \
    --expected-reg-sha256 "$MODEL_SHA" --candidate-label "$LABEL" --output-dir "$ORIGINAL_PARTS" \
    --case-offset "$case" --n-cases 1 --sign 0.02
  test ! -e "$COMPLEMENT_PARTS/case${case}.feather"
  env SLURM_PROCID=0 python -u scripts/score_anchor_response_model.py --base "$COMPLEMENT_BASE" \
    --reference "$COMPLEMENT_REFERENCE" --reference-baseline-column R_blend_lsst_r_extnbr_indom_tuned \
    --model-tag lsst_r_extnbr_indom_tuned --reg-file "$MODEL" --expected-reg-sha256 "$MODEL_SHA" \
    --candidate-label "$LABEL" --output-dir "$COMPLEMENT_PARTS" --case-offset "$case" --n-cases 1 --sign 0.02
done
echo "LEGACY_V2_ANCHOR_SCORE_DONE cases=$START-$STOP"
