#!/bin/bash
# Validate the archived response/scoring implementation on already measured
# complement case 400 before the production response chain starts.
#SBATCH --job-name=lv2as_smoke
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lv2as_smoke_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lv2as_smoke_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
LEGACY=$ROOT/archive/pre-v3
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_v2_complement_smoke_case400
REFERENCE=$LEGACY/results/anchorblend_v2_complement_smoke_case400.feather
SCORES=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/anchor_scores_fullv2_c400-899/smoke_complement
MODEL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/weighted_model.json
MODEL_SHA=3cf70b6e74ad382f3ec59c6e8a2d0a5b9b0615d4c4677c7a71dd2344cbf35553

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$LEGACY:$LEGACY/scripts:$BE:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$LEGACY"
test ! -e "$REFERENCE"
python -u scripts/build_anchorblend_response.py --base "$BASE" --cases 400 --g 0.02 \
  --primary-domain v2-complement --tags lsst_r_extnbr_indom_tuned --output "$REFERENCE"
mkdir -p "$SCORES"
test ! -e "$SCORES/case400.feather"
python -u scripts/score_anchor_response_model.py --base "$BASE" --reference "$REFERENCE" \
  --reference-baseline-column R_blend_lsst_r_extnbr_indom_tuned \
  --model-tag lsst_r_extnbr_indom_tuned --reg-file "$MODEL" --expected-reg-sha256 "$MODEL_SHA" \
  --candidate-label v2_reweighted_vector_fixed_from_v22_trial15 --output-dir "$SCORES" \
  --case-offset 400 --n-cases 1 --sign 0.02
test -s "$REFERENCE"
test -s "$SCORES/case400.feather"
echo LEGACY_V2_ANCHOR_SMOKE_SCORE_DONE
