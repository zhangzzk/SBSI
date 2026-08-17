#!/bin/bash
#SBATCH --job-name=abv2sup_an
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abv2sup_an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abv2sup_an_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1
OUTPUT=$ROOT/results/anchorblend_v2_reweighted_vector_fixed_fullv2_supplement_c400-899.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$ROOT/scripts:${PYTHONPATH:-}"
cd "$ROOT"
[ ! -e "$OUTPUT" ] || { echo "REFUSING existing $OUTPUT"; exit 1; }
python -u scripts/analyze_anchor_v2_supplement.py \
  --original-parts "$RUN/anchor_scores_fullv2_c400-899/original" \
  --complement-parts "$RUN/anchor_scores_fullv2_c400-899/complement" \
  --original-base-prefix /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c \
  --complement-base-prefix /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_v2complement_c \
  --case-min 400 --case-max 899 --output "$OUTPUT"
test -s "$OUTPUT"
test -s "${OUTPUT%.json}.cases.csv"
echo ANCHOR_V2_SUPPLEMENT_ANALYSIS_DONE
