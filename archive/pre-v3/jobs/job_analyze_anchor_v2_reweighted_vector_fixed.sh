#!/bin/bash
#SBATCH --job-name=ab_v2_rwv_an
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_v2_rwv_an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_v2_rwv_an_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/anchor_scores_c400-899
LABEL=v2_reweighted_vector_fixed_from_v22_trial15
OUTPUT=results/anchorblend_v2_reweighted_vector_fixed_g002_c400-899.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$ROOT/scripts:${PYTHONPATH:-}"
cd "$ROOT"

[ ! -e "$OUTPUT" ] || { echo "REFUSING to overwrite $OUTPUT"; exit 1; }
python -u scripts/analyze_anchor_response_weighted_family.py \
  --parts-root "$PARTS" --case-min 400 --case-max 899 \
  --tags "$LABEL" --output "$OUTPUT"
test -s "$OUTPUT"
echo ANCHOR_V2_REWEIGHTED_VECTOR_FIXED_ANALYSIS_DONE
