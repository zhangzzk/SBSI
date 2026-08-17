#!/bin/bash
#SBATCH --job-name=ab1n_ana
#SBATCH --time=02:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab1n_ana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab1n_ana_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE_U=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_u_c400-499
BASE_V=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_v_c400-499
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_oneactive_orthogonal_c400-499
COHERENT=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather
OUT_FEATHER=$ROOT/results/anchorblend_oneactive_orthogonal_v22_c400-499.feather
OUT_JSON=$ROOT/results/anchorblend_oneactive_orthogonal_v22_c400-499.json
test ! -e "$OUT_FEATHER" || { echo "REFUSING existing $OUT_FEATHER"; exit 1; }
test ! -e "$OUT_JSON" || { echo "REFUSING existing $OUT_JSON"; exit 1; }
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/analyze_anchorblend_oneactive_orthogonal.py \
  --base-u "$BASE_U" --base-v "$BASE_V" --manifest-dir "$MANIFEST" \
  --coherent "$COHERENT" --cases $(seq 400 499) --g 0.05 \
  --output-feather "$OUT_FEATHER" --output-json "$OUT_JSON"
echo ANCHORBLEND_ONEACTIVE_ANALYZE_JOB_DONE
date
