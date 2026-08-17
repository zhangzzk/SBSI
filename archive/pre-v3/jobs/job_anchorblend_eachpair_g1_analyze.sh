#!/bin/bash
#SBATCH --job-name=abep_ana
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abep_ana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abep_ana_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_eachpair_g1
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_eachpair_g1_pilot_c400-409
COHERENT=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather
OUT_ANCHOR=$ROOT/results/anchorblend_eachpair_g1_v22_pilot_c400-409.feather
OUT_PAIR=$ROOT/results/anchorblend_eachpair_g1_pairs_v22_pilot_c400-409.feather
OUT_JSON=$ROOT/results/anchorblend_eachpair_g1_v22_pilot_c400-409.json
for output in "$OUT_ANCHOR" "$OUT_PAIR" "$OUT_JSON"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/analyze_anchorblend_eachpair_g1.py \
  --output-prefix "$PREFIX" --manifest-dir "$MANIFEST" \
  --coherent "$COHERENT" --cases $(seq 400 409) --g 0.05 --max-rank 18 \
  --output-feather "$OUT_ANCHOR" --output-pairs "$OUT_PAIR" --output-json "$OUT_JSON"
echo ANCHORBLEND_EACHPAIR_G1_ANALYZE_JOB_DONE
date
