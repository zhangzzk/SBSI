#!/bin/bash
#SBATCH --job-name=ab_rlcoh
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_rlcoh_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_rlcoh_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BASE=/project/ls-gruen/users/zekang.zhang
cd "$ROOT"
OUT=${OUT:-results/anchorblend_random_layers012_vs_coherent_v22_c200-299.json}
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/analyze_anchorblend_multilayer_vs_coherent.py \
  --coherent-response results/anchorblend_g005_response_v22_c100-299.feather \
  --random10-response results/anchorblend_random_local10_response_v22_c200-299.feather \
  --random15-response results/anchorblend_random_local15_response_v22_c200-299.feather \
  --manifest-root "$BASE/lsst_sims_fs2_25876_anchorblend_random_local10_c200-299" \
  --random10-response results/anchorblend_random_layer1_local10_response_v22_c200-299.feather \
  --random15-response results/anchorblend_random_layer1_local15_response_v22_c200-299.feather \
  --manifest-root "$BASE/lsst_sims_fs2_25876_anchorblend_random_layer1_local10_c200-299" \
  --random10-response results/anchorblend_random_layer2_local10_response_v22_c200-299.feather \
  --random15-response results/anchorblend_random_layer2_local15_response_v22_c200-299.feather \
  --manifest-root "$BASE/lsst_sims_fs2_25876_anchorblend_random_layer2_local10_c200-299" \
  --case-min 200 --case-max 299 --output "$OUT"
echo ANCHORBLEND_RANDOM_LAYERS012_VS_COHERENT_JOB_DONE; date
