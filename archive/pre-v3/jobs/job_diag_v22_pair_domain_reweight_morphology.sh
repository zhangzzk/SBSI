#!/bin/bash
#SBATCH --job-name=v22_pairrwmorph
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_pairrwmorph_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22_pairrwmorph_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
BASE=/project/ls-gruen/users/zekang.zhang
OUT=$ROOT/results/v22_pair_domain_reweight_morphology_c0-39_to_anchor_c200-299.json
cd "$ROOT"
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
"$PY" -u scripts/diag_v22_pair_domain_reweight.py \
  --halfshear-catalogue "$BASE/lsst_sims_fs2_25876/response_catalogue_train.feather" \
  --anchor-base "$BASE/lsst_sims_fs2_25876_anchorblend_random_local10_c200-299" \
  --response10 results/anchorblend_random_local10_response_v22_c200-299.feather \
  --response15 results/anchorblend_random_local15_response_v22_c200-299.feather \
  --response10 results/anchorblend_random_layer1_local10_response_v22_c200-299.feather \
  --response15 results/anchorblend_random_layer1_local15_response_v22_c200-299.feather \
  --response10 results/anchorblend_random_layer2_local10_response_v22_c200-299.feather \
  --response15 results/anchorblend_random_layer2_local15_response_v22_c200-299.feather \
  --anchor-result results/anchorblend_random_layers012_radii_v22_c200-299.json \
  --feature-set morphology --output "$OUT"
echo V22_PAIR_DOMAIN_REWEIGHT_MORPHOLOGY_DONE
date
