#!/bin/bash
#SBATCH --job-name=v29cg_feat
#SBATCH --time=01:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v29cg_feat_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RBLEND=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend/rblend_const_c40-139.feather
PRIMARY=$C/const_primary_features_c40-139.feather
OUT=$C/const_features_rblend_primary_c40-139.feather
mkdir -p "$C"
for f in "$CAT" "$RBLEND"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
if [ ! -e "$PRIMARY" ]; then
  "$PY" -u scripts/compact_key_features.py \
    --catalogue "$CAT" --columns r_input_p Re_input_p \
    --min-case 40 --max-case 139 --output "$PRIMARY"
fi
if [ ! -e "$OUT" ]; then
  "$PY" -u scripts/augment_catalogue_lookup.py \
    --catalogue "$PRIMARY" --lookup "$RBLEND" --columns r_blend \
    --drop-unmatched --output "$OUT"
fi
echo V29_CONDITIONAL_FEATURES_DONE; date
