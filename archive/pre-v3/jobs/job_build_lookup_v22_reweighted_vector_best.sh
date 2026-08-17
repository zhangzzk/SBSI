#!/bin/bash
#SBATCH --job-name=lk_v22_rwvbest
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_v22_rwvbest_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lk_v22_rwvbest_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
MODEL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json
EXPECTED_SHA=01decd1335ce1c23aac1ef6ba055ae01c3950e47c4046345dcb6a1813033c21f
OUT=results/blend_lookup_v22_reweighted_vector_optuna30_best_c40-139.feather

export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
export XGB_DEVICE=cpu
cd "$ROOT"

test "$(sha256sum "$MODEL" | awk '{print $1}')" = "$EXPECTED_SHA"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/build_blend_lookup.py \
  --cases $(seq 40 139) \
  --tag lsst_r_extnbr_v22 \
  --reg-file "$MODEL" \
  --output "$OUT"
test -s "$OUT"
echo V22_REWEIGHTED_VECTOR_BEST_LOOKUP_DONE
