#!/bin/bash
#SBATCH --job-name=lk_v22_pos010
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_v22_pos010_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lk_v22_pos010_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
export XGB_DEVICE=cpu
cd "$ROOT"

OUT=results/blend_lookup_v22_rpowposa010_c40-139.feather
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/build_blend_lookup.py \
  --cases $(seq 40 139) \
  --tag lsst_r_extnbr_v22_rpowposa010 \
  --output "$OUT"
test -s "$OUT"
echo V22_POSITIVE010_LOOKUP_DONE
