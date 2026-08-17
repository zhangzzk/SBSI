#!/bin/bash
#SBATCH --job-name=ab_o3flux
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_o3flux_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_o3flux_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
OUT=results/neighbor_flux_shells_anchor_c200-299.feather
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/build_neighbor_flux_shells.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --cases $(seq 200 299) --sign 0.05 --output "$OUT"
echo ANCHOR_OTHERFLUX_LOOKUP_DONE
date
