#!/bin/bash
#SBATCH --job-name=hsv26merge
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv26merge_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v26_scene_target_c40-199
OUT=$OUTDIR/halfshear_selfresp_v26_scene_target_c40-199.feather
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/merge_halfshear_selfresp_parts.py \
  --parts "$OUTDIR/parts/part_s*.feather" --out "$OUT" \
  --min-case 40 --max-case 199 --expect-seeds 501 502 503 505
echo HS_V26_SCENE_MERGE_DONE; date
