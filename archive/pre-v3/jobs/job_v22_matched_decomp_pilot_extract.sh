#!/bin/bash
#SBATCH --job-name=v22dc_ext
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22dc_ext_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22dc_ext_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22decomp_pilot
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_pilot
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_pilot_truth.feather
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/extract_v22_matched_decomposition.py \
  --base "total=${PREFIX}_total" \
  --base "self=${PREFIX}_self" \
  --base "neighbour=${PREFIX}_neighbour" \
  --base "pair0=${PREFIX}_pair0" \
  --base "pair1=${PREFIX}_pair1" \
  --manifest-dir "$MANIFEST" --cases 300 301 302 303 --g 0.05 --output "$OUT"
echo V22_MATCHED_DECOMP_PILOT_EXTRACT_DONE

