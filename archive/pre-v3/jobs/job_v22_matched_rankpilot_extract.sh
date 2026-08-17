#!/bin/bash
#SBATCH --job-name=v22rk_ext
#SBATCH --time=08:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22rk_ext_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22rk_ext_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BROAD_PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22decomp_pilot
RANK_PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22decomp_rankpilot
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_rankpilot
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_rankpilot_truth.feather
if [ -e "$OUT" ]; then
  echo "REFUSING: rank truth already exists: $OUT"
  exit 1
fi
BASE_ARGS=(
  --base "total=${BROAD_PREFIX}_total"
  --base "self=${BROAD_PREFIX}_self"
  --base "neighbour=${BROAD_PREFIX}_neighbour"
)
for rank in $(seq 0 18); do
  BASE_ARGS+=(--base "rank${rank}=${RANK_PREFIX}_rank${rank}")
done
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/extract_v22_matched_decomposition.py \
  "${BASE_ARGS[@]}" --manifest-dir "$MANIFEST" \
  --cases 300 301 302 303 --g 0.05 --output "$OUT"
echo V22_MATCHED_RANKPILOT_EXTRACT_DONE
