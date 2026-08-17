#!/bin/bash
#SBATCH --job-name=v22rk_prep
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22rk_prep_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22rk_prep_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
RANK_PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22decomp_rankpilot
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22decomp_pilot_total
SOURCE_MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_pilot
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_rankpilot
if [ -e "$MANIFEST" ]; then
  echo "REFUSING: rank-pilot manifest already exists: $MANIFEST"
  exit 1
fi
BASE_ARGS=()
for rank in $(seq 0 18); do
  BASE_ARGS+=(--base "rank${rank}=${RANK_PREFIX}_rank${rank}")
done
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/prepare_v22_matched_rank_extension.py \
  "${BASE_ARGS[@]}" --reference-base "$REFERENCE" \
  --source-manifest "$SOURCE_MANIFEST" --manifest-dir "$MANIFEST" \
  --cases 300 301 302 303 --g 0.05 --random-seed 73021
echo V22_MATCHED_RANKPILOT_PREP_DONE
