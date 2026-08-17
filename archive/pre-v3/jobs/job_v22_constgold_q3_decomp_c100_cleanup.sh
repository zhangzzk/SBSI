#!/bin/bash
#SBATCH --job-name=v22q3c_clean
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3c_clean_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3c_clean_%j.err

set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: $0 START COUNT"; exit 2; }
START=$1
COUNT=$2
[[ "$START" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || { echo "numeric START/COUNT required"; exit 2; }
END=$((START + COUNT - 1))
(( START >= 48 && END <= 139 && COUNT >= 1 && COUNT <= 20 )) || { echo "invalid chunk $START..$END"; exit 2; }
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22cgq3_c48-139
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
TRUTH=$CACHE/v22_constgold_q3_decomp_c48-139_truth_c${START}-${END}.feather
RECEIPT=$CACHE/v22_constgold_q3_decomp_c48-139_cleanup_c${START}-${END}.json
CASES=()
for ((case=START; case<=END; case++)); do CASES+=("$case"); done
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/cleanup_v22_constgold_bin_images.py \
  --base "total=${PREFIX}_total" \
  --base "self=${PREFIX}_self" \
  --base "deployed=${PREFIX}_deployed" \
  --base "other=${PREFIX}_other" \
  --truth "$TRUTH" --cases "${CASES[@]}" --signs 0.02 -0.02 --receipt "$RECEIPT"
echo "V22_CONSTGOLD_Q3_DECOMP_C100_CLEANUP_DONE cases=$START-$END"
