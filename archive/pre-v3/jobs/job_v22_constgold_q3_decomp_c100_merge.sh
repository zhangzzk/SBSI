#!/bin/bash
#SBATCH --job-name=v22q3c_merge
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3c_merge_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3c_merge_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
PILOT=$CACHE/v22_constgold_q3_decomp_pilot_truth.feather
PREFIX=$CACHE/v22_constgold_q3_decomp_c48-139_truth
OUT=$CACHE/v22_constgold_q3_decomp_c40-139_truth.feather
CASES=({40..139})
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/merge_v22_constgold_bin_decomposition.py \
  --input "$PILOT" \
  --input "${PREFIX}_c48-67.feather" \
  --input "${PREFIX}_c68-87.feather" \
  --input "${PREFIX}_c88-107.feather" \
  --input "${PREFIX}_c108-127.feather" \
  --input "${PREFIX}_c128-139.feather" \
  --cases "${CASES[@]}" --output "$OUT"
echo V22_CONSTGOLD_Q3_DECOMP_C100_MERGE_DONE
