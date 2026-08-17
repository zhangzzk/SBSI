#!/bin/bash
#SBATCH --job-name=cmp_v22_phys2_pilot
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/cmp_v22_phys2_pilot_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BASELINE="$ROOT/results/blend_lookup_v22_c40-139.feather"
CANDIDATE="$ROOT/results/blend_lookup_v22_phys2_c40-49.feather"
OUT="$ROOT/results/v22_phys2_constgold_pilot_c40-49.json"

export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cd "$ROOT"
[ -f "$BASELINE" ] || { echo "MISSING baseline: $BASELINE"; exit 1; }
[ -f "$CANDIDATE" ] || { echo "MISSING candidate: $CANDIDATE"; exit 1; }
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.2 PHYS2 CONSTGOLD LOOKUP COMPARISON job=$SLURM_JOB_ID cases=40-49 ###"
date
"$PY" -u scripts/compare_blend_lookups.py \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE" \
  --cases $(seq 40 49) \
  --output "$OUT"
[ -f "$OUT" ] || { echo "COMPARISON_FAILED: no output"; exit 1; }
echo "COMPARISON_DONE: $OUT"
date
