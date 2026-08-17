#!/bin/bash
#SBATCH --job-name=cmp_v22_phys2_c40_89
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/cmp_v22_phys2_c40-89_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
FIRST="$ROOT/results/blend_lookup_v22_phys2_c40-49.feather"
REMAINING="$ROOT/results/blend_lookup_v22_phys2_c50-89.feather"
CANDIDATE="$ROOT/results/blend_lookup_v22_phys2_c40-89.feather"
BASELINE="$ROOT/results/blend_lookup_v22_c40-139.feather"
OUT="$ROOT/results/v22_phys2_constgold_pilot_c40-89.json"

export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cd "$ROOT"
for input in "$FIRST" "$REMAINING" "$BASELINE"; do
  [ -f "$input" ] || { echo "MISSING input: $input"; exit 1; }
done
[ ! -e "$CANDIDATE" ] || { echo "REFUSING to overwrite $CANDIDATE"; exit 1; }
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.2 PHYS2 CONSTGOLD LOOKUP ASSEMBLY/COMPARISON job=$SLURM_JOB_ID cases=40-89 ###"
date
"$PY" -u scripts/concat_blend_lookups.py \
  --inputs "$FIRST" "$REMAINING" \
  --cases $(seq 40 89) \
  --output "$CANDIDATE"
"$PY" -u scripts/compare_blend_lookups.py \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE" \
  --cases $(seq 40 89) \
  --output "$OUT"
[ -f "$CANDIDATE" ] || { echo "ASSEMBLY_FAILED: no candidate lookup"; exit 1; }
[ -f "$OUT" ] || { echo "COMPARISON_FAILED: no comparison"; exit 1; }
echo "COMPARISON_DONE: $OUT"
date
