#!/bin/bash
#SBATCH --job-name=abg002x_amp
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
OUT=$ROOT/results/anchorblend_g002_gap_by_amplitude_v22_c400-899_guarded.json
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
test ! -e "$OUT" || { echo "REFUSING existing $OUT"; exit 1; }
cd "$ROOT"

RESPONSES=()
DOMINANCE=()
for block in 400-499 500-599 600-699 700-799 800-899; do
  RESPONSES+=("results/anchorblend_g002_response_v22_c${block}.feather")
  DOMINANCE+=("results/anchorblend_g002_dominance_v22_c${block}.feather")
done
for path in "${RESPONSES[@]}" "${DOMINANCE[@]}"; do
  test -s "$path" || { echo "MISSING $path"; exit 1; }
done

python -u scripts/analyze_anchor_gap_by_amplitude.py \
  --g002 "${RESPONSES[@]}" --dominance "${DOMINANCE[@]}" \
  --ratio-threshold 20 --case-min 400 --case-max 899 \
  --edge-window 400 499 --n-bins 10 --output "$OUT"
echo ANCHOR_GAP_BY_AMPLITUDE_JOB_DONE
date
