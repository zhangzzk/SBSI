#!/bin/bash
#SBATCH --job-name=abg002x_new
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail
# The 500-case result includes c400-499, so its agreement with the published
# 100-case number is not independent.  This scores ONLY the 400 newly rendered
# cases, which share no data with the earlier block.
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
OUT=$ROOT/results/anchorblend_g002_gap_by_dominance_v22_c500-899_newonly.json
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
test ! -e "$OUT" || { echo "REFUSING existing $OUT"; exit 1; }
cd "$ROOT"
R=(); D=()
for b in 500-599 600-699 700-799 800-899; do
  R+=("results/anchorblend_g002_response_v22_c${b}.feather")
  D+=("results/anchorblend_g002_dominance_v22_c${b}.feather")
done
python -u scripts/analyze_anchor_g002_gap_by_dominance.py \
  --g002 "${R[@]}" --dominance "${D[@]}" \
  --ratio-threshold 20 --case-min 500 --case-max 899 \
  --replicate-window 500 599 --output "$OUT"
echo ANCHOR_G002_NEWCASES_DONE; date
