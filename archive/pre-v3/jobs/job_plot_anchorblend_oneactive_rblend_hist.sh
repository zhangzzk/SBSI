#!/bin/bash
#SBATCH --job-name=oneact_hist
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/py31
INPUT=$ROOT/results/anchorblend_oneactive_orthogonal_v22_c400-499.feather
OUT=$ROOT/results/anchorblend_oneactive_rblend_hist_c400-499
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$INPUT" || { echo "MISSING $INPUT"; exit 1; }
for suffix in png pdf json md; do
  test ! -e "$OUT.$suffix" || { echo "REFUSING existing $OUT.$suffix"; exit 1; }
done

python -u scripts/plot_anchorblend_oneactive_rblend_hist.py \
  --input "$INPUT" --output-prefix "$OUT" --bins 140 \
  --tail-probability 0.001
echo ONEACTIVE_RBLEND_HIST_JOB_DONE
date
