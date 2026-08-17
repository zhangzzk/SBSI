#!/bin/bash
#SBATCH --job-name=v22truthx
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/py31
SOURCE=$ROOT/results/v22_pair_residual_direct_curve_transfer_anchor_g002_c400-899.json
OUT=$ROOT/results/v22_coherent_gap_vs_measured_response_c400-899
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$SOURCE" || { echo "MISSING $SOURCE"; exit 1; }
for suffix in png pdf json md; do
  test ! -e "$OUT.$suffix" || { echo "REFUSING existing $OUT.$suffix"; exit 1; }
done
test ! -e "${OUT}_deciles.csv" || {
  echo "REFUSING existing ${OUT}_deciles.csv"; exit 1; }

python -u scripts/plot_v22_coherent_gap_vs_measured_response.py \
  --source-json "$SOURCE" --output-prefix "$OUT" --n-bins 10
echo V22_COHERENT_GAP_VS_MEASURED_RESPONSE_JOB_DONE
date
