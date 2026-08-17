#!/bin/bash
#SBATCH --job-name=anti_rbcase
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --array=0-99%20
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti_rbcase_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/anti_rbcase_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_gap_antithetic_rblend_c0-99
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$OUT"
cd "$ROOT"
python -u scripts/measure_antithetic_rblend_case.py \
  --case "$SLURM_ARRAY_TASK_ID" --output-dir "$OUT"

