#!/bin/bash
#SBATCH --job-name=hs_vecclose
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_vecclose_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_vecclose_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
python -u scripts/analyze_halfshear_vector_closure.py \
  --table /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_coherence_corrected_c0-39.feather \
  --output results/halfshear_vector_closure_v22_c0-39.json
echo HALFSHEAR_VECTOR_CLOSURE_JOB_DONE
date
