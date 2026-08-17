#!/bin/bash
#SBATCH --job-name=hs_vecamp
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_vecamp_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_vecamp_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/compare_halfshear_vector_amplitudes.py \
  --g005 /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_vector_g005_c0-39.feather \
  --g020 /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_coherence_corrected_c0-39.feather \
  --output results/halfshear_vector_amplitude_g005_vs_g020_v22_c0-39.json
echo HALFSHEAR_VECTOR_AMPLITUDE_COMPARISON_JOB_DONE
date
