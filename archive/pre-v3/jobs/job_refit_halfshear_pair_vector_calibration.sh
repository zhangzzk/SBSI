#!/bin/bash
#SBATCH --job-name=hs_pairrefit
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_pairrefit_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_pairrefit_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/refit_halfshear_pair_vector_calibration.py \
  --table /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_paircal_v22_c0-39.feather \
  --source-calibration results/halfshear_pair_vector_calibration_v22_c0-39.json \
  --output results/halfshear_pair_vector_calibration_refitall_v22_c0-39.json
echo HALFSHEAR_PAIR_VECTOR_CALIBRATION_REFIT_JOB_DONE
date
