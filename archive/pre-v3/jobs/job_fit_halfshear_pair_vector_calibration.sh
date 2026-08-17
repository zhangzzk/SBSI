#!/bin/bash
#SBATCH --job-name=hs_paircal
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_paircal_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_paircal_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TABLE=/project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_paircal_v22_c0-39.feather
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/fit_halfshear_pair_vector_calibration.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather \
  --shear 0.2 --n-cases 40 --n-bins 6 --table-output "$TABLE" \
  --output results/halfshear_pair_vector_calibration_v22_c0-39.json
echo HALFSHEAR_PAIR_VECTOR_CALIBRATION_JOB_DONE
date
