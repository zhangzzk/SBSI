#!/bin/bash
#SBATCH --job-name=hs_pair_train
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_pair_train_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_pair_train_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/audit_halfshear_pair_calibration_bins.py \
  --catalogue "$CAT" \
  --calibration results/halfshear_pair_vector_calibration_v22_c0-39.json \
  --shear 0.2 --case-min 40 --n-cases 200 --development-max 119 \
  --output results/halfshear_pair_calibration_bin_audit_v22_training_c40-199.json
echo HALFSHEAR_PAIR_CALIBRATION_TRAINING_BIN_AUDIT_DONE
