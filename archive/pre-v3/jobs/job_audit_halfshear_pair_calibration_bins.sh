#!/bin/bash
#SBATCH --job-name=hs_pairbina
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_pairbina_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_pairbina_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/audit_halfshear_pair_calibration_bins.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather \
  --calibration results/halfshear_pair_vector_calibration_v22_c0-39.json \
  --output results/halfshear_pair_calibration_bin_audit_v22_c0-39.json
echo HALFSHEAR_PAIR_CALIBRATION_BIN_AUDIT_JOB_DONE
date
