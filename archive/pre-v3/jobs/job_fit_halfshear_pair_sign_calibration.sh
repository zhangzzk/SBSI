#!/bin/bash
#SBATCH --job-name=hs_pair_sign
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_pair_sign_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_pair_sign_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/fit_halfshear_pair_vector_calibration.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather \
  --shear 0.2 --n-cases 40 --development-max 19 --n-bins 2 --fixed-zero-split \
  --table-output /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_pairsign_v22_c0-39.feather \
  --output results/halfshear_pair_sign_calibration_v22_c0-39.json
