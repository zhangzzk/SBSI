#!/bin/bash
#SBATCH --job-name=nn_lookup
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/nn_lookup_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/build_nn_distance_lookup.py \
  --cases $(seq 0 39) \
  --sign 0.02 --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --output results/nn_dist_const_c0-39.feather 2>&1 | grep -v module
echo NN_LOOKUP_DONE
