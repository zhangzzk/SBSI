#!/bin/bash
#SBATCH --job-name=crowd_lk
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/crowd_lk_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/build_crowding_lookup.py --cases $(seq 0 39) \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 --sign 0.0 \
  --output results/crowd_flux_c0-39.feather 2>&1 | grep -v module
echo CROWD_LK_DONE
