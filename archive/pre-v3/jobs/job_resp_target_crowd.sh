#!/bin/bash
#SBATCH --job-name=resp_crowd
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/resp_crowd_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
NF=${1:-6}; NS=${2:-3}; NC=${3:-5}
python -u scripts/compute_response_target_blend.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_c0-39.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 --nominal-g 0.05 \
  --crowd-col r_blend --n-flux $NF --n-size $NS --n-crowd $NC --min-count 200 \
  --output results/response_target_crowd_rblend_${NF}x${NS}x${NC}.npz 2>&1 | grep -v module
echo RESP_CROWD_DONE
