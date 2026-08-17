#!/bin/bash
#SBATCH --job-name=resp_crowd_full
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resp_crowd_full_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

NF=${1:-6}
NS=${2:-3}
NC=${3:-5}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUT=results/response_target_crowd_rblend_full_${NF}x${NS}x${NC}.npz

echo "### response target from full g0.05 crowd catalogue ###"
date
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 \
  --crowd-col r_blend \
  --n-flux $NF \
  --n-size $NS \
  --n-crowd $NC \
  --min-count 500 \
  --output $OUT 2>&1 | grep -v module
date
echo "RESP_CROWD_FULL_DONE $OUT"
