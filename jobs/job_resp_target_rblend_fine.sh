#!/bin/bash
#SBATCH --job-name=resp_rblend_fine
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/resp_rblend_fine_%j.out
set -eo pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUT=results/response_target_crowd_rblend_snc_c0-99_6x9x5.npz
echo "### RESP_RBLEND_FINE job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --size-edges 0.1,0.24,0.30,0.36,0.42,0.50,0.62,0.80,1.10,1.50 --n-crowd 5 \
  --min-count 500 --max-case 99 \
  --snc-lookup results/g0_lookup_c0-99.feather \
  --output $OUT 2>&1 | grep -v "module command"
date
echo "RESP_RBLEND_FINE_DONE $OUT"
python - <<PY
import numpy as np
d=np.load("$OUT")
R=d["Rsim"]; c=d["counts"]
w=np.where(np.isfinite(R),c,0.0); Rv=np.where(np.isfinite(R),R,0.0)
# marginal over flux&size (counts-weighted) along crowd axis
marg=(Rv*w).sum(axis=(0,1))/np.clip(w.sum(axis=(0,1)),1e-9,None)
print("shape", R.shape, " edges_crowd=", np.round(d["edges_crowd"],4).tolist())
print("crowd-axis marginal Rsim (near->far blend):", np.round(marg,4).tolist())
PY
