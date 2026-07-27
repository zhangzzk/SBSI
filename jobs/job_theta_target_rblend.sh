#!/bin/bash
#SBATCH --job-name=theta_rblend
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/theta_rblend_%j.out
set -eo pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
OUT=results/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
echo "### THETA_RBLEND job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -u scripts/build_theta_coupling_target.py \
  --grid-npz results/response_target_crowd_rblend_snc_c0-99_6x9x5.npz \
  --max-case 99 --true-re-min 0.3 --true-mag-max 26.0 --min-count 300 \
  --output $OUT 2>&1 | grep -v "module command"
date
echo "THETA_RBLEND_DONE $OUT"
python - <<PY
import numpy as np
d=np.load("$OUT")
print("keys:", list(d.keys()))
print("coupling_size shape:", d["coupling_size"].shape, " crowd_col:", str(d["crowd_col"]) if "crowd_col" in d.files else "n/a")
cs=d["coupling_size"]
# marginal b_size across crowd axis (mean over flux&size)
print("b_size marginal across crowd (near->far blend):", np.round(np.nanmean(cs,axis=(0,1)),4).tolist())
print("b_size global:", float(d["b_size_global"]))
PY
