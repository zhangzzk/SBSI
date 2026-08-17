#!/bin/bash
#SBATCH --job-name=self_lk
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/self_lk_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date; python -u scripts/build_self_lookup.py --cases $(seq -s ' ' 0 39) \
  --output results/self_lookup_const_c0-39.feather --tag lsst_r_extnbr_ho || { echo SELF_LK_FAILED; exit 1; }; date
echo SELF_LK_DONE
