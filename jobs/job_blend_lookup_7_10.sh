#!/bin/bash
#SBATCH --job-name=blk7_10
#SBATCH --time=05:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=small
#SBATCH --output=/home/z/Zekang.Zhang/logs/blk7_10_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### BLK7_10 job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
echo "---- 10\" (native extnbrho, ungated) ----"
python -u scripts/build_blend_lookup.py --cases $(seq 0 39) --tag lsst_r_extnbr_ho \
  --output results/blend_lookup_extnbrho_c0-39.feather 2>&1 | grep -v "module command"
echo "---- 7\" (gated) ----"
python -u scripts/build_blend_lookup.py --cases $(seq 0 39) --tag lsst_r_extnbr_ho --max-distance 7.0 \
  --output results/blend_lookup_extnbrho_d7_c0-39.feather 2>&1 | grep -v "module command"
echo "### BLK7_10_DONE job=$SLURM_JOB_ID ###"; date
