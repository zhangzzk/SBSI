#!/bin/bash
#SBATCH --job-name=diag_rvs
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_rvs_%j.out

# DIAGNOSTIC (read-only, tunes nothing): head-to-head of the acceptance metric's dump r_sim against a
# clean SNC forward-diff response on the SAME isolated-large objects -> is the metric TRUTH biased?
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### DIAG_RVS job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_rsim_vs_snc.py || { echo "DIAG_RVS FAILED"; exit 1; }
echo "### DIAG_RVS_DONE job=$SLURM_JOB_ID ###"; date
