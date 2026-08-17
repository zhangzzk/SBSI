#!/bin/bash
#SBATCH --job-name=diag_rnl
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_rnl_%j.out

# DIAGNOSTIC (read-only, tunes nothing): measure the shear response of large galaxies at g=0.02 and
# g=0.05 (SNC forward-diff) to test whether the FORWARD-diff-at-0.05 response TARGET grid is biased
# LOW vs the true g->0 slope (root-cause hypothesis for the +6..8% large-size selection bias).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### DIAG_RNL job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_response_nonlinearity.py || { echo "DIAG_RNL FAILED"; exit 1; }
echo "### DIAG_RNL_DONE job=$SLURM_JOB_ID ###"; date
