#!/bin/bash
#SBATCH --job-name=detresp
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/detresp_%j.out

# DIAGNOSTIC: two-leg ngmix detection-response dP/dgamma truth target (Goal 2). Streams the
# g=0.0 and g=0.05 ngmix parents (undetected retained), source-selects, and finite-differences
# the detection rate per (true-mag x true-size x blend) bin. Never touches certified m.
set -e
source /software/opt/focal/x86_64/python/3.10-2022.08/etc/profile.d/conda.sh
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### DETRESP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/derisk_detection_response.py "$@" \
  || { echo "DETRESP FAILED"; exit 1; }
echo "### DETRESP_DONE job=$SLURM_JOB_ID ###"; date
