#!/bin/bash
#SBATCH --job-name=selfresp_iso
#SBATCH --time=00:20:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/selfresp_iso_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### SELFRESP_ISO job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/selfresp_by_isolation.py
echo "### SELFRESP_ISO_DONE job=$SLURM_JOB_ID ###"; date
