#!/bin/bash
#SBATCH --job-name=anti0205
#SBATCH --time=00:20:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti0205_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### ANTI0205 job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/test_antithetic_0205.py
echo "### ANTI0205_DONE job=$SLURM_JOB_ID ###"; date
