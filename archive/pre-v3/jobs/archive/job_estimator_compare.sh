#!/bin/bash
#SBATCH --job-name=estcmp
#SBATCH --time=00:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/estcmp_%j.out

# cont.99: matched flow-closure per cut, ratio-of-means vs per-object calibration.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### ESTCMP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/estimator_compare_constgold.py || { echo "ESTCMP FAILED"; exit 1; }
echo "### ESTCMP_DONE job=$SLURM_JOB_ID ###"; date
