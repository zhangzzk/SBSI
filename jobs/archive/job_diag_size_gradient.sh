#!/bin/bash
#SBATCH --job-name=diag_szgrad
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_szgrad_%j.out

# DIAGNOSTIC (read-only, tunes nothing): fine true-size scan of the sz6eqw estimator to localise
# the large-size bias (within-bin gradient vs flat offset).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/z/Zekang.Zhang/SBSI
echo "### DIAG_SZGRAD job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_size_gradient.py || { echo "DIAG_SZGRAD FAILED"; exit 1; }
echo "### DIAG_SZGRAD_DONE job=$SLURM_JOB_ID ###"; date
