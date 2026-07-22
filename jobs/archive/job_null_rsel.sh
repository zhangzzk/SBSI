#!/bin/bash
#SBATCH --job-name=nullrsel
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/nullrsel_%j.out

# cont.98: shape-noise floor on constgold R_sel via CRN-preserving random re-orientation null.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### NULLRSEL job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/null_rsel_constgold.py || { echo "NULLRSEL FAILED"; exit 1; }
echo "### NULLRSEL_DONE job=$SLURM_JOB_ID ###"; date
