#!/bin/bash
#SBATCH --job-name=cgdecomp
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgdecomp_%j.out

# STEP 1 (cont.97): constgold +/-0.02 UNMATCHED ensemble decomposition R_total/R_sel/R_shape on
# TRUE-property gentle cuts. Firewall-safe (measures truth on constgold + compares; trains nothing).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### CGDECOMP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/decomp_constgold_ensemble.py || { echo "CGDECOMP FAILED"; exit 1; }
echo "### CGDECOMP_DONE job=$SLURM_JOB_ID ###"; date
