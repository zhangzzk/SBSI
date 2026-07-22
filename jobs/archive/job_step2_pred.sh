#!/bin/bash
#SBATCH --job-name=step2pred
#SBATCH --time=00:40:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/step2pred_%j.out

# cont.98 STEP 2: certified predicted response <R_flow+R_blend> on the constgold ensemble + m_ens.
# Firewall-safe: reads ONLY R_flow/R_blend from the dump (never r_sim); trains nothing.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### STEP2PRED job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/step2_pred_response_constgold.py || { echo "STEP2PRED FAILED"; exit 1; }
echo "### STEP2PRED_DONE job=$SLURM_JOB_ID ###"; date
