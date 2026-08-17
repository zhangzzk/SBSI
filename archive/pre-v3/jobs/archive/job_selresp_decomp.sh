#!/bin/bash
#SBATCH --job-name=selresp
#SBATCH --time=03:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/selresp_%j.out

# ORACLE-#35 FEASIBILITY: decompose the g=0.05-leg shear response of the detected, selected ensemble
# into R_shape (certified models it) + R_sel (pure selection; #35 would add it). m_cert = R_sel/R_shape
# is the certified pipeline's real-data measured-selection bias. Fully on the legs; constgold never
# read; certified m untouched. Firewall-safe.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### SELRESP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_selresp_decomp.py --n-boot 1000 \
  || { echo "SELRESP FAILED"; exit 1; }
echo "### SELRESP_DONE job=$SLURM_JOB_ID ###"; date
