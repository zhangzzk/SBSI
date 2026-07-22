#!/bin/bash
#SBATCH --job-name=diag_szresp
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_szresp_%j.out

# UNIFIED-OBJECTIVE lever diagnostic (cont.85): is the size/faint TARGET fixable by finer NON-CIRCULAR
# supervision, or a sim floor? Map dump objects to the existing fine non-circular target
# (response_target_isoblend_snc_c0-99_6x6x5, snc leg-based, constgold NEVER read) and per fine size bin
# compare <r_sim>(constgold truth) vs <R_flow>(certified) vs <R_snc>(fine target), + implied m.
# constgold r_sim is validation truth only; nothing is fit. Firewall-safe.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
echo "### DIAG_SZRESP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_size_response_profile.py || { echo "DIAG_SZRESP FAILED"; exit 1; }
echo "### DIAG_SZRESP_DONE job=$SLURM_JOB_ID ###"; date
