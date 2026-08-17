#!/bin/bash
#SBATCH --job-name=diag_szf
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_szf_%j.out

# cont.88 (task #38 follow-up): same size-response diagnostic as job_diag_size_response.sh, but on the
# REALIZED finer-target dump (fig2_perobj_s501_szfine.feather). Compares r_sim(constgold truth) vs the
# RETRAINED R_flow vs R_snc(fine target lookup) per fine size bin => quantifies how much of the fine
# target the flow actually bound to (regularization-binding gap) vs how much residual is R_blend.
# constgold r_sim is validation truth only; nothing is fit. Firewall-safe.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### DIAG_SZF job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_size_response_profile.py \
  --dump /project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_szfine.feather \
  || { echo "DIAG_SZF FAILED"; exit 1; }
echo "### DIAG_SZF_DONE job=$SLURM_JOB_ID ###"; date
