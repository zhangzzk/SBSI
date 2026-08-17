#!/bin/bash
#SBATCH --job-name=ens_close
#SBATCH --time=00:25:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ens_close_%j.out
# cont.100: aggregate the 16 per-seed R_flow dumps -> ensemble per-cut flow closure.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### ENS_CLOSE job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/ensemble_flow_closure.py || { echo "ENS_CLOSE FAILED"; exit 1; }
echo "### ENS_CLOSE_DONE job=$SLURM_JOB_ID ###"; date
