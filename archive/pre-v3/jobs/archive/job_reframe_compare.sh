#!/bin/bash
#SBATCH --job-name=reframe_cmp
#SBATCH --time=00:20:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/reframe_cmp_%j.out
# cont.101: per-cut closure comparison of certified vs szfine vs reframed-loss dumps.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### REFRAME_CMP job=$SLURM_JOB_ID ###"; date
stdbuf -oL -eL python -B -u scripts/reframe_closure_compare.py || { echo "REFRAME_CMP FAILED"; exit 1; }
echo "### REFRAME_CMP_DONE ###"; date
