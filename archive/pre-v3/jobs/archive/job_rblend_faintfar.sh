#!/bin/bash
#SBATCH --job-name=rblend_ff
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/rblend_ff_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### RBLEND_FAINTFAR job=$SLURM_JOB_ID ###"; date
python -u scripts/rblend_faintfar_halfshear.py 2>&1 | grep -vE "module command|UserWarning|warn|RuntimeWarning"
echo DONE; date
