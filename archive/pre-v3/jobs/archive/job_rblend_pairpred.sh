#!/bin/bash
#SBATCH --job-name=rblend_pp
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/rblend_pp_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### RBLEND_PAIRPRED job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_rblend_pairpred_grid.py --max-case 99 2>&1 | grep -vE "module command|UserWarning|warn"
echo DONE; date
