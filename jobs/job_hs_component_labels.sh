#!/bin/bash
#SBATCH --job-name=hs_labels
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --time=00:40:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/hs_labels_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### HS_COMPONENT_LABELS job=$SLURM_JOB_ID ###"; date
python -u scripts/halfshear_component_labels.py --max-case 99 2>&1 | grep -vE "module command"
echo DONE; date
