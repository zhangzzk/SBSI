#!/bin/bash
#SBATCH --job-name=flow_grid
#SBATCH --partition=cip
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/flow_grid_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
echo "### FLOW_SELFRESP_GRID job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_flow_selfresp_grid.py --max-case 99 --max-rows 2000000 2>&1 | grep -vE "module command"
echo DONE; date
