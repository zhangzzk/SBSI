#!/bin/bash
#SBATCH --job-name=measbin
#SBATCH --partition=inter
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/measbin_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
echo "### COMPONENT_MEASBINNED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_component_measbinned.py --max-case 99 --max-rows 2000000 2>&1 | grep -vE "module command"
echo DONE; date
