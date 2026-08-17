#!/bin/bash
#SBATCH --job-name=extnbr_tr
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/extnbr_tr_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
cd /home/z/Zekang.Zhang/blendemu
echo "### RETRAIN regression emulator on EXTENDED-NEIGHBOUR domain (relaxed secondary cuts) ###"
python -u scripts/retrain_extnbr.py 2>&1 | grep -v "module command"
echo EXTNBR_JOB_DONE
