#!/bin/bash
#SBATCH --job-name=extnbr_tr
#SBATCH --time=04:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/extnbr_tr_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/blendemu
echo "### RETRAIN regression emulator on EXTENDED-NEIGHBOUR domain (CPU hist, relaxed secondary cuts) ###"
python -u scripts/retrain_extnbr.py 2>&1 | grep -v "module command"
echo EXTNBR_JOB_DONE
