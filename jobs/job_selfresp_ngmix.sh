#!/bin/bash
#SBATCH --job-name=selfresp_ng
#SBATCH --time=00:30:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=12
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/selfresp_ng_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI
echo "### SELFRESP_NG job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/compare_selfresp_sims.py --variant ngmix --hs-max-case 19 --cg-max-case 39
echo "### SELFRESP_NG_DONE job=$SLURM_JOB_ID ###"; date
