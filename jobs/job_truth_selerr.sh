#!/bin/bash
#SBATCH --job-name=truth_selerr
#SBATCH --time=00:20:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/truth_selerr_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### TRUTH SELERR job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_truth_selerr.py
echo "TRUTH_SELERR_JOB_DONE"; date
