#!/bin/bash
#SBATCH --job-name=blend_corr
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/blend_corr_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
date; python -u scripts/build_blend_lookup_corrected.py --cases $(seq -s ' ' 0 39) \
  --output results/blend_lookup_corr_c0-39.feather || { echo BLEND_CORR_FAILED; exit 1; }; date
echo BLEND_CORR_JOB_DONE
