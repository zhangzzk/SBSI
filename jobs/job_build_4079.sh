#!/bin/bash
#SBATCH --job-name=build4079
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/build4079_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
date
echo "### blend_lookup 40-79 ###"
python -u scripts/build_blend_lookup.py --cases $(seq 40 79) --tag lsst_r_extnbr_ho \
  --output results/blend_lookup_extnbrho_c40-79.feather 2>&1 | grep -v "module command"
echo "### ood_split 40-79 ###"
python -u scripts/build_ood_split_lookup.py --cases $(seq 40 79) \
  --output results/ood_split_c40-79.feather 2>&1 | grep -v "module command"
date; echo BUILD4079_DONE
