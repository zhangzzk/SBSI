#!/bin/bash
#SBATCH --job-name=ood_build
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=small
#SBATCH --output=/home/z/Zekang.Zhang/logs/ood_build_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
OOD=results/ood_split_c0-39.feather
python -u scripts/build_ood_split_lookup.py --cases $(seq 0 39) --output $OOD 2>&1 | grep -v "module command"
echo OOD_BUILD_DONE
