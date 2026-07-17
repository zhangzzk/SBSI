#!/bin/bash
#SBATCH --job-name SBSI_ISOBIAS
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_isobias_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_isobias_%j.err

echo "START - SBSI isolate multiplicative-bias source (progressive conditioning)"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

echo "########## flexible regressor (max_leaves=63, min_leaf=200) ##########"
python -u SBSI/scripts/isolate_bias_conditioning.py \
    --g0-rows 1500000 --sheared-rows 1000000 --max-leaves 63 --min-leaf 200

echo "########## heavily regularized (max_leaves=15, min_leaf=20000) ##########"
python -u SBSI/scripts/isolate_bias_conditioning.py \
    --g0-rows 1500000 --sheared-rows 1000000 --max-leaves 15 --min-leaf 20000

echo "FINISH"; date
