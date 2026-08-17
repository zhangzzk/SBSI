#!/bin/bash
#SBATCH --job-name SBSI_ISOSEL
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_isosel_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_isosel_%j.err

echo "START - SBSI isolate SELECTION/detection contribution to m"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

python -u SBSI/scripts/isolate_selection_effect.py \
    --g0-rows 4000000 --sheared-rows 3000000 --n-size 3 --n-mag 2

echo "FINISH"; date
