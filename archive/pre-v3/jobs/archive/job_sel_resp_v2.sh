#!/bin/bash
#SBATCH --job-name SBSI_SELR_V2
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selr_v2.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selr_v2.%j.err
echo "START - response-aware selection v2 (centered diff + valid mask, lam=150)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; export OMP_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/train_selection_response.py \
    --response-target-npz results/selection_target_g0.05_4x2x4_blend.npz \
    --output models/selection_respaware_v2_lam150.pt \
    --response-weight 150 --response-delta 0.05 \
    --max-rows 10000000 --epochs 80 --batch-size 32768 --num-workers 4
echo; echo "FINISH"; date
