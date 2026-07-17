#!/bin/bash
#SBATCH --job-name SBSI_SRNG
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_srng.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_srng.%j.err
echo "START ngmix response-aware selection classifier (lam=150)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; export OMP_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/train_selection_response.py \
    --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather \
    --response-target-npz results/selection_target_g0.05_4x2x4_blend_ngmix.npz \
    --output models/selection_respaware_ngmix_lam150.pt \
    --response-weight 150 --response-delta 0.05 \
    --max-rows 10000000 --epochs 80 --batch-size 32768 --num-workers 4
echo; echo FINISH; date
