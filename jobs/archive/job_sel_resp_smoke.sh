#!/bin/bash
#SBATCH --job-name SBSI_SELRSMOKE
#SBATCH --time=00:40:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selrsmoke.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selrsmoke.%j.err
echo "START - response-aware selection SMOKE"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/train_selection_response.py \
    --response-target-npz results/selection_target_g0.05_4x2x4_blend.npz \
    --output models/selection_respaware_smoke.pt \
    --response-weight 300 --response-delta 0.05 \
    --max-rows 800000 --epochs 8 --batch-size 16384 --num-workers 4
echo; echo "FINISH"; date
