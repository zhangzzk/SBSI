#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selrfull.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selrfull.%j.err
echo "START - response-aware selection FULL (lam=${LAMBDA})"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/train_selection_response.py \
    --response-target-npz results/selection_target_g0.05_4x2x4_blend.npz \
    --output "models/selection_respaware_lam${LAMBDA}_v1.pt" \
    --response-weight "${LAMBDA}" --response-delta 0.05 \
    --max-rows "${MAXROWS:-10000000}" --epochs "${EPOCHS:-80}" --batch-size 32768 --num-workers 4
echo; echo "FINISH"; date
