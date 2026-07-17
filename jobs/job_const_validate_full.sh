#!/bin/bash
#SBATCH --job-name SBSI_CVALF
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_cvalf.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_cvalf.%j.err
echo "START - constant-shear gold m & c validation (full stats)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; export OMP_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/validate_constant_response.py \
    --measurement-model models/measurement_flow_g0_shape2d_respblend_lam1000_v1.pt \
    --max-rows "${MAXROWS:-12000000}" --n-samples "${NS:-128}" --batch-size "${BS:-8192}"
echo; echo "FINISH"; date
