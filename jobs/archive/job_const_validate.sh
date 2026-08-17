#!/bin/bash
#SBATCH --job-name SBSI_CVAL
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_cval.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_cval.%j.err
echo "START - constant-shear gold validation (flow vs antithetic response)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/validate_constant_response.py \
    --measurement-model models/measurement_flow_g0_shape2d_respblend_lam1000_v1.pt \
    --max-rows "${MAXROWS:-400000}" --n-samples "${NS:-32}"
echo; echo "FINISH"; date
