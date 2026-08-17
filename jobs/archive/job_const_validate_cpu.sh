#!/bin/bash
#SBATCH --job-name SBSI_CVAL_CPU
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_cval_cpu.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_cval_cpu.%j.err
echo "START - constant validation SMOKE (CPU)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; export OMP_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/validate_constant_response.py \
    --measurement-model models/measurement_flow_g0_shape2d_respblend_lam1000_v1.pt \
    --device cpu --max-rows 300000 --n-samples 16
echo; echo "FINISH"; date
