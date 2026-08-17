#!/bin/bash
#SBATCH --job-name=NP7_tgt
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/np7_tgt_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/compute_response_target_blend.py --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_g0.05_val.feather   --nominal-g 0.05 --target-cols measured_ngmix_g1 measured_ngmix_g2 --n-flux 6 --n-size 3 --n-dist 4   --min-count 3000 --max-rows 40000000 --output results/response_target_np7_6x3x4.npz
