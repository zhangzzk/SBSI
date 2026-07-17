#!/bin/bash
#SBATCH --job-name SBSI_RTNG
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_rtng.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_rtng.%j.err
echo "START ngmix response target"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/compute_response_target_blend.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather \
  --nominal-g 0.05 --n-flux 4 --n-size 2 --n-dist 3 --min-count 20000 \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --output results/response_target_g0.05_4x2x4_blend_ngmix.npz
echo; echo FINISH; date
