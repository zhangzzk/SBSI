#!/bin/bash
#SBATCH --job-name=AP_rtgt
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_rtgt_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_rtgt_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
echo "=== AP7 response target (g0.05, per-case-target weighted, subset) ==="; date
python -u scripts/compute_response_target_blend.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.05_val.feather \
  --nominal-g 0.05 --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --n-flux 6 --n-size 3 --n-dist 4 --min-count 3000 --max-rows 30000000 \
  --output results/response_target_ap7_g0.05_6x3x4.npz
echo "=== DONE ==="; date
