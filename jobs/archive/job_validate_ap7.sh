#!/bin/bash
#SBATCH --job-name=AP_valid
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_valid_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_valid_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
M=${1:-models/measurement_flow_g0_ngmix_ap7_respblend_lam1000_v1.pt}
echo "=== VALIDATE $M ==="; date; ls -la "$M" || { echo "MODEL MISSING"; exit 1; }
echo "--- recovery m @ g=0.05 (nearest-pair val) ---"
python -u archive/validate_heldout_shear_recovery.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather --nominal-shear 0.05 --which sheared \
  --max-rows 2000000 --output-dir results/ap7_recovery || echo "(recover 0.05 failed)"
echo "--- recovery m @ g=0.02 (INDEPENDENT test) ---"
python -u archive/validate_heldout_shear_recovery.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.02_test.feather --nominal-shear 0.02 --which sheared \
  --max-rows 2000000 --output-dir results/ap7_recovery || echo "(recover 0.02 failed)"
echo "--- constant-gold m/c (coherent) ---"
python -u scripts/validate_constant_response.py --measurement-model "$M" \
  --max-rows 4000000 --batch-size 8192 || echo "(constant validate failed)"
echo "=== VALIDATE DONE ==="; date
