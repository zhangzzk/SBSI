#!/bin/bash
#SBATCH --job-name=AP_fmprec
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_fmprec_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_fmprec_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
M=${1:-models/measurement_flow_g0_ngmix_ap7_respblend_lam300_v1.pt}
echo "###### PRECISE first-moment m (streaming, full stats): $M ######"; date
echo "--- g=0.05 (calibration shear, up to 80M pairs) ---"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.05_val.feather --nominal-g 0.05 \
  --max-rows 80000000 --n-samples 64 || echo "(g0.05 failed)"
echo "--- g=0.02 (independent test, all ~3.6M) ---"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.02_test.feather --nominal-g 0.02 \
  --max-rows 0 --n-samples 64 || echo "(g0.02 failed)"
echo DONE; date
