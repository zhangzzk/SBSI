#!/bin/bash
#SBATCH --job-name=AP_fmval
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_fmval_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_fmval_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
M=${1:-models/measurement_flow_g0_ngmix_ap7_respblend_lam1000_v1.pt}
echo "###### FIRST-MOMENT m for $M ######"
for g in 0.05 0.02; do
  cat="$CAT/det_meas_ngmix_ap7_g${g}_x"
done
echo "--- g=0.05 ---"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.05_val.feather --nominal-g 0.05 \
  --max-rows 8000000 --n-samples 128 || echo "(g0.05 failed)"
echo "--- g=0.02 (independent) ---"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.02_test.feather --nominal-g 0.02 \
  --max-rows 8000000 --n-samples 128 || echo "(g0.02 failed)"
echo DONE
