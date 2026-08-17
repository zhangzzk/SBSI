#!/bin/bash
#SBATCH --job-name=AP_snc
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_snc_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_snc_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
M=models/measurement_flow_g0_ngmix_ap7_respblend_lam300_v1.pt; LK=results/g0_lookup_c0-19.feather
echo "###### g=0.02 NO-SNC (baseline) ######"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.02_test.feather --nominal-g 0.02 --max-rows 0 --n-samples 64 2>&1 | grep -iE "GLOBAL|N_eff|N_pairs" | grep -v module
echo "###### g=0.02 WITH SNC (pair with g=0) ######"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.02_test.feather --nominal-g 0.02 --max-rows 0 --n-samples 64 --snc-lookup "$LK" 2>&1 | grep -iE "SNC ON|GLOBAL|N_eff|N_pairs" | grep -v module
echo "###### g=0.05 WITH SNC (20 matched cases, for comparison) ######"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.05_val.feather --nominal-g 0.05 --max-rows 30000000 --n-samples 64 --snc-lookup "$LK" 2>&1 | grep -iE "SNC ON|GLOBAL|N_eff" | grep -v module
echo DONE
