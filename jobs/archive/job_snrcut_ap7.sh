#!/bin/bash
#SBATCH --job-name=AP_snrcut
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_snrcut_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_snrcut_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
M=models/measurement_flow_g0_ngmix_ap7_respblend_lam300_v1.pt
for snr in 0 10 15 20 30; do
  arg=""; [ "$snr" != "0" ] && arg="--snr-min $snr"
  echo "########## S/N cut > $snr ##########"
  echo "-- g=0.02 (independent) --"
  python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
    --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.02_test.feather --nominal-g 0.02 --max-rows 0 --n-samples 64 $arg 2>&1 | grep -iE "GLOBAL|BLENDED|N_eff" | grep -v module
  echo "-- g=0.05 (calibration) --"
  python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
    --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.05_val.feather --nominal-g 0.05 --max-rows 25000000 --n-samples 64 $arg 2>&1 | grep -iE "GLOBAL|BLENDED|N_eff" | grep -v module
done
echo DONE
