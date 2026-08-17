#!/bin/bash
#SBATCH --job-name=AP_snc100
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_snc100_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_snc100_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
M=models/measurement_flow_g0_ngmix_ap7_respblend_lam300_v1.pt; LK=results/g0_lookup_c0-99.feather; C=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.02_test100.feather
echo "=== 100-case g=0.02, SNC, m at various true-mag cuts ==="; date
for tm in 99 25 24; do
  arg=""; [ "$tm" != "99" ] && arg="--true-mag-max $tm"
  echo "###### SNC + true-mag<$tm ######"
  python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
    --catalogue "$C" --nominal-g 0.02 --max-rows 0 --n-samples 64 --snc-lookup "$LK" $arg 2>&1 | grep -iE "SNC ON|GLOBAL|BLENDED|N_eff" | grep -v module
done
echo "###### NO-SNC (ref, no cut) ######"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue "$C" --nominal-g 0.02 --max-rows 0 --n-samples 64 2>&1 | grep -iE "GLOBAL|N_eff" | grep -v module
echo DONE; date
