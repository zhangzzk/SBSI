#!/bin/bash
#SBATCH --job-name=snc_tm_g02
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/snc_tm_g02_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
M=models/measurement_flow_g0_ngmix_np7_respblend_lam300_v1.pt
LK=results/g0_lookup_c0-99.feather
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_g0.02_test.feather
run () {  # $1 = true-mag-max label ("none" or a number)
  local tm="$1"; local arg=""
  [ "$tm" != "none" ] && arg="--true-mag-max $tm"
  echo "### NP7 g=0.02 SNC  true-mag-max=$tm ###"
  python -u scripts/validate_allpairs_response.py --measurement-model "$M" --catalogue "$CAT" \
    --nominal-g 0.02 --max-rows 0 --n-samples 64 --snc-lookup "$LK" $arg 2>&1 \
    | grep -iE "SNC ON|true.mag|GLOBAL|ISOLATED|nbdist|BLENDED|N_eff|kept" | grep -v module
}
run none
run 25.0
run 24.5
run 24.0
run 23.5
echo SNC_TM_G02_DONE
