#!/bin/bash
#SBATCH --job-name=np7_nosnc
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/np7_nosnc_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
M=models/measurement_flow_g0_ngmix_np7_respblend_lam300_v1.pt
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
echo "### NP7 g=0.05  NO SNC (clean, apples-to-apples vs AP7 -0.64%) ###"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue $D/det_meas_ngmix_np7_g0.05_val.feather --nominal-g 0.05 --max-rows 0 --n-samples 64 2>&1 \
  | grep -iE "GLOBAL|ISOLATED|nbdist|BLENDED|N_eff" | grep -v module
echo "### NP7 g=0.02  NO SNC ###"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue $D/det_meas_ngmix_np7_g0.02_test.feather --nominal-g 0.02 --max-rows 0 --n-samples 64 2>&1 \
  | grep -iE "GLOBAL|ISOLATED|nbdist|BLENDED|N_eff" | grep -v module
echo NP7_NOSNC_DONE
