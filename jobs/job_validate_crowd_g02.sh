#!/bin/bash
#SBATCH --job-name=valc_g02
#SBATCH --time=03:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/valc_g02_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
TAG=${1:?tag}
M=models/measurement_flow_g0_ngmix_${TAG}_lam300_v1.pt
echo "### VALIDATE $TAG : half-shear g=0.02 (cases 0-39), binned by R_blend ###"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.02_test_c0-39.feather \
  --nominal-g 0.02 --max-rows 0 --n-samples 64 \
  --blend-lookup results/blend_lookup_hs_c0-39.feather 2>&1 \
  | grep -iE "BLEND-BIN|GLOBAL|INCOHERENT binned|Rblend-bin|ISO\(|\[0\.|>=0|BLENDED-ONLY" | grep -v module
echo "VALC_G02_DONE $TAG"
