#!/bin/bash
#SBATCH --job-name=valc_full
#SBATCH --time=05:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/valc_full_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
TAG=${1:?tag}; G=${2:?shear}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
if [ "$G" = "0.05" ]; then CAT=$D/det_meas_crowd_g0.05_val_full.feather; else CAT=$D/det_meas_crowd_g0.02_test_full.feather; fi
M=models/measurement_flow_g0_ngmix_${TAG}_lam300_v1.pt
echo "### VALIDATE $TAG g=$G FULL cases, binned by R_blend ###"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" --catalogue "$CAT" \
  --nominal-g "$G" --max-rows 0 --n-samples 64 --blend-lookup results/blend_lookup_c0-199.feather 2>&1 \
  | grep -iE "BLEND-BIN|GLOBAL|INCOHERENT binned|Rblend-bin|ISO\(|\[0\.|>=0|BLENDED-ONLY" | grep -v module
echo "VALC_FULL_DONE ${TAG}_g${G}"
