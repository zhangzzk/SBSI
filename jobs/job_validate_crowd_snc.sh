#!/bin/bash
#SBATCH --job-name=valc_snc
#SBATCH --time=05:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/valc_snc_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

TAG=${1:?tag}
G=${2:?shear}
LK=${3:-results/g0_lookup_c0-99.feather}
MAXCASE=${4:-99}
LAM=${5:-300}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
if [ "$G" = "0.05" ]; then CAT=$D/det_meas_crowd_g0.05_val_full.feather; else CAT=$D/det_meas_crowd_g0.02_test_full.feather; fi
M=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### VALIDATE $TAG lam=$LAM g=$G with SNC, cases <= $MAXCASE, binned by R_blend ###"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" --catalogue "$CAT" \
  --nominal-g "$G" --max-rows 0 --max-case "$MAXCASE" --n-samples 64 \
  --snc-lookup "$LK" --blend-lookup results/blend_lookup_c0-199.feather 2>&1 \
  | grep -iE "SNC ON|BLEND-BIN|GLOBAL|INCOHERENT binned|Rblend-bin|ISO\\(|\\[0\\.|>=0|BLENDED-ONLY" | grep -v module
echo "VALC_SNC_DONE ${TAG}_g${G}"
