#!/bin/bash
#SBATCH --job-name=train_crowd_snc
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_crowd_snc_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

FS=${1:-g0_crowd_flux}
TAG=${2:-crowdflux_snc100_central02}
LAM=${3:-300}
RESP=${4:-results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz}
DELTA=${5:-0.02}
MAX_ROWS=${6:-0}
EPOCHS=${7:-80}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_g0.0_train_full.feather
OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### TRAIN SNC-CENTRAL $TAG  feature-set=$FS  lam=$LAM  delta=$DELTA  max_rows=$MAX_ROWS epochs=$EPOCHS ###"
echo "catalogue=$CAT"
echo "response_target=$RESP"
date
python -u scripts/train_measurement_model.py \
  --catalogue $CAT \
  --output $OUT \
  --target-column detected \
  --selection-name sextractor_detected \
  --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 \
  --flow-type mean_affine \
  --mean-hidden 128 \
  --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 \
  --max-rows "$MAX_ROWS" \
  --epochs "$EPOCHS" \
  --batch-size 8192 \
  --hidden-dim 256 \
  --condition-layers 3 \
  --n-flows 10 \
  --lr 0.0007 \
  --patience 10 \
  --num-workers 8 \
  --response-weight "$LAM" \
  --response-delta "$DELTA" \
  --response-difference central \
  --response-target-npz "$RESP" || { echo "TRAIN_SNC_CENTRAL $TAG FAILED"; exit 1; }
date
echo "TRAIN_CROWD_SNC_DONE $OUT"
