#!/bin/bash
#SBATCH --job-name=train_crowd
#SBATCH --time=06:00:00
#SBATCH --mem=38G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_crowd_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
FS=${1:?feature-set}; TAG=${2:?tag}; LAM=${3:-300}
RESP=${4:-results/response_target_crowd_rblend_6x3x5.npz}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
echo "### TRAIN $TAG  feature-set=$FS  lam=$LAM ###"
python -u scripts/train_measurement_model.py \
  --catalogue $D/det_meas_crowd_g0.0_train_c0-39.feather \
  --output models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt \
  --target-column detected --selection-name sextractor_detected --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 --flow-type mean_affine --mean-hidden 128 \
  --flow-blind-features e1_input_p e2_input_p --shear-case 0.0 --max-rows 10000000 \
  --epochs 100 --batch-size 8192 --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0007 --patience 12 --num-workers 8 --response-weight "$LAM" --response-delta 0.05 \
  --response-target-npz "$RESP" || { echo "TRAIN $TAG FAILED"; exit 1; }
echo "TRAIN_CROWD_DONE $TAG"
