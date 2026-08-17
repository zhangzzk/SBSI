#!/bin/bash
#SBATCH --job-name=AP_train
#SBATCH --time=10:00:00
#SBATCH --mem=38G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_train_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_train_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
LAM=${1:-1000}
OUT=models/measurement_flow_g0_ngmix_ap7_respblend_lam${LAM}_v1.pt
echo "=== AP7 response-aware flow train (all-pairs ngmix, lam=$LAM) ==="; date
python -u scripts/train_measurement_model.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.0_train.feather \
  --output "$OUT" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_oriented --target-features measured_ngmix_g1 measured_ngmix_g2 \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 6000000 --max-read-batches 1500 \
  --epochs 100 --batch-size 8192 --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0007 --patience 12 --num-workers 8 \
  --response-weight "$LAM" --response-delta 0.05 \
  --response-target-npz results/response_target_ap7_g0.05_6x3x4.npz \
  || { echo "TRAIN FAILED"; exit 1; }
echo "=== TRAIN DONE -> $OUT ==="; date
