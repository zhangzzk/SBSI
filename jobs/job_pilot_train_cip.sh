#!/bin/bash
#SBATCH --job-name=pilot_train
#SBATCH --time=05:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/pilot_train_%j.out

# cip-vGPU variant of job_pilot_train.sh (cont.49). The cip a40-16gb/24gb slices are vGPU
# (A40-24Q etc.): CUDA VMM is unsupported there, so PYTORCH_CUDA_ALLOC_CONF=expandable_segments
# fails with "CUDA driver error: operation not supported" at the first tensor.to(device).
# This copy simply does NOT set it (plain caching allocator works; ~12 GB GPU need fits 24 GB).
# Defaults sized for the 41 GB-RAM cip slice nodes: 36G/8c (train MaxRSS ~30 GB, ~1-1.5 h/seed).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
unset PYTORCH_CUDA_ALLOC_CONF
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

SEEDS=${SEEDS:?set SEEDS (space-separated, e.g. "501 502 503")}
FS=${FS:-g0_meas_crowd_conc_szfl_noz}
TAG=${TAG:?set TAG (e.g. meas_szfl_noz_lam300)}
LAM=${LAM:-300}; WD=${WD:-1e-5}; DELTA=${DELTA:-0.02}; EPOCHS=${EPOCHS:-80}
RESP=${RESP:-results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather

echo "### PILOT_TRAIN(cip) job=$SLURM_JOB_ID seeds='$SEEDS' FS=$FS TAG=$TAG LAM=$LAM WD=$WD RESP=$RESP ###"
nvidia-smi -L; date

for SEED in $SEEDS; do
  OUT=models/measurement_flow_g0_ngmix_${TAG}_s${SEED}.pt
  echo "===== TRAIN seed=$SEED -> $OUT ====="; date
  python -u scripts/train_measurement_model.py \
    --catalogue $CAT --output $OUT \
    --target-column detected --selection-name sextractor_detected \
    --feature-set "$FS" \
    --target-features measured_ngmix_g1 measured_ngmix_g2 \
    --flow-type mean_affine --mean-hidden 128 \
    --flow-blind-features e1_input_p e2_input_p \
    --shear-case 0.0 --max-rows 0 --epochs "$EPOCHS" --batch-size 8192 \
    --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
    --weight-decay "$WD" \
    --seed "$SEED" --num-workers 8 --gpu-resident \
    --response-weight "$LAM" --response-delta "$DELTA" --response-difference central \
    --response-target-npz "$RESP" || { echo "PILOT_TRAIN FAILED seed=$SEED"; exit 1; }
  echo "TRAIN_DONE seed=$SEED MODEL=$OUT"; date
done
echo "PILOT_TRAIN_ALL_DONE job=$SLURM_JOB_ID seeds='$SEEDS' TAG=$TAG"; date
