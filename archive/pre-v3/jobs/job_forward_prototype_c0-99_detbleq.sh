#!/bin/bash
#SBATCH --job-name=fwd_proto99dbeq
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99dbeq_%j.out

# TRACK 2 iteration (DIAGNOSTIC, Goal 2): blend-resolved detection supervision (--btrue-grid) PLUS
# --det-equal-weight. First blend retrain (job 15155252) under-fit: global <dp>=-0.45% vs -2.03%
# target and faint high-response cells were drowned out, because binned_mse is COUNT-weighted and
# the fine grid's high-response cells (faint/isolated) are low-count. Equal-per-cell weighting aligns
# the training loss with the (already equal-weighted) early-stop metric so faint/isolated cells pull.
# FIREWALL: dP DIAGNOSTIC/additive, never wired into certified m; grid_b a fixed a-priori target,
# not tuned on |m|; OOS by case. New tag preserves the sz6 (Track 1) + sz6detblend checkpoints.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
DETCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather
BTRUE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/btrue_detection_ngmix.npz
TARGET=results/response_target_isoblend_snc_c0-99_6x6x5.npz
SEEDS=${SEEDS:-421 422 423}
echo "### FWD_PROTO99DBEQ job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" --det-catalogue "$DETCAT" \
    --btrue-npz "$BTRUE" --btrue-grid --det-equal-weight \
    --target-npz "$TARGET" --decorrelate \
    --outdir "$OUTDIR" --tag "proto_c0-99_sz6detbleq_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 6 --n-size 6 --delta 0.05 \
    --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99DBEQ FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99DBEQ_DONE job=$SLURM_JOB_ID ###"; date
