#!/bin/bash
#SBATCH --job-name=fwd_proto99prod
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99prod_%j.out

# RETRAIN the joint forward model with the PRODUCTION-GUIDED isolated recipe (cont.60), following
# scripts/train_measurement_model.py (the isolated-calibrated production flow) as the template:
#   1. FULL isolated+blend population from the g=0 leg (det_meas_ngmix_g0.0_train) -- NOT the pairs
#      product and NOT the both-legs-finite subset (cont.59 hs, which dropped 78% and biased the
#      isolated target).  The analytic-shift response needs only the g=0 leg.
#   2. --target-npz = the ISOLATED-SEPARATED snc self-response grid (blend bin 0 = isolated),
#      a both-legs-free LOOKUP -> isolated objects get an unbiased target.
#   3. --decorrelate = the documented isolated-bias fix (g=0 shape<->size decorrelation weights).
# Identical arch/lambda to the pairs-only run; new _prod tag so pairs-only ckpts are preserved for
# a clean A/B.  EXPERIMENTAL / additive-only; never wired into certified m = R_sim/(R_flow+R_blend)-1.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
TARGET=results/response_target_isoblend_snc_c0-99_6x3x5.npz
SEEDS=${SEEDS:-421 422 423}
echo "### FWD_PROTO99PROD job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date
echo "flow-catalogue=$FLOWCAT"; echo "target-npz=$TARGET (decorrelate=ON)"

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" \
    --target-npz "$TARGET" --decorrelate \
    --outdir "$OUTDIR" --tag "proto_c0-99_prod_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 4 --n-size 3 --delta 0.05 \
    --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99PROD FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99PROD_DONE job=$SLURM_JOB_ID ###"; date
