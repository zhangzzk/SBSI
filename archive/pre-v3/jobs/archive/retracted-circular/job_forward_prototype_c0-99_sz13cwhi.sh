#!/bin/bash
#SBATCH --job-name=fwd_proto99sz13hi
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99sz13hi_%j.out
# TRACK 1 iter #8 -- FINAL fit push. sz13cw (count-weight) is the best config so far: worst additive
# 1.10%, median 0.18%, all cuts sub-0.3% EXCEPT size1.0-1.5 = -1.10% (z=5.1). Weighting is exhausted
# (equal -1.50 vs count -1.10, monotonic; count is the better endpoint). The residual is a flow-FIT
# floor: R_flow slightly OVER-predicts the isolated large-galaxy (Re>1) shear response. Last untested
# within-framework lever = fit the response HARDER with more mean-head capacity: count-weight (best) +
# mean_hidden 64->128 + lam_r 50->120, SAME 6x13x5 constgold target (no rebuild). If size1.0-1.5 -> sub
# percent without regressing GLOBAL/mag -> DONE; else the isolated-large-tail residual is the framework
# floor (owner-gated next step: smooth continuous R_flow supervision or dedicated iso-large calibration).
# FIREWALL: a-priori capacity/loss-weight choice, OOS by case, GLOBAL re-checked, R_blend rebuilt
# SEPARATELY, adoption owner-gated.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
TARGET=results/response_target_constgold_c0-99_6x13x5.npz
SEEDS=${SEEDS:-421 422 423}
echo "### FWD_PROTO99SZ13HI job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' (count-wt, mean128, lamr120) ###"; nvidia-smi -L; date
[ -f "$TARGET" ] || { echo "TARGET MISSING: $TARGET"; exit 1; }
for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" --target-npz "$TARGET" --decorrelate \
    --outdir "$OUTDIR" --tag "proto_c0-99_sz13hi_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 6 --n-size 13 --delta 0.05 --lam-r 120 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 128 --det-hidden 128 \
    || { echo "FWD_PROTO99SZ13HI FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99SZ13HI_DONE job=$SLURM_JOB_ID ###"; date
