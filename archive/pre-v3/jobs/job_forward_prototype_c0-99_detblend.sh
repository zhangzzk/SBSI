#!/bin/bash
#SBATCH --job-name=fwd_proto99detbl
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99detbl_%j.out

# TRACK 2 (owner-greenlit): DIAGNOSTIC / combined-candidate retrain. Supervises the detection-
# response head on the BLEND-RESOLVED b_true grid grid_b[flux,size,blend] (--btrue-grid) instead of
# mag-only b_true, to fix the diagnosed blend-resolved non-closure (head misses ISOLATED, over-
# weights close blends -- Goal 2).  Also carries the finer-size 6x6x5 flow target, so IF it passes
# acceptance it is a single unified production candidate (both fixes).  ngmix det catalogue +
# ngmix b_true grid for population consistency.
# FIREWALL: dP is DIAGNOSTIC / additive, NEVER wired into certified m=R_sim/(R_flow+R_blend)-1; the
# grid_b target is a fixed precomputed a-priori quantity, never tuned on |m|; OOS by case (0-79/80-99).
# New tag -> the certified sz6 (Track 1) + prod checkpoints are preserved for a clean comparison.
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
echo "### FWD_PROTO99DETBL job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date
echo "flow-cat=$FLOWCAT"; echo "det-cat=$DETCAT"; echo "btrue=$BTRUE (BLEND-RESOLVED grid)"; echo "target=$TARGET"

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" --det-catalogue "$DETCAT" \
    --btrue-npz "$BTRUE" --btrue-grid \
    --target-npz "$TARGET" --decorrelate \
    --outdir "$OUTDIR" --tag "proto_c0-99_sz6detblend_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 6 --n-size 6 --delta 0.05 \
    --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99DETBL FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99DETBL_DONE job=$SLURM_JOB_ID ###"; date
