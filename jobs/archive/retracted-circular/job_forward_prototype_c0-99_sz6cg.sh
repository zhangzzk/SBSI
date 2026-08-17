#!/bin/bash
#SBATCH --job-name=fwd_proto99sz6cg
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99sz6cg_%j.out

# TRACK 1 iteration #4 (owner-greenlit, CERTIFIED path): METRIC-CONSISTENT response target.
# Root cause of the +6..8% large-size selection bias (cont.63): the flow's response target was a
# VARIABLE-SHEAR g0.05 forward-diff grid (iso-large ~0.916) but the acceptance metric's r_sim is a
# CONSTGOLD antithetic +/-g central diff (iso-large ~1.02) -- a train/eval response-definition
# inconsistency a size-blind R_blend cannot bridge.  Fix: train on the constgold-built target
# (response_target_constgold_c0-99_6x6x5.npz, --antithetic).  Keep --flow-equal-weight (best prior
# config).  Detection supervision UNCHANGED (mag-only) so the R_flow change is attributable to the
# target swap.  FIREWALL: a-priori physics choice, OOS by case, GLOBAL m re-checked in acceptance
# before any adoption; R_blend rebuilt SEPARATELY against this R_flow; adoption as certified = owner.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
TARGET=results/response_target_constgold_c0-99_6x6x5.npz
SEEDS=${SEEDS:-421 422 423}
echo "### FWD_PROTO99SZ6CG job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date
echo "target=$TARGET (constgold antithetic 6x6x5, decorrelate=ON, flow-equal-weight=ON, det=mag-only)"
[ -f "$TARGET" ] || { echo "TARGET MISSING: $TARGET"; exit 1; }

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" \
    --target-npz "$TARGET" --decorrelate --flow-equal-weight \
    --outdir "$OUTDIR" --tag "proto_c0-99_sz6cg_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 6 --n-size 6 --delta 0.05 \
    --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99SZ6CG FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99SZ6CG_DONE job=$SLURM_JOB_ID ###"; date
