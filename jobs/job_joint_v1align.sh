#!/bin/bash
#SBATCH --job-name=v1align
#SBATCH --time=03:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/v1align_%j.out

# Gold-V2 Stage-1 DIAGNOSTIC (owner 2026-07-23): keep the ESSENTIAL reframe (4D measured outputs +
# truth-conditioning) but ALIGN the non-essential knobs to V1, to test whether the +8.2% constgold
# faint under-prediction is a knob or the reframe itself. Aligned to V1: lam_r 50->450 (pin response
# 9x harder), DROP --decorrelate, DROP --flow-equal-weight (count-weight like V1). Kept from V2:
# RAWfine 6x9x5 target, arch 256/10/128, delta 0.05, det_meas_ngmix cat, seed 421. NO theta pin
# (Stage-2; irrelevant to the shape dims). Then run the Stage-1 constgold closure (bridge=1.0).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
TARGET=results/response_target_isoblend_RAWfine_c0-99_6x9x5.npz
SEED=421; TAG=sw_v1align
echo "### V1ALIGN job=$SLURM_JOB_ID node=$SLURMD_NODENAME seed=$SEED tag=$TAG ###"; nvidia-smi -L; date

# ---- retrain: V1-aligned knobs (NO --decorrelate, NO --flow-equal-weight, lam_r 450) ----
stdbuf -oL -eL python -B -u scripts/train_joint_forward.py \
  --flow-catalogue "$FLOWCAT" \
  --target-npz "$TARGET" \
  --outdir "$OUTDIR" --tag "${TAG}_seed${SEED}" \
  --max-case 200 --train-case-max 160 \
  --det-max-case 100 --det-train-case-max 80 \
  --max-rows-flow 2400000 --max-rows-det 1600000 --max-rows-val 500000 \
  --delta 0.05 \
  --lam-r 450 --lam-s 50 --lam-d 1 \
  --epochs 60 --patience 14 --no-sensitivity --no-baselines \
  --batch-size 16384 --lr 7e-4 --seed "$SEED" \
  --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 256 \
  --flow-layers 3 --n-flows 10 --mean-hidden 128 --det-hidden 128 \
  || { echo "V1ALIGN TRAIN FAILED"; exit 1; }

# ---- Stage-1 constgold closure (bridge=1.0, no empirical factor) ----
echo; echo "=================== Stage-1 constgold closure (bridge=1.0) ==================="
python -B -u scripts/eval_constgold_closure.py \
  --ckpt-glob "$OUTDIR/forward_${TAG}_seed${SEED}_joint.pt" \
  --bridge 1.0 --blend-lookup results/blend_lookup_const_c0-40.feather \
  --max-case 40 --true-re-min 0.3 --true-mag-max 26 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/stage1_cg_${TAG}.npz \
  || { echo "V1ALIGN CLOSURE FAILED"; exit 1; }
echo "### V1ALIGN_DONE job=$SLURM_JOB_ID tag=$TAG ###"; date
