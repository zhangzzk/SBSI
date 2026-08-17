#!/bin/bash
#SBATCH --job-name=joint_re
#SBATCH --time=05:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/joint_re_%j.out

# Retrain sweep on the UNIFIED RAW target (cont.116). Owner granted autonomy + more data/arch/loss.
# Env knobs: SEED, TAG, TARGET, MAXCASE, TRAINCASEMAX, ROWSFLOW, ROWSDET, EXTRA (extra CLI args).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
TARGET=${TARGET:-results/response_target_isoblend_RAW_c0-99_6x3x5.npz}
SEED=${SEED:-421}
TAG=${TAG:-re_unified}
MAXCASE=${MAXCASE:-200}
TRAINCASEMAX=${TRAINCASEMAX:-160}
ROWSFLOW=${ROWSFLOW:-2400000}
ROWSDET=${ROWSDET:-1600000}
echo "### JOINT_RE job=$SLURM_JOB_ID node=$SLURMD_NODENAME seed=$SEED tag=$TAG ###"; nvidia-smi -L; date
echo "target=$TARGET  maxcase=$MAXCASE traincasemax=$TRAINCASEMAX rowsflow=$ROWSFLOW  EXTRA=$EXTRA"

stdbuf -oL -eL python -B -u scripts/train_joint_forward.py \
  --flow-catalogue "$FLOWCAT" \
  --target-npz "$TARGET" --decorrelate \
  --outdir "$OUTDIR" --tag "${TAG}_seed${SEED}" \
  --max-case "$MAXCASE" --train-case-max "$TRAINCASEMAX" \
  --det-max-case 100 --det-train-case-max 80 \
  --max-rows-flow "$ROWSFLOW" --max-rows-det "$ROWSDET" --max-rows-val 500000 \
  --delta 0.05 \
  --lam-r 50 --lam-s 50 --lam-d 1 \
  --epochs 60 --patience 14 --no-sensitivity --no-baselines \
  --batch-size 16384 --lr 7e-4 --seed "$SEED" \
  --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
  --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
  $EXTRA \
  || { echo "JOINT_RE FAILED seed=$SEED tag=$TAG"; exit 1; }
echo "### JOINT_RE_DONE job=$SLURM_JOB_ID seed=$SEED tag=$TAG ###"; date