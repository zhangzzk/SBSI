#!/bin/bash
#SBATCH --job-name=fwd_prodsz6
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_prodsz6_%j.out

# Goal-1 size-robustness iteration-3 (task #32): retrain the _prod joint flow with a FINER-SIZE target
# grid (6x6x5, size edges 0.1/0.177/0.241/0.315/0.412/0.592/1.5) instead of the coarse 6x3x5 (only 3
# size bins). The near-PSF residual after size-aware R_blend (size0.2-0.3 -8.6%, mag24-26xsize<0.3 -29%)
# is where R_flow is supervised by ~1 coarse size bin; finer bins should make R_flow more accurate at
# small size so R_blend absorbs a smaller/smoother residual. Grid is the NON-circular varshear
# isoblend_snc (constgold never read). Identical arch/lambda/recipe to _prod; new _prodsz6 tag.
# EXPERIMENTAL/additive; certified m untouched.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
TARGET=results/response_target_isoblend_snc_c0-99_6x6x5.npz
SEEDS=${SEEDS:-421 422 423}
echo "### FWD_PRODSZ6 job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date
echo "flow-catalogue=$FLOWCAT"; echo "target-npz=$TARGET (6 size bins, decorrelate=ON)"

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" \
    --target-npz "$TARGET" --decorrelate \
    --outdir "$OUTDIR" --tag "proto_c0-99_prodsz6_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 4 --n-size 3 --delta 0.05 \
    --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PRODSZ6 FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PRODSZ6_DONE job=$SLURM_JOB_ID ###"; date
