#!/bin/bash
#SBATCH --job-name=fwd_proto99sz6eq
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99sz6eq_%j.out

# TRACK 1 iteration (owner-greenlit, CERTIFIED path): finer-size 6x6x5 response target PLUS
# --flow-equal-weight. The sz6 retrain (job 15155245) fixed small-size but UNDER-fit the rare
# high-response LARGE-size cells (size cuts additive residual +4..+6%), because binned_mse is
# COUNT-weighted and large galaxies are rare -> their high-response bins are drowned out (same flaw
# as the detection head). Equal-per-cell weighting makes the flow fit large-size as hard as small.
# Detection supervision UNCHANGED (mag-only) so the R_flow change is attributable to the size grid +
# weighting. FIREWALL: a-priori choice (not tuned on |m|), OOS by case; GLOBAL m re-checked in the
# acceptance eval before any adoption; R_blend rebuilt SEPARATELY against this R_flow.
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
echo "### FWD_PROTO99SZ6EQ job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date
echo "target=$TARGET (6x6x5, decorrelate=ON, flow-equal-weight=ON, det=mag-only)"

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" \
    --target-npz "$TARGET" --decorrelate --flow-equal-weight \
    --outdir "$OUTDIR" --tag "proto_c0-99_sz6eqw_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 6 --n-size 6 --delta 0.05 \
    --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99SZ6EQ FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99SZ6EQ_DONE job=$SLURM_JOB_ID ###"; date
