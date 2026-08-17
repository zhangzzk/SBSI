#!/bin/bash
#SBATCH --job-name=fwd_proto99sz6cap
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99sz6cap_%j.out

# TRACK 1 iteration #3 (owner-greenlit, CERTIFIED path): attack the LARGE-size response-tail
# COMPRESSION diagnosed from sz6eqw.  The flow UNDER-FITS a CORRECT target (grid iso Rsim(size5)
# = +0.916 == dump truth ~0.91; bright-large cells target up to +1.55) -- its mean-head saturates
# ~0.83, leaving size cuts at +6..+8% additive.  Equal-weighting the CELLS (sz6eqw) did not
# decompress it, so the remaining levers are RESPONSE PRESSURE (lam_r 50->150) and MEAN-HEAD
# CAPACITY (mean-hidden 64->192): the per-cell response penalty is secondary to the NLL on a shared
# trunk, so the rare high-response cells are conceded.  FIREWALL: a-priori architectural choice (NOT
# tuned on |m|), OOS by case, --flow-equal-weight kept, GLOBAL m re-checked in acceptance before any
# adoption; R_blend rebuilt SEPARATELY against this R_flow.  Detection supervision UNCHANGED.
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
echo "### FWD_PROTO99SZ6CAP job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date
echo "target=$TARGET (6x6x5, decorrelate=ON, flow-equal-weight=ON, lam_r=150, mean-hidden=192, det=mag-only)"

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" \
    --target-npz "$TARGET" --decorrelate --flow-equal-weight \
    --outdir "$OUTDIR" --tag "proto_c0-99_sz6cap_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 6 --n-size 6 --delta 0.05 \
    --lam-r 150 --lam-s 50 --lam-d 1 \
    --epochs 60 --patience 14 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 192 --det-hidden 128 \
    || { echo "FWD_PROTO99SZ6CAP FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99SZ6CAP_DONE job=$SLURM_JOB_ID ###"; date
