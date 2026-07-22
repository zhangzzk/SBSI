#!/bin/bash
#SBATCH --job-name=fwd_proto99sz13cg
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99sz13cg_%j.out
# TRACK 1 iter #6: metric-consistent constgold target with LARGE-SIZE-RESOLVED edges (6x13x5) to close
# the within-large-bin size gradient residual (size1.0-1.5 -1.99% after sz10cg). Edges come from the npz.
# --flow-equal-weight, detection supervision UNCHANGED. FIREWALL: a-priori, OOS, GLOBAL re-checked in
# acceptance, R_blend rebuilt SEPARATELY, adoption as certified = owner.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
TARGET=results/response_target_constgold_c0-99_6x13x5.npz
SEEDS=${SEEDS:-421 422 423}
echo "### FWD_PROTO99SZ13CG job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date
[ -f "$TARGET" ] || { echo "TARGET MISSING: $TARGET"; exit 1; }
for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" --target-npz "$TARGET" --decorrelate --flow-equal-weight \
    --outdir "$OUTDIR" --tag "proto_c0-99_sz13cg_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 6 --n-size 13 --delta 0.05 --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99SZ13CG FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99SZ13CG_DONE job=$SLURM_JOB_ID ###"; date
