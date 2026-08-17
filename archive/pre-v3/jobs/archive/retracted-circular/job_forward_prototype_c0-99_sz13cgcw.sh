#!/bin/bash
#SBATCH --job-name=fwd_proto99sz13cw
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99sz13cw_%j.out
# TRACK 1 iter #7 -- WEIGHTING BRACKET. The sz13cg residual is isolated-only (R_flow) and is a smooth
# OVER-STEEPENED size slope (iso add +3.6% at 0.5 -> -6.5% at 1.4, pivot ~1.0); blended majority is
# sub-percent (R_blend vindicated). --flow-equal-weight up-weights sparse extreme-size cells whose
# antithetic targets are noisy -> hypothesised over-steepening. This run is IDENTICAL to sz13cg but
# COUNT-weighted (NO --flow-equal-weight) on the SAME 6x13x5 constgold target (no rebuild) to bracket
# the slope. If count-weight flattens the iso slope toward zero without regressing small-size/GLOBAL ->
# the knob is found; if both weightings leave a ~1-1.5% iso slope -> flow-fit floor. FIREWALL: a-priori
# weighting choice, OOS by case, GLOBAL re-checked, R_blend rebuilt SEPARATELY, adoption owner-gated.
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
echo "### FWD_PROTO99SZ13CW job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' (COUNT-weighted) ###"; nvidia-smi -L; date
[ -f "$TARGET" ] || { echo "TARGET MISSING: $TARGET"; exit 1; }
for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" --target-npz "$TARGET" --decorrelate \
    --outdir "$OUTDIR" --tag "proto_c0-99_sz13cw_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 6 --n-size 13 --delta 0.05 --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99SZ13CW FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99SZ13CW_DONE job=$SLURM_JOB_ID ###"; date
