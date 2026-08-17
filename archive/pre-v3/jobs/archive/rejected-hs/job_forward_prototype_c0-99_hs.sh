#!/bin/bash
#SBATCH --job-name=fwd_proto99hs
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99hs_%j.out

# RETRAIN the joint forward model with the ISOLATED-INCLUDING half-sim FLOW catalogue (cont.58).
# Identical config to job_forward_prototype_c0-99.sh (the pairs-only run) EXCEPT --flow-catalogue
# now points at the rebuilt half-sim catalogue (22% isolated + 78% blend) and a new _hs tag, so
# the pairs-only checkpoints are preserved for a clean A/B.  --no-baselines: the harvest only
# needs the joint checkpoint (V3 no-trade-off already established in cont.55).  EXPERIMENTAL;
# additive-only; never wired into m = R_sim/(R_flow+R_blend)-1.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/self_response_halfsim_isoblend_cases0_99.feather
SEEDS=${SEEDS:-421 422 423}
echo "### FWD_PROTO99HS job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date
echo "flow-catalogue=$FLOWCAT"

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --flow-catalogue "$FLOWCAT" \
    --outdir "$OUTDIR" --tag "proto_c0-99_hs_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 4 --n-size 3 --delta 0.05 \
    --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity --no-baselines \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99HS FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done
echo "### FWD_PROTO99HS_DONE job=$SLURM_JOB_ID ###"; date
