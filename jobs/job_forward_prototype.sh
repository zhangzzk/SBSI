#!/bin/bash
#SBATCH --job-name=fwd_proto
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto_%j.out

# EXPERIMENTAL feasibility gate (NOT certified): trains the prototype unified forward model
# (sbs_shear.forward_model.SetConditionedForwardModel) with the 4-term separable-piece loss and
# reports V1-V4 on held-out cases (train cases 0-15, validate 16-19). Additive-only: never wired
# into m = R_sim/(R_flow+R_blend)-1.  inter-partition full GPU (a40/a100), so expandable_segments
# is safe (unlike the cip vGPU slices).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI

echo "### FWD_PROTO job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date

stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
  --outdir /project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto \
  --tag proto \
  --max-case 20 --train-case-max 16 \
  --max-rows-flow 1000000 --max-rows-det 1000000 --max-rows-val 400000 \
  --n-flux 4 --n-size 3 --delta 0.05 \
  --lam-r 50 --lam-s 50 --lam-d 1 \
  --epochs 50 --patience 12 --sens-epochs 25 \
  --batch-size 16384 --lr 7e-4 --seed 421 \
  --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
  --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
  || { echo "FWD_PROTO FAILED"; exit 1; }

echo "### FWD_PROTO_ALL_DONE job=$SLURM_JOB_ID ###"; date
