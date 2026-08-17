#!/bin/bash
#SBATCH --job-name=harv_sz10cg
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/harv_sz10cg_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
echo "### HARV_SZ10CG job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
  --checkpoint $FP/forward_proto_c0-99_sz10cg_seed421_joint.pt \
               $FP/forward_proto_c0-99_sz10cg_seed422_joint.pt \
               $FP/forward_proto_c0-99_sz10cg_seed423_joint.pt \
  --min-case 40 --chunk-batches 4 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_joint_sz10cg_ens3_c40-139.npz \
  || { echo "HARV_SZ10CG FAILED"; exit 1; }
echo "### HARV_SZ10CG_DONE job=$SLURM_JOB_ID ###"; date
