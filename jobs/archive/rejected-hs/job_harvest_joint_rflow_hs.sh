#!/bin/bash
#SBATCH --job-name=harv_jrflowhs
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/harv_jrflowhs_%j.out

# Harvest the ISOLATED-INCLUDING joint model's per-object R_flow (mean-head central secant) over
# the constant cert catalogue (cases 40-139), 3-seed ensemble averaged.  Output npz ->
# eval_selection_robustness.py --rflow-override for eval B (isolated cells expected to close).
# Does NOT modify certified m.  gpu:1 = any cip a40 MIG slice (8/16/24gb ok; the cip a40s are
# MIG-partitioned so there is no plain 'a40' type to request).  The cip GPU hosts are small
# (12 CPU / 25-40G RAM), so mem stays LOW: harvest STREAMS the catalogue in record-batch chunks
# (ipc.open_file, not a full load) -> ~20G is ample.  --chunk-batches 4 keeps each GPU forward
# small enough for an 8gb slice.  Do NOT set expandable_segments on cip vGPU.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto

echo "### HARV_JRFLOWHS job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
  --checkpoint $FP/forward_proto_c0-99_hs_seed421_joint.pt \
               $FP/forward_proto_c0-99_hs_seed422_joint.pt \
               $FP/forward_proto_c0-99_hs_seed423_joint.pt \
  --min-case 40 --chunk-batches 4 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_joint_hs_ens3_c40-139.npz \
  || { echo "HARV_JRFLOWHS FAILED"; exit 1; }
echo "### HARV_JRFLOWHS_DONE job=$SLURM_JOB_ID ###"; date
