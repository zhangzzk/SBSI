#!/bin/bash
#SBATCH --job-name=harv_jrflow
#SBATCH --time=02:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/harv_jrflow_%j.out

# INFERENCE (experimental joint model): harvest the joint forward model's per-object R_flow
# (mean-head central secant) over the constant-shear cert catalogue (cases 40-139), 3-seed
# ensemble averaged. Output npz -> eval_selection_robustness.py --rflow-override for the
# head-to-head vs the certified tabular R_flow (R_blend kept SEPARATE, unchanged).
# Does NOT modify certified m or any certified artifact.
# cip full-a40 (cip-cl-nv01); do NOT set expandable_segments on cip vGPU slices.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto

echo "### HARV_JRFLOW job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
  --checkpoint $FP/forward_proto_c0-99_seed421_joint.pt \
               $FP/forward_proto_c0-99_seed422_joint.pt \
               $FP/forward_proto_c0-99_seed423_joint.pt \
  --min-case 40 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_joint_ens3_c40-139.npz \
  || { echo "HARV_JRFLOW FAILED"; exit 1; }
echo "### HARV_JRFLOW_DONE job=$SLURM_JOB_ID ###"; date
