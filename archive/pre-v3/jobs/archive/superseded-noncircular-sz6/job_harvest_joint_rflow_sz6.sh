#!/bin/bash
#SBATCH --job-name=harv_jrflowsz6
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/harv_jrflowsz6_%j.out

# TRACK 1 harvest: per-object R_flow (mean-head central secant) of the FINER-SIZE (6x6x5) joint
# model over the constant cert catalogue (cases 40-139), 3-seed ensemble. Output npz ->
# build_scene_rblend.py --rflow-override (rebuild scene R_blend against this R_flow) and then
# eval_selection_robustness.py --rflow-override. Does NOT modify certified m.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto

echo "### HARV_JRFLOWSZ6 job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
  --checkpoint $FP/forward_proto_c0-99_sz6_seed421_joint.pt \
               $FP/forward_proto_c0-99_sz6_seed422_joint.pt \
               $FP/forward_proto_c0-99_sz6_seed423_joint.pt \
  --min-case 40 --chunk-batches 4 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_joint_sz6_ens3_c40-139.npz \
  || { echo "HARV_JRFLOWSZ6 FAILED"; exit 1; }
echo "### HARV_JRFLOWSZ6_DONE job=$SLURM_JOB_ID ###"; date
