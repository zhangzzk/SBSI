#!/bin/bash
#SBATCH --job-name=harv_sz6eqw
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/harv_sz6eqw_%j.out

# TRACK 1 iteration harvest: per-object R_flow of the finer-size + EQUAL-WEIGHT joint model over the
# cert catalogue (cases 40-139), 3-seed ensemble. -> build_scene_rblend --rflow-override then
# eval_selection_robustness --rflow-override. Does NOT modify certified m.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto

echo "### HARV_SZ6EQW job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
  --checkpoint $FP/forward_proto_c0-99_sz6eqw_seed421_joint.pt \
               $FP/forward_proto_c0-99_sz6eqw_seed422_joint.pt \
               $FP/forward_proto_c0-99_sz6eqw_seed423_joint.pt \
  --min-case 40 --chunk-batches 4 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_joint_sz6eqw_ens3_c40-139.npz \
  || { echo "HARV_SZ6EQW FAILED"; exit 1; }
echo "### HARV_SZ6EQW_DONE job=$SLURM_JOB_ID ###"; date
