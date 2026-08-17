#!/bin/bash
#SBATCH --job-name=harv_prodsz6
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/harv_prodsz6_%j.out

# Harvest the _prodsz6 (finer-size) joint model's per-object R_flow over the constant cert catalogue
# (cases 40-139), 3-seed ensemble averaged -> eval_selection_robustness.py --rflow-override. Does NOT
# modify certified m. (task #32 iteration-3)
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
echo "### HARV_PRODSZ6 job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
  --checkpoint $FP/forward_proto_c0-99_prodsz6_seed421_joint.pt \
               $FP/forward_proto_c0-99_prodsz6_seed422_joint.pt \
               $FP/forward_proto_c0-99_prodsz6_seed423_joint.pt \
  --min-case 40 --chunk-batches 4 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_joint_prodsz6_ens3_c40-139.npz \
  || { echo "HARV_PRODSZ6 FAILED"; exit 1; }
echo "### HARV_PRODSZ6_DONE job=$SLURM_JOB_ID ###"; date
