#!/bin/bash
#SBATCH --job-name=harv_jrflowprod
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/harv_jrflowprod_%j.out

# Harvest the PRODUCTION-GUIDED joint model's per-object R_flow (mean-head central secant) over the
# constant cert catalogue (cases 40-139), 3-seed ensemble averaged.  Output npz ->
# eval_selection_robustness.py --rflow-override for eval B.  Does NOT modify certified m.
# Low mem: harvest STREAMS the catalogue in record-batch chunks; gpu:1 = any cip a40 MIG slice.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto

echo "### HARV_JRFLOWPROD job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
  --checkpoint $FP/forward_proto_c0-99_prod_seed421_joint.pt \
               $FP/forward_proto_c0-99_prod_seed422_joint.pt \
               $FP/forward_proto_c0-99_prod_seed423_joint.pt \
  --min-case 40 --chunk-batches 4 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_joint_prod_ens3_c40-139.npz \
  || { echo "HARV_JRFLOWPROD FAILED"; exit 1; }
echo "### HARV_JRFLOWPROD_DONE job=$SLURM_JOB_ID ###"; date
