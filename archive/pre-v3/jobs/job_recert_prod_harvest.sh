#!/bin/bash
#SBATCH --job-name=recert_harv
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/recert_harv_%j.out

# RE-CERT (task #29, banking _prod): reproduce the PRODUCTION-recipe joint R_flow harvest from the
# 3 _prod checkpoints to a NEW _recert path (does NOT overwrite the banked canonical
# rflow_joint_prod_ens3_c40-139.npz). Determinism/reproducibility check before adopting _prod as the
# candidate certified R_flow. Identical to job_harvest_joint_rflow_prod.sh except --out. Firewall:
# harvests R_flow only; certified m untouched.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto

echo "### RECERT_HARV job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_joint_rflow.py \
  --checkpoint $FP/forward_proto_c0-99_prod_seed421_joint.pt \
               $FP/forward_proto_c0-99_prod_seed422_joint.pt \
               $FP/forward_proto_c0-99_prod_seed423_joint.pt \
  --min-case 40 --chunk-batches 4 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_joint_prod_recert_c40-139.npz \
  || { echo "RECERT_HARV FAILED"; exit 1; }
echo "### RECERT_HARV_DONE job=$SLURM_JOB_ID ###"; date
