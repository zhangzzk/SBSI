#!/bin/bash
#SBATCH --job-name=detresp_head
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/detresp_head_%j.out

# DIAGNOSTIC (Goal 2): harvest the joint-prod detection head's dp = dP(detect)/dgamma per object
# on the constant cert population and run the BLEND-RESOLVED finite-difference gradient check vs
# derisk/btrue_detection_ngmix.npz (isolated -7.5% vs closest-blend -4.4%). The head was supervised
# ONLY on b_true(mag) -> the blend split is an out-of-supervision generalization test.
# INFERENCE only; never touches certified m.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
echo "### DETRESP_HEAD job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_det_response.py \
  --checkpoint $FP/forward_proto_c0-99_prod_seed421_joint.pt \
               $FP/forward_proto_c0-99_prod_seed422_joint.pt \
               $FP/forward_proto_c0-99_prod_seed423_joint.pt \
  --min-case 40 --chunk-batches 8 \
  || { echo "DETRESP_HEAD FAILED"; exit 1; }
echo "### DETRESP_HEAD_DONE job=$SLURM_JOB_ID ###"; date
