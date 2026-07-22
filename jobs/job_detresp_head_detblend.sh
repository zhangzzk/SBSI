#!/bin/bash
#SBATCH --job-name=detresp_detbl
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/detresp_detbl_%j.out

# TRACK 2 validation (Goal 2): harvest the BLEND-SUPERVISED detection head's dp=dP(detect)/dgamma
# per object on the constant cert population and run the BLEND-RESOLVED gradient check vs
# derisk/btrue_detection_ngmix.npz. Contrast with the mag-only-supervised baseline
# (detresp_head_blend.npz): the fix should make ISOLATED dp track b_true (-18%) and close blends
# stop being over-weighted. INFERENCE only; never touches certified m.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto

echo "### DETRESP_DETBL job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_det_response.py \
  --checkpoint $FP/forward_proto_c0-99_sz6detblend_seed421_joint.pt \
               $FP/forward_proto_c0-99_sz6detblend_seed422_joint.pt \
               $FP/forward_proto_c0-99_sz6detblend_seed423_joint.pt \
  --min-case 40 --chunk-batches 8 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detresp_head_detblend.npz \
  || { echo "DETRESP_DETBL FAILED"; exit 1; }
echo "### DETRESP_DETBL_DONE job=$SLURM_JOB_ID ###"; date
