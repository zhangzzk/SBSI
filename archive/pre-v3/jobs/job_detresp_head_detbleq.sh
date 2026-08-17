#!/bin/bash
#SBATCH --job-name=detresp_dbeq
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/detresp_dbeq_%j.out

# TRACK 2 iteration validation (Goal 2): harvest the EQUAL-WEIGHT blend-supervised detection head's
# dp per object and run the blend-resolved gradient check vs btrue_detection_ngmix.npz. Compare to
# detresp_head_detblend.npz (count-weighted under-fit: <dp>=-0.45%, ISO~0) and detresp_head_blend.npz
# (mag-only baseline). INFERENCE only; never touches certified m.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
FP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto

echo "### DETRESP_DBEQ job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
stdbuf -oL -eL python -B -u scripts/harvest_det_response.py \
  --checkpoint $FP/forward_proto_c0-99_sz6detbleq_seed421_joint.pt \
               $FP/forward_proto_c0-99_sz6detbleq_seed422_joint.pt \
               $FP/forward_proto_c0-99_sz6detbleq_seed423_joint.pt \
  --min-case 40 --chunk-batches 8 \
  --out /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detresp_head_detbleq.npz \
  || { echo "DETRESP_DBEQ FAILED"; exit 1; }
echo "### DETRESP_DBEQ_DONE job=$SLURM_JOB_ID ###"; date
