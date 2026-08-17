#!/bin/bash
#SBATCH --job-name=rtheta_ro
#SBATCH --time=00:40:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/rtheta_ro_%j.out

# Gold-V2 Stage-2 gate (cont.121): read out the theta-pinned models' HELD-OUT measured mag/size
# shear response (b_size log spin-2 slope vs +0.490 target; b_mag vs +0.025), confirming the
# --lam-theta pin moved them while R_shape stays ~+0.71. No retraining: loads frozen checkpoints
# and pushes +/-delta ellipticity contexts through the mean head on val cases.
# FIREWALL: half-shear g0 flow leg only; constgold never touched.
# Env: CKPTS (space-separated tags, default the sw_base/th100/th400 sweep).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

CKPTS=${CKPTS:-"sw_base_seed421 sw_th100_seed421 sw_th400_seed421"}
echo "### RTHETA_RO job=$SLURM_JOB_ID node=$SLURMD_NODENAME ckpts=$CKPTS ###"; nvidia-smi -L; date

stdbuf -oL -eL python -B -u scripts/readout_rtheta.py \
  --ckpts $CKPTS \
  --target-npz results/response_target_isoblend_RAWfine_c0-99_6x9x5.npz \
  --theta-target-npz results/response_target_theta_coupling_c0-99_6x9x5.npz \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rtheta_readout.npz \
  || { echo "RTHETA_RO FAILED"; exit 1; }
echo "### RTHETA_RO_DONE job=$SLURM_JOB_ID ###"; date
