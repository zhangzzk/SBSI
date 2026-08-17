#!/bin/bash
#SBATCH --job-name=diag_oos
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_oos_%j.out
set -e
source /software/opt/focal/x86_64/python/3.10-2022.08/etc/profile.d/conda.sh
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
echo "### DIAG_OOS job=$SLURM_JOB_ID ###"; date
stdbuf -oL -eL python -B -u scripts/diag_oos_case_split.py \
  --rflow-override  $D/rflow_joint_sz13hi_ens3_c40-139.npz \
  --rblend-override $D/rblend_scene_sz13hirflow_c40-139.npz \
  || { echo "DIAG_OOS FAILED"; exit 1; }
echo "### DIAG_OOS_DONE job=$SLURM_JOB_ID ###"; date
