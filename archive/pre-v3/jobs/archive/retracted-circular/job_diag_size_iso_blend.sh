#!/bin/bash
#SBATCH --job-name=diag_szib
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_szib_%j.out
# READ-ONLY: decompose the residual size1.0-1.5 / size0.5-1.0 bias into ISOLATED (R_flow) vs BLENDED
# (size-blind scene R_blend) using the sz13cg overrides. Locates whether the stuck -1.5% is a flow
# under-resolution (keep refining size target) or a size-blind-R_blend floor (owner-gated design).
set -e
source /software/opt/focal/x86_64/python/3.10-2022.08/etc/profile.d/conda.sh
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
echo "### DIAG_SZIB job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_size_iso_blend.py \
  --rflow-override  $D/rflow_joint_sz13cg_ens3_c40-139.npz \
  --rblend-override $D/rblend_scene_sz13cgrflow_c40-139.npz \
  --tag sz13cg \
  || { echo "DIAG_SZIB FAILED"; exit 1; }
echo "### DIAG_SZIB_DONE job=$SLURM_JOB_ID ###"; date
