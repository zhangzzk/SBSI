#!/bin/bash
#SBATCH --job-name=selfresp_v
#SBATCH --time=01:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=12
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/selfresp_v_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### SELFRESP_V job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
for V in ngmix_ap7 ngmix_np7; do
  echo; echo "############################## VARIANT=$V ##############################"
  python -B -u scripts/compare_selfresp_sims.py --variant $V --hs-max-case 19 --cg-max-case 39 \
    --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selfresp_${V}.npz \
    || echo "VARIANT $V FAILED (continuing)"
done
echo "### SELFRESP_V_DONE job=$SLURM_JOB_ID ###"; date
