#!/bin/bash
#SBATCH --job-name=btrue_ngmix
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/btrue_ngmix_%j.out

# DIAGNOSTIC (Goal 2): rebuild the SHAPE-CORRELATED detection selection-bias b_true on the
# CURRENT ngmix two-leg parents (undetected retained), replacing the deprecated-SExtractor
# population in derisk_btrue_detection.py. b_true = [<s_par|det> - <s_par|parent>]/g with
# s_par = S_gamma(e_intrinsic).ghat. Never wired into certified m.
set -e
source /software/opt/focal/x86_64/python/3.10-2022.08/etc/profile.d/conda.sh
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
DR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
echo "### BTRUE_NGMIX job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u $DR/derisk_btrue_detection.py \
  --g05 $CAT/det_meas_ngmix_g0.05_val.feather \
  --g00 $CAT/det_meas_ngmix_g0.0_train.feather \
  --out-prefix $DR/btrue_detection_ngmix \
  || { echo "BTRUE_NGMIX FAILED"; exit 1; }
echo "### BTRUE_NGMIX_DONE job=$SLURM_JOB_ID ###"; date
