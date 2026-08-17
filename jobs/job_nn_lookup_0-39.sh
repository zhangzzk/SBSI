#!/bin/bash
#SBATCH --job-name=nnlkup
#SBATCH --time=00:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/nnlkup_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### NNLKUP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/build_nn_distance_lookup.py --cases $(seq -s ' ' 0 39) \
  --sign 0.02 --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/nn_dist_c0-39.feather
echo "### NNLKUP_DONE job=$SLURM_JOB_ID ###"; date
