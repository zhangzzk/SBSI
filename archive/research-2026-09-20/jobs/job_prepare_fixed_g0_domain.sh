#!/usr/bin/env bash
#SBATCH --job-name=sbsi_fg0prep
#SBATCH --partition=inter,cip
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v1
g0=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather
raw=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
"$python" -u -m scripts.prepare_fixed_g0_domain \
  --g0-catalogue "$g0" \
  --g05-catalogue "$g05" \
  --raw-simulation-root "$raw" \
  --output-root "$root/domain" \
  --seed 501 --validation-size 0.2
