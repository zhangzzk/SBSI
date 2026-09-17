#!/usr/bin/env bash
#SBATCH --job-name=sbsi_fulldomprep
#SBATCH --partition=inter,cip
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1
g0=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$root/logs"
cd "$repo"
"$python" -u -m scripts.prepare_full_domain_flow \
  --g0-catalogue "$g0" \
  --g05-catalogue "$g05" \
  --output-root "$root/domain" \
  --seed 501 --validation-size 0.2 \
  --truth-magnitude-max 26.0
