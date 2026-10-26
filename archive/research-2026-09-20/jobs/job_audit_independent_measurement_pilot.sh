#!/usr/bin/env bash
#SBATCH --job-name=sbsi_independent_measurement
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
data=/project/ls-gruen/users/zekang.zhang
cd /home/z/Zekang.Zhang/SBSI
exec "$data/envs/py31/bin/python" -u -m scripts.audit_independent_measurement_pilot \
  --pilot-root "$data/blendemu_runs/independent_g005_subpixel_pilot_20260918_v1" \
  --raw-root "$data/lsst_sims_fs2_25876_constant_g005_c140-239" \
  --cases 200 201 \
  --output "$data/sbsi_caches/harmonized_secondary_20260917_v1/independent_measurement_pilot.json"
