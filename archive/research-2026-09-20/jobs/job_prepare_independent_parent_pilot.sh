#!/usr/bin/env bash
#SBATCH --job-name=sbsi_independent_parent
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
data=/project/ls-gruen/users/zekang.zhang
root=$data/sbsi_caches/harmonized_secondary_20260917_v1
cd /home/z/Zekang.Zhang/SBSI
exec "$data/envs/py31/bin/python" -u -m scripts.prepare_independent_truth_parent \
  --raw-root "$data/lsst_sims_fs2_25876_constant_g005_c140-239" \
  --truth-audit "$root/existing_g005_truth.json" \
  --crowd-reference "$data/sbsi_caches/derisk/v22_g005_c140-239/crowd_flux_conc_c140-239.feather" \
  --cases 200 201 --output-root "$root/independent_parent_pilot"
