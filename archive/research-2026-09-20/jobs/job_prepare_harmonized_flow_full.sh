#!/usr/bin/env bash
#SBATCH --job-name=sbsi_harmonized_fullprep
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
data=/project/ls-gruen/users/zekang.zhang
mapfile -t cases < <(seq 0 199)
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.prepare_harmonized_flow_domain \
  --measurement-root "$data/blendemu_runs/harmonized_secondary_20260917_v1" \
  --raw-root "$data/lsst_sims_fs2_25876" \
  --crowd-catalogue "$data/sbsi_caches/crowd_flux_conc_c0-199.feather" \
  --baseline-domain "$data/sbsi_caches/full_domain_true_mag26_guard_v1/domain" \
  --output-root "$data/sbsi_caches/harmonized_secondary_20260917_v1/domain" \
  --cases "${cases[@]}"
