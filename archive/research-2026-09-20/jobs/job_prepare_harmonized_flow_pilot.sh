#!/usr/bin/env bash
#SBATCH --job-name=sbsi_harmonized_prep
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
data=/project/ls-gruen/users/zekang.zhang
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.prepare_harmonized_flow_domain \
  --measurement-root "$data/blendemu_runs/common_optimizer_secondary_pilot_20260917_v1" \
  --raw-root "$data/lsst_sims_fs2_25876" \
  --crowd-catalogue "$data/sbsi_caches/crowd_flux_conc_c0-199.feather" \
  --baseline-domain "$data/sbsi_caches/full_domain_true_mag26_guard_v1/domain" \
  --output-root "$data/sbsi_caches/output_conditioned_response_20260917_v1/harmonized_pilot_domain" \
  --cases 0 40 41
