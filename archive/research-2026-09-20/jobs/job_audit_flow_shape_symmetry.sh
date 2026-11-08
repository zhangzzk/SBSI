#!/usr/bin/env bash
# Exact log-density diagnostic on held-out g0 data; CPU only, no model fitting.
#SBATCH --job-name=sbsi_shape_symmetry
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
cache=$data/sbsi_caches
root=$cache/harmonized_secondary_20260917_v1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
exec "$data/envs/py31/bin/python" -u -m scripts.audit_flow_shape_symmetry \
  --domain "$root/domain" --flow "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --flow "$root/refinement/selected.pt" --noise-catalogue "$data/lsst_sims_fs2_25876/noise.csv" \
  --rows-per-case 4096 --output "$root/shape_symmetry.json"
