#!/usr/bin/env bash
#SBATCH --job-name=sbsi_cgparent_prep
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_20260917_v1/logs/%x_%j.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
mapfile -t cases < <(seq 40 119)
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.prepare_constgold_truth_parent \
  --classifier-prepared "$cache/fixed_g0_m258_r060_v1/classifier_prepared" \
  --domain-root "$cache/full_domain_true_mag26_guard_v1/domain" \
  --output-root "$cache/constgold_truth_parent_20260917_v1/parent_v2" \
  --truth-pattern '/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather' \
  --cases "${cases[@]}"
