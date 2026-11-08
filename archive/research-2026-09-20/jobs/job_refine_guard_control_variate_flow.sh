#!/usr/bin/env bash
#SBATCH --job-name=sbsi_guard_cv_refine
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=01:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/guard_control_variate_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/guard_control_variate_20260917_v1/logs/%x_%j.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
nvidia-smi -L
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.refine_guard_control_variate_flow \
  --domain-root "$cache/full_domain_true_mag26_guard_v1/domain" \
  --flow "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --reference-root "$cache/guard_control_variate_20260917_v1/audit_v2" \
  --output-root "$cache/guard_control_variate_20260917_v1/refinement"
